"""
EnvIT - the BASE PAPER model, re-implemented faithfully (Gao et al., IEEE TNSE 2026) so that it can be trained and compared
with the proposed EGT on exactly the same data and split.

  1. Environmental aggregation : GAT (Eq. 1) on the account-account graphs of the meta-paths A-D-A, A-I-A, A-L-A (here:
                                 device / network / address); h^a = sum over meta-paths of the last-layer outputs
  2. Environmental difference  : GraphSAGE-mean (Eq. 2) on the bipartite account-entity sub-graphs (one per relation, summed),
                                 then GraphSAGE-mean on the account graph linking accounts with the same personal ID
  3. Behaviour                 : T-LSTM (Eq. 3-7) with g(dt) = 1/log(dt+e); behaviour vector = SUM of the hidden states
  4. Attentive fusion          : scaled dot-product self-attention over [h^d, h^a, h^b] (Eq. 8), rows summed
  5. Detection layer           : fully connected + softmax;  cross-entropy on labelled accounts
"""
import math
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig, RELATIONS
from .model import seg_softmax, BehaviourEncoder


class GATLayer(nn.Module):
    """alpha_ij = softmax_j( LeakyReLU( a^T [W h_i || W h_j] ) ),  h_i' = ReLU( sum_j alpha_ij W h_j )  (multi-head, concatenated)."""

    def __init__(self, d, heads):
        super().__init__()
        self.h, self.dh = heads, d // heads
        self.W = nn.Linear(d, d, bias=False)
        self.a_dst = nn.Parameter(torch.randn(heads, self.dh) * 0.1)
        self.a_src = nn.Parameter(torch.randn(heads, self.dh) * 0.1)

    def forward(self, h, src, dst):
        n = h.shape[0]
        z = self.W(h).view(n, self.h, self.dh)
        e = F.leaky_relu((z[dst] * self.a_dst).sum(-1) + (z[src] * self.a_src).sum(-1), 0.2)     # a^T[z_i || z_j]
        alpha = seg_softmax(e, dst, n)
        out = torch.zeros(n, self.h, self.dh, dtype=z.dtype).index_add_(0, dst, alpha.unsqueeze(-1) * z[src])
        return F.relu(out.reshape(n, -1))


def _mean_agg(x_src, src, dst, n_dst):
    s = torch.zeros(n_dst, x_src.shape[1], dtype=x_src.dtype).index_add_(0, dst, x_src[src])
    c = torch.zeros(n_dst, dtype=x_src.dtype).index_add_(0, dst, torch.ones(len(src), dtype=x_src.dtype))
    return s / c.clamp(min=1).unsqueeze(1)


def metapath_edges(src, dst, n_acc, n_ent):
    """Account-account edges of the meta-path A-X-A  (two accounts that use the same entity)."""
    if len(src) == 0:
        return torch.zeros(0, dtype=torch.long), torch.zeros(0, dtype=torch.long)
    S = sp.csr_matrix((np.ones(len(src), np.float32), (src.numpy(), dst.numpy())), shape=(n_acc, max(n_ent, 1)))
    P = (S @ S.T).tocoo()
    m = P.row != P.col
    return torch.from_numpy(P.col[m].astype(np.int64)), torch.from_numpy(P.row[m].astype(np.int64))     # neighbour -> target


class EnvIT(nn.Module):
    def __init__(self, cfg: ModelConfig, mode="full"):
        super().__init__()
        self.cfg, self.mode = cfg, mode          # mode: full | graph (EnvIT-Graph, no behaviour) | beh (EnvIT-Beh, behaviour only)
        d, H, R = cfg.hidden, cfg.heads, cfg.n_rel
        self.stat = nn.Sequential(nn.Linear(cfg.n_stats, d), nn.ReLU())
        self.ent_type = nn.Parameter(torch.randn(R, d) * 0.1)
        self.ent_deg = nn.ModuleList([nn.Linear(2, d) for _ in range(R)])
        self.gat = nn.ModuleList([nn.ModuleList([GATLayer(d, H) for _ in range(cfg.env_layers)]) for _ in range(R)])
        self.sage_a = nn.ModuleList([nn.ModuleList([nn.Linear(d, d) for _ in range(cfg.env_layers)]) for _ in range(R)])
        self.sage_acc = nn.ModuleList([nn.Linear(d, d) for _ in range(cfg.diff_layers)])
        self.beh = BehaviourEncoder(cfg.seq_dim, d, cfg.dropout, pool="sum")
        self.q, self.k, self.v = nn.Linear(d, d), nn.Linear(d, d), nn.Linear(d, d)
        self.drop = nn.Dropout(cfg.dropout)
        self.out = nn.Linear(d, 2)

    def forward(self, G, explain=False):
        cfg, S = self.cfg, G["n_seeds"]
        if self.mode == "beh":
            rows = G["seq_row"]; mask = rows >= 0; r0 = rows.clamp(min=0)
            h_b, _ = self.beh(G["ev_feat"][r0] * mask.unsqueeze(-1), G["ev_dt"][r0] * mask, mask)
            return self.out(self.drop(h_b))
        x = self.stat(G["stats"])
        M = x.shape[0]
        ents0 = [self.ent_type[r] + F.relu(self.ent_deg[r](G["ent_deg"][r])) for r in range(cfg.n_rel)]
        # 1) environmental aggregation: GAT on meta-path graphs, summed over meta-paths
        h_a = torch.zeros_like(x)
        for r in range(cfg.n_rel):
            ed = G["edges"][r]
            ms, md = metapath_edges(ed["src"], ed["dst"], M, ents0[r].shape[0])
            loops = torch.arange(M)
            src, dst = torch.cat([ms, loops]), torch.cat([md, loops])
            h = x
            for layer in self.gat[r]:
                h = self.drop(layer(h, src, dst))
            h_a = h_a + h
        # 2) environmental difference: GraphSAGE on bipartite sub-graphs, then on the same-ID account graph
        h_e = torch.zeros_like(x)
        for r in range(cfg.n_rel):
            ed = G["edges"][r]; e_src, e_dst = ed["src"], ed["dst"]
            he = ents0[r]                                                                   # entity state  h^0_j = x_j
            for k in range(cfg.env_layers):
                ha = self.drop(F.relu(self.sage_a[r][k](_mean_agg(he, e_dst, e_src, M))))   # Eq.(2): h_i^k = ReLU(W MEAN_j h_j^{k-1})
                he = _mean_agg(ha, e_src, e_dst, he.shape[0])                               # entity state = mean of its accounts
            h_e = h_e + ha
        keep = G["sib_src"] != G["sib_dst"]
        s_src, s_dst = G["sib_src"][keep], G["sib_dst"][keep]
        h_d = h_e
        for lin in self.sage_acc:
            h_d = self.drop(F.relu(lin(_mean_agg(h_d, s_src, s_dst, M))))
        if self.mode == "graph":
            Hm = torch.stack([h_d[:S], h_a[:S]], 1)
            att = torch.softmax(self.q(Hm) @ self.k(Hm).transpose(1, 2) / math.sqrt(Hm.shape[-1]), -1)
            return self.out(self.drop((att @ self.v(Hm)).sum(1)))
        # 3) behaviour (seeds only): T-LSTM, sum of hidden states
        rows = G["seq_row"]; mask = rows >= 0; r0 = rows.clamp(min=0)
        h_b, _ = self.beh(G["ev_feat"][r0] * mask.unsqueeze(-1), G["ev_dt"][r0] * mask, mask)
        # 4) attentive fusion: scaled dot-product self-attention over [h_d, h_a, h_b], rows summed
        Hm = torch.stack([h_d[:S], h_a[:S], h_b], 1)
        att = torch.softmax(self.q(Hm) @ self.k(Hm).transpose(1, 2) / math.sqrt(Hm.shape[-1]), -1)
        fused = (att @ self.v(Hm)).sum(1)
        # 5) detection layer
        return self.out(self.drop(fused))
