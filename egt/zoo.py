"""
Comparison models for the results chapter - plain PyTorch, same inputs, same sampler, same training loop as EGT.

  GAT       Velickovic et al. 2018     multi-head attention on the account graph (union of the A-D-A, A-I-A, A-L-A, same-ID edges)
  SAGE      Hamilton et al. 2017       GraphSAGE, mean aggregator, concatenation with the node's own state
  R-GCN     Schlichtkrull et al. 2018  one weight matrix per relation (device / network / address / same-ID) + self weight
  HAN       Wang et al. 2019           node-level GAT per meta-path, then semantic (meta-path level) attention
  HACUD     Hu et al. 2019             hierarchical attention on the attributed heterogeneous graph: account -> entity -> account
                                       (node-level attention over typed neighbours, then type-level attention)
  SOBT      sequence-only Transformer on the event sequence (no graph). The original SOBT definition could NOT be verified
            online; this is a behaviour-sequence Transformer baseline of that role - replace if your paper defines it otherwise.
  SEC-GFD   Xu et al. AAAI 2024        spectral graph fraud detection: low-pass / high-pass / band-pass filter bank with node-wise
                                       adaptive gating (simplified re-implementation of the filter-bank idea)
  EnvIT-Graph / EnvIT-Beh  the base paper's model with only its graph part / only its behaviour part (see envit.py)

All of them are SIMPLIFIED re-implementations written for this project, not the authors' code.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig
from .model import seg_softmax
from .envit import GATLayer, _mean_agg, metapath_edges


def account_graphs(G):
    """A-X-A meta-path edges for the 3 entity types and the same-personal-ID edges of a sampled sub-graph."""
    M = G["stats"].shape[0]
    paths = [metapath_edges(ed["src"], ed["dst"], M, G["ent_deg"][r].shape[0]) for r, ed in enumerate(G["edges"])]
    keep = G["sib_src"] != G["sib_dst"]
    return M, paths, (G["sib_src"][keep], G["sib_dst"][keep])


def union_edges(M, paths, sib, loops=True):
    src = torch.cat([p[0] for p in paths] + [sib[0]])
    dst = torch.cat([p[1] for p in paths] + [sib[1]])
    if len(src):
        key = torch.unique(src * M + dst)
        src, dst = key // M, key % M
    if loops:
        ar = torch.arange(M)
        src, dst = torch.cat([src, ar]), torch.cat([dst, ar])
    return src, dst


class _Base(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.hidden
        self.stat = nn.Sequential(nn.Linear(cfg.n_stats, d), nn.ReLU())
        self.drop = nn.Dropout(cfg.dropout)
        self.out = nn.Linear(d, 2)

    def readout(self, h, S):
        return self.out(self.drop(h[:S]))


class GATNet(_Base):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.layers = nn.ModuleList([GATLayer(cfg.hidden, cfg.heads) for _ in range(cfg.env_layers)])

    def forward(self, G, explain=False):
        M, paths, sib = account_graphs(G)
        src, dst = union_edges(M, paths, sib)
        h = self.stat(G["stats"])
        for l in self.layers:
            h = self.drop(l(h, src, dst))
        return self.readout(h, G["n_seeds"])


class SAGENet(_Base):
    def __init__(self, cfg):
        super().__init__(cfg)
        d = cfg.hidden
        self.lin = nn.ModuleList([nn.Linear(2 * d, d) for _ in range(cfg.env_layers)])

    def forward(self, G, explain=False):
        M, paths, sib = account_graphs(G)
        src, dst = union_edges(M, paths, sib, loops=False)
        h = self.stat(G["stats"])
        for lin in self.lin:
            h = self.drop(F.relu(lin(torch.cat([h, _mean_agg(h, src, dst, M)], 1))))
        return self.readout(h, G["n_seeds"])


class RGCNNet(_Base):
    def __init__(self, cfg):
        super().__init__(cfg)
        d, R = cfg.hidden, cfg.n_rel + 1
        self.self_w = nn.ModuleList([nn.Linear(d, d) for _ in range(cfg.env_layers)])
        self.rel_w = nn.ModuleList([nn.ModuleList([nn.Linear(d, d, bias=False) for _ in range(R)]) for _ in range(cfg.env_layers)])

    def forward(self, G, explain=False):
        M, paths, sib = account_graphs(G)
        rels = list(paths) + [sib]
        h = self.stat(G["stats"])
        for sw, rw in zip(self.self_w, self.rel_w):
            z = sw(h)
            for (s, d_), W in zip(rels, rw):
                z = z + W(_mean_agg(h, s, d_, M))
            h = self.drop(F.relu(z))
        return self.readout(h, G["n_seeds"])


class HANNet(_Base):
    def __init__(self, cfg):
        super().__init__(cfg)
        d, P = cfg.hidden, cfg.n_rel
        self.gat = nn.ModuleList([nn.ModuleList([GATLayer(d, cfg.heads) for _ in range(cfg.env_layers)]) for _ in range(P)])
        self.sw, self.sq = nn.Linear(d, d), nn.Linear(d, 1, bias=False)

    def forward(self, G, explain=False):
        M, paths, _ = account_graphs(G)
        x = self.stat(G["stats"])
        loops = torch.arange(M)
        zs = []
        for p, (s, d_) in enumerate(paths):
            h = x
            for l in self.gat[p]:
                h = self.drop(l(h, torch.cat([s, loops]), torch.cat([d_, loops])))
            zs.append(h)
        Z = torch.stack(zs, 1)                                             # [M,P,d]
        beta = torch.softmax(self.sq(torch.tanh(self.sw(Z))).squeeze(-1).mean(0), 0)       # semantic attention (meta-path level)
        h = (beta.view(1, -1, 1) * Z).sum(1)
        return self.readout(h, G["n_seeds"])


class HACUDLayer(nn.Module):
    def __init__(self, d, R, drop):
        super().__init__()
        self.wq = nn.Linear(d, d, bias=False)
        self.we = nn.ModuleList([nn.Linear(d, d, bias=False) for _ in range(R)])
        self.up = nn.ModuleList([nn.Linear(d, d) for _ in range(R)])
        self.a_i = nn.Parameter(torch.randn(R, d) * 0.1)
        self.a_e = nn.Parameter(torch.randn(R, d) * 0.1)
        self.tw, self.tq = nn.Linear(d, d), nn.Linear(d, 1, bias=False)
        self.mix = nn.Linear(2 * d, d)
        self.drop = nn.Dropout(drop)

    def forward(self, h, ents, edges, has_rel):
        M, ms, new = h.shape[0], [], []
        zi = self.wq(h)
        for r, (e, (s, t)) in enumerate(zip(ents, edges)):
            e = e + F.relu(self.up[r](_mean_agg(h, s, t, e.shape[0])))           # node-level step 1: entity <- its accounts
            ze = self.we[r](e)
            sc = F.leaky_relu((zi[s] * self.a_i[r]).sum(-1) + (ze[t] * self.a_e[r]).sum(-1), 0.2).unsqueeze(1)
            al = seg_softmax(sc, s, M)                                           # node-level step 2: account attends over its entities
            ms.append(torch.zeros(M, h.shape[1], dtype=h.dtype).index_add_(0, s, al * ze[t]))
            new.append(e)
        Z = torch.stack(ms, 1)
        lg = self.tq(torch.tanh(self.tw(Z))).squeeze(-1).masked_fill(~has_rel, -1e9)              # type-level attention
        beta = torch.softmax(lg, 1) * has_rel.any(1, keepdim=True).float()
        agg = (beta.unsqueeze(-1) * Z).sum(1)
        return self.drop(F.relu(self.mix(torch.cat([h, agg], 1)))), new


class HACUDNet(_Base):
    def __init__(self, cfg):
        super().__init__(cfg)
        d, R = cfg.hidden, cfg.n_rel
        self.ent_type = nn.Parameter(torch.randn(R, d) * 0.1)
        self.ent_deg = nn.ModuleList([nn.Linear(2, d) for _ in range(R)])
        self.layers = nn.ModuleList([HACUDLayer(d, R, cfg.dropout) for _ in range(cfg.env_layers)])

    def forward(self, G, explain=False):
        h = self.stat(G["stats"])
        ents = [self.ent_type[r] + F.relu(self.ent_deg[r](G["ent_deg"][r])) for r in range(self.cfg.n_rel)]
        edges = [(ed["src"], ed["dst"]) for ed in G["edges"]]
        for l in self.layers:
            h, ents = l(h, ents, edges, G["has_rel"])
        return self.readout(h, G["n_seeds"])


class SOBTNet(nn.Module):
    """Behaviour-sequence Transformer: event features + time-gap encoding + position -> Transformer encoder -> mean pool."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        d = cfg.hidden
        self.inp, self.dt = nn.Linear(cfg.seq_dim, d), nn.Linear(1, d)
        self.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, cfg.heads, 2 * d, cfg.dropout, batch_first=True, activation="gelu"),
                                         num_layers=cfg.env_layers, enable_nested_tensor=False)
        self.drop, self.out = nn.Dropout(cfg.dropout), nn.Linear(d, 2)

    def forward(self, G, explain=False):
        rows = G["seq_row"]; mask = rows >= 0; r0 = rows.clamp(min=0)
        x = G["ev_feat"][r0] * mask.unsqueeze(-1)
        dt = torch.log1p(G["ev_dt"][r0] * mask).unsqueeze(-1)
        B, T, _ = x.shape
        pos = torch.arange(T).float().unsqueeze(1)
        div = torch.exp(torch.arange(0, self.cfg.hidden, 2).float() * (-math.log(10000.0) / self.cfg.hidden))
        pe = torch.zeros(T, self.cfg.hidden); pe[:, 0::2] = torch.sin(pos * div); pe[:, 1::2] = torch.cos(pos * div)
        z = F.gelu(self.inp(x)) + self.dt(dt) + pe
        pad = ~mask
        pad[:, 0] = False                                                    # never mask a whole row
        z = self.enc(self.drop(z), src_key_padding_mask=pad)
        m = mask.unsqueeze(-1).float()
        pooled = (z * m).sum(1) / m.sum(1).clamp(min=1)
        return self.out(self.drop(pooled))


class SpecLayer(nn.Module):
    def __init__(self, d, drop):
        super().__init__()
        self.W = nn.ModuleList([nn.Linear(d, d) for _ in range(4)])
        self.gate = nn.Linear(d, 4)
        self.drop = nn.Dropout(drop)

    def forward(self, h, prop):
        low = prop(h); high = h - low; band = low - prop(low)               # all-pass, low-pass, high-pass, band-pass
        fs = [h, low, high, band]
        g = torch.softmax(self.gate(h), 1)
        z = sum(g[:, k:k + 1] * self.W[k](f) for k, f in enumerate(fs))
        return self.drop(F.relu(z))


class SECGFDNet(_Base):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.layers = nn.ModuleList([SpecLayer(cfg.hidden, cfg.dropout) for _ in range(cfg.env_layers)])

    def forward(self, G, explain=False):
        M, paths, sib = account_graphs(G)
        src, dst = union_edges(M, paths, sib, loops=False)
        deg = torch.zeros(M).index_add_(0, dst, torch.ones(len(dst))).clamp(min=1)
        w = (deg[src] * deg[dst]).rsqrt()                                    # symmetric normalisation D^-1/2 A D^-1/2

        def prop(x):
            return torch.zeros_like(x).index_add_(0, dst, x[src] * w.unsqueeze(1))
        h = self.stat(G["stats"])
        for l in self.layers:
            h = l(h, prop)
        return self.readout(h, G["n_seeds"])


ZOO = {"gat": GATNet, "sage": SAGENet, "rgcn": RGCNNet, "han": HANNet, "hacud": HACUDNet, "sobt": SOBTNet, "secgfd": SECGFDNet}
