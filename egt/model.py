"""
Explainable Graph Transformer (EGT) - pure PyTorch (no DGL / PyG needed).

Five modules (base paper EnvIT, upgraded from GAT/GraphSAGE/LSTM to Transformer attention):
  1. Environment-Aggregation Extractor  heterogeneous graph transformer on account<->{device, ip, address}
                                         (account->entity->account = the A-D-A / A-I-A / A-L-A meta-paths)
                                         + relation-level semantic attention
  2. Environment-Difference Extractor   graph transformer over accounts that share a personal ID; the
                                         attention is conditioned on edge features that encode how
                                         DIFFERENT the sibling environments are (Jaccard overlap, ...)
  3. Behaviour Extractor                Time-aware LSTM (T-LSTM) + attention pooling over events
  4. Attentive Feature Fusion           multi-head self-attention over {profile, aggregation, difference,
                                         behaviour} tokens + token-attention pooling
  5. Detection layer                    MLP -> 2 logits

EVERY attention map is returned on request -> used by explain.py for human-readable explanations.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig, RELATIONS


# --------------------------------------------------------------------------- sparse attention
def seg_softmax(scores, index, n):
    """softmax of scores [E,H] grouped by destination index [E]."""
    H = scores.shape[1]
    if scores.shape[0] == 0:
        return scores
    mx = torch.full((n, H), -1e30, device=scores.device, dtype=scores.dtype)
    mx = mx.scatter_reduce(0, index[:, None].expand(-1, H), scores, reduce="amax", include_self=True)
    ex = torch.exp(scores - mx[index])
    den = torch.zeros((n, H), device=scores.device, dtype=scores.dtype).index_add_(0, index, ex)
    return ex / (den[index] + 1e-9)


class SparseAttn(nn.Module):
    """Multi-head graph attention: dst nodes attend over their incoming edges (src nodes).
    Optional edge features modulate keys and values (edge-aware graph transformer)."""

    def __init__(self, dim, heads, edge_dim=0, dropout=0.0):
        super().__init__()
        assert dim % heads == 0
        self.h, self.dh = heads, dim // heads
        self.q, self.k, self.v = nn.Linear(dim, dim), nn.Linear(dim, dim), nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.e = nn.Linear(edge_dim, dim) if edge_dim else None
        self.drop = nn.Dropout(dropout)

    def forward(self, x_src, x_dst, src, dst, efeat=None):
        n_dst = x_dst.shape[0]
        if src.numel() == 0:
            return torch.zeros_like(x_dst), torch.zeros(0, self.h, device=x_dst.device)
        H, D = self.h, self.dh
        q = self.q(x_dst).view(-1, H, D)[dst]
        k = self.k(x_src).view(-1, H, D)[src]
        v = self.v(x_src).view(-1, H, D)[src]
        if self.e is not None and efeat is not None:
            ee = self.e(efeat).view(-1, H, D)
            k, v = k + ee, v + ee
        score = (q * k).sum(-1) / math.sqrt(D)                      # [E,H]
        alpha = seg_softmax(score, dst, n_dst)
        msg = (self.drop(alpha).unsqueeze(-1) * v).reshape(-1, H * D)
        out = torch.zeros(n_dst, H * D, device=x_dst.device, dtype=msg.dtype).index_add_(0, dst, msg)
        return self.o(out), alpha


class FFN(nn.Module):
    def __init__(self, d, drop):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Dropout(drop), nn.Linear(2 * d, d))

    def forward(self, x):
        return self.net(x)


# --------------------------------------------------------------------------- 1. aggregation
class EnvAggLayer(nn.Module):
    def __init__(self, d, heads, n_rel, edge_dim, drop):
        super().__init__()
        self.a2e = nn.ModuleList([SparseAttn(d, heads, edge_dim, drop) for _ in range(n_rel)])
        self.e2a = nn.ModuleList([SparseAttn(d, heads, edge_dim, drop) for _ in range(n_rel)])
        self.ln_e = nn.ModuleList([nn.LayerNorm(d) for _ in range(n_rel)])
        self.sem_w, self.sem_q = nn.Linear(d, d), nn.Linear(d, 1, bias=False)
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.ffn = FFN(d, drop)
        self.drop = nn.Dropout(drop)

    def forward(self, h, ents, edges, has_rel):
        zs, alphas, new_ents = [], [], []
        for r, (e, ed) in enumerate(zip(ents, edges)):
            src, dst, ef = ed
            upd, _ = self.a2e[r](h, e, src, dst, ef)               # accounts -> entity   (who shares me?)
            e = self.ln_e[r](e + self.drop(upd))
            out, al = self.e2a[r](e, h, dst, src, ef)              # entity -> accounts   (what does my env look like?)
            zs.append(out); alphas.append(al); new_ents.append(e)
        Z = torch.stack(zs, 1)                                       # [N,R,d]
        logit = self.sem_q(torch.tanh(self.sem_w(Z))).squeeze(-1)    # relation-level semantic attention
        logit = logit.masked_fill(~has_rel, -1e9)
        w = torch.softmax(logit, 1) * has_rel.any(1, keepdim=True).float()
        agg = (w.unsqueeze(-1) * Z).sum(1)
        h = self.ln1(h + self.drop(agg))
        h = self.ln2(h + self.drop(self.ffn(h)))
        return h, new_ents, w, alphas


# --------------------------------------------------------------------------- 2. difference
class EnvDiffLayer(nn.Module):
    def __init__(self, d, heads, edge_dim, drop):
        super().__init__()
        self.att = SparseAttn(d, heads, edge_dim, drop)
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.ffn = FFN(d, drop)
        self.drop = nn.Dropout(drop)

    def forward(self, h, src, dst, ef):
        out, al = self.att(h, h, src, dst, ef)
        h = self.ln1(h + self.drop(out))
        h = self.ln2(h + self.drop(self.ffn(h)))
        return h, al


# --------------------------------------------------------------------------- 3. behaviour
class TLSTM(nn.Module):
    """Time-aware LSTM (Baytas et al. 2017) - long-term memory kept, short-term memory discounted by g(dt)."""

    def __init__(self, din, dh):
        super().__init__()
        self.dh = dh
        self.decomp = nn.Linear(dh, dh)
        self.gates = nn.Linear(din + dh, 4 * dh)

    def forward(self, x, dt, mask):
        B, T, _ = x.shape
        h = x.new_zeros(B, self.dh); c = x.new_zeros(B, self.dh)
        outs = []
        for t in range(T):
            cs = torch.tanh(self.decomp(c))
            g = 1.0 / torch.log(dt[:, t].clamp(min=0.0) + math.e)         # g(dt) = 1 / log(dt + e)
            c_adj = (c - cs) + g.unsqueeze(1) * cs
            i, f, o, cand = self.gates(torch.cat([x[:, t], h], 1)).chunk(4, 1)
            c_new = torch.sigmoid(f) * c_adj + torch.sigmoid(i) * torch.tanh(cand)
            h_new = torch.sigmoid(o) * torch.tanh(c_new)
            m = mask[:, t].unsqueeze(1).float()
            h = m * h_new + (1 - m) * h
            c = m * c_new + (1 - m) * c
            outs.append(h_new * m)
        return torch.stack(outs, 1)                                    # [B,T,dh]


class BehaviourEncoder(nn.Module):
    def __init__(self, din, d, drop, pool="attention"):
        super().__init__()
        self.pool = pool
        self.inp = nn.Linear(din, d)
        self.lstm = TLSTM(d, d)
        if pool == "attention":
            self.pw, self.pq = nn.Linear(d, d), nn.Linear(d, 1, bias=False)
        self.drop = nn.Dropout(drop)

    def forward(self, seq, dt, mask):
        x = F.gelu(self.inp(seq))
        hs = self.lstm(self.drop(x), dt, mask)
        if self.pool == "sum":                                      # base paper: h_b = sum_t h_t
            return (hs * mask.unsqueeze(-1).float()).sum(1), mask.float() / mask.sum(1, keepdim=True).clamp(min=1)
        sc = self.pq(torch.tanh(self.pw(hs))).squeeze(-1).masked_fill(~mask, -1e9)
        a = torch.softmax(sc, 1) * mask.any(1, keepdim=True).float()  # event-level attention
        return (a.unsqueeze(-1) * hs).sum(1), a


# --------------------------------------------------------------------------- full model
class EGT(nn.Module):
    TOKENS = ["profile", "aggregation", "difference", "behaviour"]

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        d, H, dr = cfg.hidden, cfg.heads, cfg.dropout
        self.stat_enc = nn.Sequential(nn.Linear(cfg.n_stats, d), nn.GELU(), nn.Dropout(dr), nn.Linear(d, d))
        self.ent_type = nn.Parameter(torch.randn(cfg.n_rel, d) * 0.1)
        self.ent_deg = nn.ModuleList([nn.Linear(2, d) for _ in range(cfg.n_rel)])
        self.agg = nn.ModuleList([EnvAggLayer(d, H, cfg.n_rel, cfg.edge_dim, dr) for _ in range(cfg.env_layers)])
        self.diff = nn.ModuleList([EnvDiffLayer(d, H, cfg.sib_edge_dim, dr) for _ in range(cfg.diff_layers)])
        self.beh = BehaviourEncoder(cfg.seq_dim, d, dr)
        self.tok_proj = nn.ModuleList([nn.Linear(d, d) for _ in self.TOKENS])
        self.tok_type = nn.Parameter(torch.randn(len(self.TOKENS), d) * 0.1)
        self.fuse_att = nn.MultiheadAttention(d, H, dropout=dr, batch_first=True)
        self.ln_f1, self.ln_f2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.ffn_f = FFN(d, dr)
        self.pool_w, self.pool_q = nn.Linear(d, d), nn.Linear(d, 1, bias=False)
        self.cls = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Dropout(dr), nn.Linear(d, 2))
        self.drop = nn.Dropout(dr)

    # ---- stage 1: encode graph + sequence into modality tokens -----------------------------
    def encode(self, G, explain=False):
        """G is a (sub)graph produced by sampler.GraphIndex.sample: nodes 0..S-1 are the SEED accounts whose
        prediction is required; the remaining nodes are sampled neighbours that only provide context."""
        cfg = self.cfg
        S = G["n_seeds"]
        x = self.stat_enc(G["stats"])                                    # profile embedding  x_i  (all nodes)
        info = {}
        ents = [self.ent_type[r] + F.gelu(self.ent_deg[r](G["ent_deg"][r])) for r in range(cfg.n_rel)]
        edges = [(G["edges"][r]["src"], G["edges"][r]["dst"], G["edges"][r]["feat"]) for r in range(cfg.n_rel)]
        has_rel = G["has_rel"]
        rm = getattr(cfg, "rel_mask", (1, 1, 1))
        if not all(rm):                                    # meta-path study: switch single relations off
            e0 = torch.zeros(0, dtype=torch.long)
            edges = [e if rm[r] else (e0, e0, torch.zeros(0, cfg.edge_dim)) for r, e in enumerate(edges)]
            has_rel = has_rel & torch.tensor([bool(v) for v in rm])
        # 1) environment aggregation
        h = x
        if cfg.use_agg:
            for layer in self.agg:
                h, ents, w, al = layer(h, ents, edges, has_rel)
                if explain:
                    info.setdefault("rel_w", []).append(w.detach()); info.setdefault("agg_alpha", []).append([a.detach() for a in al])
            h_agg = h
        else:
            h_agg = torch.zeros_like(x)
        # 2) environment difference (initialised with the environment embedding, as in the paper)
        h_diff = h_agg if cfg.use_agg else x
        if cfg.use_diff:
            for layer in self.diff:
                h_diff, al = layer(h_diff, G["sib_src"], G["sib_dst"], G["sib_feat"])
                if explain:
                    info.setdefault("diff_alpha", []).append(al.detach())
        else:
            h_diff = torch.zeros_like(x)
        # 3) behaviour sequence (seeds only)
        if cfg.use_beh:
            rows = G["seq_row"]; mask = rows >= 0; r0 = rows.clamp(min=0)
            seq = G["ev_feat"][r0] * mask.unsqueeze(-1); dt = G["ev_dt"][r0] * mask
            h_beh, ev_alpha = self.beh(seq, dt, mask)
            if explain:
                info["ev_alpha"] = ev_alpha.detach()
        else:
            h_beh = torch.zeros_like(x[:S])
        toks = torch.stack([x[:S], h_agg[:S], h_diff[:S], h_beh], 1)      # [S,4,d]
        return toks, info

    # ---- stage 2: attentive feature fusion + detection --------------------------------------
    def fuse(self, toks, drop_token=None, explain=False):
        cfg = self.cfg
        t = torch.stack([self.tok_proj[i](toks[:, i]) for i in range(4)], 1) + self.tok_type
        active = torch.tensor([True, cfg.use_agg, cfg.use_agg and cfg.use_diff or cfg.use_diff, cfg.use_beh], device=toks.device)
        if drop_token is not None:
            active = active.clone(); active[drop_token] = False
        key_pad = ~active.unsqueeze(0).expand(t.shape[0], -1)
        t = t * active.view(1, 4, 1).float()
        if cfg.use_fusion_attn:
            a, att = self.fuse_att(t, t, t, key_padding_mask=key_pad, need_weights=True, average_attn_weights=True)
            t = self.ln_f1(t + self.drop(a))
            t = self.ln_f2(t + self.drop(self.ffn_f(t)))
            t = t * active.view(1, 4, 1).float()
            sc = self.pool_q(torch.tanh(self.pool_w(t))).squeeze(-1).masked_fill(key_pad, -1e9)
            w = torch.softmax(sc, 1)                                      # modality importance
            z = (w.unsqueeze(-1) * t).sum(1)
        else:                                                              # ablation: plain average
            w = active.float().unsqueeze(0).expand(t.shape[0], -1)
            w = w / w.sum(1, keepdim=True)
            z = (w.unsqueeze(-1) * t).sum(1); att = None
        logits = self.cls(z)
        if explain:
            return logits, {"mod_w": w.detach(), "fuse_att": None if att is None else att.detach()}
        return logits

    def forward(self, G, explain=False):
        toks, info = self.encode(G, explain)
        if explain:
            logits, info2 = self.fuse(toks, explain=True)
            info.update(info2); info["tokens"] = toks.detach()
            return logits, info
        return self.fuse(toks)


def build_model(cfg):
    """Factory: proposed EGT or the re-implemented base-paper EnvIT."""
    arch = getattr(cfg, "arch", "egt")
    if arch in ("envit", "envit_graph", "envit_beh"):
        from .envit import EnvIT
        return EnvIT(cfg, {"envit": "full", "envit_graph": "graph", "envit_beh": "beh"}[arch])
    if arch != "egt":
        from .zoo import ZOO
        return ZOO[arch](cfg)
    return EGT(cfg)
