
import numpy as np
import torch

from .config import RELATIONS

SELF_FEAT = np.array([1, 1, 1, 0, 0, 0, 0, 1], dtype=np.float32)


def _ranges(ptr, ids):
    starts = ptr[ids]
    lens = ptr[ids + 1] - starts
    total = int(lens.sum())
    owner = np.repeat(np.arange(len(ids)), lens)
    offs = np.arange(total) - np.repeat(np.cumsum(lens) - lens, lens)
    return owner, np.repeat(starts, lens) + offs


def _cap(owner, keys, cap):

    if len(owner) == 0:
        return np.zeros(0, bool)
    order = np.lexsort((keys, owner))
    o = owner[order]
    first = np.r_[0, np.flatnonzero(np.diff(o)) + 1]
    rank = np.arange(len(o)) - np.repeat(first, np.diff(np.r_[first, len(o)]))
    keep = np.zeros(len(owner), bool)
    keep[order[rank < cap]] = True
    return keep


def _hash(a, b):
    return ((a.astype(np.uint64) * np.uint64(2654435761)) ^ (b.astype(np.uint64) * np.uint64(40503))) % np.uint64(4294967291)


class GraphIndex:
    def __init__(self, gd, scaler):
        self.gd, self.N = gd, gd.n
        self.stats = torch.from_numpy(scaler.transform(gd.stats))
        self.ev_feat = torch.from_numpy(gd.ev_feat)
        self.ev_dt = torch.from_numpy(gd.ev_dt)
        self.seq_row = torch.from_numpy(gd.seq_event_row)
        self.rel = []
        for r in RELATIONS:
            ed = gd.edges[r]
            src, dst, feat = ed["src"], ed["dst"], ed["feat"]
            n_ent = len(gd.ent_names[r])
            o = np.argsort(src, kind="stable")
            a_ptr = np.r_[0, np.cumsum(np.bincount(src, minlength=self.N))].astype(np.int64)
            o2 = np.argsort(dst, kind="stable")
            e_ptr = np.r_[0, np.cumsum(np.bincount(dst, minlength=n_ent))].astype(np.int64)
            self.rel.append(dict(a_ptr=a_ptr, a_ent=dst[o], a_feat=feat[o], e_ptr=e_ptr, e_acct=src[o2], n_ent=n_ent,
                                 ent_deg=torch.from_numpy(gd.ent_deg[r]) if n_ent else torch.zeros(0, 2)))
        m = gd.sib_src != gd.sib_dst
        s, d, f = gd.sib_src[m], gd.sib_dst[m], gd.sib_feat[m]
        o = np.argsort(d, kind="stable")
        self.sib_ptr = np.r_[0, np.cumsum(np.bincount(d, minlength=self.N))].astype(np.int64)
        self.sib_nbr, self.sib_feat = s[o], f[o]

    def sample(self, seeds, cap_acct_per_ent=20, cap_ent_per_acct=30, cap_sib=10):
        seeds = np.asarray(seeds, dtype=np.int64)
        S = len(seeds)
        extra = []
        ents = []
        # siblings of the seeds
        ow, pos = _ranges(self.sib_ptr, seeds)
        keep = _cap(ow, np.arange(len(ow)), cap_sib)
        extra.append(self.sib_nbr[pos[keep]])
        for rel in self.rel:
            ow, pos = _ranges(rel["a_ptr"], seeds)
            ent = rel["a_ent"][pos]
            keep = _cap(ow, -rel["a_feat"][pos, 0], cap_ent_per_acct)
            e_set = np.unique(ent[keep])
            ents.append(e_set)
            if len(e_set):
                ow2, pos2 = _ranges(rel["e_ptr"], e_set)
                accs = rel["e_acct"][pos2]
                keep2 = _cap(ow2, _hash(accs, e_set[ow2]), cap_acct_per_ent)
                extra.append(accs[keep2])
        extra = np.unique(np.concatenate(extra)) if extra else np.zeros(0, np.int64)
        extra = extra[~np.isin(extra, seeds)]
        nodes = np.concatenate([seeds, extra])
        M = len(nodes)
        loc = np.full(self.N, -1, np.int64)
        loc[nodes] = np.arange(M)
        G_edges, has_rel = [], np.zeros((M, len(RELATIONS)), bool)
        for ri, rel in enumerate(self.rel):
            e_set = ents[ri]
            loc_e = np.full(max(rel["n_ent"], 1), -1, np.int64)
            loc_e[e_set] = np.arange(len(e_set))
            ow, pos = _ranges(rel["a_ptr"], nodes)
            de = loc_e[rel["a_ent"][pos]]
            k = de >= 0
            src, dst, ft = ow[k], de[k], rel["a_feat"][pos[k]]
            has_rel[src, ri] = True
            G_edges.append(dict(src=torch.from_numpy(src), dst=torch.from_numpy(dst), feat=torch.from_numpy(ft)))
        ow, pos = _ranges(self.sib_ptr, nodes)
        nb = loc[self.sib_nbr[pos]]
        k = nb >= 0
        s_src = np.concatenate([np.arange(M), nb[k]])
        s_dst = np.concatenate([np.arange(M), ow[k]])
        s_feat = np.concatenate([np.tile(SELF_FEAT, (M, 1)), self.sib_feat[pos[k]]])
        G = {
            "stats": self.stats[nodes],
            "ent_deg": [rel["ent_deg"][ents[ri]] for ri, rel in enumerate(self.rel)],
            "edges": G_edges, "has_rel": torch.from_numpy(has_rel),
            "sib_src": torch.from_numpy(s_src), "sib_dst": torch.from_numpy(s_dst), "sib_feat": torch.from_numpy(s_feat),
            "n_seeds": S, "seq_row": self.seq_row[seeds], "ev_feat": self.ev_feat, "ev_dt": self.ev_dt,
            "_nodes": nodes, "_ents": ents,
        }
        return G


def predict_logits(model, gi, ids, batch=512, **caps):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(ids), batch):
            G = gi.sample(ids[i:i + batch], **caps)
            out.append(model(G))
    return torch.cat(out).numpy() if out else np.zeros((0, 2), np.float32)
