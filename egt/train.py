"""Mini-batch training (semi-supervised: unlabeled accounts take part in message passing only)."""
import copy, time
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from .config import ModelConfig, TrainConfig
from .features import Scaler
from .model import EGT, build_model
from .risk import fit_temperature, derive_thresholds
from .sampler import GraphIndex, predict_logits


def set_seed(s):
    np.random.seed(s); torch.manual_seed(s)


def make_split(labels, seed, ratio=(3, 1, 1)):
    """Stratified random split of the LABELED accounts (paper: 3:1:1)."""
    rng = np.random.default_rng(seed)
    tr, va, te = [], [], []
    tot = float(sum(ratio))
    for c in (0, 1):
        idx = np.where(labels == c)[0]
        rng.shuffle(idx)
        a = int(round(len(idx) * ratio[0] / tot)); b = int(round(len(idx) * (ratio[0] + ratio[1]) / tot))
        tr += list(idx[:a]); va += list(idx[a:b]); te += list(idx[b:])
    return np.array(sorted(tr)), np.array(sorted(va)), np.array(sorted(te))


def split_from_data(gd, seed, ratio=(3, 1, 1)):
    """Use the split fixed during data preparation (leak-free preprocessing) if present, else a random 3:1:1 split."""
    if gd.split is not None and (gd.split >= 0).any():
        lab = gd.labels >= 0
        return tuple(np.where((gd.split == k) & lab)[0] for k in (0, 1, 2))
    return make_split(gd.labels, seed, ratio)


def _caps(tc):
    return dict(cap_acct_per_ent=tc.cap_acct_per_ent, cap_sib=tc.cap_sib)


def train_model(gd, model_cfg: ModelConfig = None, train_cfg: TrainConfig = None, split=None, verbose=True, log=print,
                progress=None, partial_ckpt=None):
    model_cfg = copy.deepcopy(model_cfg or ModelConfig())
    tc = train_cfg or TrainConfig()
    set_seed(tc.seed)
    model_cfg.n_stats, model_cfg.seq_dim = gd.stats.shape[1], gd.seq_dim
    tr, va, te = split if split is not None else split_from_data(gd, tc.seed, tc.split)
    scaler = Scaler().fit(gd.stats[tr])                      # preprocessing statistics from TRAIN accounts only
    gi = GraphIndex(gd, scaler)
    caps = _caps(tc)
    y = np.where(gd.labels < 0, 0, gd.labels)
    yt = torch.as_tensor(y, dtype=torch.long)
    model = build_model(model_cfg)
    opt = torch.optim.Adam(model.parameters(), lr=tc.lr, weight_decay=tc.weight_decay)
    pos, neg = int((y[tr] == 1).sum()), int((y[tr] == 0).sum())
    w = torch.tensor([1.0, 1.0 if tc.undersample else float(np.clip(np.sqrt(neg / max(pos, 1)), 1, tc.weight_clip))])
    rng = np.random.default_rng(tc.seed)
    best, best_state, best_ep, bad, hist = -1.0, None, 0, 0, []
    t0 = time.time()
    for ep in range(1, tc.epochs + 1):
        model.train()
        ids = tr
        if tc.undersample:                                   # paper: under-sample the majority class every epoch
            fr, nm = tr[y[tr] == 1], tr[y[tr] == 0]
            ids = np.concatenate([fr, rng.choice(nm, size=min(len(nm), tc.undersample * len(fr)), replace=False)])
        ids = rng.permutation(ids)
        if tc.max_batches:
            ids = ids[: tc.max_batches * tc.batch_size]
        tot, nb = 0.0, 0
        for i in range(0, len(ids), tc.batch_size):
            b = ids[i:i + tc.batch_size]
            G = gi.sample(b, **caps)
            opt.zero_grad()
            loss = F.cross_entropy(model(G), yt[b], weight=w)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot += float(loss.item()); nb += 1
        lg = predict_logits(model, gi, va, tc.batch_size, **caps)
        p = torch.softmax(torch.from_numpy(lg), 1)[:, 1].numpy()
        auc = roc_auc_score(y[va], p)
        hist.append(dict(epoch=ep, loss=tot / max(nb, 1), val_auc=float(auc), seconds=time.time() - t0))
        if auc > best:
            best, bad, best_ep = auc, 0, ep
            best_state = copy.deepcopy(model.state_dict())
            if partial_ckpt:
                torch.save(dict(state=best_state, epoch=ep, val_auc=best), partial_ckpt)
        else:
            bad += 1
        if verbose and (ep % 2 == 0 or ep == 1):
            log(f"  epoch {ep:3d} | loss {tot/max(nb,1):.4f} | val AUC {auc:.4f} | best {best:.4f} (ep {best_ep}) | {time.time()-t0:.0f}s")
        if progress:
            progress(ep / tc.epochs, f"epoch {ep}  val-AUC {auc:.3f}")
        if tc.max_minutes and (time.time() - t0) / 60 >= tc.max_minutes:
            if verbose:
                log(f"  time budget of {tc.max_minutes:g} min reached after epoch {ep} - keeping the best epoch ({best_ep}, val AUC {best:.4f})")
            break
        if bad >= tc.patience:
            if verbose:
                log(f"  early stopping at epoch {ep} (best val AUC {best:.4f} at epoch {best_ep})")
            break
    train_seconds = time.time() - t0
    model.load_state_dict(best_state); model.eval()
    # calibration + thresholds from VALIDATION only
    lg_va = predict_logits(model, gi, va, tc.batch_size, **caps)
    T = fit_temperature(lg_va[:, 1] - lg_va[:, 0], y[va])
    p_va = 1 / (1 + np.exp(-(lg_va[:, 1] - lg_va[:, 0]) / T))
    thr = derive_thresholds(p_va, y[va])
    t1 = time.time()
    lg_all = predict_logits(model, gi, np.arange(gd.n), tc.batch_size, **caps)
    infer_seconds = time.time() - t1
    probs = 1 / (1 + np.exp(-(lg_all[:, 1] - lg_all[:, 0]) / T))
    return dict(model=model, scaler=scaler, model_cfg=model_cfg, train_cfg=tc, T=T, thresholds=thr, probs=probs,
                split=(tr, va, te), history=hist, best_val_auc=float(best), best_epoch=best_ep,
                train_seconds=train_seconds, infer_seconds=infer_seconds, n_params=sum(p.numel() for p in model.parameters()))


def save_checkpoint(res, path, extra=None, gd=None):
    tc = res["train_cfg"]
    meta = dict(extra or {})
    meta.update(caps=_caps(tc), best_epoch=res["best_epoch"], best_val_auc=res["best_val_auc"], history=res["history"],
                train_seconds=res["train_seconds"], infer_seconds=res["infer_seconds"], n_params=res["n_params"],
                train_cfg=vars(tc) if hasattr(tc, "__dict__") else {})
    if gd is not None:
        meta.update(feat_keys=list(gd.feat_keys), seq_dim=gd.seq_dim)
    torch.save(dict(state=res["model"].state_dict(), model_cfg=res["model_cfg"].to_dict(), scaler=res["scaler"].to_dict(),
                    T=res["T"], thresholds=res["thresholds"], extra=meta), path)


def load_checkpoint(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ModelConfig(**ck["model_cfg"])
    model = build_model(cfg); model.load_state_dict(ck["state"]); model.eval()
    return dict(model=model, scaler=Scaler.from_dict(ck["scaler"]), model_cfg=cfg, T=ck["T"],
                thresholds=ck["thresholds"], extra=ck.get("extra", {}))
