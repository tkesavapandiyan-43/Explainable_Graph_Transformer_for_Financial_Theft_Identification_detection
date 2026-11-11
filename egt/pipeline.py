"""High-level API: load model -> score any events table -> thresholds/tiers -> explain -> timeline."""
import os
import numpy as np
import pandas as pd
import torch

from .features import build_graph, FEAT_KEYS
from .risk import assign_tier, apply_policy
from .sampler import GraphIndex, predict_logits
from .train import load_checkpoint

BASE_COLS = ["event_id", "account_id", "person_id", "timestamp", "event_type", "device_id", "ip", "address", "amount", "label", "split"]


class ScoreResult:
    def __init__(self, gd, gi, prob):
        self.gd, self.gi, self.prob = gd, gi, prob
        self._ego = {}


class Predictor:
    def __init__(self, ckpt_path):
        ck = load_checkpoint(ckpt_path)
        self.path = ckpt_path
        self.model, self.scaler, self.T = ck["model"], ck["scaler"], ck["T"]
        self.thresholds, self.extra, self.cfg = ck["thresholds"], ck["extra"], ck["model_cfg"]
        self.caps = self.extra.get("caps", dict(cap_acct_per_ent=20, cap_sib=10))
        self.model.eval()
        self.preprocessor = None
        pp = os.path.join(os.path.dirname(ckpt_path), self.extra.get("preprocessor_file", "")) if self.extra.get("preprocessor_file") else None
        if pp and os.path.exists(pp):
            from .vesta import VestaPreprocessor
            self.preprocessor = VestaPreprocessor.load(pp)

    @property
    def is_demo_model(self):
        return str(self.extra.get("source", "")).startswith("synthetic")

    # ------------------------------------------------------------------ scoring
    def check_compat(self, gd):
        want = self.extra.get("feat_keys")
        if gd.stats.shape[1] != self.cfg.n_stats or (want is not None and list(gd.feat_keys) != list(want)):
            raise ValueError(
                f"The input produces {gd.stats.shape[1]} account features but this model was trained with {self.cfg.n_stats}. "
                f"The model expects the data format it was trained on ({self.extra.get('source', 'unknown')}). "
                + ("Provide Vesta/IEEE-CIS style transaction (+identity) files so the saved preprocessor can build the features."
                   if self.preprocessor is not None else "Use the same pre-processing as in training."))

    def score(self, events: pd.DataFrame, progress=None) -> ScoreResult:
        return self.score_graph(build_graph(events), progress)

    def score_graph(self, gd, progress=None) -> ScoreResult:
        self.check_compat(gd)
        gi = GraphIndex(gd, self.scaler)
        n = gd.n
        lg = np.zeros((n, 2), np.float32)
        bs = 512
        for i in range(0, n, bs * 8):                        # chunks only to report progress
            ids = np.arange(i, min(i + bs * 8, n))
            lg[ids] = predict_logits(self.model, gi, ids, bs, **self.caps)
            if progress:
                progress(min(1.0, (i + len(ids)) / n))
        d = (lg[:, 1] - lg[:, 0]).astype(np.float64)
        return ScoreResult(gd, gi, 1.0 / (1.0 + np.exp(-d / self.T)))

    def score_vesta(self, transaction_csv, identity_csv=None, progress=None):
        if self.preprocessor is None:
            raise ValueError("This checkpoint has no saved Vesta preprocessor (it was trained on the generic event schema).")
        ev = self.preprocessor.transform_files(transaction_csv, identity_csv)
        return self.score(ev, progress)

    def input_warnings(self, res: ScoreResult, tbl: pd.DataFrame = None):
        gd, w = res.gd, []
        env_cov = float(gd.has_rel.any(1).mean())
        if env_cov < 0.3:
            w.append(f"Only {env_cov:.0%} of accounts have device / network / address data - the graph modules have little to work with.")
        if tbl is not None and (tbl.prediction == "FRAUD").mean() > 0.5:
            w.append("More than half of the accounts are flagged - the data probably differs strongly from the training data.")
        if self.is_demo_model:
            w.append("GENERATED DATA: this model was trained on generated stand-in data that imitates the known fraud patterns, not on real Meituan / Vesta data (see README).")
        return w

    def table(self, res: ScoreResult, mode="f1", manual=0.5, topk=0.05) -> pd.DataFrame:
        """Dynamic risk table: risk score (0-100), tier, and fraud flag under the chosen threshold policy."""
        gd = res.gd
        flag, cut = apply_policy(res.prob, self.thresholds, mode, manual, topk)
        df = pd.DataFrame({
            "account_id": gd.account_ids, "person_id": gd.person_ids,
            "risk_score": np.round(res.prob * 100, 2),
            "risk_tier": assign_tier(res.prob, self.thresholds["tiers"]),
            "prediction": np.where(flag, "FRAUD", "LEGITIMATE"),
            "n_events": np.expm1(gd.stats[:, 0]).round().astype(int),
        })
        df["given_label"] = np.where(gd.labels >= 0, np.where(gd.labels == 1, "fraud", "normal"), "")
        if gd.split is not None:
            df["split"] = np.array(["train", "val", "test", "-"])[np.where(gd.split >= 0, gd.split, 3)]
        df.attrs["threshold"], df.attrs["mode"] = cut, mode
        return df.sort_values("risk_score", ascending=False).reset_index(drop=True)

    # ------------------------------------------------------------------ dynamic risk over time
    def timeline(self, events_norm: pd.DataFrame, account_id: str, n_points=6, min_events=3, progress=None):
        """Re-scores the account (inside the whole graph) on snapshots that contain only the events seen so far."""
        ev = events_norm[events_norm["account_id"] == account_id].sort_values("timestamp")
        cols = ["cutoff", "events_seen", "devices", "networks", "addresses", "max_other_accounts_sharing_an_entity",
                "night_share", "burst_share", "siblings", "risk_score", "risk_tier"]
        if len(ev) < min_events:
            return pd.DataFrame(columns=cols)
        idx = np.unique(np.linspace(min_events - 1, len(ev) - 1, n_points).round().astype(int))
        cuts = ev["timestamp"].iloc[idx].tolist()
        rows = []
        for k, c in enumerate(cuts):
            snap = events_norm[events_norm["timestamp"] <= c]
            r = self.score(snap)
            i = r.gd.acct_index.get(account_id)
            if i is None:
                continue
            st = r.gd.stats[i]; ix = {k_: j for j, k_ in enumerate(r.gd.feat_keys)}
            share = max(np.expm1(st[ix[k_]]) for k_ in ("max_dev_deg", "max_ip_deg", "max_addr_deg"))
            rows.append(dict(cutoff=c, events_seen=int((ev["timestamp"] <= c).sum()),
                             devices=int(round(np.expm1(st[ix["n_device"]]))), networks=int(round(np.expm1(st[ix["n_ip"]]))),
                             addresses=int(round(np.expm1(st[ix["n_address"]]))), max_other_accounts_sharing_an_entity=int(round(share)),
                             night_share=float(st[ix["night_ratio"]]), burst_share=float(st[ix["frac_gap_lt_10min"]]),
                             siblings=int(round(np.expm1(st[ix["n_siblings"]]))), risk_score=round(float(r.prob[i]) * 100, 2),
                             risk_tier=str(assign_tier([r.prob[i]], self.thresholds["tiers"])[0])))
            if progress:
                progress((k + 1) / len(cuts))
        return pd.DataFrame(rows, columns=cols)


def normalized_events(res: ScoreResult) -> pd.DataFrame:
    ev = res.gd.events
    cols = [c for c in BASE_COLS if c in ev.columns] + [c for c in ev.columns if c.startswith("x_")]
    return ev[cols].copy()


# ---------------------------------------------------------------------------------- locating model / cached results
def ckpt_id(path):
    import hashlib
    h = hashlib.md5(open(path, "rb").read()).hexdigest()
    return f"{os.path.getsize(path)}-{h}"


def find_study_model(root, ds):
    d = os.path.join(root, "study", "deploy", ds)
    ck = os.path.join(d, "egt.pt")
    return (ck, d, d, True) if os.path.exists(ck) else (None, None, None, None)


def find_model(root):
    """(checkpoint, outputs_dir, cache_dir, is_demo).  The real-data (Vesta) model is preferred; the demo model
    (trained on synthetic data) is used only if no real model exists."""
    real = os.path.join(root, "models", "egt_vesta.pt")
    if os.path.exists(real):
        return real, os.path.join(root, "outputs"), os.path.join(root, "data", "cache"), False
    st = find_study_model(root, "meituan")
    if st[0]:
        return st
    demo = os.path.join(root, "models", "demo", "egt_demo.pt")
    if os.path.exists(demo):
        return demo, os.path.join(root, "demo", "outputs"), os.path.join(root, "demo", "cache"), True
    return None, None, None, None


def load_cached_result(pred, cache_dir, out_dir):
    """The prepared dataset (graph.pkl) with the scores written at the end of training - no re-scoring needed."""
    import json, pickle
    g, pj, pn = os.path.join(cache_dir, "graph.pkl"), os.path.join(out_dir, "final_probs.json"), os.path.join(out_dir, "final_probs.npy")
    if not (os.path.exists(g) and os.path.exists(pj) and os.path.exists(pn)):
        return None
    meta = json.load(open(pj))
    if meta.get("ckpt_id") != ckpt_id(pred.path):
        return None
    with open(g, "rb") as f:
        gd = pickle.load(f)
    prob = np.load(pn)
    if len(prob) != gd.n:
        return None
    return ScoreResult(gd, GraphIndex(gd, pred.scaler), prob)
