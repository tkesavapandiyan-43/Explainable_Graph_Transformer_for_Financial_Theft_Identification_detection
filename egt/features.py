"""
Events table  ->  graph + features.

Builds, from ANY user supplied transaction log:
  * account-level statistical features (profile)
  * per-account behaviour sequences (last T events)             -> Behaviour extractor
  * heterogeneous account <-> {device, ip, address} edges        -> Environment-aggregation extractor
  * homogeneous account <-> account edges (same personal ID) with
    environment-difference edge features                         -> Environment-difference extractor
"""
import re
from dataclasses import dataclass, field
import numpy as np
import pandas as pd

from .config import RELATIONS, REL_COLUMN, SEQ_LEN, SEQ_DIM

# key, human label, is_log (stored as log1p), unit
FEATURES = [
    ("n_events", "total number of events", True, ""),
    ("n_login", "number of logins", True, ""),
    ("n_payment", "number of payments", True, ""),
    ("n_order", "number of orders", True, ""),
    ("active_days", "number of active days", True, "days"),
    ("span_days", "activity span (first -> last event)", True, "days"),
    ("events_per_active_day", "events per active day", True, ""),
    ("mean_gap_min", "average gap between events", True, "min"),
    ("median_gap_min", "median gap between events", True, "min"),
    ("frac_gap_lt_1min", "share of events <1 min after the previous one", False, ""),
    ("frac_gap_lt_10min", "share of events <10 min after the previous one", False, ""),
    ("std_log_gap", "irregularity of timing (std of log-gap)", False, ""),
    ("night_ratio", "share of activity at night (00:00-05:00)", False, ""),
    ("max_events_1h", "max events inside a single hour", True, ""),
    ("max_events_1d", "max events inside a single day", True, ""),
    ("n_device", "distinct devices used", True, ""),
    ("n_ip", "distinct IP networks (/24) used", True, ""),
    ("n_address", "distinct delivery addresses", True, ""),
    ("device_per_event", "devices per event", False, ""),
    ("ip_per_event", "IP networks per event", False, ""),
    ("addr_per_event", "addresses per event", False, ""),
    ("device_switch_rate", "rate of switching device between events", False, ""),
    ("ip_switch_rate", "rate of switching IP network between events", False, ""),
    ("addr_switch_rate", "rate of switching address between events", False, ""),
    ("amount_mean", "average payment amount", True, ""),
    ("amount_max", "largest payment amount", True, ""),
    ("amount_cv", "variability of payment amounts", False, ""),
    ("pay_ratio", "share of events that are payments", False, ""),
    ("hours_to_first_payment", "hours from first event to first payment", True, "h"),
    ("frac_first_48h", "share of all events within first 48h of the account", False, ""),
    ("n_siblings", "other accounts verified with the same personal ID", True, ""),
]
N_PROFILE = len(FEATURES)          # profile statistics; graph-structure features are appended after them
FEATURES += [
    ("max_dev_deg", "other accounts on its most-shared device", True, ""),
    ("mean_dev_deg", "avg. other accounts per device used", True, ""),
    ("max_ip_deg", "other accounts on its most-shared IP network", True, ""),
    ("mean_ip_deg", "avg. other accounts per IP network used", True, ""),
    ("max_addr_deg", "other accounts at its most-shared address", True, ""),
    ("mean_addr_deg", "avg. other accounts per address used", True, ""),
    ("sib_jacc_dev", "device overlap with same-ID sibling accounts", False, ""),
    ("sib_jacc_ip", "IP overlap with same-ID sibling accounts", False, ""),
    ("sib_jacc_addr", "address overlap with same-ID sibling accounts", False, ""),
]
FEAT_KEYS = [f[0] for f in FEATURES]
FEAT_LABEL = {f[0]: f[1] for f in FEATURES}
FEAT_ISLOG = {f[0]: f[2] for f in FEATURES}
FEAT_UNIT = {f[0]: f[3] for f in FEATURES}

GRAPH_BASELINE_KEYS = ["max_dev_deg", "mean_dev_deg", "max_ip_deg", "mean_ip_deg", "max_addr_deg",
                       "mean_addr_deg", "sib_jacc_dev", "sib_jacc_ip", "sib_jacc_addr"]


def describe_x(col):
    """Readable name of a dataset-specific event feature (x_*), e.g. 'x_vpca03' -> 'Vesta V-features, principal component 4'."""
    c = col[2:].lower()
    rules = [(r"vpca(\d+)", lambda m: f"Vesta V-features, principal component {int(m[1]) + 1}"),
             (r"c(\d+)", lambda m: f"counting feature C{m[1]}"), (r"d(\d+)", lambda m: f"time-delta feature D{m[1]}"),
             (r"m(\d+)", lambda m: f"match flag M{m[1]}"), (r"id_(\d+)", lambda m: f"identity field id_{m[1]}"),
             (r"freq_p_emaildomain", lambda m: "popularity of the purchaser e-mail domain"), (r"freq_r_emaildomain", lambda m: "popularity of the recipient e-mail domain"),
             (r"freq_id_30", lambda m: "popularity of the operating system"), (r"freq_id_31", lambda m: "popularity of the browser"),
             (r"freq_devicetype", lambda m: "popularity of the device type"), (r"freq_id_23", lambda m: "popularity of the proxy type"),
             (r"freq_(id_\d+)", lambda m: f"popularity of identity field {m[1]}"), (r"amt_log", lambda m: "log transaction amount"),
             (r"amt_cents", lambda m: "cents part of the amount"), (r"dist(\d)_log", lambda m: f"log distance {m[1]}"),
             (r"prod_(\w)", lambda m: f"product code {m[1].upper()}"), (r"card4_(\w+)", lambda m: f"card network {m[1]}"),
             (r"card6_(\w+)", lambda m: f"{m[1]} card"), (r"miss_(\w)", lambda m: f"missing-value rate of the {m[1].upper()} features"),
             (r"email_match", lambda m: "purchaser and recipient e-mail domain equal"), (r"has_identity", lambda m: "identity record available"),
             (r"screen_log", lambda m: "log screen size")]
    for pat, fn in rules:
        m = re.fullmatch(pat, c)
        if m:
            return fn(m)
    return c.replace("_", " ")


@dataclass
class GraphData:
    account_ids: np.ndarray
    person_ids: np.ndarray
    stats: np.ndarray                 # [N, F] (log-transformed where flagged), unscaled
    ev_feat: np.ndarray               # [n_events, D] per-event feature matrix (sequence input)
    ev_dt: np.ndarray                 # [n_events] gap to the previous event of the same account, minutes
    seq_event_row: np.ndarray         # [N, T] row index into ev_feat / events (-1 = padding); chronological
    labels: np.ndarray                # [N]  1 / 0 / -1
    edges: dict                       # rel -> dict(src, dst, feat)
    ent_names: dict                   # rel -> np.ndarray of names
    ent_deg: dict                     # rel -> np.ndarray [n_ent, 2]  (log accounts, log events)
    sib_src: np.ndarray
    sib_dst: np.ndarray
    sib_feat: np.ndarray
    events: pd.DataFrame              # normalised/sorted events (with acct_idx)
    graph_feats: np.ndarray           # [N, G] hand-crafted graph features (baselines only)
    has_rel: np.ndarray               # [N, R] bool
    feat_keys: list = field(default_factory=list)      # names of the columns of `stats`
    feat_labels: list = field(default_factory=list)    # human readable names
    feat_islog: list = field(default_factory=list)
    feat_units: list = field(default_factory=list)
    split: np.ndarray = None                           # 0 train / 1 val / 2 test per account (if the data prep defined it)
    acct_index: dict = field(default_factory=dict)

    @property
    def n(self):
        return len(self.account_ids)

    @property
    def seq_dim(self):
        return int(self.ev_feat.shape[1])

    def raw_value(self, j, stored):
        """Convert a stored (possibly log1p'd) statistic back to a human readable value."""
        return float(np.expm1(stored)) if self.feat_islog[j] else float(stored)


class Scaler:
    """Standardiser for the statistics block (fit on training accounts only)."""

    def __init__(self, mean=None, std=None):
        self.mean, self.std = mean, std

    def fit(self, x):
        self.mean = x.mean(0)
        self.std = x.std(0) + 1e-6
        return self

    def transform(self, x):
        return np.clip((x - self.mean) / self.std, -6, 6).astype(np.float32)

    def to_dict(self):
        return {"mean": np.asarray(self.mean), "std": np.asarray(self.std)}

    @classmethod
    def from_dict(cls, d):
        return cls(d["mean"], d["std"])


# --------------------------------------------------------------------------- helpers
def _norm_ip(s):
    s = str(s)
    parts = s.split(".")
    if len(parts) == 4 and all(p.isdigit() for p in parts):
        return ".".join(parts[:3])                      # paper: first three bytes of the IP
    return s


def normalise_events(df: pd.DataFrame) -> pd.DataFrame:
    """Validate the user's table and bring it to the canonical schema."""
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    alias = {"user_id": "account_id", "acct": "account_id", "account": "account_id", "customer_id": "account_id",
             "time": "timestamp", "datetime": "timestamp", "date": "timestamp", "transactiondt": "timestamp",
             "device": "device_id", "ip_address": "ip", "ipaddr": "ip", "delivery_address": "address",
             "addr": "address", "type": "event_type", "transaction_amt": "amount", "amt": "amount",
             "id_number": "person_id", "personal_id": "person_id", "identity_id": "person_id", "fraud": "label",
             "is_fraud": "label", "isfraud": "label"}
    df = df.rename(columns={k: v for k, v in alias.items() if k in df.columns and v not in df.columns})
    missing = [c for c in ("account_id", "timestamp") if c not in df.columns]
    if missing:
        raise ValueError(f"Input is missing required column(s): {missing}. "
                         f"Required: account_id, timestamp. Optional: person_id, event_type, device_id, ip, address, amount, label.")
    if "person_id" not in df.columns:
        df["person_id"] = df["account_id"]
    for c, default in (("event_type", "login"), ("device_id", ""), ("ip", ""), ("address", ""), ("amount", 0.0), ("label", -1)):
        if c not in df.columns:
            df[c] = default
    if pd.api.types.is_numeric_dtype(df["timestamp"]):
        unit = "ms" if df["timestamp"].abs().max() > 1e11 else "s"
        ts = pd.to_datetime(df["timestamp"], unit=unit, errors="coerce")
    else:
        ts = pd.to_datetime(df["timestamp"], errors="coerce")
    df["timestamp"] = ts
    df = df.dropna(subset=["account_id", "timestamp"]).copy()
    df["account_id"] = df["account_id"].astype(str)
    df["person_id"] = df["person_id"].fillna(df["account_id"]).astype(str)
    for c in ("device_id", "address"):
        df[c] = df[c].fillna("").astype(str).replace({"nan": "", "None": "", "NaN": ""})
    df["ip"] = df["ip"].fillna("").astype(str).replace({"nan": "", "None": ""}).map(lambda s: _norm_ip(s) if s else "")
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0.0).clip(lower=0)
    df["label"] = pd.to_numeric(df["label"], errors="coerce").fillna(-1).astype(int)
    et = df["event_type"].fillna("other").astype(str).str.lower()
    df["etype"] = np.where(et.str.contains("log"), 0, np.where(et.str.contains("pay|purch|transfer|txn|trans"), 1, 2))
    df = df.sort_values(["account_id", "timestamp"], kind="mergesort").reset_index(drop=True)
    if "event_id" not in df.columns:
        df["event_id"] = [f"EV_{i:07d}" for i in range(len(df))]
    return df


def _jaccard(a, b):
    if not a and not b:
        return 0.0
    u = len(a | b)
    return len(a & b) / u if u else 0.0


# --------------------------------------------------------------------------- main builder
def build_graph(raw_events: pd.DataFrame, T: int = SEQ_LEN, max_sibs: int = 20) -> GraphData:
    ev = normalise_events(raw_events)
    acc_ids, acc_idx = np.unique(ev["account_id"].values, return_inverse=True)
    N = len(acc_ids)
    ev["a"] = acc_idx
    g = ev.groupby("a", sort=True)

    t = ev["timestamp"].astype("int64").values / 1e9                        # seconds
    ev["t"] = t
    ev["hour"] = ev["timestamp"].dt.hour.values
    ev["day"] = (t // 86400).astype(np.int64)
    prev_t = g["t"].shift(1)
    dt = (ev["t"] - prev_t).fillna(0.0).values                               # seconds
    ev["dt"] = dt
    first_t = g["t"].transform("min")
    ev["since_first"] = ev["t"] - first_t
    ev["is_night"] = ((ev["hour"] >= 0) & (ev["hour"] < 5)).astype(np.float32)

    def changed(col):
        prev = g[col].shift(1)
        return ((ev[col] != prev) & prev.notna()).astype(np.float32).values

    ev["dev_ch"], ev["ip_ch"], ev["addr_ch"] = changed("device_id"), changed("ip"), changed("address")
    ev["new_dev"] = (ev.groupby(["a", "device_id"]).cumcount() == 0).astype(np.float32) * (ev["device_id"] != "")

    # ---- statistics ------------------------------------------------------------------------
    cnt = g.size().reindex(range(N)).values.astype(float)
    S = {}
    S["n_events"] = cnt
    for k, code in (("n_login", 0), ("n_payment", 1), ("n_order", 2)):
        S[k] = (ev["etype"] == code).groupby(ev["a"]).sum().reindex(range(N)).fillna(0).values
    S["active_days"] = g["day"].nunique().reindex(range(N)).values
    S["span_days"] = g["since_first"].max().reindex(range(N)).values / 86400.0
    S["events_per_active_day"] = cnt / np.maximum(S["active_days"], 1)
    gap_min = np.where(ev["dt"].values > 0, ev["dt"].values, np.nan) / 60.0
    ev["gap_min"] = gap_min
    ev["has_gap"] = (~np.isnan(gap_min))
    S["mean_gap_min"] = np.nan_to_num(g["gap_min"].mean().reindex(range(N)).values, nan=0.0)
    S["median_gap_min"] = np.nan_to_num(g["gap_min"].median().reindex(range(N)).values, nan=0.0)
    ngap = np.maximum(g["has_gap"].sum().reindex(range(N)).values, 1)
    ev["lt1"] = (ev["gap_min"] < 1).astype(float)
    ev["lt10"] = (ev["gap_min"] < 10).astype(float)
    S["frac_gap_lt_1min"] = g["lt1"].sum().reindex(range(N)).values / ngap
    S["frac_gap_lt_10min"] = g["lt10"].sum().reindex(range(N)).values / ngap
    ev["lgap"] = np.log1p(np.nan_to_num(ev["gap_min"].values, nan=0.0))
    ev["lgap"] = np.where(ev["has_gap"], ev["lgap"], np.nan)
    S["std_log_gap"] = np.nan_to_num(g["lgap"].std().reindex(range(N)).values, nan=0.0)
    S["night_ratio"] = g["is_night"].mean().reindex(range(N)).values
    hb = ev.groupby(["a", (ev["t"] // 3600).astype(np.int64)]).size().groupby(level=0).max()
    S["max_events_1h"] = hb.reindex(range(N)).fillna(1).values
    db = ev.groupby(["a", "day"]).size().groupby(level=0).max()
    S["max_events_1d"] = db.reindex(range(N)).fillna(1).values
    for key, col in (("n_device", "device_id"), ("n_ip", "ip"), ("n_address", "address")):
        nn = ev[ev[col] != ""].groupby("a")[col].nunique()
        S[key] = nn.reindex(range(N)).fillna(0).values
    S["device_per_event"] = S["n_device"] / cnt
    S["ip_per_event"] = S["n_ip"] / cnt
    S["addr_per_event"] = S["n_address"] / cnt
    S["device_switch_rate"] = g["dev_ch"].sum().reindex(range(N)).values / np.maximum(cnt - 1, 1)
    S["ip_switch_rate"] = g["ip_ch"].sum().reindex(range(N)).values / np.maximum(cnt - 1, 1)
    S["addr_switch_rate"] = g["addr_ch"].sum().reindex(range(N)).values / np.maximum(cnt - 1, 1)
    pay = ev[ev["etype"] == 1]
    S["amount_mean"] = pay.groupby("a")["amount"].mean().reindex(range(N)).fillna(0).values
    S["amount_max"] = pay.groupby("a")["amount"].max().reindex(range(N)).fillna(0).values
    cv = (pay.groupby("a")["amount"].std() / (pay.groupby("a")["amount"].mean() + 1e-6)).reindex(range(N))
    S["amount_cv"] = np.nan_to_num(cv.values, nan=0.0)
    S["pay_ratio"] = S["n_payment"] / cnt
    fp = pay.groupby("a")["since_first"].min().reindex(range(N))
    S["hours_to_first_payment"] = np.nan_to_num(fp.values, nan=0.0) / 3600.0
    ev["f48"] = (ev["since_first"] <= 48 * 3600).astype(float)
    S["frac_first_48h"] = g["f48"].mean().reindex(range(N)).values
    # siblings
    person_codes, person_inv = np.unique(ev.groupby("a")["person_id"].first().reindex(range(N)).values, return_inverse=True)
    person_of = ev.groupby("a")["person_id"].first().reindex(range(N)).values
    per_size = np.bincount(person_inv)
    S["n_siblings"] = (per_size[person_inv] - 1).astype(float)

    stats = np.zeros((N, N_PROFILE), dtype=np.float32)
    for j, (key, _, islog, _) in enumerate(FEATURES[:N_PROFILE]):
        v = np.nan_to_num(np.asarray(S[key], dtype=float), nan=0.0, posinf=0.0)
        stats[:, j] = np.log1p(np.maximum(v, 0)) if islog else v

    # ---- sequences -------------------------------------------------------------------------
    ev["rank"] = g.cumcount()
    ev["n_in"] = g["a"].transform("size")
    keep = ev["rank"] >= (ev["n_in"] - T)
    pos = (ev["rank"] - np.maximum(ev["n_in"] - T, 0)).values
    hr = ev["hour"].values
    x_cols = [c for c in ev.columns if c.startswith("x_")]            # optional dataset-specific event features
    F = np.zeros((len(ev), SEQ_DIM + len(x_cols)), dtype=np.float32)
    F[:, 0] = np.log1p(ev["dt"].values / 60.0) / 8.0
    F[:, 1] = np.sin(2 * np.pi * hr / 24)
    F[:, 2] = np.cos(2 * np.pi * hr / 24)
    F[:, 3] = ev["is_night"].values
    F[np.arange(len(ev)), 4 + ev["etype"].values] = 1.0
    F[:, 7] = np.log1p(ev["amount"].values) / 8.0
    F[:, 8], F[:, 9], F[:, 10] = ev["dev_ch"].values, ev["ip_ch"].values, ev["addr_ch"].values
    F[:, 11] = ev["new_dev"].values
    if x_cols:
        F[:, SEQ_DIM:] = np.nan_to_num(ev[x_cols].to_numpy(dtype=np.float32), nan=0.0)
    ev_dt = (ev["dt"].values / 60.0).astype(np.float32)
    seq_row = -np.ones((N, T), dtype=np.int64)
    kk = keep.values
    seq_row[ev["a"].values[kk], pos[kk]] = np.nonzero(kk)[0]

    # ---- bipartite edges -------------------------------------------------------------------
    edges, ent_names, ent_deg = {}, {}, {}
    acct_sets = {r: [set() for _ in range(N)] for r in RELATIONS}
    has_rel = np.zeros((N, len(RELATIONS)), dtype=bool)
    graph_feats = np.zeros((N, len(GRAPH_BASELINE_KEYS)), dtype=np.float32)
    for ri, r in enumerate(RELATIONS):
        col = REL_COLUMN[r]
        sub = ev[ev[col] != ""]
        if len(sub) == 0:
            edges[r] = dict(src=np.zeros(0, np.int64), dst=np.zeros(0, np.int64), feat=np.zeros((0, 2), np.float32))
            ent_names[r], ent_deg[r] = np.array([], dtype=object), np.zeros((0, 2), np.float32)
            continue
        codes, uniq = pd.factorize(sub[col])
        pair = pd.DataFrame({"a": sub["a"].values, "e": codes}).groupby(["a", "e"]).size().reset_index(name="c")
        n_ev_acct = cnt[pair["a"].values]
        feat = np.stack([np.log1p(pair["c"].values), pair["c"].values / n_ev_acct], 1).astype(np.float32)
        edges[r] = dict(src=pair["a"].values.astype(np.int64), dst=pair["e"].values.astype(np.int64), feat=feat)
        ent_names[r] = np.asarray(uniq, dtype=object)
        deg_acc = np.bincount(pair["e"].values, minlength=len(uniq))
        deg_ev = np.bincount(codes, minlength=len(uniq))
        ent_deg[r] = np.stack([np.log1p(deg_acc), np.log1p(deg_ev)], 1).astype(np.float32)
        has_rel[np.unique(pair["a"].values), ri] = True
        for a_, e_ in zip(pair["a"].values, pair["e"].values):
            acct_sets[r][a_].add(e_)
        d = deg_acc[pair["e"].values].astype(float)
        dfp = pd.DataFrame({"a": pair["a"].values, "d": d}).groupby("a")["d"]
        graph_feats[dfp.max().index.values, 2 * ri] = dfp.max().values - 1
        graph_feats[dfp.mean().index.values, 2 * ri + 1] = dfp.mean().values - 1

    # ---- sibling (same personal ID) edges ----------------------------------------------------
    order = np.argsort(person_inv, kind="stable")
    bounds = np.flatnonzero(np.diff(person_inv[order])) + 1
    nl = {"device": stats[:, FEAT_KEYS.index("n_device")], "ip": stats[:, FEAT_KEYS.index("n_ip")],
          "address": stats[:, FEAT_KEYS.index("n_address")]}
    night = stats[:, FEAT_KEYS.index("night_ratio")]
    jsum = np.zeros((N, 3)); jcnt = np.zeros(N)
    src_l, dst_l, sf_l = [np.arange(N)], [np.arange(N)], [np.tile(np.array([1, 1, 1, 0, 0, 0, 0, 1], np.float32), (N, 1))]
    for members in np.split(order, bounds):
        if len(members) < 2:
            continue
        members = sorted(members.tolist(), key=lambda i: -cnt[i])[:max_sibs]
        s_, d_, f_ = [], [], []
        for i in members:
            for j in members:
                if j == i:
                    continue
                jac = [_jaccard(acct_sets[r][i], acct_sets[r][j]) for r in RELATIONS]
                f_.append(jac + [abs(nl["device"][i] - nl["device"][j]), abs(nl["ip"][i] - nl["ip"][j]),
                                 abs(nl["address"][i] - nl["address"][j]), abs(night[i] - night[j]), 0])
                s_.append(j); d_.append(i)
                jsum[i] += jac; jcnt[i] += 1
        src_l.append(np.array(s_)); dst_l.append(np.array(d_)); sf_l.append(np.array(f_, dtype=np.float32))
    sib_src = np.concatenate(src_l).astype(np.int64); sib_dst = np.concatenate(dst_l).astype(np.int64)
    sib_feat = np.concatenate(sf_l).astype(np.float32).reshape(-1, 8)
    has_sib = jcnt > 0
    graph_feats[has_sib, 6:9] = (jsum[has_sib] / jcnt[has_sib, None])
    graph_feats[~has_sib, 6:9] = -1.0

    gf = graph_feats.copy()
    gf[:, :6] = np.log1p(np.maximum(gf[:, :6], 0))
    keys = list(FEAT_KEYS); labels = [FEAT_LABEL[k] for k in keys]; islog = [FEAT_ISLOG[k] for k in keys]; units = [FEAT_UNIT[k] for k in keys]
    blocks = [stats, gf]
    if x_cols:                                       # account-level mean / max of every dataset-specific event feature
        gx = ev.groupby("a")[x_cols]
        mean_x = gx.mean().reindex(range(N)).to_numpy(np.float32)
        max_x = gx.max().reindex(range(N)).to_numpy(np.float32)
        blocks += [np.nan_to_num(mean_x), np.nan_to_num(max_x)]
        for tag, word in (("mean", "average"), ("max", "maximum")):
            for c in x_cols:
                keys.append(f"{tag}_{c}"); labels.append(f"{word} of {describe_x(c)}"); islog.append(False); units.append("")
    stats = np.hstack(blocks).astype(np.float32)
    split_arr = ev.groupby("a")["split"].first().reindex(range(N)).fillna(-1).astype(int).values if "split" in ev.columns else None
    lab = ev[ev["label"] >= 0].groupby("a")["label"].max().reindex(range(N)).fillna(-1).astype(int).values
    return GraphData(account_ids=acc_ids, person_ids=person_of, stats=stats, ev_feat=F, ev_dt=ev_dt,
                     seq_event_row=seq_row, labels=lab, edges=edges,
                     ent_names=ent_names, ent_deg=ent_deg, sib_src=sib_src, sib_dst=sib_dst, sib_feat=sib_feat,
                     events=ev, graph_feats=graph_feats, has_rel=has_rel, feat_keys=keys, feat_labels=labels,
                     feat_islog=islog, feat_units=units, split=split_arr, acct_index={a: i for i, a in enumerate(acc_ids)})
