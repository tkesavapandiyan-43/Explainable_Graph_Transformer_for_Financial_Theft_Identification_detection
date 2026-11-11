"""
SHAP (Shapley-value) explanation of one account  -  NOVELTY: explainable decisions.

Algorithm = interventional permutation SHAP (Strumbelj & Kononenko 2014; Lundberg & Lee 2017):
  f(z)    : the trained model's fraud log-odds for the account when its feature vector is z (its graph neighbourhood is fixed)
  base    : the "average account" (all standardised features = 0, i.e. the training mean)
  phi_j   : average, over random feature orderings, of the change f(z + feature j) - f(z) when feature j is switched from the
            baseline value to the account's real value
  efficiency property (checked in tests):   sum_j phi_j  =  f(x) - f(base)      (exact for every ordering)
Features are explained in semantic GROUPS (activity volume, timing & bursts, sharing with other accounts, ...) because
correlated features (e.g. mean and maximum of the same quantity) otherwise split their credit arbitrarily; with
by="feature" every single feature is explained (slower).  Antithetic (reversed) orderings are used for lower variance.
This is our own implementation of the SHAP estimator, not the `shap` package, so the project has no extra dependency.
"""
import numpy as np
import pandas as pd
import torch

GROUP_RULES = [
    ("Activity volume", ("n_events", "n_login", "n_payment", "n_order", "active_days", "span_days", "events_per_active_day", "pay_ratio", "max_events_1h", "max_events_1d")),
    ("Timing & bursts", ("mean_gap_min", "median_gap_min", "frac_gap_lt_1min", "frac_gap_lt_10min", "std_log_gap", "hours_to_first_payment", "frac_first_48h")),
    ("Night-time activity", ("night_ratio",)),
    ("Devices / networks / addresses used", ("n_device", "n_ip", "n_address", "device_per_event", "ip_per_event", "addr_per_event")),
    ("Switching devices / networks / addresses", ("device_switch_rate", "ip_switch_rate", "addr_switch_rate")),
    ("Payment amounts", ("amount_mean", "amount_max", "amount_cv")),
    ("Sharing with other accounts (graph)", ("max_dev_deg", "mean_dev_deg", "max_ip_deg", "mean_ip_deg", "max_addr_deg", "mean_addr_deg")),
    ("Sibling accounts, same personal ID", ("n_siblings", "sib_jacc_dev", "sib_jacc_ip", "sib_jacc_addr")),
]
def _family(body):
    b = body.lower()
    if b.startswith("vpca"): return "Vesta V-feature components"
    if len(b) > 1 and b[0] == "c" and b[1:].isdigit(): return "Counting features C1-C14"
    if len(b) > 1 and b[0] == "d" and b[1:].isdigit(): return "Time-delta features D1-D15"
    if len(b) > 1 and b[0] == "m" and b[1:].isdigit(): return "Match flags M1-M9"
    if b.startswith("id_"): return "Identity fields id_*"
    if b.startswith("freq_"): return "Popularity (frequency) features"
    if b.startswith(("amt", "prod_", "card")): return "Amount / product / card features"
    if b.startswith("miss_"): return "Missing-value rates"
    return "Other transaction features"


def feature_groups(gd, by="group"):
    keys = list(gd.feat_keys)
    if by == "feature":
        return {gd.feat_labels[j]: [j] for j in range(len(keys))}
    out = {}
    fixed = {k: g for g, ks in GROUP_RULES for k in ks}
    for j, k in enumerate(keys):
        if k in fixed:
            g = fixed[k]
        else:
            body = k.split("_", 1)[1] if k.startswith(("mean_", "max_")) else k
            body = body[2:] if body.startswith("x_") else body
            g = _family(body)
        out.setdefault(g, []).append(j)
    return out


def shap_values(pred, res, i, by="group", n_perm=12, seed=0):
    """Returns a DataFrame (group, value summary, shap) and the exactness check. shap is in calibrated fraud log-odds."""
    from .explain import _ego
    key = (i, by, n_perm)
    if key in res.__dict__.setdefault("_shap", {}):
        return res._shap[key]
    G0, *_ = _ego(pred, res, i)
    x = G0["stats"][0].clone()
    base = torch.zeros_like(x)
    groups = feature_groups(res.gd, by)
    names = list(groups)
    idx = [torch.tensor(groups[n]) for n in names]
    model = pred.model; model.eval()

    def f(z):
        st = G0["stats"].clone(); st[0] = z
        G = dict(G0); G["stats"] = st
        with torch.no_grad():
            lg = model(G)
        return float(lg[0, 1] - lg[0, 0])
    rng = np.random.default_rng(seed)
    phi = np.zeros(len(names)); cnt = 0
    f_base, f_x = f(base), f(x)
    for _ in range(n_perm):
        order = rng.permutation(len(names))
        for ordr in (order, order[::-1]):                         # antithetic pair
            z, prev = base.clone(), f_base
            for g in ordr:
                z[idx[g]] = x[idx[g]]
                cur = f(z)
                phi[g] += cur - prev; prev = cur
            cnt += 1
    phi = phi / cnt / pred.T
    gd = res.gd
    rows = []
    for n, ix in zip(names, [groups[n] for n in names]):
        top = max(ix, key=lambda j: abs(float(x[j])))
        rows.append(dict(group=n, n_features=len(ix), strongest_feature=gd.feat_labels[top],
                         value=gd.raw_value(top, gd.stats[i, top]), shap=float(phi[names.index(n)])))
    df = pd.DataFrame(rows).sort_values("shap", key=np.abs, ascending=False).reset_index(drop=True)
    out = dict(table=df, f_x=f_x / pred.T, f_base=f_base / pred.T, sum_shap=float(phi.sum()), by=by)
    res._shap[key] = out
    return out
