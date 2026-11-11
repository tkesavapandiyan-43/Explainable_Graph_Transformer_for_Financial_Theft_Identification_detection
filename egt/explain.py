"""
Explainability for individual accounts - answers "WHY was this account judged fraud / legitimate?"

Everything is computed from the TRAINED model on the account's own sampled neighbourhood (the same sub-graph
procedure that is used for training and scoring):
  1. Feature attribution   Integrated Gradients on the account's own features (baseline = the average account)
  2. Module attribution    counterfactual: change of fraud log-odds when one fusion module is removed + fusion attention
  3. Graph evidence        attention over the account's devices / networks / addresses and over same-ID sibling accounts
  4. Behaviour evidence    attention of the T-LSTM pooling over the account's individual events
and a plain-English verdict that combines them.

Why Integrated Gradients and not SHAP / LIME / GNNExplainer?  IG is a path-attribution method with the
completeness property (attributions add up to the change in the model output), needs only gradients of the
trained network, and is exact for the quantity we explain (the account's own feature vector). KernelSHAP would
need thousands of full graph forward passes per account; LIME is a local surrogate with no guarantee of faithfulness
for a graph model; GNNExplainer learns an edge mask per instance, but our model already exposes its attention
over edges. They are therefore NOT implemented (documented replacement, not a missing feature).
"""
import numpy as np
import pandas as pd
import torch

from .config import RELATIONS

SHORT_MODULES = ["Profile", "Env. aggregation", "Env. difference", "Behaviour"]
MODULES = ["Profile statistics", "Environment aggregation (shared devices / networks / addresses)",
           "Environment difference (same-ID sibling accounts)", "Behaviour sequence (timing & activity)"]
REL_TITLE = {"device": "Device", "ip": "Network", "address": "Address"}


def _ego(pred, res, i):
    """Forward pass on the sampled neighbourhood of account i (cached). Returns (G, toks, info)."""
    if i not in res._ego:
        G = res.gi.sample([i], **pred.caps)
        pred.model.eval()
        with torch.no_grad():
            toks, info = pred.model.encode(G, explain=True)
            logits, info2 = pred.model.fuse(toks, explain=True)
        info.update(info2)
        res._ego[i] = (G, toks.detach(), info, logits)
    return res._ego[i]


def feature_attribution(pred, res, i, top=None, steps=12):
    """Integrated Gradients on the seed account's scaled feature vector (neighbours held fixed).
    Signed contribution to the fraud log-odds: positive -> FRAUD, negative -> LEGITIMATE."""
    G0, *_ = _ego(pred, res, i)
    x = G0["stats"][0].clone()
    total = torch.zeros_like(x)
    pred.model.eval()
    for s in range(1, steps + 1):
        st = G0["stats"].clone()
        st[0] = x * ((s - 0.5) / steps)
        st.requires_grad_(True)
        G = dict(G0); G["stats"] = st
        lg = pred.model(G)
        (lg[0, 1] - lg[0, 0]).backward()
        total += st.grad[0]
    attr = (total / steps * x).detach().numpy() / pred.T
    gd = res.gd
    med = np.median(gd.stats if gd.n <= 200000 else gd.stats[:200000], 0)
    rows = []
    for j, key in enumerate(gd.feat_keys):
        val, typ = gd.raw_value(j, gd.stats[i, j]), gd.raw_value(j, med[j])
        if key.startswith("sib_jacc") and val < 0:
            val = typ = float("nan")                           # no sibling accounts
        rows.append(dict(feature=key, label=gd.feat_labels[j], value=val, typical=typ, unit=gd.feat_units[j], attribution=float(attr[j])))
    df = pd.DataFrame(rows).sort_values("attribution", key=np.abs, ascending=False)
    return df.head(top) if top else df


def module_attribution(pred, res, i):
    G, toks, info, _ = _ego(pred, res, i)
    with torch.no_grad():
        full = pred.model.fuse(toks)
        d_full = float(full[0, 1] - full[0, 0])
        rows = []
        for k in range(4):
            lg = pred.model.fuse(toks, drop_token=k)
            d = float(lg[0, 1] - lg[0, 0])
            rows.append(dict(module=SHORT_MODULES[k], attention_weight=float(info["mod_w"][0, k]), effect_on_risk=(d_full - d) / pred.T))
    return pd.DataFrame(rows)


def entity_evidence(pred, res, i, thr_flag):
    G, _, info, _ = _ego(pred, res, i)
    gd, gi, prob = res.gd, res.gi, res.prob
    out = []
    alphas = info.get("agg_alpha")
    for r, rel in enumerate(RELATIONS):
        ed = G["edges"][r]
        m = (ed["src"] == 0).nonzero().flatten().numpy()
        if len(m) == 0:
            continue
        al = alphas[-1][r].mean(1).numpy()[m] if alphas else np.ones(len(m)) / len(m)
        ents = G["_ents"][r][ed["dst"][m].numpy()]
        for k, e in enumerate(ents):
            R = gi.rel[r]
            members = R["e_acct"][R["e_ptr"][e]:R["e_ptr"][e + 1]]
            others = members[members != i]
            out.append(dict(relation=REL_TITLE[rel], entity=str(gd.ent_names[rel][e]), attention=float(al[k]),
                            n_events=int(round(np.expm1(float(ed["feat"][m[k], 0])))), other_accounts=int(len(others)),
                            other_flagged=int((prob[others] >= thr_flag).sum()) if len(others) else 0,
                            avg_other_risk=float(prob[others].mean() * 100) if len(others) else 0.0))
    df = pd.DataFrame(out)
    return df.sort_values("attention", ascending=False) if len(df) else df


def sibling_evidence(pred, res, i, thr_flag):
    G, _, info, _ = _ego(pred, res, i)
    gd = res.gd
    m = ((G["sib_dst"] == 0) & (G["sib_src"] != 0)).nonzero().flatten().numpy()
    if len(m) == 0:
        return pd.DataFrame()
    al = info["diff_alpha"][-1].mean(1).numpy()
    rows = []
    for e in m:
        j = int(G["_nodes"][int(G["sib_src"][e])]); f = G["sib_feat"][e].numpy()
        rows.append(dict(sibling=str(gd.account_ids[j]), attention=float(al[e]), device_overlap=float(f[0]), network_overlap=float(f[1]),
                         address_overlap=float(f[2]), sibling_risk=float(res.prob[j] * 100), sibling_flagged=bool(res.prob[j] >= thr_flag)))
    return pd.DataFrame(rows).sort_values("attention", ascending=False)


def event_evidence(pred, res, i, top=8):
    G, _, info, _ = _ego(pred, res, i)
    gd = res.gd
    a = info["ev_alpha"][0].numpy(); rows_ = G["seq_row"][0].numpy()
    out = []
    for t in np.argsort(-a)[:top]:
        r = rows_[t]
        if r < 0:
            continue
        e = gd.events.iloc[r]
        out.append(dict(time=str(e["timestamp"]), type=["login", "payment", "order/other"][int(e["etype"])], device=e["device_id"],
                        network=e["ip"], address=e["address"], amount=float(e["amount"]), gap_min=float(e["dt"] / 60.0), attention=float(a[t])))
    return pd.DataFrame(out)


def _fmt(v, unit=""):
    s = f"{v:.0f}" if abs(v) >= 10 or float(v).is_integer() else f"{v:.2f}"
    return f"{s} {unit}".strip()


def explain_account(pred, res, account_id, mode="f1", manual=0.5, topk=0.05, tbl=None):
    gd = res.gd
    i = gd.acct_index[account_id]
    tbl = tbl if tbl is not None else pred.table(res, mode, manual, topk)
    row = tbl[tbl.account_id == account_id].iloc[0]
    thr = tbl.attrs["threshold"]
    is_fraud = row.prediction == "FRAUD"
    p = float(res.prob[i])
    fa = feature_attribution(pred, res, i)
    ma = module_attribution(pred, res, i)
    ent = entity_evidence(pred, res, i, thr)
    sib = sibling_evidence(pred, res, i, thr)
    evt = event_evidence(pred, res, i)
    from .shap_explain import shap_values
    sh = shap_values(pred, res, i)

    tot = fa.attribution.abs().sum() + 1e-9
    sig = fa[(fa.attribution.abs() / tot > 0.03) & fa.value.notna()]
    fraud_f, legit_f = sig[sig.attribution > 0].head(4), sig[sig.attribution < 0].head(4)
    reasons_f = [f"{r.label}: {_fmt(r.value, r.unit)} (typical {_fmt(r.typical, r.unit)})" for _, r in fraud_f.iterrows()]
    reasons_l = [f"{r.label}: {_fmt(r.value, r.unit)} (typical {_fmt(r.typical, r.unit)})" for _, r in legit_f.iterrows()]
    graph_f, graph_l = [], []
    base_flag = float((res.prob >= thr).mean())                     # share of flagged accounts in the whole input
    if len(ent):
        for _, r in ent.iterrows():
            share = r.other_flagged / max(r.other_accounts, 1)
            if r.other_accounts >= 2 and r.other_flagged >= 2 and share >= max(0.2, 3 * base_flag):      # clearly more fraud than average
                graph_f.append(f"{r.relation} {r.entity} is shared with {r.other_accounts} other account(s), {r.other_flagged} of them also flagged "
                               f"({share:.0%} vs {base_flag:.0%} in the whole input)")
        if ent.iloc[0].other_accounts == 0:
            graph_l.append("its device / network / address evidence is not shared with any other account")
    if len(sib):
        none = sib[sib[["device_overlap", "network_overlap", "address_overlap"]].max(axis=1) < 0.05]
        same = sib[sib[["device_overlap", "network_overlap", "address_overlap"]].max(axis=1) >= 0.4]
        if len(none):
            ex_ids = ", ".join(none.sibling.head(2))
            graph_f.append(f"{len(none)} sibling account(s) with the same personal ID share NO device/network/address with it "
                           f"({int(none.sibling_flagged.sum())} of them flagged), e.g. {ex_ids}")
        if len(same):
            graph_l.append(f"it shares its environment with {len(same)} sibling account(s) (e.g. {same.sibling.iloc[0]}, overlap "
                           f"{same[['device_overlap', 'network_overlap', 'address_overlap']].max(axis=1).iloc[0]:.0%}) - consistent with one real person")
    mod_top = ma.sort_values("effect_on_risk", ascending=False).iloc[0]
    lines = [f"Account {account_id} is predicted {'FRAUDULENT' if is_fraud else 'LEGITIMATE'} "
             f"(risk {p*100:.0f}/100, tier {row.risk_tier}, decision threshold {thr*100:.0f})."]
    if is_fraud:
        lines.append("WHY FRAUD:")
        lines += ["  + " + s for s in graph_f + reasons_f]
        if len(evt):
            hrs = pd.to_datetime(evt.head(5)["time"]).dt.hour
            nn = int((hrs < 5).sum())
            if nn >= 3:
                lines.append(f"  + {nn} of its 5 most influential events happened between 00:00 and 05:00")
        lines.append(f"  Strongest signal came from the module: {mod_top.module} (+{mod_top.effect_on_risk:.1f} log-odds).")
        if reasons_l or graph_l:
            lines.append("SIGNS OF LEGITIMACY (outweighed):")
            lines += ["  - " + s for s in graph_l + reasons_l[:2]]
    else:
        lines.append("WHY LEGITIMATE:")
        lines += ["  - " + s for s in graph_l + reasons_l]
        if reasons_f or graph_f:
            lines.append("MINOR WARNING SIGNS (not enough to flag):")
            lines += ["  + " + s for s in graph_f + reasons_f[:2]]
        lines.append(f"  Module with the strongest influence: {mod_top.module} ({mod_top.effect_on_risk:+.1f} log-odds).")
    st_ = sh["table"]
    pos, neg = st_[st_.shap > 0].head(2), st_[st_.shap < 0].head(1)
    lines.append("SHAP (Shapley values, log-odds): " + "; ".join(f"{r.group} {r.shap:+.2f}" for _, r in pd.concat([pos, neg]).iterrows()))
    return dict(account_id=account_id, risk=p * 100, tier=row.risk_tier, prediction=row.prediction, threshold=thr,
                narrative="\n".join(lines), features=fa, modules=ma, entities=ent, siblings=sib, events=evt, shap=sh, given_label=row.given_label)
