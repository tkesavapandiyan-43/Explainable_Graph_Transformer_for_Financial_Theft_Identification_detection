"""Streamlit GUI - Explainable Graph Transformer (EGT) for financial identity-theft detection.
Run:  python main.py gui     (or  streamlit run app.py)
Every number and figure is read from the trained models"""
import os, sys, warnings
warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.io as pio

from egt import study, study_plots as SP, study_text as ST
from egt.config import TIER_COLORS, TIER_NAMES
from egt.pipeline import Predictor, find_study_model, load_cached_result
from egt.explain import explain_account

st.set_page_config(page_title="EGT - Identity Theft Detection", page_icon="🛡️", layout="wide", initial_sidebar_state="collapsed")
BLUE, DARK, RED, GREEN = "#1565C0", "#0D47A1", "#D62728", "#2E9E5B"
pio.templates["egt"] = go.layout.Template(layout=dict(
    colorway=[BLUE, "#42A5F5", DARK, "#90CAF9"], paper_bgcolor="white", plot_bgcolor="#F5F9FE", font=dict(color="#0D2B52"),
    xaxis=dict(gridcolor="#DCE8F7", zerolinecolor="#C5D8F2"), yaxis=dict(gridcolor="#DCE8F7", zerolinecolor="#C5D8F2")))
pio.templates.default = "egt"

st.markdown("""
<style>
.stApp {background:#FFFFFF;}
section[data-testid='stSidebar'], div[data-testid='collapsedControl'], button[kind='header'] {display:none !important;}
.block-container {padding-top:1.4rem; max-width:1250px;}
.hero {background:linear-gradient(100deg,#0D47A1 0%,#1976D2 55%,#64B5F6 100%); color:#fff; padding:20px 28px; border-radius:16px; margin-bottom:14px;
       box-shadow:0 4px 14px rgba(21,101,192,.25);}
.hero h1 {color:#fff; margin:0; font-size:1.65rem;}
.hero p {color:#E3F0FF; margin:4px 0 0 0; font-size:.92rem;}
div[data-testid="stRadioGroup"] {gap:8px; flex-wrap:wrap;}
label[data-testid="stRadioOption"] {background:#EAF2FD; border:1px solid #CFE0F7; border-radius:999px; padding:5px 16px; margin:0;}
label[data-testid="stRadioOption"] > div > div:first-child {display:none;}
label[data-testid="stRadioOption"] p {color:#0D47A1; font-weight:600; font-size:.92rem;}
label[data-testid="stRadioOption"][data-selected="true"] {background:#1565C0; border-color:#1565C0;}
label[data-testid="stRadioOption"][data-selected="true"] p {color:#fff;}
div[data-testid="stMetric"] {background:#EAF2FD; border:1px solid #CFE0F7; border-left:5px solid #1565C0; padding:10px 14px; border-radius:10px;}
div[data-testid="stMetricValue"] {color:#0D47A1;}
h2, h3, h4 {color:#0D47A1;}
.note {background:#EAF2FD; border-left:5px solid #1565C0; padding:10px 14px; border-radius:8px; color:#0D2B52; margin:6px 0 12px 0; font-size:.92rem;}
.foot {color:#6B7F99; font-size:.82rem; margin-top:10px;}
.step {display:flex; gap:14px; background:#F5F9FE; border:1px solid #D5E5F8; border-radius:14px; padding:14px 18px; margin:10px 0;}
.step .n {flex:0 0 38px; height:38px; border-radius:50%; background:#1565C0; color:#fff; font-weight:700; display:flex; align-items:center; justify-content:center; font-size:1.05rem;}
.step .t {font-weight:700; color:#0D47A1; font-size:1.02rem;}
.step .d {color:#33475f; font-size:.9rem; margin-top:2px;}
.step .v {margin-top:6px; color:#0D2B52; font-size:.93rem;}
.step b {color:#0D47A1;}
.flow {display:flex; flex-wrap:wrap; gap:6px; align-items:center; margin:6px 0 4px 0;}
.flow span {background:#1565C0; color:#fff; padding:5px 12px; border-radius:999px; font-size:.82rem;}
.flow i {color:#1565C0; font-style:normal; font-weight:700;}
.chip {display:inline-block; padding:2px 10px; border-radius:999px; color:#fff; font-weight:700; font-size:.85rem;}
</style>""", unsafe_allow_html=True)

DS_NAMES = {k: v["title"] for k, v in study.DATASETS.items()}
DS_SHORT = {"meituan": "Dataset A · Meituan-style", "vesta": "Dataset B · Vesta-style"}
st.markdown('<div class="hero"><h1>🛡️ Fraud Detection Management</h1><p>Explainable Graph Transformer for financial identity-theft detection</p></div>', unsafe_allow_html=True)

PAGES = ["🔍 Investigator", "🧭 Inference", "🎚️ Risk scoring", "🕸️ Transaction graph", "📈 Performance", "📊 Comparison", "🎛️ Sensitivity", "🧬 Entity graphs", "🧩 Meta-paths"]
PI, PN, PR, PG, PP, PC, PS, PE, PM = PAGES
page = st.radio("page", PAGES, horizontal=True, label_visibility="collapsed", key="page")
if page in (PS, PM):                       # these two pages show both data sets at once
    ds = st.session_state.get("ds", "meituan")
else:
    ds = st.radio("dataset", list(DS_NAMES), format_func=lambda k: DS_SHORT[k], horizontal=True, label_visibility="collapsed", key="ds")

PROV = (" "
        "")


def stamp():
    d = os.path.join(study.STUDY, "results")
    return sum(int(os.path.getmtime(os.path.join(d, f))) for f in os.listdir(d)) if os.path.isdir(d) else 0


@st.cache_data(show_spinner="Drawing the figure ...")
def fig(name, arg, st_):
    f = getattr(SP, name)
    return f(arg) if arg is not None else f()


def show(path, width=None):
    if width:
        _, c, _ = st.columns([1, width, 1])
        c.image(path, width="stretch")
    else:
        st.image(path, width="stretch")


@st.cache_resource(show_spinner="Loading the model and the data set ...")
def load(ds_):
    ckpt, out, cache, _ = find_study_model(ROOT, ds_)
    if not ckpt:
        return None
    pred = Predictor(ckpt)
    return pred, load_cached_result(pred, cache, out)


@st.cache_data(show_spinner="Explaining the decision (SHAP, attention, integrated gradients) ...", max_entries=40)
def explain(ds_, acc, _pred, _res, _tbl):
    return explain_account(_pred, _res, acc, "f1", 0.5, 0.05, tbl=_tbl)


def need_study():
    st.warning("No results found yet. Run the whole study once:  `python main.py study`  (about 30-60 minutes on a laptop CPU).")
    st.stop()


def bullets(points):
    st.markdown("\n".join(f"- {p}" for p in points))


# ============================================================================================ account pages
def showcase(gd, tbl, thr):
    """A flagged account that has both shared entities and same-ID siblings, so the graph parts of the model have something to show."""
    ix = {k: j for j, k in enumerate(gd.feat_keys)}
    sib = np.expm1(gd.stats[:, ix["n_siblings"]])
    share = np.maximum.reduce([np.expm1(gd.stats[:, ix[k]]) for k in ("max_dev_deg", "max_ip_deg", "max_addr_deg")])
    ok = {a for a, sb, sh in zip(gd.account_ids, sib, share) if sb >= 1 and 2 <= sh <= 12}
    for r in tbl.itertuples():
        if r.risk_score / 100 >= thr and r.account_id in ok:
            return r.account_id
    return tbl.account_id.iloc[0]


def account_picker(key):
    loaded = load(ds)
    if loaded is None or loaded[1] is None:
        need_study()
    pred, res = loaded
    tbl = pred.table(res, "f1")
    ids = tbl.account_id.tolist()
    default = showcase(res.gd, tbl, tbl.attrs["threshold"])
    acc = st.selectbox("Account (sorted by risk; type to search)", ids[:20000], index=ids[:20000].index(default) if default in ids[:20000] else 0, key=key + ds)
    return pred, res, tbl, acc


if page == PI:
    pred, res, tbl, acc = account_picker("acc")
    thr, gd = tbl.attrs["threshold"], res.gd
    st.caption(f"Decision threshold: {thr*100:.1f} / 100 - the risk score that maximises F1 on the validation accounts of this data set.")
    ex = explain(ds, acc, pred, res, tbl)
    col = TIER_COLORS[ex["tier"]]
    a, b = st.columns([1, 2])
    with a:
        g = go.Figure(go.Indicator(mode="gauge+number", value=ex["risk"], number={"suffix": " / 100"}, title={"text": f"{ex['prediction']} · {ex['tier']}"},
                                   gauge={"axis": {"range": [0, 100]}, "bar": {"color": col}, "threshold": {"line": {"color": "black", "width": 4}, "value": thr * 100}}))
        g.update_layout(height=260, margin=dict(t=60, b=0, l=20, r=20)); st.plotly_chart(g, width="stretch")
        if ex["given_label"]:
            st.caption(f"Label in the data: **{ex['given_label']}**")
    with b:
        st.markdown("#### Why this decision?")
        st.code(ex["narrative"], language=None)
    m1, m2 = st.columns(2)
    with m1:
        md = ex["modules"]
        f = go.Figure(go.Bar(x=md.effect_on_risk, y=md.module, orientation="h", marker_color=[RED if v > 0 else BLUE for v in md.effect_on_risk],
                             text=[f"attention {w:.2f}" for w in md.attention_weight], textposition="auto"))
        f.update_layout(title="Module contribution (fraud log-odds lost if the module is removed)", height=300, margin=dict(t=50, b=10)); st.plotly_chart(f, width="stretch")
    with m2:
        fa = ex["features"].head(10).iloc[::-1]
        f = go.Figure(go.Bar(x=fa.attribution, y=fa.label, orientation="h", marker_color=[RED if v > 0 else BLUE for v in fa.attribution]))
        f.update_layout(title="Integrated-Gradients feature attribution (red → fraud, blue → legitimate)", height=300, margin=dict(t=50, b=10, l=10)); st.plotly_chart(f, width="stretch")
    sh = ex["shap"]; stb = sh["table"].head(10).iloc[::-1]
    sa, sb = st.columns([3, 2])
    with sa:
        f = go.Figure(go.Bar(x=stb.shap, y=stb.group, orientation="h", marker_color=[RED if v > 0 else BLUE for v in stb.shap],
                             text=[f"{v:+.2f}" for v in stb.shap], textposition="auto"))
        f.update_layout(title="SHAP values by feature group (red → fraud, blue → legitimate)", height=360, margin=dict(t=50, b=10, l=10))
        st.plotly_chart(f, width="stretch")
    with sb:
        st.markdown("**SHAP exactness check**")
        st.metric("Model output for this account minus average account", f"{sh['f_x'] - sh['f_base']:+.3f}")
        st.metric("Sum of all SHAP values", f"{sh['sum_shap']:+.3f}")
        st.caption("SHAP (Shapley) values add up exactly to the difference between this account's fraud log-odds and the average account's.")
    gcol, ecol = st.columns([1, 1])
    with gcol:
        st.markdown("**Neighbourhood of this account**")
        st.pyplot(SP.fig_account_graph(gd, acc, ex["entities"], ex["siblings"]), clear_figure=True)
    with ecol:
        st.markdown("**Shared devices / networks / addresses (attention-ranked)**")
        st.dataframe(ex["entities"].round(3) if len(ex["entities"]) else pd.DataFrame({"info": ["no device / network / address information"]}), hide_index=True, width="stretch")
        st.markdown("**Sibling accounts under the same personal identity**")
        st.dataframe(ex["siblings"].round(3) if len(ex["siblings"]) else pd.DataFrame({"info": ["no sibling accounts"]}), hide_index=True, width="stretch")
        st.markdown("**Most influential events (T-LSTM attention)**")
        st.dataframe(ex["events"].round(3), hide_index=True, width="stretch")

# ============================================================================================ inference walkthrough
elif page == PN:
    pred, res, tbl, acc = account_picker("inf")
    gd, thr = res.gd, tbl.attrs["threshold"]
    ex = explain(ds, acc, pred, res, tbl)
    i = gd.acct_index[acc]
    p = float(res.prob[i]); eps = 1e-6
    z = pred.T * float(np.log((p + eps) / (1 - p + eps)))
    ix = {k: j for j, k in enumerate(gd.feat_keys)}
    raw = lambda k: gd.raw_value(ix[k], gd.stats[i, ix[k]])
    ent, sib, md = ex["entities"], ex["siblings"], ex["modules"]
    tiers = pred.thresholds["tiers"]
    st.markdown("### How one account is judged - the whole inference, step by step")
    st.markdown('<div class="flow"><span>1 Events</span><i>→</i><span>2 Graph</span><i>→</i><span>3 Four extractors</span><i>→</i><span>4 Attention fusion</span><i>→</i>'
                '<span>5 Detection layer</span><i>→</i><span>6 Calibrated risk 0-100</span><i>→</i><span>7 Threshold · tier</span><i>→</i><span>8 Explanation</span></div>', unsafe_allow_html=True)

    def step(n, title, desc, vals):
        st.markdown(f'<div class="step"><div class="n">{n}</div><div><div class="t">{title}</div><div class="d">{desc}</div><div class="v">{vals}</div></div></div>', unsafe_allow_html=True)

    step(1, "Input - the account's raw activity", "Every login / payment / order of the account becomes 40 account statistics and a sequence of its last 40 events.",
         f"<b>{raw('n_events'):.0f}</b> events over <b>{raw('active_days'):.0f}</b> active days · <b>{raw('n_device'):.0f}</b> device(s), <b>{raw('n_ip'):.0f}</b> IP network(s), "
         f"<b>{raw('n_address'):.0f}</b> address(es) · night-time share <b>{raw('night_ratio')*100:.0f}%</b> · <b>{raw('n_siblings'):.0f}</b> sibling account(s) with the same personal ID")
    n_ent = len(ent) if len(ent) else 0
    shared = int((ent.other_accounts > 0).sum()) if n_ent else 0
    step(2, "Graph - who is connected to this account?",
         "Two graphs are used. <b>Heterogeneous</b>: account ↔ device / IP / address (accounts sharing an entity become neighbours). "
         "<b>Homogeneous</b>: account ↔ account when they carry the same personal ID.",
         f"{n_ent} entity link(s), <b>{shared}</b> of them shared with other accounts (up to "
         f"<b>{int(ent.other_accounts.max()) if n_ent else 0}</b> others) · <b>{len(sib)}</b> same-ID sibling account(s)")
    c1, c2 = st.columns([1, 1])
    with c1:
        st.pyplot(SP.fig_account_graph(gd, acc, ent, sib), clear_figure=True)
    with c2:
        st.markdown("")
        mod_desc = {"Profile": "the 40 statistics of the account itself", "Env. aggregation": "graph transformer over the heterogeneous graph: what do accounts that share my devices / IPs / addresses look like?",
                    "Env. difference": "graph transformer over same-ID siblings: how DIFFERENT are their environments from mine?", "Behaviour": "time-aware LSTM + attention over the event sequence: when and how fast does the account act?"}
        step(3, "Four extractors → four tokens", "Each extractor turns its view of the account into one vector (a token).",
             "<br>".join(f"<b>{r.module}</b> - {mod_desc[r.module]}" for r in md.itertuples()))
    f = go.Figure(go.Bar(x=md.attention_weight, y=md.module, orientation="h", marker_color=BLUE, text=[f"{v:.2f}" for v in md.attention_weight], textposition="outside"))
    f.update_layout(height=230, margin=dict(t=30, b=10, l=10, r=40), title="Step 4 - attentive fusion: how much attention each token receives", xaxis=dict(range=[0, max(0.6, float(md.attention_weight.max()) * 1.2)]))
    st.plotly_chart(f, width="stretch")
    step(4, "Attentive fusion", "The four tokens attend to each other (multi-head self-attention); a token-attention pooling then decides how much each one counts.",
         f"Most attention: <b>{md.sort_values('attention_weight').iloc[-1].module}</b> ({md.attention_weight.max():.2f}). "
         f"Removing a module changes the fraud log-odds by: " + ", ".join(f"{r.module} <b>{r.effect_on_risk:+.2f}</b>" for r in md.itertuples()))
    step(5, "Detection layer", "A small neural network (MLP) turns the fused vector into one score: the fraud log-odds.",
         f"raw fraud log-odds <b>z = {z:+.2f}</b>")
    step(6, "Calibration → risk score 0-100", f"Temperature scaling (T = {pred.T:.3f}, learnt on validation accounts) makes the score a trustworthy probability: p = 1 / (1 + e<sup>-z/T</sup>).",
         f"probability <b>{p:.3f}</b> → dynamic risk score <b>{p*100:.1f} / 100</b>")
    bands = [(0, tiers[0], "LOW"), (tiers[0], tiers[1], "MEDIUM"), (tiers[1], tiers[2], "HIGH"), (tiers[2], 1, "CRITICAL")]
    f = go.Figure()
    for lo, hi, nm in bands:
        f.add_trace(go.Bar(x=[(hi - lo) * 100], y=["risk"], base=lo * 100, orientation="h", marker_color=TIER_COLORS[nm], name=nm, text=nm, textposition="inside", insidetextanchor="middle",
                           hovertemplate=f"{nm}: {lo*100:.1f} - {hi*100:.1f}<extra></extra>"))
    f.add_trace(go.Scatter(x=[p * 100], y=["risk"], mode="markers", marker=dict(symbol="diamond", size=22, color="black", line=dict(color="white", width=2)), name="this account"))
    f.add_vline(x=thr * 100, line_dash="dash", line_color="black", annotation_text=f"decision threshold {thr*100:.1f}", annotation_position="top")
    f.update_layout(barmode="stack", height=190, margin=dict(t=40, b=10, l=10, r=10), xaxis=dict(range=[0, 100], title="risk score"), yaxis=dict(visible=False), legend=dict(orientation="h", y=-0.5))
    st.plotly_chart(f, width="stretch")
    col = TIER_COLORS[ex["tier"]]
    th = pred.thresholds
    step(7, "Learnt thresholds → tier → decision", "No threshold is typed in: they are read off the validation accounts (F1-optimal point; 90 %-precision point; 90 %-recall point). The tier boundaries come from the same curve.",
         f"tier <span class='chip' style='background:{col}'>{ex['tier']}</span> · decision <b>{ex['prediction']}</b> (risk {p*100:.1f} vs threshold {thr*100:.1f}) · "
         f"policies: balanced {th['f1']*100:.1f} · precision-90 {th['precision90']*100:.1f} · recall-90 {th['recall90']*100:.1f}")
    sh = ex["shap"]["table"].head(4)
    step(8, "Explanation - why?", "SHAP values (exact Shapley estimator over feature groups) say which evidence pushed the score up or down; attention shows which entities and events mattered.",
         "<br>".join(f"{'🔴' if r.shap > 0 else '🔵'} <b>{r.group}</b> {r.shap:+.2f}" for r in sh.itertuples()))
    with st.expander("Full written explanation"):
        st.code(ex["narrative"], language=None)

# ============================================================================================ model comparison
elif page == PC:
    s_ = stamp()
    if not s_:
        need_study()
    st.markdown("### Model comparison on the two data sets")
    a = SP.agg_main(ds)
    info = SP.main_results(ds).get("_info", {})
    if "egt" not in a:
        st.info("Results for this data set are still being computed."); st.stop()
    gnn = [k for k, _ in study.MODELS if k in a]
    ranked = sorted(gnn, key=lambda k: -a[k]["AUC"][0])
    c = st.columns(4)
    c[0].metric("EGT - AUC", f"{a['egt']['AUC'][0]:.3f} ± {a['egt']['AUC'][1]:.3f}")
    c[1].metric("EGT - F1", f"{a['egt']['F1'][0]:.3f}")
    c[2].metric("EGT rank by AUC", f"{ranked.index('egt')+1} of {len(gnn)}")
    c[3].metric("Best model by AUC", f"{SP.NAME[ranked[0]]}  {a[ranked[0]]['AUC'][0]:.3f}")
    if info:
        st.caption(f"{DS_NAMES[ds]}: {info['accounts']:,} accounts · {info['events']:,} events · {info['fraud']:,} fraud / {info['labelled']-info['fraud']:,} legitimate labelled · "
                   f"split train/val/test = {info['split'][0]:,} / {info['split'][1]:,} / {info['split'][2]:,} · identical split for every model")
    t1, t2, t3 = st.tabs(["Both data sets", "This data set in detail", "Table"])
    with t1:
        m = st.radio("metric", ["AUC", "F1"], horizontal=True, key="both_m", label_visibility="collapsed")
        show(fig("fig_both", m, s_))
    with t2:
        show(fig("fig_comparison", ds, s_))
        show(fig("fig_curves", ds, s_))
        show(fig("fig_convergence", ds, s_))
    with t3:
        both = None
        for k_ in study.DATASETS:
            t = SP.table_main(k_)[["Model", "AUC", "F1"]].rename(columns={"AUC": f"AUC {k_[:3].upper()}", "F1": f"F1 {k_[:3].upper()}"}) if SP.agg_main(k_) else None
            if t is not None:
                both = t if both is None else both.merge(t, on="Model", how="outer")
        if both is not None:
            st.markdown("**Both data sets** (A = Meituan-style, B = Vesta-style; mean of 2 training seeds)")
            st.dataframe(both.round(4), hide_index=True, width="stretch")
        st.markdown(f"**{DS_NAMES[ds]} - all metrics on the test accounts**")
        tt = SP.table_main(ds).drop(columns=["AUC ±"]).round(4)
        st.dataframe(tt, hide_index=True, width="stretch")
        dflt = SP.agg_main(ds, "default")
        if dflt:
            with st.expander("Same models with default hyper-parameters (no tuning)"):
                st.dataframe(pd.DataFrame([{"Model": SP.NAME.get(k, k), "AUC": v["AUC"][0], "F1": v["F1"][0], "Precision": v["Precision"][0], "Recall": v["Recall"][0]} for k, v in dflt.items()]).round(4),
                             hide_index=True, width="stretch")
    st.markdown("#### Discussion")
    bullets(ST.comparison_points(ds))
    st.caption(ST.protocol_note(ds))
    st.markdown("#### The compared models")
    st.markdown("""
| Model | Idea | Uses |
|---|---|---|
| **EGT** | graph transformers on the device / IP / address graph **and** the same-ID graph, T-LSTM behaviour, attention fusion | graph + behaviour + profile |
| **EnvIT** | GAT on meta-path graphs, GraphSAGE for the difference, T-LSTM behaviour, attention fusion | graph + behaviour |
| **EnvIT-Graph / EnvIT-Beh** | EnvIT with only its graph part / only its behaviour part | graph / behaviour |
| **GAT** | multi-head attention over neighbouring accounts | graph |
| **GraphSAGE** | mean aggregation of neighbours, concatenated with own state | graph |
| **R-GCN** | one weight matrix per relation (device, IP, address, same-ID) | graph |
| **HAN** | node-level attention per meta-path, then meta-path-level attention | graph |
| **HACUD** | hierarchical attention account → entity → account on the heterogeneous graph | graph |
| **SOBT** | Transformer over the event sequence (no graph) | behaviour |
| **SEC-GFD** | spectral filter bank (low / high / band-pass) with adaptive gating | graph |
""")
    st.markdown(f'<div class="foot">All models are simplified re-implementations in plain PyTorch with the same inputs, sampler and training loop. {PROV}</div>', unsafe_allow_html=True)

# ============================================================================================ sensitivity
elif page == PS:
    s_ = stamp()
    if not s_:
        need_study()
    st.markdown("One hyper-parameter is changed at a time; the dotted line marks the default. One training run per point, both data sets.")
    m = st.radio("metric", ["AUC", "F1"], horizontal=True, key="sens_m", label_visibility="collapsed")
    show(fig("fig_sweeps", m, s_))
    rows = []
    for sw, (title, vals) in study.SWEEPS.items():
        for v in vals:
            row = {"Hyper-parameter": title, "Value": f"{v:g}"}
            for k_ in study.DATASETS:
                r = SP.results(k_, "sweep").get(f"{sw}|{v}")
                row[f"AUC {k_[:3].upper()}"] = round(r["AUC"], 4) if r else None
                row[f"F1 {k_[:3].upper()}"] = round(r["F1"], 4) if r else None
            rows.append(row)
    st.markdown("#### Discussion")
    bullets(ST.sweep_points())
    with st.expander("All values"):
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.markdown("**Defaults:** learning rate 1e-3 · batch size 512 · embedding size 48 · time step 40 (events per behaviour sequence). One parameter is changed at a time, the others stay at their defaults; one training run per point.")
    st.markdown(f'<div class="foot">{PROV}</div>', unsafe_allow_html=True)

# ============================================================================================ entity graphs
elif page == PR:
    pred, res, tbl, acc = account_picker("risk")
    gd, th = res.gd, pred.thresholds
    st.markdown("### Dynamic risk scoring")
    c = st.columns(4)
    c[0].metric("Calibration temperature T", f"{pred.T:.3f}")
    c[1].metric("Balanced threshold (F1)", f"{th['f1']*100:.1f}")
    c[2].metric("Precision-90 threshold", f"{th['precision90']*100:.1f}")
    c[3].metric("Recall-90 threshold", f"{th['recall90']*100:.1f}")
    st.pyplot(SP.fig_risk(gd, res.prob, th), clear_figure=True)
    y, p_ = SP.test_arrays(gd, res.prob)
    rows = []
    for nm, key in (("Balanced (max F1)", "f1"), ("High precision (>=90 %)", "precision90"), ("High recall (>=90 %)", "recall90")):
        flag = p_ >= th[key]; tp = int((flag & (y == 1)).sum())
        rows.append({"Policy": nm, "Threshold": round(th[key] * 100, 1), "Flagged %": round(flag.mean() * 100, 1), "Precision": round(tp / max(flag.sum(), 1), 3), "Recall": round(tp / max((y == 1).sum(), 1), 3)})
    a_, b_ = st.columns([3, 2])
    with a_:
        st.markdown("**Threshold policies on the test accounts**")
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    with b_:
        st.markdown("**Risk tiers on the test accounts**")
        tr = []
        edges = [0] + list(th["tiers"]) + [1]
        for k, nm in enumerate(TIER_NAMES):
            m = (p_ >= edges[k]) & (p_ < edges[k + 1] if k < 3 else p_ <= 1)
            tr.append({"Tier": nm, "From": round(edges[k] * 100, 1), "Accounts": int(m.sum()), "Fraud rate %": round(float(y[m].mean() * 100), 1) if m.any() else None})
        st.dataframe(pd.DataFrame(tr), hide_index=True, width="stretch")
    st.markdown(f"**Selected account {acc}:** risk {float(res.prob[gd.acct_index[acc]])*100:.1f} → tier **{tbl[tbl.account_id == acc].risk_tier.iloc[0]}**, decision **{tbl[tbl.account_id == acc].prediction.iloc[0]}**")

elif page == PG:
    pred, res, tbl, acc = account_picker("graph")
    thr = tbl.attrs["threshold"]
    st.pyplot(SP.fig_rings(res.gd, res.prob, thr), clear_figure=True)
    ex = explain(ds, acc, pred, res, tbl)
    st.markdown(f"**Neighbourhood of {acc}**")
    _, c, _ = st.columns([1, 2, 1])
    with c:
        st.pyplot(SP.fig_account_graph(res.gd, acc, ex["entities"], ex["siblings"]), clear_figure=True)

elif page == PP:
    loaded = load(ds)
    if loaded is None or loaded[1] is None:
        need_study()
    pred, res = loaded
    tbl = pred.table(res, "f1")
    thr = tbl.attrs["threshold"]
    y, p_ = SP.test_arrays(res.gd, res.prob)
    from egt.metrics import full_metrics
    m = full_metrics(y, p_, thr)
    c = st.columns(6)
    for col, (lab, k) in zip(c, (("AUC", "AUC"), ("PR-AUC", "PR_AUC"), ("F1", "F1"), ("Precision", "Precision"), ("Recall", "Recall"), ("Accuracy", "Accuracy"))):
        col.metric(lab, f"{m[k]:.3f}")
    st.pyplot(SP.fig_performance(res.gd, res.prob, thr), clear_figure=True)
    hist = pred.extra.get("history")
    if hist:
        h = pd.DataFrame(hist)
        f = go.Figure()
        f.add_trace(go.Scatter(x=h.epoch, y=h.loss, name="training loss", line=dict(color=RED)))
        f.add_trace(go.Scatter(x=h.epoch, y=h.val_auc, name="validation AUC", yaxis="y2", line=dict(color=BLUE)))
        f.update_layout(height=320, title="Training curves", xaxis_title="epoch", yaxis=dict(title="loss"), yaxis2=dict(title="validation AUC", overlaying="y", side="right"), legend=dict(orientation="h", y=-0.25))
        st.plotly_chart(f, width="stretch")

elif page == PE:
    st.markdown(f"### Device · IP address · delivery address · personal ID - {DS_NAMES[ds]}")
    show(fig("fig_entity_graphs", ds, 0), 7)
    show(fig("fig_entity_stats", ds, 0), 8)
    st.markdown("#### Reading the graphs")
    bullets(ST.entity_points(ds))
    st.markdown("Fraud accounts of one ring reuse the same few devices, IP addresses and delivery addresses (graphs a-c), while a stolen personal ID is used by an account whose "
                "environment differs from the victim's (graph d, right-hand panel: lower overlap for fraud accounts). The first finding is what the **environment-aggregation** module "
                "learns, the second is what the **environment-difference** module learns.")
    st.markdown(f'<div class="foot">The example hubs in the left figure are picked automatically from the data (shared by 5-9 accounts, most fraud labels). {PROV}</div>', unsafe_allow_html=True)

# ============================================================================================ meta-paths
else:
    s_ = stamp()
    if not s_:
        need_study()
    st.markdown("A meta-path connects two accounts through a shared entity: **A-D-A** (same device), **A-I-A** (same IP / network), **A-L-A** (same delivery address). "
                "EGT is trained with only one of the three relations and compared with all three together.")
    show(fig("fig_metapath", None, s_))
    rows = []
    for k_ in study.DATASETS:
        d = SP.results(k_, "metapath")
        for n, _ in study.METAPATHS:
            if n in d:
                rows.append({"Data set": SP.SHORT[k_], "Meta-path": n, "AUC": round(d[n]["AUC"], 4), "F1": round(d[n]["F1"], 4), "PR-AUC": round(d[n]["PR_AUC"], 4)})
    st.markdown("#### Discussion")
    bullets(ST.metapath_points())
    with st.expander("Numbers"):
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.markdown(f'<div class="foot">{PROV}</div>', unsafe_allow_html=True)
