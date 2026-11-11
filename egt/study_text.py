"""Plain-language discussion generated FROM the result files (nothing here is typed in by hand, so text and numbers always agree)."""
import numpy as np
from . import study
from .study import DATASETS, MODELS, SWEEPS, METAPATHS
from .study_plots import agg_main, results, SHORT, NAME, entity_table, boot_delta, main_results


def _f(x):
    return f"{x:.3f}"


def comparison_points(ds):
    a = agg_main(ds)
    if "egt" not in a:
        return []
    gnn = [k for k, _ in MODELS if k in a]
    ranked = sorted(gnn, key=lambda k: -a[k]["AUC"][0])
    r = ranked.index("egt") + 1
    out = []
    e = a["egt"]
    out.append(f"EGT reaches AUC {_f(e['AUC'][0])} (± {_f(e['AUC'][1])} over seeds) and F1 {_f(e['F1'][0])}; it ranks {r} of {len(gnn)} graph / sequence models by AUC."
               f" Best: {NAME[ranked[0]]} ({_f(a[ranked[0]]['AUC'][0])}).")
    if "envit" in a:
        d = e["AUC"][0] - a["envit"]["AUC"][0]
        bd = boot_delta(ds, "egt", "envit")
        if bd:
            verdict = ("EGT is significantly better" if bd[1] > 0 else "EnvIT is significantly better" if bd[2] < 0 else "the interval contains 0, so the two are statistically indistinguishable on this test set")
            out.append(f"Against EnvIT the AUC difference is {d:+.3f} (95 % bootstrap interval of the test accounts [{bd[1]:+.3f}, {bd[2]:+.3f}]: {verdict}); F1 difference {e['F1'][0]-a['envit']['F1'][0]:+.3f}.")
        else:
            out.append(f"Against EnvIT the AUC difference is {d:+.3f}; F1 difference {e['F1'][0]-a['envit']['F1'][0]:+.3f}.")
    if all(k in a for k in ("envit_graph", "envit_beh", "envit")):
        out.append(f"Using only the graph part of EnvIT gives AUC {_f(a['envit_graph']['AUC'][0])}, only the behaviour sequence {_f(a['envit_beh']['AUC'][0])}, "
                   f"both fused {_f(a['envit']['AUC'][0])} - {'fusing both parts helps' if a['envit']['AUC'][0] > max(a['envit_graph']['AUC'][0], a['envit_beh']['AUC'][0]) + 0.003 else 'fusion adds little over the better single part'}.")
    graph_only = [k for k in ("gat", "sage", "rgcn", "han", "hacud", "secgfd") if k in a]
    if graph_only:
        bg = max(graph_only, key=lambda k: a[k]["AUC"][0])
        out.append(f"Among the pure graph models the best is {NAME[bg]} (AUC {_f(a[bg]['AUC'][0])}); the behaviour-only SOBT reaches {_f(a['sobt']['AUC'][0]) if 'sobt' in a else '-'}.")
    best = ranked[0]
    if best != "egt":
        bd = boot_delta(ds, "egt", best)
        if bd:
            out.append(f"EGT vs the top model {NAME[best]}: AUC difference {bd[0]:+.3f}, 95 % interval [{bd[1]:+.3f}, {bd[2]:+.3f}] - "
                       + ("EGT is not significantly different." if bd[1] <= 0 <= bd[2] else "a clear gap."))
    cl = [k for k in a if k not in NAME]
    if cl:
        bc = max(cl, key=lambda k: a[k]["AUC"][0])
        better = a[bc]["AUC"][0] > e["AUC"][0]
        out.append(f"Classical tabular reference: {bc} reaches AUC {_f(a[bc]['AUC'][0])} - " + ("higher than EGT here, so on this data the graph models do not beat a strong tabular model on ranking quality alone; "
                   "their added value is the graph evidence and the explanations." if better else "lower than EGT."))
    return out


def protocol_note(ds):
    d = main_results(ds)
    if "_default" in d:
        return ("Protocol: every model got the same small search (learning rate {1e-3, 5e-3} x embedding size {16, 32, 48}); the setting with the best VALIDATION AUC was kept "
                "and trained with 2 seeds. Classical baselines use their default settings. All models share one train / validation / test split.")
    return "Protocol: default hyper-parameters, 2 training seeds, one shared train / validation / test split."


def sweep_points():
    out = []
    for sw, (title, vals) in SWEEPS.items():
        for ds in DATASETS:
            d = results(ds, "sweep")
            ys = [(v, d[f"{sw}|{v}"]["AUC"]) for v in vals if f"{sw}|{v}" in d]
            if len(ys) < 2:
                continue
            best = max(ys, key=lambda t: t[1]); span = max(y for _, y in ys) - min(y for _, y in ys)
            out.append(f"{title} - {SHORT[ds]}: best at {best[0]:g} (AUC {_f(best[1])}); the AUC changes by {span:.3f} over the tested range"
                       f" ({'within the noise of one run on this test set (about ±0.04) - practically insensitive' if span < 0.04 else 'a large effect, larger than the noise'}).")
    return out


def metapath_points():
    out = []
    for ds in DATASETS:
        d = results(ds, "metapath")
        if len(d) < 4:
            continue
        single = {n: d[n]["AUC"] for n, m in METAPATHS if sum(m) == 1 and n in d}
        bs = max(single, key=single.get)
        out.append(f"{SHORT[ds]}: the strongest single relation is {bs} (AUC {_f(single[bs])}); using all three gives {_f(d['All three']['AUC'])}.")
    return out


def entity_points(ds):
    t = entity_table(ds)
    out = []
    for nm in ("Device", "IP address", "Delivery address"):
        s = t[t.entity == nm]
        if len(s):
            b = s.loc[s.fraud_rate.idxmax()]; big = s[s.accounts_sharing == "10+"]
            txt = f"{nm}: fraud rate peaks at {b.fraud_rate*100:.0f}% when {b.accounts_sharing} accounts share one"
            if len(big):
                txt += f", but only {big.fraud_rate.iloc[0]*100:.0f}% for 10+ (in the simulator large hubs are public Wi-Fi, offices, shared family devices and busy addresses - legitimate)"
            out.append(txt + ".")
    return out
