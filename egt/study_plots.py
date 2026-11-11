"""Clean static figures for the results chapter (matplotlib).  Every figure is drawn from study/results/*.json or from the data set itself."""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from . import study
from .study import DATASETS, MODELS, SWEEPS, METAPATHS

NAME = dict(MODELS)
C = dict(dark="#0D47A1", blue="#1565C0", light="#90CAF9", pale="#E3F0FF", grey="#90A4AE", lgrey="#CFD8DC", text="#0D2B52",
         fraud="#D62728", legit="#2E9E5B", unk="#B0BEC5", orange="#EF6C00", teal="#00897B")
SHORT = {"meituan": "Dataset A (Meituan-style)", "vesta": "Dataset B (Vesta-style)"}
DS_COLOR = {"meituan": C["blue"], "vesta": C["orange"]}
FOOT = "Generated stand-in data (the real Meituan data is private; Kaggle-Vesta not downloaded) - numbers are from the runs in study/results."


def _style():
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": C["lgrey"], "axes.labelcolor": C["text"], "text.color": C["text"],
                         "xtick.color": C["text"], "ytick.color": C["text"], "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.color": "#E6EEF7", "grid.linewidth": 0.8, "axes.axisbelow": True,
                         "figure.facecolor": "white", "axes.facecolor": "white", "font.family": "DejaVu Sans"})


def fig_dir():
    d = os.path.join(study.STUDY, "figures")
    os.makedirs(d, exist_ok=True)
    return d


def _save(fig, name):
    p = os.path.join(fig_dir(), name + ".png")
    fig.savefig(p, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return p


def results(ds, name):
    return study._load(ds, name)


def main_results(ds):
    """Headline results = the tuned protocol (each model's lr / embedding size chosen on validation AUC) once it is complete,
    otherwise the default-hyper-parameter run.  Classical baselines always come from the default run."""
    m, t = results(ds, "main"), results(ds, "tuned")
    if t and "secgfd|43" in t:
        d = {k: v for k, v in t.items()}
        d["classical"] = m.get("classical", {})
        d["_info"] = dict(m.get("_info", {}), **t.get("_info", {}))
        d["_default"] = m
        return d
    return m


def agg_main(ds, which="headline"):
    """{key: dict(metric -> (mean, std))}  over the training seeds."""
    d = main_results(ds) if which == "headline" else results(ds, "main")
    out = {}
    for key, _ in MODELS:
        runs = [v for k, v in d.items() if k.startswith(key + "|") and k.split("|")[1].isdigit() and isinstance(v, dict)]
        if runs:
            out[key] = {m: (float(np.mean([r[m] for r in runs])), float(np.std([r[m] for r in runs]))) for m in
                        ("AUC", "PR_AUC", "F1", "Precision", "Recall", "Accuracy", "R@90%P", "MCC", "train_s", "params")}
            out[key]["n"] = len(runs)
    for k, v in d.get("classical", {}).items():
        out[k] = {m: (v[m], 0.0) for m in ("AUC", "PR_AUC", "F1", "Precision", "Recall", "Accuracy", "R@90%P", "MCC", "train_s") if m in v}
        out[k]["n"] = 1
    return out


def tuned_cfg(ds):
    d = main_results(ds)
    return {k: d.get(f"{k}|cfg") for k, _ in MODELS if d.get(f"{k}|cfg")}


def table_main(ds):
    import pandas as pd
    a = agg_main(ds)
    cfg = tuned_cfg(ds)
    rows = []
    for k, v in a.items():
        rows.append({"Model": NAME.get(k, k), "lr": cfg[k]["lr"] if k in cfg else None, "Embedding": cfg[k]["hidden"] if k in cfg else None, "AUC": v["AUC"][0], "AUC ±": v["AUC"][1], "PR-AUC": v["PR_AUC"][0], "F1": v["F1"][0], "Precision": v["Precision"][0],
                     "Recall": v["Recall"][0], "Accuracy": v["Accuracy"][0], "Recall@90%P": v["R@90%P"][0],
                     "Params": int(v["params"][0]) if "params" in v else None, "Type": "graph/sequence model" if k in NAME else "classical (tabular)"})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------------------ 1. model comparison
_boot = {}


def boot_delta(ds, a, b, n=1000, seed=0):
    """AUC(a) - AUC(b) on the test accounts (seed-42 models): mean and 95 % bootstrap interval."""
    from sklearn.metrics import roc_auc_score
    key = (ds, a, b)
    if key in _boot:
        return _boot[key]
    d = main_results(ds)
    y = np.array(d.get("_info", {}).get("ytest", []))
    if not len(y) or f"{a}|42" not in d or "ptest" not in d[f"{a}|42"] or f"{b}|42" not in d:
        return None
    pa, pb = np.array(d[f"{a}|42"]["ptest"]), np.array(d[f"{b}|42"]["ptest"])
    rng = np.random.default_rng(seed)
    ds_ = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].sum() in (0, len(i)):
            continue
        ds_.append(roc_auc_score(y[i], pa[i]) - roc_auc_score(y[i], pb[i]))
    _boot[key] = (float(np.mean(ds_)), float(np.percentile(ds_, 2.5)), float(np.percentile(ds_, 97.5)))
    return _boot[key]


def fig_comparison(ds, save=True):
    _style()
    a = agg_main(ds)
    keys = [k for k, _ in MODELS if k in a] + [k for k in a if k not in NAME]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.9), sharey=True)
    for ax, m, lab in zip(axes, ("AUC", "F1"), ("AUC", "F1 score")):
        vals = [a[k][m][0] for k in keys][::-1]
        errs = [a[k][m][1] for k in keys][::-1]
        kk = keys[::-1]
        cols = [C["dark"] if k == "egt" else (C["light"] if k in NAME else C["lgrey"]) for k in kk]
        ax.barh(range(len(kk)), vals, xerr=errs, color=cols, height=0.68, error_kw=dict(ecolor=C["grey"], lw=1, capsize=2))
        for i, (v, k) in enumerate(zip(vals, kk)):
            ax.text(v + 0.012, i, f"{v:.3f}", va="center", fontsize=8.5, color=C["dark"] if k == "egt" else C["text"], fontweight="bold" if k == "egt" else "normal")
        ax.set_yticks(range(len(kk))); ax.set_yticklabels([NAME.get(k, k) for k in kk])
        for t, k in zip(ax.get_yticklabels(), kk):
            if k == "egt": t.set_fontweight("bold"); t.set_color(C["dark"])
        ax.set_xlim(0, 1.0); ax.set_xlabel(lab); ax.grid(axis="y", visible=False)
        ax.set_title(lab + " on the test accounts", loc="left", fontsize=11, color=C["dark"], fontweight="bold")
    fig.legend(handles=[Line2D([0], [0], color=C["dark"], lw=8), Line2D([0], [0], color=C["light"], lw=8), Line2D([0], [0], color=C["lgrey"], lw=8)],
               labels=["EGT", "graph / sequence models", "classical tabular models"], loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.03))
    fig.suptitle(f"Model comparison - {SHORT[ds]}", x=0.01, ha="left", fontsize=13, fontweight="bold", color=C["dark"])
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    return _save(fig, f"comparison_{ds}") if save else fig


# ------------------------------------------------------------------------------------------------ 2. ROC / PR
def fig_curves(ds, save=True, show=("egt", "envit", "gat", "sage", "rgcn", "han", "hacud", "secgfd")):
    _style()
    d = main_results(ds)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    pal = ["#6A1B9A", "#00897B", "#EF6C00", "#8D6E63", "#C2185B", "#546E7A", "#7CB342"]
    j = 0
    for k in show:
        r = d.get(f"{k}|42")
        if not r or "roc" not in r:
            continue
        egt = k == "egt"
        col = C["dark"] if egt else pal[j % len(pal)]
        j += 0 if egt else 1
        for ax, key in zip(axes, ("roc", "pr")):
            x, y = r[key]
            ax.plot(x, y, color=col, lw=2.8 if egt else 1.3, alpha=1 if egt else 0.85, label=f"{NAME[k]}  (AUC {r['AUC']:.3f})" if key == "roc" else None, zorder=5 if egt else 2)
    axes[0].plot([0, 1], [0, 1], ls="--", color=C["grey"], lw=1)
    axes[0].set_xlabel("False-positive rate"); axes[0].set_ylabel("True-positive rate"); axes[0].set_title("ROC curve", loc="left", fontweight="bold", color=C["dark"])
    axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision"); axes[1].set_title("Precision-recall curve", loc="left", fontweight="bold", color=C["dark"])
    axes[0].legend(loc="lower right", fontsize=8, frameon=True, framealpha=0.95)
    for ax in axes: ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    fig.suptitle(f"Ranking quality (seed-42 models) - {SHORT[ds]}", x=0.01, ha="left", fontsize=13, fontweight="bold", color=C["dark"])
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return _save(fig, f"curves_{ds}") if save else fig


# ------------------------------------------------------------------------------------------------ 3. convergence
def fig_convergence(ds, save=True, show=("egt", "envit", "gat", "han", "hacud", "sobt")):
    _style()
    d = main_results(ds)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    pal = ["#6A1B9A", "#00897B", "#EF6C00", "#8D6E63", "#546E7A"]
    j = 0
    for k in show:
        r = d.get(f"{k}|42")
        if not r:
            continue
        h = np.array(r["history"])
        egt = k == "egt"
        col = C["dark"] if egt else pal[j % len(pal)]
        j += 0 if egt else 1
        for ax, c in zip(axes, (1, 2)):
            ax.plot(h[:, 0], h[:, c], color=col, lw=2.8 if egt else 1.3, label=NAME[k] if c == 1 else None)
    axes[0].set_title("Training loss", loc="left", fontweight="bold", color=C["dark"]); axes[1].set_title("Validation AUC", loc="left", fontweight="bold", color=C["dark"])
    for ax in axes: ax.set_xlabel("epoch")
    axes[0].legend(fontsize=8, frameon=False)
    fig.suptitle(f"Convergence (seed-42 models) - {SHORT[ds]}", x=0.01, ha="left", fontsize=13, fontweight="bold", color=C["dark"])
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return _save(fig, f"convergence_{ds}") if save else fig


# ------------------------------------------------------------------------------------------------ 4. four sensitivity graphs
def sweep_series(ds, sw):
    d = results(ds, "sweep")
    vals = SWEEPS[sw][1]
    return [(v, d.get(f"{sw}|{v}")) for v in vals]


def fig_sweeps(metric="AUC", save=True):
    _style()
    defaults = {"lr": 1e-3, "batch": 512, "emb": 48, "steps": 40}
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.2))
    allv = []
    for ds in DATASETS:
        for sw in SWEEPS:
            allv += [r[metric] for _, r in sweep_series(ds, sw) if r]
    lo, hi = (min(allv) - 0.03, max(allv) + 0.03) if allv else (0, 1)
    letters = "abcd"
    for ax, (sw, (title, vals)), L in zip(axes.ravel(), SWEEPS.items(), letters):
        x = np.arange(len(vals))
        for ds in DATASETS:
            ser = sweep_series(ds, sw)
            ys = [r[metric] if r else np.nan for _, r in ser]
            ax.plot(x, ys, marker="o", ms=7, lw=2.2, color=DS_COLOR[ds], label=SHORT[ds])
            if np.isfinite(ys).any():
                b = int(np.nanargmax(ys))
                ax.annotate(f"{ys[b]:.3f}", (x[b], ys[b]), textcoords="offset points", xytext=(0, 9), ha="center", fontsize=8.5, color=DS_COLOR[ds], fontweight="bold")
        dv = defaults[sw]
        if dv in vals:
            ax.axvline(vals.index(dv), color=C["grey"], ls=":", lw=1.2)
            pass
        lab = [("%g" % v) for v in vals]
        ax.set_xticks(x); ax.set_xticklabels(lab); ax.set_xlabel(title); ax.set_ylabel(metric if metric != "F1" else "F1 score")
        ax.set_ylim(lo, hi + 0.015)
        ax.set_title(f"({L}) {title}", loc="left", fontweight="bold", color=C["dark"])
    axes[0, 0].legend(frameon=False, fontsize=9, loc="lower right")
    fig.suptitle(f"EGT sensitivity ({metric})", x=0.01, ha="left", fontsize=13, fontweight="bold", color=C["dark"])
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return _save(fig, f"sensitivity_{metric}") if save else fig


# ------------------------------------------------------------------------------------------------ 5. meta-path graphs
def fig_metapath(save=True):
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=False)
    names = [n for n, _ in METAPATHS]
    for ax, metric in zip(axes, ("AUC", "F1")):
        w = 0.38
        for j, ds in enumerate(DATASETS):
            d = results(ds, "metapath")
            ys = [d[n][metric] if n in d else np.nan for n in names]
            b = ax.bar(np.arange(len(names)) + (j - 0.5) * w, ys, w, color=DS_COLOR[ds], label=SHORT[ds])
            for rect, v in zip(b, ys):
                if np.isfinite(v):
                    ax.text(rect.get_x() + rect.get_width() / 2, v + 0.005, f"{v:.3f}", ha="center", fontsize=8)
        ax.set_xticks(range(len(names))); ax.set_xticklabels([n.replace(" (", "\n(") for n in names], fontsize=9)
        ax.set_title(metric if metric == "AUC" else "F1 score", loc="left", fontweight="bold", color=C["dark"])
        allv = [results(ds, "metapath")[n][metric] for ds in DATASETS for n in names if n in results(ds, "metapath")]
        if allv: ax.set_ylim(max(0, min(allv) - 0.08), min(1.0, max(allv) + 0.06))
        ax.grid(axis="x", visible=False)
    axes[0].legend(frameon=False, fontsize=9, loc="upper left")
    fig.suptitle("Meta-path study ", x=0.01, ha="left", fontsize=13, fontweight="bold", color=C["dark"])
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return _save(fig, "metapaths") if save else fig


# ------------------------------------------------------------------------------------------------ 6. the four entity graphs
_gd_cache = {}


def get_gd(ds):
    if ds not in _gd_cache:
        from .features import build_graph
        _gd_cache[ds] = build_graph(study.load_events(ds))
    return _gd_cache[ds]


def _members(gd, rel):
    ed = gd.edges[rel]
    order = np.argsort(ed["dst"], kind="stable")
    dst, src = ed["dst"][order], ed["src"][order]
    cut = np.flatnonzero(np.diff(dst)) + 1
    groups = np.split(src, cut)
    ids = dst[np.r_[0, cut]] if len(dst) else []
    return dict(zip(ids, groups))


def _pick_hub(gd, rel, lo=5, hi=9):
    """Entity shared by lo..hi accounts with the most fraud labels (tidy and informative picture)."""
    best, score = None, -1
    for e, accs in _members(gd, rel).items():
        if not (lo <= len(accs) <= hi):
            continue
        lab = gd.labels[accs]
        s = (lab == 1).sum() * 10 + (lab == 0).sum() - abs(len(accs) - 7)
        if (lab == 1).sum() >= 2 and s > score:
            best, score = (e, accs), s
    if best is None:
        for e, accs in _members(gd, rel).items():
            if 3 <= len(accs) <= hi and (gd.labels[accs] == 1).any():
                best = (e, accs); break
    return best


def _col(l):
    return C["fraud"] if l == 1 else (C["legit"] if l == 0 else C["unk"])


def _draw_hub(ax, gd, rel, title, hub_label):
    pick = _pick_hub(gd, rel)
    ax.set_axis_off(); ax.set_xlim(-1.5, 1.5); ax.set_ylim(-1.5, 1.5); ax.set_aspect("equal")
    ax.set_title(title, loc="left", fontweight="bold", color=C["dark"], fontsize=11.5)
    if pick is None:
        ax.text(0, 0, "no suitable shared entity in this data", ha="center"); return
    e, accs = pick
    n = len(accs)
    ang = np.linspace(np.pi / 2, np.pi / 2 + 2 * np.pi, n, endpoint=False)
    for a, acc in zip(ang, accs):
        x, y = 1.12 * np.cos(a), 1.12 * np.sin(a)
        ax.plot([0, x], [0, y], color=C["lgrey"], lw=1.6, zorder=1)
        ax.scatter([x], [y], s=520, color=_col(gd.labels[acc]), edgecolor="white", linewidth=2, zorder=3)
        ax.text(x * 1.27, y * 1.27, str(gd.account_ids[acc]).replace("ACC_", "A-"), ha="center", va="center", fontsize=8, color=C["text"])
    ax.scatter([0], [0], s=2600, marker="s", color=C["dark"], edgecolor="white", linewidth=2, zorder=4)
    ax.text(0, 0, hub_label, ha="center", va="center", color="white", fontsize=9, fontweight="bold", zorder=5)
    lab = gd.labels[accs]


def _draw_person(ax, gd, title):
    ax.set_axis_off(); ax.set_xlim(-1.7, 1.7); ax.set_ylim(-1.7, 1.7); ax.set_aspect("equal")
    ax.set_title(title, loc="left", fontweight="bold", color=C["dark"], fontsize=11.5)
    from .config import RELATIONS
    mem = {r: None for r in RELATIONS}
    by_person = {}
    for i, p in enumerate(gd.person_ids):
        by_person.setdefault(p, []).append(i)
    best, bs = None, -1
    rel_of = []
    for r in RELATIONS:
        ed = gd.edges[r]
        d = {}
        for s_, d_ in zip(ed["src"], ed["dst"]):
            d.setdefault(int(s_), set()).add(int(d_))
        rel_of.append(d)
    for p, accs in by_person.items():
        if not (2 <= len(accs) <= 3):
            continue
        lab = gd.labels[accs]
        if (lab == 1).any() and (lab == 0).any():
            # prefer persons whose sibling accounts use clearly different environments
            ok = all(any(len(rel_of[r].get(a, ())) for r in range(3)) for a in accs)
            sets = [set().union(*[{(r, e) for e in rel_of[r].get(a, ())} for r in range(3)]) for a in accs]
            overlap = len(set.intersection(*sets)) if sets else 0
            s = (2 if ok else 0) - overlap + min(len(sets[0]), 3) * 0.1
            if s > bs:
                best, bs = (p, accs), s
    if best is None:
        ax.text(0, 0, "no person with a fraud and a legitimate account", ha="center"); return
    p, accs = best
    n = len(accs)
    ang = np.linspace(np.pi / 2, np.pi / 2 + 2 * np.pi, n, endpoint=False)
    marks = {0: ("D", "device"), 1: ("^", "IP network"), 2: ("o", "address")}
    for a, acc in zip(ang, accs):
        x, y = 0.78 * np.cos(a), 0.78 * np.sin(a)
        ax.plot([0, x], [0, y], color=C["grey"], lw=1.8, zorder=1)
        ents = [(r, e) for r in range(3) for e in sorted(rel_of[r].get(acc, ()))][:4]
        spread = np.linspace(-0.55, 0.55, max(len(ents), 1)) if len(ents) > 1 else [0.0]
        for (r, e), off in zip(ents, spread):
            ex, ey = 1.45 * np.cos(a + off), 1.45 * np.sin(a + off)
            ax.plot([x, ex], [y, ey], color=C["lgrey"], lw=1.2, zorder=1)
            ax.scatter([ex], [ey], s=170, marker=marks[r][0], color="white", edgecolor=C["blue"], linewidth=1.8, zorder=3)
        ax.scatter([x], [y], s=620, color=_col(gd.labels[acc]), edgecolor="white", linewidth=2, zorder=3)
        ax.text(x + (0.3 if x >= 0 else -0.3) * 1.0, y, str(gd.account_ids[acc]).replace("ACC_", "A-"), ha="center", va="center", fontsize=7.5, color=C["text"], zorder=5)
    ax.scatter([0], [0], s=2600, marker="h", color=C["dark"], edgecolor="white", linewidth=2, zorder=4)
    ax.text(0, 0, "Person\nID", ha="center", va="center", color="white", fontsize=8.5, fontweight="bold", zorder=5)


def fig_entity_graphs(ds, save=True):
    _style()
    gd = get_gd(ds)
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 10.2))
    _draw_hub(axes[0, 0], gd, "device", "(a) Device", "Device")
    _draw_hub(axes[0, 1], gd, "ip", "(b) IP address / network", "IP")
    _draw_hub(axes[1, 0], gd, "address", "(c) Delivery address", "Address")
    _draw_person(axes[1, 1], gd, "(d) Personal ID")
    fig.legend(handles=[Line2D([0], [0], marker="o", color="w", markerfacecolor=C["fraud"], ms=12, label="fraud account"),
                        Line2D([0], [0], marker="o", color="w", markerfacecolor=C["legit"], ms=12, label="legitimate account"),
                        Line2D([0], [0], marker="o", color="w", markerfacecolor=C["unk"], ms=12, label="unlabelled account"),
                        Line2D([0], [0], marker="s", color="w", markerfacecolor=C["dark"], ms=12, label="shared entity")],
               loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(f"Shared entities - {SHORT[ds]}", x=0.01, ha="left", fontsize=13, fontweight="bold", color=C["dark"])
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    return _save(fig, f"entity_graphs_{ds}") if save else fig


BUCKETS = [("1", 1, 1), ("2", 2, 2), ("3-4", 3, 4), ("5-9", 5, 9), ("10+", 10, 10 ** 9)]


def entity_sharing(gd, kind):
    """per account: the largest number of accounts that share one entity of this kind (incl. the account itself)."""
    if kind == "person":
        ids = np.unique(gd.person_ids, return_inverse=True)[1]
        return np.bincount(ids)[ids]
    ed = gd.edges[kind]
    cnt = np.bincount(ed["dst"], minlength=len(gd.ent_names[kind]))
    out = np.zeros(gd.n, int)
    np.maximum.at(out, ed["src"], cnt[ed["dst"]])
    return out


def entity_table(ds):
    import pandas as pd
    gd = get_gd(ds)
    rows = []
    for kind, nm in (("device", "Device"), ("ip", "IP address"), ("address", "Delivery address"), ("person", "Personal ID")):
        deg = entity_sharing(gd, kind)
        lab = gd.labels
        for b, lo, hi in BUCKETS:
            m = (deg >= lo) & (deg <= hi) & (lab >= 0)
            if m.sum():
                rows.append(dict(entity=nm, accounts_sharing=b, labelled_accounts=int(m.sum()), fraud_rate=float((lab[m] == 1).mean())))
    return pd.DataFrame(rows)


def fig_entity_stats(ds, save=True):
    _style()
    t = entity_table(ds)
    gd = get_gd(ds)
    overall = float((gd.labels[gd.labels >= 0] == 1).mean())
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 6.8))
    for ax, nm, L in zip(axes.ravel()[:3], ["Device", "IP address", "Delivery address"], "abc"):
        s_ = t[t.entity == nm]
        x = np.arange(len(s_))
        bars = ax.bar(x, s_.fraud_rate * 100, color=C["blue"], width=0.62)
        for r, b in zip(s_.itertuples(), bars):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.5, f"{r.fraud_rate*100:.0f}%", ha="center", fontsize=8.5, fontweight="bold", color=C["dark"])
        ax.axhline(overall * 100, color=C["fraud"], ls="--", lw=1.2)
        ax.text(len(s_) - 0.45, overall * 100 + 2, f"all accounts {overall*100:.0f}%", color=C["fraud"], fontsize=8, ha="right")
        ax.set_xticks(x); ax.set_xticklabels([f"{a}\n(n={n})" for a, n in zip(s_.accounts_sharing, s_.labelled_accounts)], fontsize=8.5)
        ax.set_ylim(0, 112)
        ax.set_xlabel("accounts sharing the same " + nm.lower().replace("delivery ", "").replace("ip address", "IP address"), labelpad=6)
        ax.set_ylabel("fraud rate (%)"); ax.set_title(f"({L}) {nm}", loc="left", fontweight="bold", color=C["dark"])
        ax.grid(axis="x", visible=False)
    ax = axes[1, 1]
    ix = {k: i for i, k in enumerate(gd.feat_keys)}
    m = (gd.stats[:, ix["n_siblings"]] > 0) & (gd.labels >= 0)
    w = 0.36
    for j, (lab, col, nm) in enumerate(((1, C["fraud"], "fraud accounts"), (0, C["legit"], "legitimate accounts"))):
        vals = [float(gd.stats[m & (gd.labels == lab), ix[k]].mean()) for k in ("sib_jacc_dev", "sib_jacc_ip", "sib_jacc_addr")]
        b = ax.bar(np.arange(3) + (j - 0.5) * w, vals, w, color=col, label=nm)
        for rect, v in zip(b, vals):
            ax.text(rect.get_x() + rect.get_width() / 2, v + 0.015, f"{v:.2f}", ha="center", fontsize=8.5)
    ax.set_xticks(range(3)); ax.set_xticklabels(["device", "IP network", "address"]); ax.set_ylim(0, 1)
    ax.set_ylabel("overlap with same-ID siblings"); ax.set_xlabel("environment type", labelpad=14)
    ax.set_title("(d) Personal ID: environment overlap", loc="left", fontweight="bold", color=C["dark"]); ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    ax.grid(axis="x", visible=False)
    fig.suptitle(f"Sharing statistics - {SHORT[ds]}", x=0.01, ha="left", fontsize=13, fontweight="bold", color=C["dark"])
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return _save(fig, f"entity_stats_{ds}") if save else fig


# ------------------------------------------------------------------------------------------------ both data sets side by side
def fig_both(metric="AUC", save=True):
    _style()
    aggs = {ds: agg_main(ds) for ds in DATASETS}
    keys = [k for k, _ in MODELS if all(k in a for a in aggs.values())]
    fig, ax = plt.subplots(figsize=(11, 5.4))
    x = np.arange(len(keys)); w = 0.38
    for j, ds in enumerate(DATASETS):
        vals = [aggs[ds][k][metric][0] for k in keys]
        errs = [aggs[ds][k][metric][1] for k in keys]
        b = ax.bar(x + (j - 0.5) * w, vals, w, yerr=errs, color=DS_COLOR[ds], label=SHORT[ds], error_kw=dict(ecolor=C["grey"], lw=1, capsize=2))
        for rect, v in zip(b, vals):
            ax.text(rect.get_x() + rect.get_width() / 2, v + 0.012, f"{v:.3f}", ha="center", fontsize=7.5, rotation=90, va="bottom")
    ax.set_xticks(x); ax.set_xticklabels([NAME[k] for k in keys], rotation=30, ha="right")
    for t, k in zip(ax.get_xticklabels(), keys):
        if k == "egt": t.set_fontweight("bold"); t.set_color(C["dark"])
    allv = [aggs[ds][k][metric][0] for ds in DATASETS for k in keys]
    ax.set_ylim(max(0, min(allv) - 0.12), min(1.05, max(allv) + 0.12)); ax.set_ylabel(metric if metric != "F1" else "F1 score"); ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, ncol=2, loc="upper right")
    ax.set_title(f"{metric if metric != 'F1' else 'F1 score'} of every model on the two data sets", loc="left", fontsize=13, fontweight="bold", color=C["dark"])
    fig.tight_layout()
    return _save(fig, f"both_{metric}") if save else fig


# ------------------------------------------------------------------------------------------------ neighbourhood of ONE account (tidy star, not a hairball)
def fig_account_graph(gd, acc, ent, sib, max_ent=6, max_sib=4):
    """ent / sib = the evidence tables of explain_account (attention-ranked)."""
    _style()
    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    ax.set_axis_off(); ax.set_xlim(-1.75, 1.75); ax.set_ylim(-1.45, 1.45); ax.set_aspect("equal")
    items = []
    if ent is not None and len(ent):
        for r in ent.head(max_ent).itertuples():
            items.append(("ent", r))
    if sib is not None and len(sib):
        for r in sib.head(max_sib).itertuples():
            items.append(("sib", r))
    n = len(items)
    marks = {"Device": "D", "Network": "^", "Address": "o"}
    ax.scatter([0], [0], s=2300, color=C["dark"], edgecolor="white", linewidth=2.5, zorder=5)
    ax.text(0, 0, str(acc).replace("ACC_", "A-"), color="white", ha="center", va="center", fontsize=8.5, fontweight="bold", zorder=6)
    if n == 0:
        ax.text(0, -0.5, "no device / network / address / sibling links", ha="center", color=C["grey"])
    mx = max([r.attention for _, r in items] + [1e-9])
    for k, (kind, r) in enumerate(items):
        a = np.pi / 2 + 2 * np.pi * k / max(n, 1)
        x, y = 1.15 * np.cos(a), 1.0 * np.sin(a)
        lw = 1 + 4.5 * r.attention / mx
        ax.plot([0, x], [0, y], color=C["lgrey"] if kind == "ent" else C["light"], lw=lw, zorder=1, solid_capstyle="round")
        if kind == "ent":
            hot = r.other_flagged >= 2
            ax.scatter([x], [y], s=620, marker=marks.get(r.relation, "o"), color=C["fraud"] if hot else C["blue"], edgecolor="white", linewidth=2, zorder=4)
            lab = f"{r.relation}\nshared with {r.other_accounts}" + (f" ({r.other_flagged} flagged)" if r.other_accounts else "")
        else:
            ax.scatter([x], [y], s=620, color=C["fraud"] if r.sibling_flagged else C["legit"], edgecolor="white", linewidth=2, zorder=4)
            lab = f"{str(r.sibling).replace('ACC_', 'A-')}\nsame ID, risk {r.sibling_risk:.0f}"
        ax.text(x * 1.3 + (0.03 if x >= 0 else -0.03), y * 1.3, lab.replace("\\n", "\n"), ha="left" if x > 0.05 else ("right" if x < -0.05 else "center"),
                va="center", fontsize=7.8, color=C["text"])
    ax.legend(handles=[Line2D([0], [0], marker="s", color="w", markerfacecolor=C["blue"], ms=10, label="entity, little fraud around"),
                       Line2D([0], [0], marker="s", color="w", markerfacecolor=C["fraud"], ms=10, label="entity with other flagged accounts"),
                       Line2D([0], [0], marker="o", color="w", markerfacecolor=C["legit"], ms=10, label="sibling, not flagged"),
                       Line2D([0], [0], marker="o", color="w", markerfacecolor=C["fraud"], ms=10, label="sibling, flagged")],
              loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=2, frameon=False, fontsize=8)
    fig.tight_layout()
    return fig


# ------------------------------------------------------------------------------------------------ transaction graph: a few tidy fraud-ring views
def _rings(gd, prob, thr, n=4, max_hub=12, lo=4, hi=14):
    import networkx as nx
    from .config import RELATIONS
    G = nx.Graph()
    for r in RELATIONS:
        ed = gd.edges[r]
        cnt = np.bincount(ed["dst"], minlength=len(gd.ent_names[r]))
        for a, e in zip(ed["src"], ed["dst"]):
            if 2 <= cnt[e] <= max_hub:
                G.add_edge(("A", int(a)), (r, int(e)))
    comps = []
    for c in nx.connected_components(G):
        accs = [x[1] for x in c if x[0] == "A"]
        if lo <= len(accs) <= hi:
            fl = int((prob[accs] >= thr).sum())
            comps.append((fl, len(accs), G.subgraph(c).copy()))
    comps.sort(key=lambda t: (-t[0], -t[1]))
    return comps[:n]


def fig_rings(gd, prob, thr, n=4):
    import networkx as nx
    _style()
    comps = _rings(gd, prob, thr, n)
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 9.4))
    mk = {"device": "s", "ip": "^", "address": "h"}
    cmap = matplotlib.colormaps["RdYlGn_r"]
    for ax, (fl, na, H) in zip(axes.ravel(), comps):
        pos = nx.kamada_kawai_layout(H)
        for u, v in H.edges():
            ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]], color=C["lgrey"], lw=1.4, zorder=1)
        for node in H.nodes():
            x, y = pos[node]
            if node[0] == "A":
                pr = float(prob[node[1]])
                ax.scatter([x], [y], s=520, color=cmap(pr), edgecolor=C["dark"] if pr >= thr else "white", linewidth=2.4 if pr >= thr else 1.5, zorder=3)
                ax.text(x, y - 0.13, str(gd.account_ids[node[1]]).replace("ACC_", ""), ha="center", fontsize=6.8, color=C["text"], zorder=4)
            else:
                ax.scatter([x], [y], s=300, marker=mk[node[0]], color="white", edgecolor=C["blue"], linewidth=2, zorder=2)
        ne = sum(1 for x in H.nodes() if x[0] != "A")
        ax.set_title(f"{na} accounts · {ne} shared entities · {fl} flagged", loc="left", fontsize=10.5, fontweight="bold", color=C["dark"])
        ax.set_axis_off(); ax.margins(0.12)
    for ax in axes.ravel()[len(comps):]:
        ax.set_axis_off()
    from matplotlib.cm import ScalarMappable
    fig.legend(handles=[Line2D([0], [0], marker="s", color="w", markerfacecolor="white", markeredgecolor=C["blue"], ms=10, label="device"),
                        Line2D([0], [0], marker="^", color="w", markerfacecolor="white", markeredgecolor=C["blue"], ms=10, label="IP network"),
                        Line2D([0], [0], marker="h", color="w", markerfacecolor="white", markeredgecolor=C["blue"], ms=10, label="address"),
                        Line2D([0], [0], marker="o", color="w", markerfacecolor=cmap(0.95), markeredgecolor=C["dark"], ms=11, label="account, flagged (colour = risk)"),
                        Line2D([0], [0], marker="o", color="w", markerfacecolor=cmap(0.1), ms=11, label="account, not flagged")],
               loc="lower center", ncol=5, frameon=False, fontsize=8.5, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle("Transaction graph: the four riskiest fraud rings", x=0.01, ha="left", fontsize=13, fontweight="bold", color=C["dark"])
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    return fig


# ------------------------------------------------------------------------------------------------ model performance of the deployed model
def test_arrays(gd, prob):
    te = np.where(gd.split == 2)[0]
    te = te[gd.labels[te] >= 0]
    return gd.labels[te], prob[te]


def fig_performance(gd, prob, thr):
    from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score, confusion_matrix
    _style()
    y, p = test_arrays(gd, prob)
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.4))
    f, t, _ = roc_curve(y, p)
    ax = axes[0, 0]; ax.plot(f, t, color=C["dark"], lw=2.6); ax.plot([0, 1], [0, 1], "--", color=C["grey"], lw=1)
    ax.fill_between(f, t, alpha=0.08, color=C["blue"]); ax.set_xlabel("False-positive rate"); ax.set_ylabel("True-positive rate")
    ax.set_title(f"ROC  (AUC {roc_auc_score(y, p):.3f})", loc="left", fontweight="bold", color=C["dark"])
    pr, rc, _ = precision_recall_curve(y, p)
    ax = axes[0, 1]; ax.plot(rc, pr, color=C["dark"], lw=2.6); ax.axhline(y.mean(), ls="--", color=C["grey"], lw=1)
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.set_ylim(0, 1.02)
    ax.set_title(f"Precision-recall  (AP {average_precision_score(y, p):.3f})", loc="left", fontweight="bold", color=C["dark"])
    ax = axes[1, 0]
    edges = np.linspace(0, 1, 9); mid, obs = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (p >= a) & (p < b if b < 1 else p <= b)
        if m.sum() >= 5:
            mid.append(p[m].mean()); obs.append(y[m].mean())
    ax.plot([0, 1], [0, 1], "--", color=C["grey"], lw=1); ax.plot(mid, obs, "o-", color=C["dark"], lw=2.2, ms=7)
    ax.set_xlabel("Predicted risk"); ax.set_ylabel("Observed fraud rate"); ax.set_title(f"Calibration  (ECE {ece(y, p):.3f})", loc="left", fontweight="bold", color=C["dark"])
    ax = axes[1, 1]
    cm = confusion_matrix(y, (p >= thr).astype(int), labels=[0, 1])
    ax.imshow(cm, cmap="Blues"); ax.grid(False)
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center", fontsize=16, fontweight="bold", color="white" if v > cm.max() / 2 else C["dark"])
    ax.set_xticks([0, 1]); ax.set_xticklabels(["legitimate", "fraud"]); ax.set_yticks([0, 1]); ax.set_yticklabels(["legitimate", "fraud"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title(f"Confusion matrix at threshold {thr*100:.1f}", loc="left", fontweight="bold", color=C["dark"])
    fig.suptitle("Deployed EGT on the test accounts", x=0.01, ha="left", fontsize=13, fontweight="bold", color=C["dark"])
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


def ece(y, p, bins=10):
    from .metrics import ece as _e
    return _e(y, p, bins)


# ------------------------------------------------------------------------------------------------ dynamic risk scoring
def fig_risk(gd, prob, thresholds):
    from .config import TIER_NAMES, TIER_COLORS
    _style()
    lab = gd.labels
    tiers = thresholds["tiers"]
    fig, ax = plt.subplots(figsize=(10.5, 4.6))
    bins = np.linspace(0, 100, 41)
    for l, col, nm in ((0, C["legit"], "legitimate"), (1, C["fraud"], "fraud")):
        ax.hist(prob[lab == l] * 100, bins=bins, color=col, alpha=0.75, label=nm)
    edges = [0] + [t * 100 for t in tiers] + [100]
    ymax = ax.get_ylim()[1]
    for k, nm in enumerate(TIER_NAMES):
        ax.axvspan(edges[k], edges[k + 1], color=TIER_COLORS[nm], alpha=0.10, lw=0)
        ax.text((edges[k] + edges[k + 1]) / 2, ymax * 1.6, nm, ha="center", fontsize=8.5, color=TIER_COLORS[nm], fontweight="bold")
    ax.axvline(thresholds["f1"] * 100, color="black", ls="--", lw=1.4)
    ax.text(thresholds["f1"] * 100 - 0.8, 1.3, "decision threshold", rotation=90, ha="right", fontsize=8.5)
    ax.set_yscale("log"); ax.set_ylim(0.8, ymax * 3)
    ax.set_xlabel("risk score (0-100)"); ax.set_ylabel("accounts (log scale)"); ax.legend(frameon=False, loc="center left")
    ax.set_title("Risk score distribution and tiers", loc="left", fontsize=12, fontweight="bold", color=C["dark"])
    fig.tight_layout()
    return fig


def make_all():
    out = []
    for ds in DATASETS:
        for f in (fig_comparison, fig_curves, fig_convergence, fig_entity_graphs, fig_entity_stats):
            try:
                out.append(f(ds))
            except Exception as e:
                print("skip", f.__name__, ds, e)
    for m in ("AUC", "F1"):
        out.append(fig_sweeps(m)); out.append(fig_both(m))
    out.append(fig_metapath())
    return out
