#!/usr/bin/env python3
"""
Explainable Graph Transformer for Financial Identity Theft Detection - command line interface.

REAL-DATA WORKFLOW (Kaggle IEEE-CIS Fraud Detection = Vesta; put the two files in data/raw/)
  python main.py run-all --transaction data/raw/train_transaction.csv --identity data/raw/train_identity.csv
      = report (outputs/dataset_report.txt) + prepare (features, graph, cache) + train (EGT + baselines + reports)
  or step by step:
  python main.py report  --transaction ... --identity ...
  python main.py prepare --transaction ... --identity ... [--split time|random]
  python main.py train   [--seeds 42 43 44] [--epochs 60]

RESULTS STUDY (two data sets, 11 models, 4 sensitivity sweeps, meta-path study)  - everything the GUI shows
  python main.py study                        runs it all (both data sets in parallel, 30-60 min), then draws the figures
  python main.py study --dataset meituan --stage main      (stages: data main sweep metapath deploy)   [--quick = tiny smoke test]
  python main.py figures                      redraws the figures in study/figures from study/results
  python main.py compare [--dataset vesta]    prints the model comparison table

USING THE TRAINED MODEL
  python main.py predict  [--vesta-transaction X.csv --vesta-identity Y.csv | --input events.csv]   (default: prepared dataset)
                          [--mode f1|precision90|recall90|manual|topk] [--threshold 0.7] [--topk 0.05]
  python main.py explain  --account A_123 [--save]
  python main.py graph    [--view all|risky|focus] [--account A_123] [--html]
  python main.py timeline --account A_123
  python main.py gui

  python main.py demo-train      python main.py test
"""
import argparse, json, os, pickle, shutil, subprocess, sys, warnings
warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
CACHE = os.path.join(ROOT, "data", "cache")
WARN = []


def _load(a):
    import pandas as pd
    from egt.pipeline import Predictor, find_model, load_cached_result
    ckpt, out_dir, cache_dir, is_dev = find_model(ROOT)
    if getattr(a, "dataset", None):
        from egt.pipeline import find_study_model
        ckpt, out_dir, cache_dir, is_dev = find_study_model(ROOT, a.dataset)
    if a.model:
        ckpt = a.model
    if not ckpt or not os.path.exists(ckpt):
        sys.exit("[!] no trained model found.\n    Real data:  python main.py run-all --transaction data/raw/train_transaction.csv --identity data/raw/train_identity.csv"
                 "\n    Demo model (synthetic data):  python main.py demo-train")
    pred = Predictor(ckpt)
    if getattr(a, "vesta_transaction", None):
        res = pred.score_vesta(a.vesta_transaction, a.vesta_identity)
    elif getattr(a, "input", None):
        res = pred.score(pd.read_csv(a.input))
    else:
        res = load_cached_result(pred, cache_dir or CACHE, out_dir or os.path.join(ROOT, "outputs"))
        if res is None:
            sys.exit("[!] no prepared dataset for this model. Give --vesta-transaction/--input, or run prepare + train first.")
    tbl = pred.table(res, a.mode, a.threshold, a.topk)
    WARN[:] = pred.input_warnings(res, tbl)
    return pred, res, tbl


def _print_table(tbl, n):
    import pandas as pd
    pd.set_option("display.width", 220); pd.set_option("display.max_columns", 30)
    for w in WARN:
        print("[!] WARNING:", w)
    print(f"\nPolicy: {tbl.attrs['mode']}  |  decision threshold = {tbl.attrs['threshold']*100:.1f} / 100")
    print(tbl.head(n).to_string(index=False))
    vc = tbl.risk_tier.value_counts().reindex(["CRITICAL", "HIGH", "MEDIUM", "LOW"]).fillna(0).astype(int)
    print("\nRisk tiers :", "  ".join(f"{k}={v}" for k, v in vc.items()), f"|  flagged FRAUD: {(tbl.prediction=='FRAUD').sum():,} of {len(tbl):,} accounts")
    lab = tbl[tbl.given_label != ""]
    if "split" in tbl.columns:
        lab = lab[lab.split == "test"] if (lab.split == "test").any() else lab
        what = "TEST-split accounts"
    else:
        what = "labelled accounts"
    if len(lab) and lab.given_label.nunique() == 2:
        from egt.metrics import full_metrics
        m = full_metrics((lab.given_label == "fraud").astype(int).values, lab.risk_score.values / 100, tbl.attrs["threshold"])
        print(f"Metrics on {what} ({len(lab):,}): " + "  ".join(f"{k}={m[k]:.3f}" for k in ("AUC", "PR_AUC", "Precision", "Recall", "F1", "Accuracy")))


# ---------------------------------------------------------------------------------------- real-data workflow
def cmd_report(a):
    from egt.vesta import dataset_report
    dataset_report(a.transaction, a.identity, os.path.join(ROOT, "outputs", "dataset_report.txt"), a.chunksize)
    print("\nsaved -> outputs/dataset_report.txt")


def cmd_prepare(a):
    from egt.vesta import VestaPreprocessor
    from egt.features import build_graph
    import time
    os.makedirs(CACHE, exist_ok=True)
    pp = VestaPreprocessor()
    ev, info = pp.fit_transform(a.transaction, a.identity, a.split, a.seed, chunksize=a.chunksize)
    t = time.time(); gd = build_graph(ev); del ev
    print(f"graph built in {time.time()-t:.0f}s: {gd.n:,} accounts, {gd.stats.shape[1]} account features, {gd.seq_dim} event features, "
          f"{len(gd.edges['device']['src']):,}/{len(gd.edges['ip']['src']):,}/{len(gd.edges['address']['src']):,} account-device/network/address edges")
    pp.save(os.path.join(CACHE, "vesta_preprocessor.pkl"))
    with open(os.path.join(CACHE, "graph.pkl"), "wb") as f:
        pickle.dump(gd, f, protocol=4)
    info.update(source=f"Kaggle IEEE-CIS (Vesta): {os.path.basename(a.transaction)} + {os.path.basename(a.identity) if a.identity else 'no identity file'}")
    json.dump(info, open(os.path.join(CACHE, "prep_info.json"), "w"), indent=1)
    print("saved prepared dataset ->", CACHE)


def cmd_train(a):
    from egt.evaluate import run_experiment
    gp = os.path.join(CACHE, "graph.pkl")
    if not os.path.exists(gp):
        sys.exit("[!] run  python main.py prepare ...  first (it needs the real data files).")
    with open(gp, "rb") as f:
        gd = pickle.load(f)
    info = json.load(open(os.path.join(CACHE, "prep_info.json")))
    ckpt = a.model or os.path.join(ROOT, "models", "egt_vesta.pt")
    shutil.copy(os.path.join(CACHE, "vesta_preprocessor.pkl"), os.path.join(os.path.dirname(ckpt), "vesta_preprocessor.pkl"))
    run_experiment(gd, os.path.join(ROOT, "outputs"), ckpt, tuple(a.seeds), a.epochs, a.patience, a.batch_size, not a.no_baselines,
                   not a.no_ablation, True, a.max_minutes, info["source"], info, "vesta_preprocessor.pkl")
    print("\nreports -> outputs/dataset_report.txt (run `report`), outputs/training_report.txt, outputs/benchmark_table.csv")


def cmd_run_all(a):
    cmd_report(a)
    cmd_prepare(a)
    cmd_train(a)


# ---------------------------------------------------------------------------------------- using the model
def cmd_predict(a):
    pred, res, tbl = _load(a)
    _print_table(tbl, a.top)
    os.makedirs(a.out, exist_ok=True)
    f = os.path.join(a.out, "risk_scores.csv"); tbl.to_csv(f, index=False); print("saved ->", f)
    if a.explain:
        cmd_explain(a, (pred, res, tbl), a.explain)


def cmd_explain(a, loaded=None, account=None):
    import pandas as pd
    from egt.explain import explain_account
    pred, res, tbl = loaded or _load(a)
    acc = account or a.account
    if acc not in res.gd.acct_index:
        sys.exit(f"[!] account {acc} not found. Examples: {list(res.gd.account_ids[:5])}")
    ex = explain_account(pred, res, acc, a.mode, a.threshold, a.topk, tbl=tbl)
    pd.set_option("display.width", 220); pd.set_option("display.max_columns", 30)
    print("\n" + "=" * 100 + "\n" + ex["narrative"] + "\n" + "=" * 100)
    if ex["given_label"]:
        print(f"(label in the data: {ex['given_label']})")
    print("\nModule contribution (fraud log-odds lost if the module is removed; attention = fusion weight):")
    print(ex["modules"].round(3).to_string(index=False))
    print("\nSHAP values by feature group (Shapley, calibrated log-odds; positive = towards FRAUD; they add up to f(x)-f(average account) = %.3f):" % (ex["shap"]["f_x"] - ex["shap"]["f_base"]))
    print(ex["shap"]["table"][["group", "strongest_feature", "value", "shap"]].round(3).to_string(index=False))
    print("\nTop feature attributions - Integrated Gradients (positive = towards FRAUD):")
    print(ex["features"].head(8)[["label", "value", "typical", "attribution"]].round(3).to_string(index=False))
    if len(ex["entities"]):
        print("\nShared devices / networks / addresses (attention-ranked):"); print(ex["entities"].head(8).round(3).to_string(index=False))
    if len(ex["siblings"]):
        print("\nSibling accounts (same personal identity):"); print(ex["siblings"].round(3).to_string(index=False))
    if len(ex["events"]):
        print("\nMost influential events:"); print(ex["events"].round(3).to_string(index=False))
    if getattr(a, "save", False):
        os.makedirs(a.out, exist_ok=True)
        from egt.viz import render_graph, draw_matplotlib
        H, pos, note = render_graph(res, tbl, "focus", acc, 2)
        f = os.path.join(a.out, f"explain_{acc}.png"); draw_matplotlib(H, pos, f"Transaction graph around {acc}", note, focus=acc).savefig(f, bbox_inches="tight")
        print("\ngraph saved ->", f)


def cmd_graph(a):
    pred, res, tbl = _load(a)
    from egt.viz import render_graph, draw_matplotlib, draw_plotly
    H, pos, note = render_graph(res, tbl, a.view, a.account, a.hops, a.max_nodes, min_tier=a.min_tier)
    title = {"all": "Transaction graph - all accounts", "risky": "Transaction graph - suspicious groups", "focus": f"Transaction graph around {a.account}"}[a.view]
    fig = draw_matplotlib(H, pos, title, note, focus=a.account if a.view == "focus" else None)
    os.makedirs(a.out, exist_ok=True)
    f = os.path.join(a.out, f"graph_{a.view}{'_' + a.account if a.account else ''}.png")
    fig.savefig(f, bbox_inches="tight"); print("graph saved ->", f, "|", note)
    if a.html:
        h = f.replace(".png", ".html"); draw_plotly(H, pos, title, note).write_html(h); print("interactive ->", h)


def cmd_timeline(a):
    from egt.pipeline import normalized_events
    import pandas as pd
    pd.set_option("display.width", 220); pd.set_option("display.max_columns", 30)
    pred, res, tbl = _load(a)
    print(pred.timeline(normalized_events(res), a.account).to_string(index=False))


def cmd_gui(a):
    sys.exit(subprocess.call([sys.executable, "-m", "streamlit", "run", os.path.join(ROOT, "app.py")]))


# ---------------------------------------------------------------------------------------- development only
def cmd_demo_train(a):
    from egt.data_synth import generate_events
    from egt.features import build_graph
    from egt.evaluate import run_experiment
    ev, _ = generate_events(a.accounts, seed=142)
    gd = build_graph(ev)
    os.makedirs(os.path.join(ROOT, "demo", "cache"), exist_ok=True); os.makedirs(os.path.join(ROOT, "models", "demo"), exist_ok=True)
    run_experiment(gd, os.path.join(ROOT, "demo", "outputs"), os.path.join(ROOT, "models", "demo", "egt_demo.pt"), (42, 43, 44), 40, 8, 512, True, True, True, 0.0,
                   "synthetic (demo model - generated data that imitates the paper's fraud patterns)", {"n_accounts": gd.n})
    with open(os.path.join(ROOT, "demo", "cache", "graph.pkl"), "wb") as f:     # after training: gd.split is now filled
        pickle.dump(gd, f, protocol=4)
    print("\nDemo model written to models/demo (trained on generated data). Use `python main.py gui` or `python main.py predict`.")


def cmd_study(a):
    import subprocess
    stages = a.stage or ["data", "main", "sweep", "metapath", "deploy"]
    if a.dataset == "all" and not a.quick:
        procs = [subprocess.Popen([sys.executable, os.path.abspath(__file__), "study", "--dataset", d, "--stage", *stages]) for d in ("meituan", "vesta")]
        for pr in procs:
            pr.wait()
        return cmd_figures(a)
    from egt import study
    for d in (["meituan", "vesta"] if a.dataset == "all" else [a.dataset]):
        study.run(d, tuple(stages), quick=a.quick)
    if not a.stage or a.dataset == "all":
        cmd_figures(a)


def cmd_figures(a):
    from egt import study_plots
    for f in study_plots.make_all():
        print("saved ->", f)


def cmd_compare(a):
    import pandas as pd
    from egt import study_plots
    pd.set_option("display.width", 200)
    for d in ([a.dataset] if getattr(a, "dataset", None) else ["meituan", "vesta"]):
        print(f"\n== {d} ==")
        print(study_plots.table_main(d).drop(columns=["Type"]).round(4).to_string(index=False))


def cmd_results(a):
    """Prints every study result in the terminal: comparison, sensitivity, meta-paths, deployed-model performance and risk scoring."""
    import numpy as np, pandas as pd
    from egt import study_plots as SP, pipeline
    from egt.study import SWEEPS
    pd.set_option("display.width", 220)
    dss = [a.dataset] if a.dataset else ["meituan", "vesta"]
    nm = {"meituan": "Dataset A (Meituan-style, generated)", "vesta": "Dataset B (Vesta-style, generated)"}
    bar = "=" * 100
    for d in dss:
        print(f"\n{bar}\n {nm[d]} - MODEL COMPARISON (test set, tuned, mean of seeds)\n{bar}")
        print(SP.table_main(d).drop(columns=["Type"]).round(4).to_string(index=False))
    print(f"\n{bar}\n SENSITIVITY (test AUC; single run per point)\n{bar}")
    for sw, (label, vals) in SWEEPS.items():
        for d in dss:
            row = {v: (r or {}).get("AUC") for v, r in SP.sweep_series(d, sw)}
            print(f"  {label:<14s}{d:<9s}" + "  ".join(f"{v}:{x:.3f}" for v, x in row.items() if x is not None))
    print(f"\n{bar}\n META-PATH STUDY (EGT, test AUC / F1)\n{bar}")
    for d in dss:
        for k, r in SP.results(d, "metapath").items():
            print(f"  {d:<9s}{k:<24s}AUC {r['AUC']:.3f}   F1 {r['F1']:.3f}")
    for d in dss:
        ckpt, out, cache, _ = pipeline.find_study_model(ROOT, d)
        if not ckpt:
            continue
        pred = pipeline.Predictor(ckpt); res = pipeline.load_cached_result(pred, cache, out)
        tbl = pred.table(res, "f1"); thr = tbl.attrs["threshold"]
        y, p = SP.test_arrays(res.gd, res.prob)
        from egt.metrics import full_metrics
        m = full_metrics(y, p, thr)
        print(f"\n{bar}\n DEPLOYED EGT - {nm[d]}\n{bar}")
        print("  " + "  ".join(f"{k} {m[k]:.3f}" for k in ("AUC", "PR_AUC", "F1", "Precision", "Recall", "Accuracy", "ECE")))
        print(f"  confusion  TP {m['TP']}  FP {m['FP']}  TN {m['TN']}  FN {m['FN']}")
        th = pred.thresholds
        print(f"  calibration T = {pred.T:.4f}")
        for k, lab in (("f1", "balanced (max F1)"), ("precision90", "precision >= 90%"), ("recall90", "recall >= 90%")):
            f = p >= th[k]; tp = int((f & (y == 1)).sum())
            print(f"  threshold {lab:<20s}{th[k]*100:6.1f}   flagged {f.mean()*100:5.1f}%   precision {tp/max(f.sum(),1):.3f}   recall {tp/max((y==1).sum(),1):.3f}")
        e = [0] + list(th["tiers"]) + [1]
        for i, t in enumerate(["LOW", "MEDIUM", "HIGH", "CRITICAL"]):
            mk = (p >= e[i]) & ((p < e[i + 1]) if i < 3 else (p <= 1))
            print(f"  tier {t:<9s}from {e[i]*100:5.1f}   accounts {int(mk.sum()):4d}   fraud rate {(y[mk].mean()*100 if mk.any() else float('nan')):5.1f}%")


def cmd_test(a):
    sys.exit(subprocess.call([sys.executable, os.path.join(ROOT, "tests", "test_pipeline.py")]))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def raw(sp, need=True):
        sp.add_argument("--transaction", required=need, help="path to train_transaction.csv")
        sp.add_argument("--identity", default=None, help="path to train_identity.csv")
        sp.add_argument("--chunksize", type=int, default=100_000)

    def trn(sp):
        sp.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44]); sp.add_argument("--epochs", type=int, default=60)
        sp.add_argument("--patience", type=int, default=10); sp.add_argument("--batch-size", dest="batch_size", type=int, default=512)
        sp.add_argument("--max-minutes", dest="max_minutes", type=float, default=0.0, help="time budget per training run in minutes (0 = unlimited)"); sp.add_argument("--no-baselines", action="store_true"); sp.add_argument("--no-ablation", action="store_true"); sp.add_argument("--model", default=None)

    def use(sp):
        sp.add_argument("--vesta-transaction", dest="vesta_transaction", default=None); sp.add_argument("--vesta-identity", dest="vesta_identity", default=None)
        sp.add_argument("--input", default=None, help="generic events CSV (account_id,timestamp,...) for generic-schema models"); sp.add_argument("--model", default=None)
        sp.add_argument("--dataset", default=None, choices=["meituan", "vesta"], help="use the model trained for this study data set (default: Dataset A)")
        sp.add_argument("--mode", default="f1", choices=["f1", "precision90", "recall90", "manual", "topk"])
        sp.add_argument("--threshold", type=float, default=0.5, help="manual threshold 0-1"); sp.add_argument("--topk", type=float, default=0.05)
        sp.add_argument("--out", default=os.path.join(ROOT, "outputs"))

    s = sub.add_parser("report"); raw(s); s.set_defaults(f=cmd_report)
    s = sub.add_parser("prepare"); raw(s); s.add_argument("--split", default="time", choices=["time", "random"]); s.add_argument("--seed", type=int, default=42); s.set_defaults(f=cmd_prepare)
    s = sub.add_parser("train"); trn(s); s.set_defaults(f=cmd_train)
    s = sub.add_parser("run-all"); raw(s); trn(s); s.add_argument("--split", default="time", choices=["time", "random"]); s.add_argument("--seed", type=int, default=42); s.set_defaults(f=cmd_run_all)
    s = sub.add_parser("predict"); use(s); s.add_argument("--top", type=int, default=20); s.add_argument("--explain", default=None, help="also explain this account id"); s.add_argument("--save", action="store_true", help="with --explain: save the account graph PNG"); s.set_defaults(f=cmd_predict)
    s = sub.add_parser("explain"); use(s); s.add_argument("--account", required=True); s.add_argument("--save", action="store_true"); s.set_defaults(f=lambda a: cmd_explain(a))
    s = sub.add_parser("graph"); use(s); s.add_argument("--view", default="all", choices=["all", "risky", "focus"]); s.add_argument("--account", default=None)
    s.add_argument("--hops", type=int, default=2); s.add_argument("--max-nodes", dest="max_nodes", type=int, default=1500); s.add_argument("--min-tier", dest="min_tier", default="HIGH")
    s.add_argument("--html", action="store_true"); s.set_defaults(f=cmd_graph)
    s = sub.add_parser("timeline"); use(s); s.add_argument("--account", required=True); s.set_defaults(f=cmd_timeline)
    s = sub.add_parser("gui"); s.set_defaults(f=cmd_gui)
    s = sub.add_parser("demo-train"); s.add_argument("--accounts", type=int, default=6000); s.set_defaults(f=cmd_demo_train)
    s = sub.add_parser("study"); s.add_argument("--dataset", default="all", choices=["all", "meituan", "vesta"]); s.add_argument("--stage", nargs="+", choices=["data", "main", "sweep", "metapath", "deploy"])
    s.add_argument("--quick", action="store_true"); s.set_defaults(f=cmd_study)
    s = sub.add_parser("figures"); s.set_defaults(f=cmd_figures)
    s = sub.add_parser("compare"); s.add_argument("--dataset", default=None, choices=["meituan", "vesta"]); s.set_defaults(f=cmd_compare)
    s = sub.add_parser("results"); s.add_argument("--dataset", default=None, choices=["meituan", "vesta"]); s.set_defaults(f=cmd_results)
    s = sub.add_parser("test"); s.set_defaults(f=cmd_test)
    a = p.parse_args(); a.f(a)


if __name__ == "__main__":
    main()
