"""
Results-chapter experiments (everything is resumable; every number in the GUI / figures is read from study/results/*.json).

  stage data      two generated data sets (the real Meituan data is private and Kaggle-Vesta cannot be downloaded in the
                  build environment):  "meituan" = fraud-ring heavy, 17 % fraud;  "vesta" = rarer (10 %), noisier, harder.
  stage main      EGT vs EnvIT, EnvIT-Graph, EnvIT-Beh, GAT, GraphSAGE, R-GCN, HAN, HACUD, SOBT, SEC-GFD + classical baselines
  stage sweep     sensitivity of EGT to learning rate, batch size, embedding size and time step (sequence length)
  stage metapath  EGT with one relation only (A-D-A / A-I-A / A-L-A) versus all three

All models use the same events, the same labelled split (fixed with seed 42) and the same training loop.
"""
import json, os, time, warnings
import numpy as np
import pandas as pd
import torch

from .config import ModelConfig, TrainConfig
from .features import build_graph
from .train import train_model, make_split
from .metrics import full_metrics
from .risk import derive_thresholds
from .baselines import run_baselines

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STUDY = os.path.join(ROOT, "study")
DATASETS = {
    "meituan": dict(title="Dataset A - Meituan-style (generated)", n=4000, fraud_rate=0.17, difficulty=1.0, label_noise=0.03, seed=11),
    "vesta": dict(title="Dataset B - Vesta-style (generated)", n=4000, fraud_rate=0.10, difficulty=1.35, label_noise=0.05, seed=22),
}
MODELS = [  # key, display name, role
    ("egt", "EGT"), ("envit", "EnvIT"), ("envit_graph", "EnvIT-Graph"), ("envit_beh", "EnvIT-Beh"),
    ("gat", "GAT"), ("sage", "GraphSAGE"), ("rgcn", "R-GCN"), ("han", "HAN"), ("hacud", "HACUD"), ("sobt", "SOBT"), ("secgfd", "SEC-GFD")]
SWEEPS = {
    "lr": ("Learning rate", [1e-4, 5e-4, 1e-3, 5e-3, 1e-2]),
    "batch": ("Batch size", [64, 128, 256, 512, 1024]),
    "emb": ("Embedding size", [16, 32, 48, 64, 96]),
    "steps": ("Time step (events per sequence)", [10, 20, 30, 40, 50]),
}
METAPATHS = [("A-D-A (device)", (1, 0, 0)), ("A-I-A (IP / network)", (0, 1, 0)), ("A-L-A (address)", (0, 0, 1)), ("All three", (1, 1, 1))]


def data_path(ds):
    return os.path.join(STUDY, "data", f"{ds}.csv.gz")


def make_data(ds, quick=False):
    from .data_synth import generate_events
    c = DATASETS[ds]
    ev, _ = generate_events(600 if quick else c["n"], fraud_rate=c["fraud_rate"], seed=c["seed"], difficulty=c["difficulty"], label_noise=c["label_noise"])
    os.makedirs(os.path.dirname(data_path(ds)), exist_ok=True)
    ev.to_csv(data_path(ds), index=False, compression="gzip")
    return ev


def load_events(ds):
    ev = pd.read_csv(data_path(ds))
    ev["timestamp"] = pd.to_datetime(ev["timestamp"])
    return ev


def _rpath(ds, name):
    return os.path.join(STUDY, "results", f"{name}_{ds}.json")


def _load(ds, name):
    p = _rpath(ds, name)
    return json.load(open(p)) if os.path.exists(p) else {}


def _save(ds, name, obj):
    os.makedirs(os.path.dirname(_rpath(ds, name)), exist_ok=True)
    json.dump(obj, open(_rpath(ds, name), "w"), indent=1)


def _tc(quick, **kw):
    base = dict(epochs=2 if quick else 30, patience=8, seed=42, batch_size=512)
    base.update(kw)
    return TrainConfig(**base)


def _metrics(gd, split, probs, thr):
    te = split[2]
    return full_metrics(gd.labels[te], probs[te], thr)


def _curves(y, p):
    from sklearn.metrics import roc_curve, precision_recall_curve
    g = np.linspace(0, 1, 101)
    fpr, tpr, _ = roc_curve(y, p)
    pr, rc, _ = precision_recall_curve(y, p)
    prec = [float(pr[rc >= r].max()) if (rc >= r).any() else 0.0 for r in g]
    return dict(roc=[g.tolist(), np.interp(g, fpr, tpr).round(4).tolist()], pr=[g.tolist(), np.round(prec, 4).tolist()])


def _run_one(gd, split, arch, tc, **mkw):
    cfg = ModelConfig(arch=arch, **mkw)
    r = train_model(gd, cfg, tc, split=split, verbose=False)
    m = _metrics(gd, split, r["probs"], r["thresholds"]["f1"])
    m.update(_curves(gd.labels[split[2]], r["probs"][split[2]]), ptest=np.round(r["probs"][split[2]], 4).tolist(), val_auc=r["best_val_auc"], best_epoch=r["best_epoch"], epochs_run=len(r["history"]), train_s=r["train_seconds"],
             infer_s=r["infer_seconds"], params=r["n_params"], history=[(h["epoch"], h["loss"], h["val_auc"]) for h in r["history"]])
    return m


def stage_main(ds, seeds=(42, 43), quick=False, log=print):
    ev = load_events(ds)
    gd = build_graph(ev)
    split = make_split(gd.labels, 42)
    out = _load(ds, "main")
    out["_info"] = dict(accounts=int(gd.n), events=int(len(ev)), labelled=int((gd.labels >= 0).sum()), fraud=int((gd.labels == 1).sum()),
                        split=[int(len(s)) for s in split], test_fraud=int((gd.labels[split[2]] == 1).sum()))
    for key, name in MODELS:
        for sd in seeds:
            k = f"{key}|{sd}"
            if k in out:
                continue
            t = time.time()
            out[k] = _run_one(gd, split, key, _tc(quick, seed=sd))
            _save(ds, "main", out)
            log(f"[{ds}] {name:<20s} seed {sd}  AUC {out[k]['AUC']:.4f}  F1 {out[k]['F1']:.4f}  ({time.time()-t:.0f}s)")
    if "classical" not in out:
        res = {}
        for name, r in run_baselines(gd, split, 42, ("all",), log=lambda *a: None).items():
            thr = derive_thresholds(r["probs"][split[1]], gd.labels[split[1]])["f1"]
            m = _metrics(gd, split, r["probs"], thr); m.update(_curves(gd.labels[split[2]], r["probs"][split[2]]), train_s=r["fit_s"], infer_s=r["infer_s"])
            res[name.replace(" (all features)", "")] = m
        out["classical"] = res
        _save(ds, "main", out)
        log(f"[{ds}] classical baselines done")


def stage_sweep(ds, quick=False, log=print):
    ev = load_events(ds)
    gd0 = build_graph(ev)
    split = make_split(gd0.labels, 42)
    out = _load(ds, "sweep")
    for sw, (title, vals) in SWEEPS.items():
        for v in (vals[:2] if quick else vals):
            k = f"{sw}|{v}"
            if k in out:
                continue
            t = time.time()
            gd = build_graph(ev, T=int(v)) if sw == "steps" else gd0
            kw_t, kw_m = {}, {}
            if sw == "lr": kw_t["lr"] = v
            if sw == "batch": kw_t["batch_size"] = int(v)
            if sw == "emb": kw_m["hidden"] = int(v)
            out[k] = _run_one(gd, split, "egt", _tc(quick, **kw_t), **kw_m)
            _save(ds, "sweep", out)
            log(f"[{ds}] {title} = {v}: AUC {out[k]['AUC']:.4f}  F1 {out[k]['F1']:.4f}  ({time.time()-t:.0f}s)")


def stage_metapath(ds, quick=False, log=print):
    gd = build_graph(load_events(ds))
    split = make_split(gd.labels, 42)
    out = _load(ds, "metapath")
    for name, mask in METAPATHS:
        if name in out:
            continue
        out[name] = _run_one(gd, split, "egt", _tc(quick), rel_mask=mask)
        _save(ds, "metapath", out)
        log(f"[{ds}] {name}: AUC {out[name]['AUC']:.4f}  F1 {out[name]['F1']:.4f}")


GRID = [(lr, h) for lr in (1e-3, 5e-3) for h in (16, 32, 48)]


def stage_tune(ds, seeds=(42, 43), quick=False, log=print):
    """Fair protocol: every model gets the same small hyper-parameter search (learning rate x embedding size = 6 settings),
    the setting with the best VALIDATION AUC (seed 42) is kept, and that setting is then trained with each seed."""
    gd = build_graph(load_events(ds))
    split = make_split(gd.labels, 42)
    out = _load(ds, "tuned")
    out["_info"] = dict(accounts=int(gd.n), labelled=int((gd.labels >= 0).sum()), fraud=int((gd.labels == 1).sum()), split=[int(len(x)) for x in split],
                        test_fraud=int((gd.labels[split[2]] == 1).sum()), ytest=[int(v) for v in gd.labels[split[2]]], grid=[list(g) for g in GRID])
    for key, name in MODELS:
        t0 = time.time()
        for lr, h in (GRID[:2] if quick else GRID):
            k = f"{key}|grid|{lr}|{h}"
            if k not in out:
                r = _run_one(gd, split, key, _tc(quick, seed=42, lr=lr), hidden=h)
                out[k] = {m: r[m] for m in ("val_auc", "AUC", "F1", "train_s", "params", "best_epoch")}
                out[k]["_run"] = r
                _save(ds, "tuned", out)
        cand = [(out[f"{key}|grid|{lr}|{h}"]["val_auc"], -h, lr, h) for lr, h in (GRID[:2] if quick else GRID)]
        _, _, blr, bh = max(cand)
        out[f"{key}|cfg"] = dict(lr=blr, hidden=bh)
        for sd in seeds:
            k = f"{key}|{sd}"
            if k in out:
                continue
            out[k] = out[f"{key}|grid|{blr}|{bh}"]["_run"] if sd == 42 else _run_one(gd, split, key, _tc(quick, seed=sd, lr=blr), hidden=bh)
            _save(ds, "tuned", out)
        log(f"[{ds}] {name:<12s} best lr={blr:g} hidden={bh}   AUC seed42 {out[f'{key}|42']['AUC']:.4f}   ({time.time()-t0:.0f}s)")
    for k in [k for k in out if "|grid|" in k]:
        out[k].pop("_run", None)           # keep the file small
    _save(ds, "tuned", out)


def deploy_dir(ds):
    return os.path.join(STUDY, "deploy", ds)


def stage_deploy(ds, quick=False, log=print):
    """Train the EGT that the GUI / command line use for this data set (same recipe and seed as the main study) and save everything they need."""
    import pickle
    from .train import save_checkpoint
    from .pipeline import ckpt_id
    d = deploy_dir(ds)
    ck = os.path.join(d, "egt.pt")
    if os.path.exists(os.path.join(d, "graph.pkl")) and os.path.exists(ck):
        return
    gd = build_graph(load_events(ds))
    split = make_split(gd.labels, 42)
    cfg = _load(ds, "tuned").get("egt|cfg", {})                      # the setting chosen on validation in the tuned comparison (default if absent)
    r = train_model(gd, ModelConfig(arch="egt", hidden=cfg.get("hidden", 48)), _tc(quick, seed=42, lr=cfg.get("lr", 1e-3)), split=split, verbose=False)
    arr = -np.ones(gd.n, int); arr[split[0]], arr[split[1]], arr[split[2]] = 0, 1, 2
    gd.split = arr
    os.makedirs(d, exist_ok=True)
    m = _metrics(gd, split, r["probs"], r["thresholds"]["f1"])
    save_checkpoint(r, ck, gd=gd, extra=dict(source=f"synthetic ({DATASETS[ds]['title']})", test_metrics={k: v for k, v in m.items()}, n_accounts=int(gd.n)))
    np.save(os.path.join(d, "final_probs.npy"), r["probs"])
    json.dump(dict(ckpt_id=ckpt_id(ck), n_accounts=int(gd.n)), open(os.path.join(d, "final_probs.json"), "w"))
    with open(os.path.join(d, "graph.pkl"), "wb") as f:
        pickle.dump(gd, f, protocol=4)
    log(f"[{ds}] deployed EGT: test AUC {m['AUC']:.4f}  F1 {m['F1']:.4f}  threshold {r['thresholds']['f1']:.4f}")


def run(ds, stages=("data", "main", "sweep", "metapath"), quick=False):
    torch.set_num_threads(1)
    global STUDY
    if quick:
        STUDY = os.path.join(ROOT, "study_quick")
    if "data" in stages and (quick or not os.path.exists(data_path(ds))):
        make_data(ds, quick)
    if "main" in stages: stage_main(ds, quick=quick)
    if "tune" in stages: stage_tune(ds, quick=quick)
    if "sweep" in stages: stage_sweep(ds, quick=quick)
    if "metapath" in stages: stage_metapath(ds, quick=quick)
    if "deploy" in stages: stage_deploy(ds, quick=quick)
