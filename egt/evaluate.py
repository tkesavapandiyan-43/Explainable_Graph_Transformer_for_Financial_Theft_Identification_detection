"""Complete experiment on the real data: multi-seed EGT training, baselines on the same split, ablations, under-sampling
comparison, Welch tests, plots, predictions for the GUI and the training report. Every number comes from a run."""
import json, os, platform, sys, time
import numpy as np
import pandas as pd

from .config import ModelConfig, TrainConfig
from .features import N_PROFILE
from .metrics import full_metrics
from .risk import derive_thresholds, assign_tier
from .train import train_model, save_checkpoint, split_from_data
from .baselines import run_baselines

METRIC_COLS = ["AUC", "PR_AUC", "Precision", "Recall", "F1", "Accuracy", "R@90%P"]


def _module_params(model):
    groups = {"profile encoder": ("stat_enc",), "environment aggregation": ("agg", "ent_type", "ent_deg"), "environment difference": ("diff",),
              "behaviour (T-LSTM + attention pooling)": ("beh",), "attentive fusion": ("tok_proj", "tok_type", "fuse_att", "ln_f", "ffn_f", "pool_"),
              "detection layer": ("cls",)}
    out = {}
    for g, pref in groups.items():
        out[g] = sum(p.numel() for n, p in model.named_parameters() if n.startswith(pref))
    return out


def _versions():
    import torch, sklearn, scipy, networkx
    v = dict(python=sys.version.split()[0], torch=torch.__version__, numpy=np.__version__, pandas=pd.__version__,
             sklearn=sklearn.__version__, scipy=scipy.__version__, networkx=networkx.__version__)
    for m in ("streamlit", "plotly", "matplotlib"):
        try:
            v[m] = __import__(m).__version__
        except Exception:
            pass
    return v


def _hardware():
    import torch
    mem = ""
    try:
        for line in open("/proc/meminfo"):
            if line.startswith("MemTotal"):
                mem = f"{int(line.split()[1]) / 1024 / 1024:.1f} GB RAM"
    except Exception:
        pass
    cpu = platform.processor() or platform.machine()
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip(); break
    except Exception:
        pass
    return f"{platform.platform()} | CPU: {cpu} ({os.cpu_count()} logical cores, torch threads {torch.get_num_threads()}) | {mem} | GPU: " + \
           ("yes" if torch.cuda.is_available() else "none (CPU only)")


def run_experiment(gd, out_dir="outputs", ckpt_path="models/egt_vesta.pt", seeds=(42,), epochs=60, patience=10, batch_size=512,
                   baselines=True, ablations=True, undersample_run=True, max_minutes=0.0, source="", dataset_info=None, preprocessor_file=None, log=print, envit=True):
    os.makedirs(out_dir, exist_ok=True); os.makedirs(os.path.dirname(ckpt_path) or ".", exist_ok=True)
    split = split_from_data(gd, seeds[0])
    tr, va, te = split
    if gd.split is None:                                   # remember which accounts are train / val / test (for display + honest metrics)
        arr = -np.ones(gd.n, int); arr[tr], arr[va], arr[te] = 0, 1, 2; gd.split = arr
    y = gd.labels
    t_all = time.time()
    log(f"split  train {len(tr):,} (fraud {int((y[tr]==1).sum()):,}) | val {len(va):,} (fraud {int((y[va]==1).sum()):,}) | test {len(te):,} (fraud {int((y[te]==1).sum()):,})")
    base_tc = dict(epochs=epochs, patience=patience, batch_size=batch_size, max_minutes=max_minutes)
    rows, probs_te, runs = {}, {}, []
    # ---------------------------------------------------------------- EGT, one run per seed
    for sd in seeds:
        log(f"== EGT training, seed {sd}")
        res = train_model(gd, ModelConfig(), TrainConfig(seed=sd, **base_tc), split=split, log=log)
        m = full_metrics(y[te], res["probs"][te], res["thresholds"]["f1"])
        m.update(train_s=res["train_seconds"], infer_ms_per_1000=res["infer_seconds"] / gd.n * 1000, best_epoch=res["best_epoch"], val_auc=res["best_val_auc"])
        runs.append((res, m))
        log(f"   seed {sd}: val AUC {res['best_val_auc']:.4f} (epoch {res['best_epoch']}) | TEST AUC {m['AUC']:.4f}  PR-AUC {m['PR_AUC']:.4f}  F1 {m['F1']:.4f}  R@90%P {m['R@90%P']:.4f}  | {res['train_seconds']:.0f}s")
    best_res, best_m = max(runs, key=lambda r: r[0]["best_val_auc"])      # model selection on VALIDATION only
    log(f"selected seed by validation AUC: {best_res['train_cfg'].seed}")
    save_checkpoint(best_res, ckpt_path, gd=gd, extra=dict(source=source, test_metrics=best_m, n_accounts=int(gd.n), preprocessor_file=preprocessor_file,
                                                           seeds=list(seeds), dataset_info=dataset_info or {}))
    rows["EGT (proposed)"] = dict(seed_metrics=[m for _, m in runs]); probs_te["EGT (proposed)"] = best_res["probs"][te]
    # ---------------------------------------------------------------- the BASE PAPER model (EnvIT: GAT + GraphSAGE + T-LSTM), same data and split
    if envit:
        ms, last = [], None
        for sd in seeds:
            log(f"== EnvIT (base paper, re-implemented) training, seed {sd}")
            r = train_model(gd, ModelConfig(arch="envit"), TrainConfig(seed=sd, **base_tc), split=split, verbose=False)
            m = full_metrics(y[te], r["probs"][te], r["thresholds"]["f1"])
            m.update(train_s=r["train_seconds"], infer_ms_per_1000=r["infer_seconds"] / gd.n * 1000); ms.append(m); last = r
            log(f"   seed {sd}: TEST AUC {m['AUC']:.4f}  PR-AUC {m['PR_AUC']:.4f}  F1 {m['F1']:.4f}")
        rows["EnvIT (base paper, re-implemented)"] = dict(seed_metrics=ms); probs_te["EnvIT (base paper, re-implemented)"] = last["probs"][te]
    # ---------------------------------------------------------------- baselines on the same split
    if baselines:
        bl = {}
        for sd in seeds:
            log(f"== baselines, seed {sd}")
            for k, v in run_baselines(gd, split, sd, ("profile", "all"), log).items():
                bl.setdefault(k, []).append(v)
        for k, vs in bl.items():
            ms = []
            for v in vs:
                thr = derive_thresholds(v["probs"][va], y[va])["f1"]
                m = full_metrics(y[te], v["probs"][te], thr); m.update(train_s=v["fit_s"], infer_ms_per_1000=v["infer_s"] / len(te) * 1000)
                ms.append(m)
            rows[k] = dict(seed_metrics=ms); probs_te[k] = vs[0]["probs"][te]
    # ---------------------------------------------------------------- ablations (first seed)
    sd0 = seeds[0]
    if ablations:
        variants = {"EGT w/o environment aggregation": dict(cfg=dict(use_agg=False)), "EGT w/o environment difference": dict(cfg=dict(use_diff=False)),
                    "EGT w/o behaviour (T-LSTM)": dict(cfg=dict(use_beh=False)), "EGT w/o attentive fusion (plain average)": dict(cfg=dict(use_fusion_attn=False))}
        if undersample_run:
            variants["EGT + paper's under-sampling (5 normal : 1 fraud)"] = dict(cfg={}, tc=dict(undersample=5))
        for name, v in variants.items():
            log(f"== ablation: {name}")
            tcv = dict(base_tc)
            if "undersample" in v.get("tc", {}):             # an under-sampled epoch is ~6x shorter -> give it proportionally more epochs
                n_use = (1 + v["tc"]["undersample"]) * max(int((y[tr] == 1).sum()), 1)
                f = max(1, int(round(len(tr) / n_use)))
                tcv.update(epochs=base_tc["epochs"] * f, patience=base_tc["patience"] * f)
            r = train_model(gd, ModelConfig(**v["cfg"]), TrainConfig(seed=sd0, **tcv, **v.get("tc", {})), split=split, verbose=False)
            m = full_metrics(y[te], r["probs"][te], r["thresholds"]["f1"]); m.update(train_s=r["train_seconds"], infer_ms_per_1000=r["infer_seconds"] / gd.n * 1000)
            rows[name] = dict(seed_metrics=[m]); probs_te[name] = r["probs"][te]
            log(f"   TEST AUC {m['AUC']:.4f}  PR-AUC {m['PR_AUC']:.4f}  F1 {m['F1']:.4f}")
    # ---------------------------------------------------------------- tables
    table = []
    for name, d in rows.items():
        ms = d["seed_metrics"]; row = {"Model": name, "runs": len(ms)}
        for c in METRIC_COLS:
            vals = np.array([m[c] for m in ms]); row[c] = f"{vals.mean():.4f}" + (f" ± {vals.std(ddof=1):.4f}" if len(vals) > 1 else "")
        row["Train time (s)"] = round(float(np.mean([m["train_s"] for m in ms])), 1)
        row["Inference (ms / 1000 accounts)"] = round(float(np.mean([m["infer_ms_per_1000"] for m in ms])), 1)
        table.append(row)
    tdf = pd.DataFrame(table); tdf.to_csv(os.path.join(out_dir, "benchmark_table.csv"), index=False)
    # Welch's t-test (paper) on the per-seed AUCs
    welch = []
    if len(seeds) > 1:
        from scipy.stats import ttest_ind
        e = [m["AUC"] for m in rows["EGT (proposed)"]["seed_metrics"]]
        for name, d in rows.items():
            if name == "EGT (proposed)" or len(d["seed_metrics"]) < 2:
                continue
            b = [m["AUC"] for m in d["seed_metrics"]]
            t, p = ttest_ind(e, b, equal_var=False)
            welch.append(dict(baseline=name, egt_auc_mean=np.mean(e), baseline_auc_mean=np.mean(b), t=float(t), p_value=float(p), significant_p_lt_0_05=bool(p < 0.05)))
    pd.DataFrame(welch).to_csv(os.path.join(out_dir, "welch_tests.csv"), index=False)
    json.dump({k: v["seed_metrics"] for k, v in rows.items()}, open(os.path.join(out_dir, "benchmark.json"), "w"), indent=1, default=float)
    # predictions for the GUI (ROC / PR / confusion matrix at any threshold)
    np.savez_compressed(os.path.join(out_dir, "predictions_test.npz"), y_test=y[te], account_ids=gd.account_ids[te],
                        names=np.array(list(probs_te)), **{f"p{i}": v for i, v in enumerate(probs_te.values())},
                        y_val=y[va], p_val=best_res["probs"][va])
    # final scores of every account with the selected model
    th = best_res["thresholds"]
    sc = pd.DataFrame({"account_id": gd.account_ids, "person_id": gd.person_ids, "split": np.array(["train", "val", "test", "-"])[np.where(gd.split >= 0, gd.split, 3)] if gd.split is not None else "",
                       "label": gd.labels, "fraud_probability": best_res["probs"], "risk_score": np.round(best_res["probs"] * 100, 2),
                       "risk_tier": assign_tier(best_res["probs"], th["tiers"])})
    sc.to_csv(os.path.join(out_dir, "final_scores.csv"), index=False)
    np.save(os.path.join(out_dir, "final_probs.npy"), best_res["probs"])
    from .pipeline import ckpt_id
    json.dump(dict(ckpt_id=ckpt_id(ckpt_path), n_accounts=int(gd.n)), open(os.path.join(out_dir, "final_probs.json"), "w"))
    _plots(out_dir, probs_te, y[te], best_res, th)
    ctx = dict(gd=gd, split=split, runs=runs, best=(best_res, best_m), table=tdf, welch=pd.DataFrame(welch), seeds=seeds, source=source,
               dataset_info=dataset_info or {}, ckpt_path=ckpt_path, total_seconds=time.time() - t_all, epochs=epochs, batch_size=batch_size)
    write_training_report(os.path.join(out_dir, "training_report.txt"), ctx)
    log("\n" + tdf.to_string(index=False))
    return ctx


def _plots(out, probs, y, best, th):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, precision_recall_curve
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    for name, p in probs.items():
        if "w/o" in name or "under-sampling" in name:
            continue
        fpr, tpr, _ = roc_curve(y, p); pr, rc, _ = precision_recall_curve(y, p)
        lw = 2.8 if name.startswith("EGT") else 1.1
        ax[0].plot(fpr, tpr, lw=lw, label=name); ax[1].plot(rc, pr, lw=lw, label=name)
    ax[0].plot([0, 1], [0, 1], "k:", lw=.8); ax[0].set_xlabel("False positive rate"); ax[0].set_ylabel("True positive rate"); ax[0].set_title("ROC curve (test accounts)")
    ax[1].axhline(0.9, color="gray", ls=":", lw=.8); ax[1].set_xlabel("Recall"); ax[1].set_ylabel("Precision"); ax[1].set_title("Precision-Recall curve (test accounts)")
    ax[0].legend(fontsize=7); ax[0].grid(alpha=.3); ax[1].grid(alpha=.3)
    fig.tight_layout(); fig.savefig(os.path.join(out, "roc_pr.png"), dpi=130); plt.close(fig)
    h = best["history"]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot([x["epoch"] for x in h], [x["loss"] for x in h]); ax[0].set_title("Training loss"); ax[0].set_xlabel("epoch")
    ax[1].plot([x["epoch"] for x in h], [x["val_auc"] for x in h]); ax[1].axvline(best["best_epoch"], color="r", ls="--"); ax[1].set_title("Validation AUC (red = selected epoch)"); ax[1].set_xlabel("epoch")
    fig.tight_layout(); fig.savefig(os.path.join(out, "training_curves.png"), dpi=130); plt.close(fig)


def write_training_report(path, c):
    gd, (tr, va, te) = c["gd"], c["split"]
    best, bm = c["best"]
    y = gd.labels
    th = best["thresholds"]
    L = []; P = L.append
    P("TRAINING REPORT  (generated automatically by egt.evaluate from the actual run)")
    P("=" * 100)
    P(f"Dataset / source           : {c['source']}")
    di = c["dataset_info"]
    for k in ("n_transactions", "n_accounts", "n_persons", "fraud_transactions", "fraud_accounts", "legit_accounts", "split_mode", "n_event_features"):
        if k in di:
            P(f"  {k:<26s}: {di[k]}")
    P(f"\nRECORDS AND ACCOUNTS")
    P(f"  events (transactions) processed : {len(gd.events):,}")
    P(f"  accounts                        : {gd.n:,}   (labelled fraud {int((y==1).sum()):,} | legitimate {int((y==0).sum()):,} | unlabeled {int((y<0).sum()):,})")
    P(f"\nGRAPH SCHEMA")
    P(f"  node types   : Account {gd.n:,} | Device {len(gd.ent_names['device']):,} | Network {len(gd.ent_names['ip']):,} | Address {len(gd.ent_names['address']):,}"
      f" (personal identity = {len(set(gd.person_ids)):,} groups; used as account-account sibling edges)")
    ns = int((gd.sib_src != gd.sib_dst).sum())
    P(f"  edge types   : Account-Device {len(gd.edges['device']['src']):,} | Account-Network {len(gd.edges['ip']['src']):,} | Account-Address {len(gd.edges['address']['src']):,}"
      f" | Account-Account (same identity) {ns:,} directed (+{gd.n:,} self-loops)")
    P(f"  features     : {gd.stats.shape[1]} account-level features ({len(gd.feat_keys)} named) | {gd.ev_feat.shape[1]} features per event, last 40 events per account"
      f" | edge features: 2 (account-entity), 8 (account-account difference)")
    P(f"\nMODEL ARCHITECTURE (Explainable Graph Transformer)")
    cfg = best["model_cfg"]
    P(f"  hidden {cfg.hidden}, heads {cfg.heads}, env-aggregation layers {cfg.env_layers}, env-difference layers {cfg.diff_layers}, dropout {cfg.dropout}")
    P(f"  parameters: {best['n_params']:,}   " + " | ".join(f"{k}: {v:,}" for k, v in _module_params(best["model"]).items()))
    tc = best["train_cfg"]
    P(f"\nHYPERPARAMETERS")
    P(f"  optimiser Adam, lr {tc.lr}, weight decay {tc.weight_decay}, batch {tc.batch_size} seed accounts, neighbour caps: {tc.cap_acct_per_ent} accounts/entity, {tc.cap_sib} siblings,"
      f" max epochs {c['epochs']}, early-stopping patience {tc.patience}, loss: class-weighted cross-entropy, gradient clipping 5.0")
    P(f"\nSPLIT (accounts)")
    for n, i in (("train", tr), ("validation", va), ("test", te)):
        P(f"  {n:<11s}: {len(i):,}   fraud {int((y[i]==1).sum()):,}  legitimate {int((y[i]==0).sum()):,}  fraud rate {(y[i]==1).mean():.3%}")
    P(f"\nTRAINING")
    P(f"  seeds run: {list(c['seeds'])}  | selected (best validation AUC): seed {tc.seed}")
    P(f"  epochs run: {len(best['history'])} | best epoch: {best['best_epoch']} | training time (selected run): {best['train_seconds']:.0f}s | total experiment time: {c['total_seconds']:.0f}s")
    P(f"  calibration temperature T = {best['T']:.4f}")
    P(f"  thresholds (from validation): F1 {th['f1']:.4f} | precision-90 {th['precision90']:.4f} | recall-90 {th['recall90']:.4f} | tiers {[round(t,4) for t in th['tiers']]}")
    for pol, v in th["policy_val"].items():
        P(f"     policy {pol:<12s} on validation: precision {v['precision']:.3f}, recall {v['recall']:.3f}, flags {v['flag_rate']:.2%} of accounts")
    for t, v in th["tier_stats"].items():
        P(f"     tier {t:<9s} on validation: {v['n']:,} accounts, empirical fraud rate {v['fraud_rate']:.2%}")
    P(f"\nVALIDATION  : best AUC {best['best_val_auc']:.4f}")
    P("TEST METRICS (selected model, threshold = validation F1 threshold):")
    for k in ("AUC", "PR_AUC", "Precision", "Recall", "F1", "Accuracy", "MCC", "R@80%P", "R@85%P", "R@90%P", "Brier", "ECE", "TP", "FP", "TN", "FN"):
        P(f"  {k:<10s}: {bm[k]:.4f}" if isinstance(bm[k], float) else f"  {k:<10s}: {bm[k]}")
    P("\nMODEL COMPARISON (mean ± std over seeds where several seeds were run; same split for every model)")
    P(c["table"].to_string(index=False))
    if len(c["welch"]):
        P("\nWELCH'S T-TEST ON TEST AUC (EGT vs baseline, over seeds)")
        P(c["welch"].to_string(index=False))
    else:
        P("\nWelch's t-test: needs >= 2 seeds (run with --seeds 42 43 44).")
    P(f"\nHARDWARE : {_hardware()}")
    P("SOFTWARE : " + ", ".join(f"{k} {v}" for k, v in _versions().items()))
    sz = os.path.getsize(c["ckpt_path"]) / 1024 / 1024
    P(f"CHECKPOINT: {c['ckpt_path']}  ({sz:.2f} MB)")
    open(path, "w").write("\n".join(L))
