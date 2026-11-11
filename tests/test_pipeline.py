#!/usr/bin/env python3
"""Automated tests.  Run:  python tests/test_pipeline.py     (or  python main.py test)
They use SYNTHETIC data / a random Vesta-LAYOUT mock purely to test the software; no result is produced from them."""
import os, sys, tempfile, time, traceback, warnings
warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "tests"))
import numpy as np, pandas as pd, torch

from egt.config import ModelConfig, TrainConfig, RELATIONS
from egt.data_synth import generate_events
from egt.features import build_graph, Scaler
from egt.sampler import GraphIndex, predict_logits
from egt.model import EGT
from egt.train import train_model, save_checkpoint, load_checkpoint, split_from_data
from egt.risk import derive_thresholds, apply_policy, assign_tier
from egt.pipeline import Predictor, normalized_events
from egt.explain import explain_account, feature_attribution, _ego

S = {}


def setup():
    ev, _ = generate_events(1200, seed=5)
    S["ev"] = ev; S["gd"] = build_graph(ev)
    S["res"] = train_model(S["gd"], train_cfg=TrainConfig(epochs=12, patience=6), verbose=False)
    S["dir"] = tempfile.mkdtemp()
    S["ckpt"] = os.path.join(S["dir"], "m.pt")
    save_checkpoint(S["res"], S["ckpt"], extra=dict(source="synthetic-dev"), gd=S["gd"])
    S["pred"] = Predictor(S["ckpt"]); S["score"] = S["pred"].score(ev); S["tbl"] = S["pred"].table(S["score"])


def test_graph_construction():
    gd = S["gd"]
    assert gd.stats.shape == (gd.n, len(gd.feat_keys)) and not np.isnan(gd.stats).any()
    for r in RELATIONS:
        e = gd.edges[r]; assert len(e["src"]) == len(e["dst"]) > 0 and e["src"].max() < gd.n and e["dst"].max() < len(gd.ent_names[r])
    assert (gd.sib_src == gd.sib_dst).sum() == gd.n                       # one self loop per account
    assert gd.seq_event_row.shape == (gd.n, 40) and gd.ev_feat.shape[1] == 12


def test_sampler_properties():
    gi = GraphIndex(S["gd"], S["res"]["scaler"])
    ids = np.array([3, 17, 40]); G = gi.sample(ids)
    assert G["n_seeds"] == 3 and (G["_nodes"][:3] == ids).all()
    for r in range(3):
        assert int(G["edges"][r]["dst"].max()) < len(G["_ents"][r])
    a = predict_logits(S["res"]["model"], gi, ids, 512); b = predict_logits(S["res"]["model"], gi, ids, 512)
    assert np.allclose(a, b)                                              # deterministic


def test_every_module_participates_in_prediction():
    gd = S["gd"]; gi = GraphIndex(gd, S["res"]["scaler"]); G = gi.sample(np.arange(64))
    m = EGT(ModelConfig(n_stats=gd.stats.shape[1], seq_dim=gd.seq_dim)); m.train()
    m(G)[:, 1].sum().backward()
    groups = {"profile": "stat_enc", "aggregation": "agg", "difference": "diff", "T-LSTM behaviour": "beh", "fusion": "fuse_att", "detection": "cls"}
    for name, pref in groups.items():
        g = sum(float(p.grad.abs().sum()) for n, p in m.named_parameters() if n.startswith(pref) and p.grad is not None)
        assert g > 0, f"module {name} receives no gradient -> not connected to the output"
    # switching a module off must change the output
    base = predict_logits(S["res"]["model"], gi, np.arange(64))
    for flag in ("use_agg", "use_diff", "use_beh"):
        cfg = ModelConfig(**{**S["res"]["model_cfg"].to_dict(), flag: False}); m2 = EGT(cfg); m2.load_state_dict(S["res"]["model"].state_dict()); m2.eval()
        with torch.no_grad():
            assert not np.allclose(m2(gi.sample(np.arange(64))).numpy(), base), flag


def test_base_paper_envit_model():
    from egt.model import build_model
    gd = S["gd"]; gi = GraphIndex(gd, S["res"]["scaler"]); G = gi.sample(np.arange(48))
    m = build_model(ModelConfig(arch="envit", n_stats=gd.stats.shape[1], seq_dim=gd.seq_dim)); m.train()
    out = m(G); assert out.shape == (48, 2)
    out[:, 1].sum().backward()
    dead = [n for n, p in m.named_parameters() if p.grad is None or float(p.grad.abs().sum()) == 0]
    assert not dead, dead                                   # GAT, GraphSAGE, T-LSTM, fusion, detection all take part


def test_shap_efficiency():
    from egt.shap_explain import shap_values
    acc = S["tbl"].iloc[3].account_id; i = S["gd"].acct_index[acc]
    for by in ("group", "feature"):
        o = shap_values(S["pred"], S["score"], i, by=by, n_perm=3)
        assert abs(o["sum_shap"] - (o["f_x"] - o["f_base"])) < 1e-3, (by, o["sum_shap"], o["f_x"] - o["f_base"])
    assert len(shap_values(S["pred"], S["score"], i)["table"]) >= 6


def test_training_learns():
    assert S["res"]["best_val_auc"] > 0.7, S["res"]["best_val_auc"]
    tr, va, te = S["res"]["split"]; assert not (set(tr) & set(va)) and not (set(tr) & set(te)) and not (set(va) & set(te))


def test_scaler_fit_on_train_only():
    gd = S["gd"]; tr = S["res"]["split"][0]
    assert np.allclose(S["res"]["scaler"].mean, gd.stats[tr].mean(0))


def test_checkpoint_roundtrip():
    ck = load_checkpoint(S["ckpt"]); assert abs(ck["T"] - S["res"]["T"]) < 1e-9
    again = S["pred"].score(S["ev"]); assert np.allclose(again.prob, S["score"].prob)


def test_thresholds_policies_tiers():
    gd, res = S["gd"], S["res"]; va = res["split"][1]
    th = res["thresholds"]; t = th["tiers"]
    assert t[0] < t[1] < t[2]
    p = S["score"].prob
    f_low, _ = apply_policy(p, th, "manual", 0.2); f_high, _ = apply_policy(p, th, "manual", 0.8)
    assert f_low.sum() > f_high.sum()                                     # changing the threshold changes decisions
    f_k, cut = apply_policy(p, th, "topk", topk=0.05); assert abs(f_k.mean() - 0.05) < 0.02
    tiers = assign_tier(p, t); assert set(tiers) <= {"LOW", "MEDIUM", "HIGH", "CRITICAL"}
    ts = th["tier_stats"]; assert ts["CRITICAL"]["fraud_rate"] >= ts["LOW"]["fraud_rate"]
    # thresholds were derived from the validation split only
    thr2 = derive_thresholds(res["probs"][va], gd.labels[va]); assert abs(thr2["f1"] - th["f1"]) < 1e-9


def test_risk_table():
    t = S["tbl"]; assert t.risk_score.between(0, 100).all() and (t.risk_score.diff().dropna() <= 1e-9).all()
    assert len(t) == S["gd"].n


def test_explanation():
    acc = S["tbl"].iloc[0].account_id
    ex = explain_account(S["pred"], S["score"], acc, tbl=S["tbl"])
    assert len(ex["narrative"]) > 50 and len(ex["modules"]) == 4 and len(ex["features"]) == len(S["gd"].feat_keys)
    # Integrated-Gradients completeness: sum of attributions ~ f(x) - f(baseline)
    i = S["gd"].acct_index[acc]; G, *_ = _ego(S["pred"], S["score"], i)
    with torch.no_grad():
        f1 = S["pred"].model(G)[0]; G0 = dict(G); st = G["stats"].clone(); st[0] = 0; G0["stats"] = st; f0 = S["pred"].model(G0)[0]
    delta = float((f1[1] - f1[0]) - (f0[1] - f0[0])) / S["pred"].T
    total = float(feature_attribution(S["pred"], S["score"], i, steps=40).attribution.sum())
    assert abs(delta - total) <= 0.15 * abs(delta) + 0.05, (delta, total)


def test_graph_visualisation():
    from egt.viz import render_graph, draw_matplotlib, draw_plotly
    for mode in ("all", "risky", "focus"):
        H, pos, note = render_graph(S["score"], S["tbl"], mode, S["tbl"].iloc[0].account_id, 2, 400)
        assert len(H) > 0 and set(pos) == set(H.nodes)
        fig = draw_matplotlib(H, pos, mode, note); fig.savefig(os.path.join(S["dir"], f"{mode}.png")); draw_plotly(H, pos, mode, note)


def test_timeline():
    acc = S["tbl"].iloc[0].account_id
    tl = S["pred"].timeline(normalized_events(S["score"]), acc, n_points=3)
    assert len(tl) >= 2 and tl.events_seen.is_monotonic_increasing and tl.risk_score.between(0, 100).all()


def test_vesta_pipeline():
    from make_vesta_like_mock import make_mock
    from egt.vesta import VestaPreprocessor, dataset_report
    d = os.path.join(S["dir"], "vm"); tx, idf = make_mock(8000, d, seed=3)
    rep = dataset_report(tx, idf, os.path.join(d, "report.txt"), log=lambda x: None); assert "accounts" in rep and "unique operating systems" in rep
    pp = VestaPreprocessor(); ev, info = pp.fit_transform(tx, idf, "time", log=lambda x: None)
    assert info["n_transactions"] == 8000 and info["n_event_features"] > 80
    gd = build_graph(ev); assert gd.stats.shape[1] > 100 and gd.split is not None
    tr, va, te = split_from_data(gd, 0)
    # time-aware split: no leakage of the future into training
    first = ev.groupby("account_id")["timestamp"].min()
    ft = lambda ids: first.loc[gd.account_ids[ids]]
    assert ft(tr).max() <= ft(va).min() and ft(va).max() <= ft(te).min()
    # re-transforming the same file gives identical features
    ev2 = pp.transform_files(tx, idf); xc = [c for c in ev.columns if c.startswith("x_")]
    assert np.allclose(ev[xc].to_numpy(), ev2[xc].to_numpy(), atol=1e-5)
    r = train_model(gd, train_cfg=TrainConfig(epochs=3, patience=3), verbose=False)
    pp.save(os.path.join(d, "pp.pkl")); ck = os.path.join(d, "v.pt")
    save_checkpoint(r, ck, extra=dict(source="mock-unit-test", preprocessor_file="pp.pkl"), gd=gd)
    p = Predictor(ck); s = p.score_vesta(tx, idf); assert len(s.prob) == gd.n and p.preprocessor is not None
    try:                                                  # a graph with a different feature layout must be rejected, not mis-scored
        p.check_compat(S["gd"])
        raise AssertionError("incompatible input must be rejected")
    except ValueError:
        pass


def test_comparison_models():
    """all 11 models of the results study run, give finite [S,2] logits and receive gradients"""
    from egt.model import build_model
    gd = S["gd"]
    G = GraphIndex(gd, S["res"]["scaler"]).sample(np.arange(32), cap_acct_per_ent=20, cap_sib=10)
    for arch in ("egt", "envit", "envit_graph", "envit_beh", "gat", "sage", "rgcn", "han", "hacud", "sobt", "secgfd"):
        m = build_model(ModelConfig(arch=arch, n_stats=gd.stats.shape[1], seq_dim=gd.seq_dim, hidden=16))
        out = m(G); assert out.shape == (32, 2) and torch.isfinite(out).all(), arch
        out[:, 1].sum().backward()
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in m.parameters()), arch


def test_relation_mask():
    """meta-path study: switching relations off changes the output, switching none off does not"""
    from egt.model import build_model
    gd = S["gd"]
    G = GraphIndex(gd, S["res"]["scaler"]).sample(np.arange(16), cap_acct_per_ent=20, cap_sib=10)
    torch.manual_seed(0); a = build_model(ModelConfig(arch="egt", n_stats=gd.stats.shape[1], seq_dim=gd.seq_dim)).eval()
    cfg2 = ModelConfig(arch="egt", n_stats=gd.stats.shape[1], seq_dim=gd.seq_dim, rel_mask=(1, 0, 0))
    b = build_model(cfg2).eval(); b.load_state_dict(a.state_dict())
    with torch.no_grad():
        assert not torch.allclose(a(G), b(G))
        cfg3 = ModelConfig(arch="egt", n_stats=gd.stats.shape[1], seq_dim=gd.seq_dim, rel_mask=(1, 1, 1))
        c = build_model(cfg3).eval(); c.load_state_dict(a.state_dict())
        assert torch.allclose(a(G), c(G))


def test_temperature_robust():
    """calibration must not return NaN on almost constant (early-epoch) logits"""
    from egt.risk import fit_temperature
    t = fit_temperature(np.full(200, -0.7) + np.random.default_rng(0).normal(0, 1e-4, 200), (np.arange(200) % 9 == 0).astype(int))
    assert np.isfinite(t) and 0.2 <= t <= 10


def test_experiment_outputs():
    from egt.evaluate import run_experiment
    out = os.path.join(S["dir"], "exp"); ck = os.path.join(S["dir"], "exp", "m.pt")
    ctx = run_experiment(S["gd"], out, ck, (1, 2), 5, 3, 512, True, False, False, 0.0, "synthetic-dev test", {}, None, log=lambda x: None)
    for f in ("training_report.txt", "benchmark_table.csv", "welch_tests.csv", "predictions_test.npz", "final_scores.csv", "roc_pr.png", "training_curves.png"):
        assert os.path.exists(os.path.join(out, f)), f
    t = pd.read_csv(os.path.join(out, "benchmark_table.csv")); assert {"Logistic Regression (all features)", "Random Forest (all features)", "Gradient Boosting (all features)", "EGT (proposed)"} <= set(t.Model)
    assert "Train time (s)" in t.columns and "Inference (ms / 1000 accounts)" in t.columns


if __name__ == "__main__":
    t0 = time.time(); setup(); fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            t = time.time()
            try:
                fn(); print(f"PASS  {name:<45s} {time.time()-t:5.1f}s")
            except Exception:
                fails += 1; print(f"FAIL  {name}"); traceback.print_exc()
    print(f"\n{'ALL TESTS PASSED' if not fails else str(fails) + ' TEST(S) FAILED'}  ({time.time()-t0:.0f}s)")
    sys.exit(1 if fails else 0)
