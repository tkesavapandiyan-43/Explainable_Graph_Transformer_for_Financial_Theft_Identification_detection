"""Classical (non-graph) baselines, trained on the SAME split as the graph model."""
import time
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from .features import N_PROFILE


def baseline_specs(seed):
    return {
        "Logistic Regression": lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=500, class_weight="balanced")),
        "Random Forest": lambda: RandomForestClassifier(200, min_samples_leaf=2, class_weight="balanced_subsample", n_jobs=-1, random_state=seed),
        "Gradient Boosting": lambda: HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, class_weight="balanced", random_state=seed),
    }


def run_baselines(gd, split, seed=42, feature_sets=("all",), log=print):
    """Returns {name: dict(probs=<all accounts>, fit_s, infer_s)}.
       feature set 'all'     = every account-level feature the EGT profile token receives (profile + graph-degree + dataset aggregates)
       feature set 'profile' = only the 31 behaviour/profile statistics (no graph-derived numbers)"""
    tr, va, te = split
    y = gd.labels
    X = {"all": gd.stats, "profile": gd.stats[:, :N_PROFILE]}
    out = {}
    for fs in feature_sets:
        for name, mk in baseline_specs(seed).items():
            key = f"{name} ({fs} features)"
            t = time.time(); m = mk().fit(X[fs][tr], y[tr]); fit_s = time.time() - t
            t = time.time(); m.predict_proba(X[fs][te]); infer_s = time.time() - t
            out[key] = dict(probs=m.predict_proba(X[fs])[:, 1], fit_s=fit_s, infer_s=infer_s, n_test=len(te))
            log(f"   {key:<44s} fit {fit_s:6.1f}s")
    return out
