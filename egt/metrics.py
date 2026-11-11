import numpy as np
from sklearn.metrics import (roc_auc_score, average_precision_score, accuracy_score, precision_score,
                             recall_score, f1_score, matthews_corrcoef, confusion_matrix, brier_score_loss,
                             precision_recall_curve)


def recall_at_precision(y, p, target):
    pr, rc, _ = precision_recall_curve(y, p)
    ok = rc[pr >= target]
    return float(ok.max()) if len(ok) else 0.0


def ece(y, p, bins=10):
    y, p = np.asarray(y), np.asarray(p)
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for a, b in zip(edges[:-1], edges[1:]):
        m = (p >= a) & (p < b if b < 1 else p <= b)
        if m.any():
            e += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(e)


def full_metrics(y, p, thr):
    y = np.asarray(y).astype(int); p = np.asarray(p)
    yh = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yh, labels=[0, 1]).ravel()
    return {
        "AUC": float(roc_auc_score(y, p)), "PR_AUC": float(average_precision_score(y, p)),
        "R@80%P": recall_at_precision(y, p, 0.80), "R@85%P": recall_at_precision(y, p, 0.85),
        "R@90%P": recall_at_precision(y, p, 0.90),
        "Accuracy": float(accuracy_score(y, yh)), "Precision": float(precision_score(y, yh, zero_division=0)),
        "Recall": float(recall_score(y, yh, zero_division=0)), "F1": float(f1_score(y, yh, zero_division=0)),
        "MCC": float(matthews_corrcoef(y, yh)) if len(set(yh)) > 1 else 0.0,
        "Brier": float(brier_score_loss(y, p)), "ECE": ece(y, p),
        "TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn), "threshold": float(thr),
    }
