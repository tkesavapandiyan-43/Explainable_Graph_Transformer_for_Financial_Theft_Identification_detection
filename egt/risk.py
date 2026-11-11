"""Calibration + dynamic risk scoring (threshold policies & risk tiers) - every number is learnt on VALIDATION data."""
import numpy as np
import torch
from sklearn.metrics import precision_recall_curve

from .config import TIER_NAMES


def fit_temperature(logit_diff, y, iters=200):
    """Temperature scaling on validation logits (fraud-vs-legit log-odds). Ranking (AUC) is unchanged."""
    z = torch.tensor(logit_diff, dtype=torch.float32)
    t = torch.tensor(y, dtype=torch.float32)
    logT = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([logT], lr=0.1, max_iter=iters)

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(z / logT.exp(), t)
        loss.backward()
        return loss
    try:
        opt.step(closure)
        T = float(logT.exp().clamp(0.2, 10.0).item())
    except Exception:
        T = float("nan")
    if not np.isfinite(T):                    # LBFGS can diverge on almost-constant early logits -> robust 1-D grid search
        grid = torch.exp(torch.linspace(np.log(0.2), np.log(10.0), 200))
        losses = [torch.nn.functional.binary_cross_entropy_with_logits(z / g, t).item() for g in grid]
        T = float(grid[int(np.argmin(losses))])
    return T


def derive_thresholds(p, y, precision_target=0.90, recall_target=0.90):
    """Operating points learnt on the validation split.
       f1          threshold maximising F1                                   (balanced policy)
       precision90 lowest threshold with precision >= 90 %                    (paper's R@90%P operating point)
       recall90    highest threshold that still catches >= 90 % of the fraud  (review-queue policy)
       tiers       [MEDIUM, HIGH, CRITICAL] boundaries:
                   HIGH = F1 point, CRITICAL = lowest threshold above HIGH with precision >= 90 %,
                   MEDIUM = max(recall-90 point, point where precision first reaches 2 x prevalence)
       tier_stats / policy_val : what each tier / policy achieves on the validation split (so they can be audited)"""
    y = np.asarray(y).astype(int); p = np.asarray(p, dtype=float)
    prev = float(y.mean())
    if y.sum() == 0 or y.sum() == len(y):
        raise ValueError("validation split needs both classes to derive thresholds")
    pr, rc, th = precision_recall_curve(y, p)
    pr, rc = pr[:-1], rc[:-1]
    f1 = 2 * pr * rc / np.maximum(pr + rc, 1e-12)
    t_f1 = float(th[np.argmax(f1)])
    ok = np.where(pr >= precision_target)[0]
    t_p90 = float(th[ok[0]]) if len(ok) else float(th[np.argmax(pr)])
    ok = np.where(rc >= recall_target)[0]
    t_r90 = float(th[ok[-1]]) if len(ok) else float(th[0])
    above = np.where((th > t_f1) & (pr >= precision_target))[0]
    t_crit = float(th[above[0]]) if len(above) else float((t_f1 + th.max()) / 2)
    two = np.where(pr >= 2 * prev)[0]
    t_two = float(th[two[0]]) if len(two) else float(th[0])
    t_med = max(t_r90, t_two)
    if t_med >= t_f1:
        t_med = t_f1 / 2
    tiers = [float(t_med), float(t_f1), float(t_crit)]

    def at(t):
        flag = p >= t
        tp = int((flag & (y == 1)).sum())
        return dict(threshold=float(t), precision=tp / max(int(flag.sum()), 1), recall=tp / max(int(y.sum()), 1),
                    flag_rate=float(flag.mean()))
    names = np.array(TIER_NAMES)[np.digitize(p, tiers)]
    tier_stats = {t: dict(n=int((names == t).sum()), fraud_rate=float(y[names == t].mean()) if (names == t).any() else 0.0)
                  for t in TIER_NAMES}
    return dict(f1=t_f1, precision90=t_p90, recall90=t_r90, tiers=tiers, prevalence=prev, tier_stats=tier_stats,
                policy_val={"f1": at(t_f1), "precision90": at(t_p90), "recall90": at(t_r90)})


def assign_tier(p, tiers):
    idx = np.digitize(np.asarray(p), tiers)          # 0..3
    return np.array(TIER_NAMES)[idx]


def apply_policy(p, th, mode="f1", manual=0.5, topk=0.05):
    """Return (flag array, threshold actually used).
       modes: f1 | precision90 | recall90 | manual | topk (alert budget: flag the riskiest K % of THIS batch)"""
    p = np.asarray(p)
    if mode == "topk":
        k = max(1, int(round(len(p) * topk)))
        cut = float(np.sort(p)[::-1][min(k, len(p)) - 1])
        return p >= cut, cut
    cut = {"f1": th["f1"], "precision90": th["precision90"], "recall90": th["recall90"], "manual": manual}[mode]
    return p >= cut, float(cut)
