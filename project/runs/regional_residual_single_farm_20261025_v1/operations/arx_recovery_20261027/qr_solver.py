"""Exact orthogonal compression for the original multitask ElasticNet objective.

For centered A=[X,Y]=QR, use sqrt(m/n)*R as m synthetic rows, m=p+t.
This preserves the complete normalized augmented Gram matrix, including Y'Y,
so the loss, group penalty, coordinate updates and normalized dual gap agree.
No observations, outputs or singular directions are dropped.
"""
import numpy as np
from scipy.linalg import qr
from sklearn.linear_model import MultiTaskElasticNet


def compress(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x_mean, y_mean = x.mean(axis=0), y.mean(axis=0)
    n, p = x.shape
    a = np.asfortranarray(np.concatenate([x - x_mean, y - y_mean], axis=1))
    r = qr(a, mode='r', overwrite_a=True, check_finite=True)[0]
    m = min(n, p + y.shape[1])
    r = r[:m] * np.sqrt(m / n)
    return np.asfortranarray(r[:, :p]), np.asfortranarray(r[:, p:]), x_mean, y_mean


def model(alpha=.003, max_iter=3000, tol=1e-4):
    return MultiTaskElasticNet(alpha=alpha, l1_ratio=.8, max_iter=max_iter,
                               tol=tol, warm_start=True, selection='cyclic',
                               fit_intercept=False)


def certificate(x, y, coef, alpha, l1_ratio=.8):
    """Independent sufficient-statistic primal/dual gap (per original sample)."""
    n = len(x)
    w = coef.T
    g = x.T @ x / n
    c = x.T @ y / n
    yy = float(np.sum(y * y) / n)
    gw = g @ w
    wn = float(np.sum(w * w))
    residual_sq = max(0., yy - 2 * float(np.sum(w * c)) + float(np.sum(w * gw)))
    l1, l2 = alpha * l1_ratio, alpha * (1 - l1_ratio)
    gradient = c - gw - l2 * w
    dual_norm = float(np.max(np.linalg.norm(gradient, axis=1), initial=0.))
    scale = min(1., l1 / dual_norm) if dual_norm else 1.
    group_norm = float(np.linalg.norm(w, axis=1).sum())
    rty = yy - float(np.sum(w * c))
    gap = .5 * (residual_sq * (1 + scale * scale) + l2 * wn * (1 + scale * scale)) + l1 * group_norm - scale * rty
    objective = .5 * residual_sq + l1 * group_norm + .5 * l2 * wn
    return dict(objective=objective, dual_gap=gap, target_energy=yy)
