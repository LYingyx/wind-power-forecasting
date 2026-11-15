"""Validate equivalence before any original ARX task is restarted."""
from pathlib import Path
import sys, json, time, warnings, os
os.nice(15)
HERE = Path(__file__).resolve().parent
RUN = HERE.parents[1]
sys.path.insert(0, str(RUN / 'code'))
from qr_solver import compress, model, certificate
from sklearn.linear_model import MultiTaskElasticNet
import numpy as np
import torch
from sparse_arx import features
from physical_data import FarmData
from settings import atomic_json, now
torch.set_num_threads(1)


def compare(x, y, iterations, tol, label):
    t = time.monotonic()
    xc, yc, xm, ym = compress(x, y)
    qr_s = time.monotonic() - t
    ac = np.concatenate([xc, yc], axis=1)
    a = np.concatenate([x - xm, y - ym], axis=1)
    gram_error = float(np.max(np.abs(a.T @ a / len(a) - ac.T @ ac / len(ac))))
    assert gram_error < 1e-10
    original = MultiTaskElasticNet(alpha=.003, l1_ratio=.8, max_iter=iterations,
        tol=tol, warm_start=True, selection='cyclic')
    reduced = model(max_iter=iterations, tol=tol)
    rows = []
    for alpha in [.003, .001, .0003]:
        original.set_params(alpha=alpha); reduced.set_params(alpha=alpha)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            t = time.monotonic(); original.fit(x, y); raw_s = time.monotonic() - t
            t = time.monotonic(); reduced.fit(xc, yc); compressed_s = time.monotonic() - t
        intercept = ym - xm @ reduced.coef_.T
        coefficient_error = float(np.max(np.abs(original.coef_ - reduced.coef_)))
        prediction_error = float(np.max(np.abs(original.predict(x[:200]) - (x[:200] @ reduced.coef_.T + intercept))))
        gap_error = abs(float(original.dual_gap_) - float(reduced.dual_gap_))
        cert = certificate(x - xm, y - ym, reduced.coef_, alpha)
        assert coefficient_error < 1e-8 and prediction_error < 1e-8 and gap_error < 1e-10
        assert abs(cert['dual_gap'] - reduced.dual_gap_) < 1e-10
        rows.append(dict(alpha=alpha,raw_seconds=raw_s,compressed_seconds=compressed_s,
                         coefficient_max_error=coefficient_error,prediction_max_error=prediction_error,
                         dual_gap_error=gap_error,original_iterations=int(original.n_iter_),
                         compressed_iterations=int(reduced.n_iter_),certificate=cert))
    result = dict(label=label,shape=list(x.shape),compressed_rows=len(xc),qr_seconds=qr_s,
                  gram_max_error=gram_error,rows=rows)
    print(json.dumps(result), flush=True)
    return result


rng = np.random.default_rng(42)
x = rng.normal(size=(2000, 45)); x[:, 10:15] = x[:, :5]
y = x[:, :4] @ rng.normal(size=(4, 6)) + rng.normal(size=(2000, 6)) * .1
results = [compare(x, y, 30000, 1e-4, 'synthetic_collinear_converged')]
train = FarmData('train', 0, 'N0')
y = (train.y.numpy() - train.h.numpy()[:, -1:]).astype('f8')
all_x = features(train)
for name in ['LOCAL', 'REG_PAIR']:
    x = all_x[name].astype('f8'); mu=x.mean(0); sd=x.std(0); active=sd>1e-6
    x = np.asfortranarray(((x - mu) / np.maximum(sd,1e-6))[:,active])
    results.append(compare(x, y, 50, 0., 'full_2022_site00_' + name))
atomic_json(HERE / 'QR_EQUIVALENCE.json',dict(timestamp=now(),passed=True,tests=results,
    objective_unchanged=True,alpha_grid=[.003,.001,.0003],l1_ratio=.8,tolerance=1e-4,
    data_access='train/2022 only; no 2024'))
