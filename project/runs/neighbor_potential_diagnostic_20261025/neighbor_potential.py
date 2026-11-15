"""Zero-training headroom diagnostic for neighbor-farm information (2023 validation only).

For each target farm, residuals of an existing independent model (formal E2C_CBH, seed 42)
are regressed on information available at the forecast origin:
  SELF      : the target's own recently realized forecast errors
  SELF+RAW  : + neighbors' raw recent power level and 1 h change (what NEIGH_GENERIC sees)
  SELF+ERR  : + neighbors' recently realized forecast errors
  SELF+BOTH : + both neighbor blocks
Ridge models are fit per farm and per lead on Jan-Jun 2023 and evaluated on Jul-Dec 2023.
The corrected forecast is prediction + fitted residual. Output: relative MAE/RMSE change of
each corrected forecast vs the uncorrected prediction. 2024 is never read.

Realized error of farm j at time t: observed CF at t minus the farm's own forecast for t
issued one hour earlier (lead 4). Neighbor information is delayed by 15 minutes.
"""
import json, sys
from pathlib import Path
import numpy as np

ROOT = Path('/shared_data_ssd2/liying/wind_cluster_project_20261018/project')
RUN = ROOT / 'runs/independent_single_farm_20261021_v1'
CACHE = ROOT / 'data/model_cache_pad5_f8'
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'runs/physics_observation_single_farm_20261024_v1/code'))
from physics import neighbor_geometry  # same neighbor definition as NEIGH_GENERIC

N_SITES, LEADS, K = 56, 24, 8
STEP = 15 * 60


def ridge_fit_predict(xtr, ytr, xte, lam=1.0, l1=False):
    mu, sd = xtr.mean(0), xtr.std(0) + 1e-6
    a, b = (xtr - mu) / sd, (xte - mu) / sd
    a = np.hstack([a, np.ones((len(a), 1))]); b = np.hstack([b, np.ones((len(b), 1))])
    reg = lam * np.eye(a.shape[1]); reg[-1, -1] = 0
    w = np.linalg.solve(a.T @ a + reg, a.T @ ytr)
    if l1:  # iteratively reweighted least squares approximating least absolute deviation
        for _ in range(15):
            r = np.abs(ytr - a @ w).clip(min=1e-3)
            aw = a / r[:, None]
            w = np.linalg.solve(a.T @ aw + reg, aw.T @ ytr)
    return b @ w


def main():
    origin = np.load(CACHE / 'val/origin_unix.npy')
    hist = np.load(CACHE / 'val/station_history.npy', mmap_mode='r')  # [n,56,24] CF, last = origin
    pos = {int(t): i for i, t in enumerate(origin)}
    lat = np.load(CACHE / 'static/station_latitude.npy'); lon = np.load(CACHE / 'static/station_longitude.npy')
    ids, geo, mask = neighbor_geometry(lat, lon, maximum_k=K, radius_km=150.)

    pred = np.stack([np.load(RUN / f'jobs/formal_E2C_CBH_s{s:02d}_r42/val_prediction.npy') for s in range(N_SITES)], 1)
    target = np.stack([np.load(RUN / f'prepared/val/{s:02d}/target.npy') for s in range(N_SITES)], 1)
    hist = np.asarray(hist)

    def index_of(offset_seconds):
        return np.array([pos.get(int(t) + offset_seconds, -1) for t in origin])

    # realized error at time (origin - delay): obs(t) - forecast for t issued 1 h before t (lead 4)
    def realized(delay_steps):
        t_idx = index_of(-delay_steps * STEP)               # sample whose origin is t
        src = index_of(-(delay_steps + 4) * STEP)            # sample issued 1 h before t
        ok = (t_idx >= 0) & (src >= 0)
        obs = np.where(ok[:, None], hist[np.maximum(t_idx, 0), :, -1], np.nan)
        fc = np.where(ok[:, None], pred[np.maximum(src, 0), :, 3], np.nan)
        return obs - fc                                      # [n,56]

    r_self0 = realized(0)                                    # own farm, no delay
    r_self1 = realized(1)
    r_nb = realized(1)                                       # neighbors, 15 min delay
    raw_level = hist[:, :, -2]                               # neighbors' power 15 min before origin
    raw_change = hist[:, :, -2] - hist[:, :, -6]             # neighbors' 1 h change, delayed 15 min

    half = origin < np.datetime64('2023-07-01').astype('datetime64[s]').astype(np.int64)
    results = []
    for s in range(N_SITES):
        nb = ids[s][mask[s]]
        if len(nb) == 0:
            continue
        blocks = {
            'SELF': [r_self0[:, [s]], r_self1[:, [s]]],
            'SELF+RAW': [r_self0[:, [s]], r_self1[:, [s]], raw_level[:, nb], raw_change[:, nb]],
            'SELF+ERR': [r_self0[:, [s]], r_self1[:, [s]], r_nb[:, nb]],
            'SELF+BOTH': [r_self0[:, [s]], r_self1[:, [s]], r_nb[:, nb], raw_level[:, nb], raw_change[:, nb]],
        }
        err = target[:, s] - pred[:, s]                       # [n,24]
        for name, parts in blocks.items():
            x = np.hstack(parts)
            ok = np.isfinite(x).all(1) & np.isfinite(err).all(1)
            tr, te = ok & half, ok & ~half
            for fit in ('L2', 'L1'):
                fitted = np.column_stack([ridge_fit_predict(x[tr], err[tr, L], x[te], l1=fit == 'L1') for L in range(LEADS)])
                base = err[te]; corr = err[te] - fitted
                for band, sl in [('all', slice(0, 24)), ('0_1h', slice(0, 4)), ('1_3h', slice(4, 12)), ('3_6h', slice(12, 24))]:
                    b, c = base[:, sl], corr[:, sl]
                    results.append(dict(fit=fit, site=s, n_neighbors=int(len(nb)), nearest_km=float(geo[s, 0, 2]), info=name, horizon=band,
                                        n_test=int(te.sum()), mae_base=float(np.abs(b).mean()), mae_corr=float(np.abs(c).mean()),
                                        rmse_base=float(np.sqrt((b ** 2).mean())), rmse_corr=float(np.sqrt((c ** 2).mean()))))
    import csv
    with open(OUT / 'neighbor_potential_per_farm.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(results[0])); w.writeheader(); w.writerows(results)
    print('farms with neighbors:', len({r['site'] for r in results}))


if __name__ == '__main__':
    main()
