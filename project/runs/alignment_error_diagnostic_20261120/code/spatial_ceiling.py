"""Q4: value of the spatial neighbourhood (grid -> farm representativeness).
Per farm GBDT on hourly power (fit 2022, eval 2023 excl. Nov) using
  point  : bilinear speed at the farm (100m ERA5 / 925+10 Pangu)
  patch5 : speeds at the 5x5 nearest grid nodes (+ point)
  patch5+dir : plus u/v at the farm
ERA5 = perfect-forecast ceiling; Pangu = operational (daily protocol issue rule).
Also: wrong-patch control (5x5 patch of a distant farm) to separate 'more inputs' from spatial alignment."""
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from common import *
from power_tests import load_power, split_masks

rng = np.random.default_rng(0)


def patch_ids(sr, sc, k=5):
    r0 = np.clip(np.round(sr).astype(int) - k // 2, 0, 24 - k); c0 = np.clip(np.round(sc).astype(int) - k // 2, 0, 25 - k)
    return [(np.arange(r, r + k)[:, None] * 25 + np.arange(c, c + k)[None, :]).ravel() for r, c in zip(r0, c0)]


if __name__ == '__main__':
    t15, cf15 = load_power()
    n = (T1 - T0) // 3600
    hidx = (t15 - T0) // 3600; ph = ((t15 - T0) % 3600) // 900
    cf_h = np.full((n, cf15.shape[1]), np.nan, np.float32); cf_h[hidx[ph == 0]] = cf15[ph == 0]
    tr, te = split_masks(T0 + np.arange(n) * 3600)
    era, eidx, w100 = load_era5()
    raw, issue, lead = load_pangu()
    slat, slon, _ = stations(); sr, sc = frac_index(slat, slon)
    S = len(slat)
    ids = patch_ids(sr, sc)
    # ERA5 hourly speed fields
    E100 = speed(np.asarray(w100[:, 0]), np.asarray(w100[:, 1])).reshape(n, -1)
    # Pangu operational fields per hour
    vh = T0 + np.arange(n) * 3600
    iss = ((vh - 15 * 3600) // 86400) * 86400; ld = (vh - iss) // 3600
    key = {(int(i), int(l)): k for k, (i, l) in enumerate(zip(issue, lead))}
    fr = np.array([key.get((int(i), int(l)), -1) for i, l in zip(iss, ld)])
    ok = fr >= 0
    u, inv = np.unique(fr[ok], return_inverse=True)
    PA = np.asarray(raw[u])[:, [CI['u925'], CI['v925'], CI['u10'], CI['v10']]][inv]
    P925 = np.full((n, 600), np.nan, np.float32); P10 = np.full((n, 600), np.nan, np.float32)
    P925[ok] = speed(PA[:, 0], PA[:, 1]).reshape(-1, 600); P10[ok] = speed(PA[:, 2], PA[:, 3]).reshape(-1, 600)
    del PA
    far = np.array([np.argmax(np.hypot(slat - slat[s], slon - slon[s])) for s in range(S)])
    res = {}
    def fit(X, y):
        a = tr & np.isfinite(X).all(1) & np.isfinite(y); b = te & np.isfinite(X).all(1) & np.isfinite(y)
        m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, l2_regularization=1.0, random_state=0).fit(X[a], y[a])
        e = m.predict(X[b]) - y[b]
        return float(np.sqrt(np.mean(e ** 2))), float(1 - np.mean(e ** 2) / np.var(y[b]))
    for src in ['E100', 'P925+P10']:
        rows = {}
        for s in range(S):
            y = cf_h[:, s]
            R = np.broadcast_to(sr[s], (n, 1)); C = np.broadcast_to(sc[s], (n, 1))
            if src == 'E100':
                F = [E100]
            else:
                F = [P925, P10]
            point = [bilinear(f.reshape(n, 24, 25), R, C)[0][:, 0] for f in F]
            sets = dict(point=np.stack(point, 1),
                        patch5=np.concatenate([np.stack(point, 1)] + [f[:, ids[s]] for f in F], 1),
                        patch5_WRONGFARM=np.concatenate([np.stack(point, 1)] + [f[:, ids[far[s]]] for f in F], 1))
            for k, X in sets.items():
                rows.setdefault(k, []).append(fit(X, y))
        res[src] = {k: dict(rmse=float(np.mean([r[0] for r in v])), r2=float(np.mean([r[1] for r in v])),
                            r2_per_farm=[r[1] for r in v]) for k, v in rows.items()}
        save_json('Q4_spatial_ceiling.json', res)
        print(src, {k: (round(v['rmse'], 4), round(v['r2'], 4)) for k, v in res[src].items()}, flush=True)
