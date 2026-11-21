"""Q8b follow-up to cldas_check C2/C4: WHY do CLDAS neighbourhood features help?
(a) ablation of the spatial features: point, +3x3 mean, +9x9 mean, +17x17 mean, +gradient only, +9x9+gradient,
    each with a wrong-farm control for the added features (same-time field at the farthest farm).
(b) same ablation with the same-product split (2022 RE01 only, odd months fit / even months evaluate).
(c) which time scales the gain lives in: band correlation (>24h, 6-24h, 1-6h, <1h) of hourly GBDT predictions
    (point vs point+9x9) linearly interpolated to 15 min; and ERA5 100 m vs ERA5 100 m + CLDAS 9x9.
All GBDT include hour-of-day. Fit 2022, evaluate 2023 excl. Nov (unless stated)."""
import numpy as np, datetime as dt
from sklearn.ensemble import HistGradientBoostingRegressor
from common import *
from power_tests import load_power, split_masks, station_weather
from fluctuation_info import bands

D = 0.0625


def gb(): return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, random_state=0)


def fitpred(X, y, tr, te):
    a = tr & np.isfinite(X).all(1) & np.isfinite(y)
    p = np.full(len(y), np.nan); ok = np.isfinite(X).all(1)
    p[ok] = gb().fit(X[a], y[a]).predict(X[ok])
    b = te & ok & np.isfinite(y)
    return 1 - np.mean((p[b] - y[b]) ** 2) / np.var(y[b]), p


if __name__ == '__main__':
    C = np.load(OUT / 'cldas_crop.npy', mmap_mode='r')
    clat = np.load(OUT / 'cldas_lat.npy'); clon = np.load(OUT / 'cldas_lon.npy')
    n = (T1 - T0) // 3600; hours = T0 + np.arange(n) * 3600
    WIN = np.asarray(C[:, 0])
    slat, slon, _ = stations(); S = len(slat)
    r = (slat - clat[0]) / D; c = (slon - clon[0]) / D
    ri = np.round(r).astype(int); ci = np.round(c).astype(int)
    R = np.broadcast_to(r, (n, S)); Cc = np.broadcast_to(c, R.shape)
    PT = bilinear(WIN, R, Cc)[0]
    def pm(k): return np.stack([np.nanmean(WIN[:, max(ri[s] - k, 0):ri[s] + k + 1, max(ci[s] - k, 0):ci[s] + k + 1], (1, 2)) for s in range(S)], 1)
    P1, P4, P8 = pm(1), pm(4), pm(8)
    gy = np.full_like(WIN, np.nan); gx = np.full_like(WIN, np.nan)
    gy[:, 1:-1] = (WIN[:, 2:] - WIN[:, :-2]) / (2 * D * 111.2)
    gx[:, :, 1:-1] = (WIN[:, :, 2:] - WIN[:, :, :-2]) / (2 * D * 111.2 * np.cos(np.deg2rad(44.5)))
    GX, GY = bilinear(gx, R, Cc)[0], bilinear(gy, R, Cc)[0]; del gx, gy
    t15, cf15 = load_power()
    hidx = (t15 - T0) // 3600; ph = ((t15 - T0) % 3600) // 900
    cf_h = np.full((n, S), np.nan, np.float32); cf_h[hidx[ph == 0]] = cf15[ph == 0]
    tr, te = split_masks(hours)
    odd = np.array([dt.datetime.utcfromtimestamp(int(x) + 8 * 3600).month % 2 == 1 for x in hours])
    Wx = station_weather(); E100 = Wx['E_spd100']
    hod = (hours // 3600) % 24
    cyc = [np.sin(2 * np.pi * hod / 24), np.cos(2 * np.pi * hod / 24)]
    far = np.array([np.argmax(np.hypot(slat - slat[s], slon - slon[s])) for s in range(S)])
    res = {'ablation_2022to2023': {}, 'ablation_RE01only': {}}
    keep = {}
    for s in range(S):
        y = cf_h[:, s]; f = far[s]
        base = [PT[:, s]] + cyc
        sets = {'point': base,
                'point+3x3': base + [P1[:, s]], 'point+9x9': base + [P4[:, s]], 'point+17x17': base + [P8[:, s]],
                'point+grad': base + [GX[:, s], GY[:, s]], 'point+9x9+grad': base + [P4[:, s], GX[:, s], GY[:, s]],
                'point+9x9_WRONGFARM': base + [P4[:, f]], 'point+grad_WRONGFARM': base + [GX[:, f], GY[:, f]],
                '9x9_only': [P4[:, s]] + cyc,
                'era5_100m': [E100[:, s]] + cyc, 'era5_100m+cldas9x9': [E100[:, s], P4[:, s]] + cyc,
                'era5_100m+cldas9x9_WRONGFARM': [E100[:, s], P4[:, f]] + cyc}
        for name, cols in sets.items():
            X = np.stack(cols, 1)
            v, p = fitpred(X, y, tr, te)
            res['ablation_2022to2023'].setdefault(name, []).append(v)
            if name in ('point', 'point+9x9', 'era5_100m', 'era5_100m+cldas9x9'):
                keep.setdefault(name, np.full((n, S), np.nan))[:, s] = p
            v2, _ = fitpred(X, y, tr & odd, tr & ~odd)
            res['ablation_RE01only'].setdefault(name, []).append(v2)
        if s % 10 == 0: print(s, flush=True)
    for k in ['ablation_2022to2023', 'ablation_RE01only']:
        res[k] = {name: float(np.mean(v)) for name, v in res[k].items()}
    save_json('Q8b_cldas_followup.json', res); print(res, flush=True)
    k15 = (t15 - T0) // 900; h0 = k15 // 4; fr = (k15 % 4) / 4.0; nx = np.minimum(h0 + 1, n - 1)
    te15 = te[h0]; obsB = bands(cf15.astype(float))
    res['bands_corr'] = {}
    for name, P in keep.items():
        B = bands((1 - fr)[:, None] * P[h0] + fr[:, None] * P[nx])
        for b in obsB:
            o = obsB[b][te15]; p = B[b][te15]; rr = []
            for s in range(S):
                ok = np.isfinite(o[:, s]) & np.isfinite(p[:, s])
                if ok.sum() > 500: rr.append(np.corrcoef(o[ok, s], p[ok, s])[0, 1])
            res['bands_corr'].setdefault(b, {})[name] = float(np.mean(rr))
    save_json('Q8b_cldas_followup.json', res); print(res['bands_corr'], flush=True)
