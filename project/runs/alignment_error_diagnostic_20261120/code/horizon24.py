"""Q7: manuscript protocol (la/ieee_bigdia_zh_en problem_zh.tex): origin t0 = D-1 23:00 BJT (15 UTC), Pangu issued
D-1 08:00 BJT (00 UTC), targets D 00:00..23:00 BJT = leads 16..39 h. Which information matters by horizon?
Per farm GBDT pooled over horizons (horizon as feature); fit 2022 days, evaluate 2023 days (excl. Nov).
Sets: CLIM (horizon + target hour + month), PERSIST (y = cf(t0)), OWN (cf(t0), 1h change, 6h mean),
      W (decoupled Pangu predictor of Q5 at target), W+OWN, W+OWN+NBR, W_ERA5 (perfect-forecast g(ERA5 100m))."""
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.ensemble import HistGradientBoostingRegressor
from common import *
from power_tests import load_power, split_masks, station_weather


def gb(): return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, random_state=0)


if __name__ == '__main__':
    t15, cf15 = load_power()
    n = (T1 - T0) // 3600
    hidx = (t15 - T0) // 3600; ph = ((t15 - T0) % 3600) // 900
    S = cf15.shape[1]
    cf_h = np.full((n, S), np.nan, np.float32); cf_h[hidx[ph == 0]] = cf15[ph == 0]
    hours = T0 + np.arange(n) * 3600
    tr, te = split_masks(hours)
    W = station_weather()
    PST, frame_of = W['PST'], W['frame_of']
    feats = ['spd10', 'spd1000', 'spd925', 'u10', 'v10', 'u925', 'v925', 't2m', 't925', 'msl']
    hod = (hours // 3600) % 24
    slat, slon, _ = stations()
    d = np.hypot((slat[:, None] - slat[None]) * 111, (slon[:, None] - slon[None]) * 111 * np.cos(np.deg2rad(44.5)))
    nbr = np.argsort(d, 1)[:, 1:9]
    # origins: 15 UTC each day
    org = np.flatnonzero(hod == 15)
    org = org[org + 24 < n]
    H = np.arange(1, 25)
    O = np.repeat(org, 24); Hh = np.tile(H, len(org)); T = O + Hh
    iss = ((hours[O] - 15 * 3600) // 86400) * 86400          # = 00 UTC same day
    f_tgt = frame_of(iss, (hours[T] - iss) // 3600)            # leads 16..39
    trm = tr[O] & tr[T]; tem = te[O] & te[T]
    bucket = np.digitize(Hh, [4, 7, 13])  # 1-3,4-6,7-12,13-24
    names = ['1-3h', '4-6h', '7-12h', '13-24h']
    rows = {}
    for s in range(S):
        y = cf_h[T, s]
        # decoupled Pangu predictor trained on hourly 2022 weather (no power) + ERA5 power curve (2022 power)
        u = W['E_spd100'][:, s]
        a = tr & np.isfinite(u) & np.isfinite(cf_h[:, s])
        g = IsotonicRegression(out_of_bounds='clip').fit(u[a], cf_h[a, s])
        Xh = np.stack([W['P_' + f][:, s] for f in feats] + [np.sin(2 * np.pi * hod / 24), np.cos(2 * np.pi * hod / 24)], 1)
        fx = np.isfinite(Xh).all(1); a1 = tr & fx & np.isfinite(u)
        m1 = gb().fit(Xh[a1], u[a1])
        Xt = np.stack([np.where(f_tgt >= 0, PST[f][np.maximum(f_tgt, 0), s], np.nan) for f in feats]
                      + [np.sin(2 * np.pi * hod[T] / 24), np.cos(2 * np.pi * hod[T] / 24)], 1)
        okx = np.isfinite(Xt).all(1)
        wp = np.full(len(T), np.nan); wp[okx] = g.predict(m1.predict(Xt[okx]))
        we = np.where(np.isfinite(u[T]), g.predict(np.nan_to_num(u[T])), np.nan)
        c0 = cf_h[O, s]; c1 = cf_h[O - 1, s]; c6 = np.nanmean(np.stack([cf_h[O - k, s] for k in range(6)], 1), 1)
        nb0 = np.nanmean(cf_h[O][:, nbr[s]], 1)
        month = ((hours[T] + 8 * 3600) // 86400 % 365) / 365.0
        cal = [Hh, np.sin(2 * np.pi * hod[T] / 24), np.cos(2 * np.pi * hod[T] / 24), np.sin(2 * np.pi * month), np.cos(2 * np.pi * month)]
        own = [c0, c0 - c1, c6]
        sets = {'CLIM': cal[:3], 'OWN': [Hh] + own, 'W': [Hh, wp], 'W+OWN': [Hh, wp] + own, 'W+OWN+NBR': [Hh, wp] + own + [nb0],
                'W_ERA5': [Hh, we], 'W_ERA5+OWN': [Hh, we] + own}
        for name, cols in sets.items():
            X = np.stack(cols, 1)
            a = trm & np.isfinite(X).all(1) & np.isfinite(y); b = tem & np.isfinite(X).all(1) & np.isfinite(y)
            p = gb().fit(X[a], y[a]).predict(X[b])
            e = p - y[b]
            for k, nm in enumerate(names):
                mk = bucket[b] == k
                rows.setdefault((name, nm), []).append((np.sqrt(np.mean(e[mk] ** 2)), np.mean(np.abs(e[mk]))))
        b = tem & np.isfinite(c0) & np.isfinite(y)
        e = c0[b] - y[b]
        for k, nm in enumerate(names):
            mk = bucket[b] == k
            rows.setdefault(('PERSIST', nm), []).append((np.sqrt(np.mean(e[mk] ** 2)), np.mean(np.abs(e[mk]))))
    res = {}
    for (name, nm), v in rows.items():
        res.setdefault(name, {})[nm] = dict(rmse=float(np.mean([x[0] for x in v])), mae=float(np.mean([x[1] for x in v])))
    save_json('Q7_horizon24.json', res)
    for name in res:
        print(f'{name:12s}', '  '.join(f"{nm}:{res[name][nm]['rmse']:.4f}" for nm in names))
