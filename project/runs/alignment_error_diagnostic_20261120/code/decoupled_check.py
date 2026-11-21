"""Q5: decoupled 'weather-side alignment -> power conversion' vs direct regression.
(a) direct: GBDT power ~ Pangu features (fit 2022 power)
(b) decoupled: GBDT ERA5-100m-speed ~ Pangu features (weather-side, fit 2022), then ERA5-fitted power curve g (2022 power)
    point = g(E[u|f]) and, errors-in-variables, E[g(u)|f] via empirical residual distribution of stage 1
(c) power-data scarcity: same with only the first 3 months of 2022 power (weather side still full 2022).
Eval 2023 excl. Nov, hourly, Pangu daily-protocol frames."""
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.ensemble import HistGradientBoostingRegressor
from common import *
from power_tests import load_power, split_masks, station_weather

if __name__ == '__main__':
    t15, cf15 = load_power()
    n = (T1 - T0) // 3600
    hidx = (t15 - T0) // 3600; ph = ((t15 - T0) % 3600) // 900
    S = cf15.shape[1]
    cf_h = np.full((n, S), np.nan, np.float32); cf_h[hidx[ph == 0]] = cf15[ph == 0]
    tr, te = split_masks(T0 + np.arange(n) * 3600)
    hours = T0 + np.arange(n) * 3600
    tr3 = tr & (hours < 1648771200)  # 2022-04-01
    W = station_weather()
    feats = ['P_spd10', 'P_spd1000', 'P_spd925', 'P_u10', 'P_v10', 'P_u925', 'P_v925', 'P_t2m', 'P_t925', 'P_msl']
    hod = (hours // 3600) % 24
    res = {}
    def gb(): return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, random_state=0)
    out = {k: [] for k in ['direct_full', 'direct_3m', 'dec_point_full', 'dec_eiv_full', 'dec_point_3m', 'dec_eiv_3m', 'stage1_rmse_ms', 'pangu925_rmse_vs_e100']}
    for s in range(S):
        X = np.stack([W[f][:, s] for f in feats] + [np.sin(2 * np.pi * hod / 24), np.cos(2 * np.pi * hod / 24)], 1)
        y = cf_h[:, s]; u = W['E_spd100'][:, s]
        fx = np.isfinite(X).all(1)
        b = te & fx & np.isfinite(y)
        def r2(p): e = p - y[b]; return 1 - np.mean(e ** 2) / np.var(y[b])
        for tag, trm in [('full', tr), ('3m', tr3)]:
            a = trm & fx & np.isfinite(y)
            out[f'direct_{tag}'].append(r2(gb().fit(X[a], y[a]).predict(X[b])))
        # stage 1 weather side (full 2022, no power used)
        a1 = tr & fx & np.isfinite(u)
        m1 = gb().fit(X[a1], u[a1])
        uhat_tr = m1.predict(X[a1]); resid = u[a1] - uhat_tr
        uhat = m1.predict(X[b])
        out['stage1_rmse_ms'].append(float(np.sqrt(np.mean((uhat - u[b]) ** 2))))
        out['pangu925_rmse_vs_e100'].append(float(np.sqrt(np.mean((X[b, 2] - u[b]) ** 2))))
        # residual quantiles conditional on predicted-speed bin (heteroscedastic EIV)
        bins = np.quantile(uhat_tr, np.linspace(0, 1, 11)); bins[-1] += 1e-6
        bi_tr = np.clip(np.searchsorted(bins, uhat_tr) - 1, 0, 9); bi = np.clip(np.searchsorted(bins, uhat) - 1, 0, 9)
        qs = np.linspace(0.025, 0.975, 39)
        rq = np.stack([np.quantile(resid[bi_tr == k], qs) for k in range(10)])
        for tag, trm in [('full', tr), ('3m', tr3)]:
            a = trm & np.isfinite(u) & np.isfinite(y)
            g = IsotonicRegression(out_of_bounds='clip').fit(u[a], y[a])   # power conversion learned on ERA5 100 m
            out[f'dec_point_{tag}'].append(r2(g.predict(uhat)))
            samples = np.clip(uhat[:, None] + rq[bi], 0, None)
            out[f'dec_eiv_{tag}'].append(r2(g.predict(samples.ravel()).reshape(samples.shape).mean(1)))
    res = {k: dict(mean=float(np.mean(v)), per_farm=[float(x) for x in v]) for k, v in out.items()}
    save_json('Q5_decoupled.json', res)
    print({k: round(v['mean'], 4) for k, v in res.items()})
