"""Route-1 pre-registered check (doc SINGLE_FARM_CROSS_RESOLUTION_RESEARCH_20261120 sec.6/9):
advection-sampled vs linearly interpolated wind at 15-min points, stratified by spatial gradient at the farm and by
hourly power ramp. Best case uses ERA5 100 m with its own wind (perfect forecast); also Pangu 925 with 925 wind.
Metric: correlation of the intra-hour power residual with (g(U_adv)-g(U_lin)) and RMSE change. 2023 eval excl. Nov."""
import numpy as np
from sklearn.isotonic import IsotonicRegression
from common import *
from power_tests import load_power, split_masks
from g0_interp import interp

if __name__ == '__main__':
    t15, cf15 = load_power()
    n = (T1 - T0) // 3600
    hidx = (t15 - T0) // 3600; ph = ((t15 - T0) % 3600) // 900
    S = cf15.shape[1]
    cf_h = np.full((n, S), np.nan, np.float32); cf_h[hidx[ph == 0]] = cf15[ph == 0]
    tr, te = split_masks(T0 + np.arange(n) * 3600)
    era, eidx, w100 = load_era5()
    slat, slon, _ = stations(); sr, sc = frac_index(slat, slon)
    U = speed(np.asarray(w100[:, 0]), np.asarray(w100[:, 1]))  # [n,24,25]
    R = np.broadcast_to(sr, (n, S)); C = np.broadcast_to(sc, (n, S))
    Up = bilinear(U, R, C)[0]
    curves = []
    for s in range(S):
        a = tr & np.isfinite(Up[:, s]) & np.isfinite(cf_h[:, s])
        curves.append(IsotonicRegression(out_of_bounds='clip').fit(Up[a, s], cf_h[a, s]))
    g = lambda X: np.stack([curves[s].predict(np.nan_to_num(X[:, s])) for s in range(S)], 1)
    # spatial gradient magnitude at farm (m/s per km)
    gy = np.zeros_like(U); gx = np.zeros_like(U)
    gy[:, 1:-1] = (U[:, 2:] - U[:, :-2]) / (2 * DY_KM); gx[:, :, 1:-1] = (U[:, :, 2:] - U[:, :, :-2]) / (2 * dx_km(44.5))
    G = bilinear(np.sqrt(gx ** 2 + gy ** 2), R, C)[0]
    sel = (ph > 0) & (hidx + 1 < n)
    h0 = hidx[sel]; fr = ph[sel] / 4.0; y = cf15[sel]
    m = te[h0] & te[h0 + 1]
    h0, fr, y = h0[m], fr[m], y[m]
    F = fr[:, None] * np.ones((1, S))
    Rr = np.broadcast_to(sr, (len(h0), S)); Cc = np.broadcast_to(sc, Rr.shape)
    lin, adv, ok = interp(U[h0], U[h0 + 1], np.asarray(w100[h0, 0]), np.asarray(w100[h0 + 1, 0]),
                          np.asarray(w100[h0, 1]), np.asarray(w100[h0 + 1, 1]), Rr, Cc, F, 1.0)
    GL, GA = g(lin), g(adv)
    rP = y - ((1 - F) * cf_h[h0] + F * cf_h[h0 + 1])
    ramp = np.abs(cf_h[h0 + 1] - cf_h[h0])
    grad = 0.5 * (G[h0] + G[h0 + 1])
    base = np.isfinite(rP) & np.isfinite(GL) & np.isfinite(GA) & ok
    res = {}
    qg = np.nanpercentile(grad[base], [50, 90, 99])
    strata = {'all': base, 'grad>p50': base & (grad > qg[0]), 'grad>p90': base & (grad > qg[1]), 'grad>p99': base & (grad > qg[2]),
              'ramp>0.1': base & (ramp > 0.1), 'ramp>0.2': base & (ramp > 0.2), 'ramp>0.3': base & (ramp > 0.3),
              'grad>p90&ramp>0.2': base & (grad > qg[1]) & (ramp > 0.2)}
    for k, mm in strata.items():
        d = (GA - GL)[mm]
        res[k] = dict(n=int(mm.sum()), intra_resid_std=float(rP[mm].std()),
                      corr_rP_vs_adv_minus_lin=float(np.corrcoef(rP[mm], d)[0, 1]) if mm.sum() > 50 else None,
                      rmse_lin=float(np.sqrt(((GL - y)[mm]) ** 2).mean()) if False else float(np.sqrt((((GL - y)[mm]) ** 2).mean())),
                      rmse_adv=float(np.sqrt((((GA - y)[mm]) ** 2).mean())),
                      mean_abs_adv_minus_lin_cf=float(np.abs(d).mean()))
        res[k]['rmse_change_pct'] = 100 * (res[k]['rmse_adv'] / res[k]['rmse_lin'] - 1)
    res['gradient_percentiles_m_s_per_km'] = qg.tolist()
    # z1000 vs station elevation (is 1000 hPa below ground?)
    try:
        elev = np.load(BUNDLE / 'physical/elevation.npy')
        z = np.asarray(era[eidx[eidx >= 0][:2000]])[:, CI['z1000']] / 9.80665
        zf = bilinear(z, np.broadcast_to(sr, (len(z), S)), np.broadcast_to(sc, (len(z), S)))[0]
        res['z1000_vs_elevation'] = dict(station_elev_m=np.asarray(elev).ravel().tolist(),
                                         z1000_height_mean_m=zf.mean(0).tolist(),
                                         frac_time_1000hPa_below_ground=(zf < np.asarray(elev).ravel()[None, :S]).mean(0).tolist() if np.asarray(elev).size >= S else None)
    except Exception as e:
        res['z1000_vs_elevation'] = str(e)
    save_json('ROUTE1_stratified.json', res)
    for k, v in res.items():
        if isinstance(v, dict) and 'n' in v:
            print(k, v['n'], round(v['corr_rP_vs_adv_minus_lin'] or 0, 4), round(v['rmse_change_pct'], 3), round(v['mean_abs_adv_minus_lin_cf'], 4))
