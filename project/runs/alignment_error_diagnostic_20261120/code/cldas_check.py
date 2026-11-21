"""Q8: is CLDAS-2.0 (CMA land-surface analysis, 0.0625 deg, hourly; station-assimilated, not a model forecast)
informative for farm power, and does it carry spatial / temporal gradient information beyond ERA5?
C0 time convention (SWDN diurnal peak + lagged corr of 10 m speed with ERA5)
C1 point agreement with ERA5 10 m; hourly-change structure (3-h periodicity, zero changes, lag-1 autocorr of changes)
C2 power information (hourly, isotonic and GBDT): CLDAS native point, CLDAS coarsened to 0.25 deg, ERA5 10/100 m,
   CLDAS multivariable, CLDAS + ERA5 100 m; spatial features (local gradient, patch means) with wrong-farm control
C3 spatial structure: sub-0.25 deg variance share and its static (terrain-locked) fraction; gradient energy vs ERA5
C4 temporal: corr of 1/3/6 h changes with power changes; band decomposition (as Q6)
C5 forecasting roles on the 24 h manuscript protocol: (a) Pangu->CLDAS weather-side target vs Pangu->ERA5 100 m;
   (b) CLDAS at origin as extra observation input.
Fit 2022, evaluate 2023 excl. Nov."""
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.ensemble import HistGradientBoostingRegressor
from common import *
from power_tests import load_power, split_masks, station_weather
from fluctuation_info import bands

D = 0.0625
VN = ['WIN', 'TMP', 'PRS', 'RHU', 'SRA', 'PRE']


def gb(it=300): return HistGradientBoostingRegressor(max_iter=it, learning_rate=0.05, max_leaf_nodes=31, random_state=0)


def iso_r2(x, y, tr, te):
    a = tr & np.isfinite(x) & np.isfinite(y); b = te & np.isfinite(x) & np.isfinite(y)
    g = IsotonicRegression(out_of_bounds='clip').fit(x[a], y[a])
    e = g.predict(x[b]) - y[b]
    return 1 - np.mean(e ** 2) / np.var(y[b]), g


def gb_r2(X, y, tr, te):
    a = tr & np.isfinite(X).all(1) & np.isfinite(y); b = te & np.isfinite(X).all(1) & np.isfinite(y)
    p = gb().fit(X[a], y[a]).predict(X[b]); e = p - y[b]
    return 1 - np.mean(e ** 2) / np.var(y[b])


def corr(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    return float(np.corrcoef(a[m], b[m])[0, 1]) if m.sum() > 50 else np.nan


if __name__ == '__main__':
    import sys, json
    C5ONLY = 'c5only' in sys.argv     # reuse saved C2-C4, recompute C0/C1 (cheap) and C5
    res = json.loads((OUT / 'Q8_cldas.json').read_text()) if C5ONLY else {}
    C = np.load(OUT / 'cldas_crop.npy', mmap_mode='r')
    clat = np.load(OUT / 'cldas_lat.npy'); clon = np.load(OUT / 'cldas_lon.npy')
    n = (T1 - T0) // 3600
    hours = T0 + np.arange(n) * 3600
    # ---------- C0 time convention ----------
    sra = np.asarray(C[:24 * 365, 4, 54, 57])
    diurnal = np.nanmean(sra.reshape(365, 24), 0)
    peak = int(np.nanargmax(diurnal))
    shift = 0 if peak <= 7 else 8          # 125E solar noon ~04 UTC; ~12 if file names are BJT
    res['C0'] = dict(sra_mean_by_filehour=diurnal.tolist(), sra_peak_filehour=peak, assumed_shift_h=shift)
    slat, slon, _ = stations(); S = len(slat)
    r = (slat - clat[0]) / D; c = (slon - clon[0]) / D
    print('loading CLDAS', flush=True)
    Cv = {k: np.asarray(C[:, i]) for i, k in enumerate(VN[:5])}
    if shift:   # aligned index: UTC hour h -> file hour h + shift
        for k in Cv:
            Cv[k] = np.concatenate([Cv[k][shift:], np.full((shift,) + Cv[k].shape[1:], np.nan, np.float32)])
    WIN = Cv['WIN']
    res['C0']['win_missing_hours'] = int(np.isnan(WIN[:, 54, 57]).sum())

    def samp(F, rr=r, cc=c):
        R = np.broadcast_to(rr, (F.shape[0], len(rr))); Cc = np.broadcast_to(cc, R.shape)
        return bilinear(F, R, Cc)[0]
    CS = {k: samp(Cv[k]) for k in Cv}                     # station series [n,S]
    Hc, Wc = WIN.shape[1] // 4, WIN.shape[2] // 4         # coarse (0.25 deg block mean) CLDAS wind
    WINc = np.nanmean(WIN[:, :Hc * 4, :Wc * 4].reshape(n, Hc, 4, Wc, 4), (2, 4))
    CSc = samp(WINc, (slat - clat[0] - 1.5 * D) / 0.25, (slon - clon[0] - 1.5 * D) / 0.25)
    t15, cf15 = load_power()
    hidx = (t15 - T0) // 3600; ph = ((t15 - T0) % 3600) // 900
    cf_h = np.full((n, S), np.nan, np.float32); cf_h[hidx[ph == 0]] = cf15[ph == 0]
    tr, te = split_masks(hours)
    Wx = station_weather()
    E10, E100 = Wx['E_spd10'], Wx['E_spd100']
    lags = {}
    for L in range(-9, 10):   # corr(CLDAS[t+L], ERA5[t])
        a = CS['WIN'][max(L, 0):n + min(L, 0)]; b = E10[max(-L, 0):n - max(L, 0)]
        lags[L] = float(np.nanmean([corr(a[:, s], b[:, s]) for s in range(S)]))
    res['C0']['corr_vs_era5_10m_by_lag_h'] = lags
    print('C0', peak, shift, max(lags, key=lags.get), flush=True)
    # ---------- C1 ----------
    m = te[:, None] & np.isfinite(CS['WIN']) & np.isfinite(E10)
    d = CS['WIN'] - E10
    res['C1'] = dict(corr_vs_era5_10m=float(np.mean([corr(CS['WIN'][m[:, s], s], E10[m[:, s], s]) for s in range(S)])),
                     rmse_vs_era5_10m=float(np.sqrt(np.nanmean(d[m] ** 2))), bias=float(np.nanmean(d[m])),
                     std_cldas=float(np.nanmean(np.nanstd(np.where(m, CS['WIN'], np.nan), 0))),
                     std_era5=float(np.nanmean(np.nanstd(np.where(m, E10, np.nan), 0))))
    dC = np.diff(CS['WIN'], axis=0); dE = np.diff(E10, axis=0)
    hod1 = (hours[1:] // 3600) % 24
    res['C1']['change_std_ratio_cldas_over_era5'] = float(np.nanstd(dC[te[1:]]) / np.nanstd(dE[te[1:]]))
    res['C1']['change_std_by_utc_hour_mod3'] = {k: float(np.nanstd(dC[te[1:] & (hod1 % 3 == k)])) for k in range(3)}
    res['C1']['change_std_by_utc_hour'] = [float(np.nanstd(dC[te[1:] & (hod1 == k)])) for k in range(24)]
    res['C1']['frac_zero_change'] = float(np.nanmean(np.abs(dC[te[1:]]) < 1e-4))
    res['C1']['lag1_autocorr_of_changes'] = dict(cldas=float(np.nanmean([corr(dC[:-1, s], dC[1:, s]) for s in range(S)])),
                                                 era5=float(np.nanmean([corr(dE[:-1, s], dE[1:, s]) for s in range(S)])))
    save_json('Q8_cldas.json', res); print('C1', res['C1'], flush=True)
    if not C5ONLY:
        # ---------- C2 power information ----------
        hod = (hours // 3600) % 24
        cyc = [np.sin(2 * np.pi * hod / 24), np.cos(2 * np.pi * hod / 24)]
        keys = ['iso_cldas_native', 'iso_cldas_coarse025', 'iso_cldas_best_cell_5x5', 'iso_era5_10m', 'iso_era5_100m',
                'gbdt_cldas_point', 'gbdt_cldas+grad+patch', 'gbdt_cldas+grad+patch_WRONGFARM', 'gbdt_cldas_all',
                'gbdt_era5_100m+10m', 'gbdt_era5_100m+cldas_all',
                'RE01only_iso_cldas_native', 'RE01only_iso_cldas_coarse025', 'RE01only_iso_era5_10m', 'RE01only_iso_era5_100m']
        out = {k: [] for k in keys}
        import datetime as dt
        odd = np.array([dt.datetime.utcfromtimestamp(int(x) + 8 * 3600).month % 2 == 1 for x in hours])
        # product-switch diagnostic: CLDAS-minus-ERA5 10 m bias/std by month
        dmon = np.array([dt.datetime.utcfromtimestamp(int(x) + 8 * 3600).strftime('%Y-%m') for x in hours])
        res['C1']['cldas_minus_era5_10m_by_month'] = {mo: [float(np.nanmean((CS['WIN'] - E10)[dmon == mo])),
                                                           float(np.nanmean([corr(CS['WIN'][dmon == mo, s], E10[dmon == mo, s]) for s in range(S)]))]
                                                      for mo in sorted(set(dmon))}
        gy = np.full_like(WIN, np.nan); gx = np.full_like(WIN, np.nan)
        gy[:, 1:-1] = (WIN[:, 2:] - WIN[:, :-2]) / (2 * D * 111.2)
        gx[:, :, 1:-1] = (WIN[:, :, 2:] - WIN[:, :, :-2]) / (2 * D * 111.2 * np.cos(np.deg2rad(44.5)))
        GX, GY = samp(gx), samp(gy); del gx, gy
        ri = np.round(r).astype(int); ci = np.round(c).astype(int)

        def patch_mean(k):
            return np.stack([np.nanmean(WIN[:, max(ri[s] - k, 0):ri[s] + k + 1, max(ci[s] - k, 0):ci[s] + k + 1], (1, 2)) for s in range(S)], 1)
        P1, P4 = patch_mean(1), patch_mean(4)
        far = np.array([np.argmax(np.hypot(slat - slat[s], slon - slon[s])) for s in range(S)])
        h1 = tr & (hours < 1656633600); h2 = tr & (hours >= 1656633600)
        for s in range(S):
            y = cf_h[:, s]
            out['iso_cldas_native'].append(iso_r2(CS['WIN'][:, s], y, tr, te)[0])
            out['iso_cldas_coarse025'].append(iso_r2(CSc[:, s], y, tr, te)[0])
            out['iso_era5_10m'].append(iso_r2(E10[:, s], y, tr, te)[0])
            out['iso_era5_100m'].append(iso_r2(E100[:, s], y, tr, te)[0])
            # same-product robustness (2022 RE01 only): odd months fit, even months evaluate
            out['RE01only_iso_cldas_native'].append(iso_r2(CS['WIN'][:, s], y, tr & odd, tr & ~odd)[0])
            out['RE01only_iso_cldas_coarse025'].append(iso_r2(CSc[:, s], y, tr & odd, tr & ~odd)[0])
            out['RE01only_iso_era5_10m'].append(iso_r2(E10[:, s], y, tr & odd, tr & ~odd)[0])
            out['RE01only_iso_era5_100m'].append(iso_r2(E100[:, s], y, tr & odd, tr & ~odd)[0])
            best, bv = None, -9
            for di in range(-2, 3):          # best single cell within +-2 cells, selected on 2022 H1 -> H2
                for dj in range(-2, 3):
                    x = WIN[:, np.clip(ri[s] + di, 0, WIN.shape[1] - 1), np.clip(ci[s] + dj, 0, WIN.shape[2] - 1)]
                    v = iso_r2(x, y, h1, h2)[0]
                    if v > bv: bv, best = v, x
            out['iso_cldas_best_cell_5x5'].append(iso_r2(best, y, tr, te)[0])
            call = np.stack([CS[k][:, s] for k in VN[:5]] + cyc, 1)
            out['gbdt_cldas_all'].append(gb_r2(call, y, tr, te))
            out['gbdt_era5_100m+10m'].append(gb_r2(np.stack([E100[:, s], E10[:, s]] + cyc, 1), y, tr, te))
            out['gbdt_era5_100m+cldas_all'].append(gb_r2(np.column_stack([E100[:, s], call]), y, tr, te))
            base = np.stack([CS['WIN'][:, s]] + cyc, 1)
            out['gbdt_cldas_point'].append(gb_r2(base, y, tr, te))
            out['gbdt_cldas+grad+patch'].append(gb_r2(np.column_stack([base, GX[:, s], GY[:, s], P1[:, s], P4[:, s]]), y, tr, te))
            f = far[s]
            out['gbdt_cldas+grad+patch_WRONGFARM'].append(gb_r2(np.column_stack([base, GX[:, f], GY[:, f], P1[:, f], P4[:, f]]), y, tr, te))
            if s % 10 == 0: print('C2', s, flush=True)
        res['C2'] = {k: float(np.mean(v)) for k, v in out.items()}
        res['C2_per_farm'] = {k: [float(x) for x in v] for k, v in out.items()}
        save_json('Q8_cldas.json', res); print(res['C2'], flush=True)
        # ---------- C3 spatial structure ----------
        ev = np.flatnonzero(te & ~np.isnan(WIN[:, 54, 57]))[::3]
        Wf = WIN[ev][:, :Hc * 4, :Wc * 4]
        sub = Wf - np.repeat(np.repeat(WINc[ev], 4, 1), 4, 2)
        tot_var = float(np.nanmean(np.nanvar(Wf.reshape(len(ev), -1), 1)))
        sub_var = float(np.nanmean(np.nanvar(sub.reshape(len(ev), -1), 1)))
        mean_pat = np.nanmean(sub, 0); static_var = float(np.nanvar(mean_pat))
        res['C3'] = dict(spatial_var_total=tot_var, subgrid_var=sub_var, subgrid_share=sub_var / tot_var,
                         subgrid_static_share=static_var / sub_var,
                         subgrid_dynamic_rms_ms=float(np.sqrt(np.nanmean((sub - mean_pat) ** 2))),
                         subgrid_static_std_ms=float(np.sqrt(static_var)))
        era, eidx, w100 = load_era5()
        hh = ev[eidx[ev] >= 0][::2]
        e = np.asarray(era[eidx[hh]]); ES = speed(e[:, CI['u10']], e[:, CI['v10']])

        def genergy(F, dlat):
            gyy = np.diff(F, axis=1) / (dlat * 111.2); gxx = np.diff(F, axis=2) / (dlat * 111.2 * np.cos(np.deg2rad(44.5)))
            return float(np.nanmean(gyy ** 2) + np.nanmean(gxx ** 2))
        res['C3']['grad_energy_era5_10m_025'] = genergy(ES, 0.25)
        res['C3']['grad_energy_cldas_coarse_025'] = genergy(WINc[hh], 0.25)
        res['C3']['grad_energy_cldas_native_00625'] = genergy(WIN[hh], D)
        an = CS['WIN'] - CSc
        res['C3']['farm_subgrid_anomaly'] = dict(mean_abs_time_mean=float(np.nanmean(np.abs(np.nanmean(an[te], 0)))),
                                                 mean_time_std=float(np.nanmean(np.nanstd(an[te], 0))))
        save_json('Q8_cldas.json', res); print(res['C3'], flush=True)
        # ---------- C4 temporal ----------
        blk = np.full((n * 4, S), np.nan); k15 = (t15 - T0) // 900; blk[k15] = cf15
        hm = np.nanmean(blk.reshape(n, 4, S), 1); hm[np.isfinite(blk.reshape(n, 4, S)).sum(1) < 4] = np.nan
        preds = {}
        for name, X in [('CLDAS', CS['WIN']), ('ERA5_10m', E10), ('ERA5_100m', E100)]:
            P = np.full((n, S), np.nan)
            for s in range(S):
                _, g = iso_r2(X[:, s], cf_h[:, s], tr, te); ok = np.isfinite(X[:, s]); P[ok, s] = g.predict(X[ok, s])
            preds[name] = P
        res['C4'] = {'change_corr': {}, 'bands': {}}
        for Dh in [1, 3, 6]:
            dO = hm[Dh:] - hm[:-Dh]; mk = te[:-Dh] & te[Dh:]
            row = {}
            for name, P in preds.items():
                dP = P[Dh:] - P[:-Dh]
                row[name] = float(np.nanmean([corr(dO[mk, s], dP[mk, s]) for s in range(S)]))
            gain = []
            for s in range(S):
                Xa = (preds['ERA5_100m'][Dh:] - preds['ERA5_100m'][:-Dh])[:, s]; Xb = (preds['CLDAS'][Dh:] - preds['CLDAS'][:-Dh])[:, s]
                yy = dO[:, s]; ok = mk & np.isfinite(Xa) & np.isfinite(Xb) & np.isfinite(yy)
                A1 = np.column_stack([Xa[ok], np.ones(ok.sum())]); A2 = np.column_stack([Xa[ok], Xb[ok], np.ones(ok.sum())])
                e1 = yy[ok] - A1 @ np.linalg.lstsq(A1, yy[ok], rcond=None)[0]; e2 = yy[ok] - A2 @ np.linalg.lstsq(A2, yy[ok], rcond=None)[0]
                gain.append(1 - np.var(e2) / np.var(e1))
            row['partial_R2_of_CLDAS_change_beyond_ERA5_100m'] = float(np.mean(gain))
            res['C4']['change_corr'][f'{Dh}h'] = row
        h0 = k15 // 4; fr = (k15 % 4) / 4.0; nx = np.minimum(h0 + 1, n - 1)
        te15 = te[h0]; obsB = bands(cf15.astype(float))
        for name, P in preds.items():
            B = bands((1 - fr)[:, None] * P[h0] + fr[:, None] * P[nx])
            for b in obsB:
                o = obsB[b][te15]; p = B[b][te15]
                rr = []
                for s in range(S):
                    ok = np.isfinite(o[:, s]) & np.isfinite(p[:, s])
                    if ok.sum() > 500: rr.append(np.corrcoef(o[ok, s], p[ok, s])[0, 1])
                res['C4']['bands'].setdefault(b, {})[name] = float(np.mean(rr))
        save_json('Q8_cldas.json', res); print(res['C4'], flush=True)
    hod = (hours // 3600) % 24
    cyc = [np.sin(2 * np.pi * hod / 24), np.cos(2 * np.pi * hod / 24)]
    # ---------- C5 forecasting roles, 24 h manuscript protocol ----------
    PST, frame_of = Wx['PST'], Wx['frame_of']
    feats = ['spd10', 'spd1000', 'spd925', 'u10', 'v10', 'u925', 'v925', 't2m', 't925', 'msl']
    org = np.flatnonzero(hod == 15); org = org[(org + 24 < n) & (org >= 6)]
    O = np.repeat(org, 24); Hh = np.tile(np.arange(1, 25), len(org)); T = O + Hh
    iss = ((hours[O] - 15 * 3600) // 86400) * 86400
    f_tgt = frame_of(iss, (hours[T] - iss) // 3600)
    trm = tr[O] & tr[T]; tem = te[O] & te[T]
    bucket = np.digitize(Hh, [4, 7, 13]); names = ['1-3h', '4-6h', '7-12h', '13-24h']
    regC = np.nanmean(WIN, (1, 2))
    rows = {}
    for s in range(S):
        y = cf_h[T, s]
        Xh = np.stack([Wx['P_' + f][:, s] for f in feats] + cyc, 1); fx = np.isfinite(Xh).all(1)
        Xt = np.stack([np.where(f_tgt >= 0, PST[f][np.maximum(f_tgt, 0), s], np.nan) for f in feats]
                      + [np.sin(2 * np.pi * hod[T] / 24), np.cos(2 * np.pi * hod[T] / 24)], 1)
        okx = np.isfinite(Xt).all(1)
        wp = {}
        for tag, u in [('E100', E100[:, s]), ('CLDAS', CS['WIN'][:, s])]:
            a = tr & np.isfinite(u) & np.isfinite(cf_h[:, s])
            g = IsotonicRegression(out_of_bounds='clip').fit(u[a], cf_h[a, s])
            a1 = tr & fx & np.isfinite(u)
            m1 = gb().fit(Xh[a1], u[a1])
            v = np.full(len(T), np.nan); v[okx] = g.predict(m1.predict(Xt[okx])); wp[tag] = v
        c0 = cf_h[O, s]; c1 = cf_h[O - 1, s]; c6 = np.nanmean(np.stack([cf_h[O - k, s] for k in range(6)], 1), 1)
        cw = CS['WIN'][:, s]
        own = [c0, c0 - c1, c6]; cls = [cw[O], cw[O] - cw[O - 1], cw[O] - cw[O - 6], regC[O], regC[O] - regC[O - 3]]
        sets = {'W_E100': [Hh, wp['E100']], 'W_CLDAS': [Hh, wp['CLDAS']], 'W_both': [Hh, wp['E100'], wp['CLDAS']],
                'OWN': [Hh] + own, 'OWN+CLDAS_t0': [Hh] + own + cls,
                'W+OWN': [Hh, wp['E100']] + own, 'W+OWN+CLDAS_t0': [Hh, wp['E100']] + own + cls}
        for name, cols in sets.items():
            X = np.stack(cols, 1)
            a = trm & np.isfinite(X).all(1) & np.isfinite(y); b = tem & np.isfinite(X).all(1) & np.isfinite(y)
            e = gb().fit(X[a], y[a]).predict(X[b]) - y[b]
            for k, nm in enumerate(names):
                mk = bucket[b] == k
                rows.setdefault(name, {}).setdefault(nm, []).append(np.sqrt(np.mean(e[mk] ** 2)))
        if s % 10 == 0: print('C5', s, flush=True)
    res['C5_rmse_24h_protocol'] = {k: {nm: float(np.mean(v)) for nm, v in d.items()} for k, d in rows.items()}
    save_json('Q8_cldas.json', res)
    for k, d in res['C5_rmse_24h_protocol'].items(): print(f'{k:16s}', '  '.join(f'{nm}:{d[nm]:.4f}' for nm in names))
