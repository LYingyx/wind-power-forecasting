"""Power-based zero-training tests (fit 2022, evaluate 2023 excluding November; no 2024).
Q1 variable relevance: which level/variables explain farm power (ERA5 = perfect-forecast upper bound; Pangu operational).
Q2 intra-hour signal: is the 15-min deviation of power from its hourly linear path explainable by hourly weather
    interpolated linearly or by advection?  (ERA5 best case and Pangu.)
Q3 history/neighbour power: does it add information beyond weather, and is that 'state' or simply more information?
    Controls: wrong-day neighbours (same clock/month), shuffled-station neighbours."""
import numpy as np, h5py
from sklearn.isotonic import IsotonicRegression
from sklearn.ensemble import HistGradientBoostingRegressor
from common import *

rng = np.random.default_rng(0)


def load_power():
    with h5py.File(POWER_H5, 'r') as f:
        t = f['power/timestamp_unix'][:]; mw = f['power/station_mw'][:]; cap = f['stations/capacity_mw'][:]
    m = (t >= T0) & (t < T1)
    cf = mw[m] / cap[None, :]
    cf[(cf < -0.05) | (cf > 1.2)] = np.nan
    return t[m], np.clip(cf, 0, 1.05)


def station_weather():
    """hourly station series [hours, station] for ERA5 and Pangu-operational (same issue rule as daily protocol)."""
    era, eidx, w100 = load_era5()
    raw, issue, lead = load_pangu()
    slat, slon, _ = stations()
    sr, sc = frac_index(slat, slon)
    n = len(eidx)
    ok = eidx >= 0
    E = np.full((n, 14), np.nan, np.float32)
    chans = list(range(14))
    out = {}
    def samp(F):
        R = np.broadcast_to(sr, (F.shape[0], len(sr))); C = np.broadcast_to(sc, R.shape)
        return bilinear(F, R, C)[0]
    hrs = np.flatnonzero(ok)
    e = np.asarray(era[eidx[hrs]])
    for name in ['u10', 'v10', 'u1000', 'v1000', 'u925', 'v925', 't2m', 't925', 'msl', 'q925']:
        x = np.full((n, len(sr)), np.nan, np.float32); x[hrs] = samp(e[:, CI[name]]); out['E_' + name] = x
    for k, name in enumerate(['u100', 'v100']):
        out['E_' + name] = samp(np.asarray(w100[:, k]))
    # Pangu operational: valid hour v uses issue floor((v-15h)/24h)*24h, lead 15..38
    vh = T0 + np.arange(n) * 3600
    iss = ((vh - 15 * 3600) // 86400) * 86400
    ld = (vh - iss) // 3600
    table = np.full((int((issue.max() - T0) // 86400) + 2, 46), -1, np.int64)
    okf = issue >= T0 - 86400
    table[(issue[okf] - (T0 - 86400)) // 86400, lead[okf]] = np.flatnonzero(okf)
    def frame_of(iss_unix, ld_h):
        d = (iss_unix - (T0 - 86400)) // 86400
        good = (d >= 0) & (d < len(table)) & (ld_h >= 0) & (ld_h <= 45)
        f = np.full(len(iss_unix), -1, np.int64); f[good] = table[d[good], ld_h[good]]
        return f
    fr = frame_of(iss, ld)
    allf = np.flatnonzero(okf)
    PA = np.asarray(raw[allf])
    PST = {}
    for name in ['u10', 'v10', 'u1000', 'v1000', 'u925', 'v925', 't2m', 't925', 'msl', 'q925']:
        x = np.full((len(issue), len(sr)), np.nan, np.float32); x[allf] = samp(PA[:, CI[name]]); PST[name] = x
    del PA
    for lev in ['10', '1000', '925']:
        PST['spd' + lev] = speed(PST['u' + lev], PST['v' + lev])
    good = fr >= 0
    for name in ['u10', 'v10', 'u1000', 'v1000', 'u925', 'v925', 't2m', 't925', 'msl', 'q925']:
        x = np.full((n, len(sr)), np.nan, np.float32); x[good] = PST[name][fr[good]]; out['P_' + name] = x
    out['PST'] = PST; out['frame_of'] = frame_of
    for src in ['E', 'P']:
        for lev in ['10', '1000', '925'] + (['100'] if src == 'E' else []):
            out[f'{src}_spd{lev}'] = speed(out[f'{src}_u{lev}'], out[f'{src}_v{lev}'])
    out['pangu_frame_of_hour'] = fr
    return out


def split_masks(hours_unix):
    import datetime as dt
    y = np.array([dt.datetime.utcfromtimestamp(int(x)).year for x in hours_unix])
    mo = np.array([dt.datetime.utcfromtimestamp(int(x) + 8 * 3600).month for x in hours_unix])
    return y == 2022, (y == 2023) & (mo != 11)


def rmse(a): return float(np.sqrt(np.nanmean(a ** 2)))


def q1(W, cf_h, tr, te):
    res = {}
    S = cf_h.shape[1]
    hours = T0 + np.arange(cf_h.shape[0]) * 3600
    hod = ((hours // 3600) % 24)[:, None] * np.ones((1, S)); doy = ((hours // 86400) % 365)[:, None] * np.ones((1, S))
    cyc = [np.sin(2 * np.pi * hod / 24), np.cos(2 * np.pi * hod / 24), np.sin(2 * np.pi * doy / 365), np.cos(2 * np.pi * doy / 365)]
    for src, levs in [('E', ['10', '100', '1000', '925']), ('P', ['10', '1000', '925'])]:
        for lev in levs:
            errs, r2 = [], []
            for s in range(S):
                x = W[f'{src}_spd{lev}'][:, s]; y = cf_h[:, s]
                a = tr & np.isfinite(x) & np.isfinite(y); b = te & np.isfinite(x) & np.isfinite(y)
                g = IsotonicRegression(out_of_bounds='clip').fit(x[a], y[a])
                e = g.predict(x[b]) - y[b]
                errs.append(rmse(e)); r2.append(1 - np.mean(e ** 2) / np.var(y[b]))
            res[f'{src}_iso_spd{lev}'] = dict(rmse_mean=float(np.mean(errs)), r2_mean=float(np.mean(r2)))
        # GBDT: speed-only (all levels) vs all variables
        base = [f'{src}_spd{l}' for l in levs]
        full = base + [f'{src}_{v}' for v in ['u10', 'v10', 'u925', 'v925', 't2m', 'msl', 'q925']] + ['stab']
        W[f'{src}_stab'] = W[f'{src}_t925'] - W[f'{src}_t2m']
        for tag, cols in [('speeds', base), ('speeds+dir+thermo+cycle', full)]:
            errs, r2 = [], []
            for s in range(S):
                X = np.stack([W[c if c != 'stab' else f'{src}_stab'][:, s] for c in cols] + ([c[:, s] for c in cyc] if 'cycle' in tag else []), 1)
                y = cf_h[:, s]
                a = tr & np.isfinite(X).all(1) & np.isfinite(y); b = te & np.isfinite(X).all(1) & np.isfinite(y)
                m = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, max_leaf_nodes=31, random_state=0).fit(X[a], y[a])
                e = m.predict(X[b]) - y[b]
                errs.append(rmse(e)); r2.append(1 - np.mean(e ** 2) / np.var(y[b]))
            res[f'{src}_gbdt_{tag}'] = dict(rmse_mean=float(np.mean(errs)), r2_mean=float(np.mean(r2)))
        save_json('Q1_variables.json', res)
    return res


def q2(W, t15, cf15, tr_h, te_h):
    """intra-hour tests at :15/:30/:45."""
    slat, slon, _ = stations()
    sr, sc = frac_index(slat, slon)
    era, eidx, w100 = load_era5()
    S = cf15.shape[1]
    hidx = (t15 - T0) // 3600; ph = ((t15 - T0) % 3600) // 900
    res = {}
    cf_h = np.full((len(eidx), S), np.nan, np.float32)
    on = ph == 0; cf_h[hidx[on]] = cf15[on]
    for src, lev in [('E', '100'), ('E', '10'), ('P', '1000'), ('P', '10')]:
        U = W[f'{src}_spd{lev}']
        curves = []
        for s in range(S):
            a = tr_h & np.isfinite(U[:, s]) & np.isfinite(cf_h[:, s])
            curves.append(IsotonicRegression(out_of_bounds='clip').fit(U[a, s], cf_h[a, s]))
        sel = (ph > 0) & (hidx + 1 < len(eidx))
        h0 = hidx[sel]; frac = ph[sel] / 4.0
        evalm = te_h[h0] & te_h[h0 + 1]
        h0 = h0[evalm]; frac = frac[evalm]; y = cf15[sel][evalm]
        U0 = U[h0]; U1 = U[h0 + 1]
        lin = (1 - frac[:, None]) * U0 + frac[:, None] * U1
        # advection interpolation needs fields; ERA5 100m uses own wind; others approximate with own-level fields
        if src == 'E' and lev == '100':
            F0 = speed(w100[h0, 0], w100[h0, 1]); F1 = speed(w100[h0 + 1, 0], w100[h0 + 1, 1])
            from g0_interp import interp
            R = np.broadcast_to(sr, (len(h0), S)); C = np.broadcast_to(sc, R.shape)
            fs = np.broadcast_to(frac[:, None], R.shape)
            _, adv, _ = interp(F0, F1, w100[h0, 0], w100[h0 + 1, 0], w100[h0, 1], w100[h0 + 1, 1], R, C, fs, 1.0)
        else:
            adv = None
        P0 = cf_h[h0]; P1 = cf_h[h0 + 1]
        rP = y - ((1 - frac[:, None]) * P0 + frac[:, None] * P1)
        out = {}
        g = lambda X: np.stack([curves[s].predict(np.nan_to_num(X[:, s])) for s in range(S)], 1)
        G0, G1, GL = g(U0), g(U1), g(lin)
        lerpG = (1 - frac[:, None]) * G0 + frac[:, None] * G1
        ok = np.isfinite(rP) & np.isfinite(U0) & np.isfinite(U1)
        out['intra_hour_power_residual_var_share'] = float(np.nanvar(rP[ok]) / np.nanvar(y[ok]))
        out['intra_hour_power_residual_std'] = float(np.nanstd(rP[ok]))
        out['corr_rP_vs_curve_nonlinearity(g(lerpU)-lerp g(U))'] = float(np.corrcoef(rP[ok], (GL - lerpG)[ok])[0, 1])
        out['rmse_cf_linear_weather'] = rmse((GL - y)[ok])
        if adv is not None:
            GA = g(adv); ok2 = ok & np.isfinite(adv)
            out['corr_rP_vs_advection_signal(g(Uadv)-lerp g(U))'] = float(np.corrcoef(rP[ok2], (GA - lerpG)[ok2])[0, 1])
            out['corr_rP_vs_advection_minus_linear(g(Uadv)-g(Ulin))'] = float(np.corrcoef(rP[ok2], (GA - GL)[ok2])[0, 1])
            out['rmse_cf_advect_weather'] = rmse((GA - y)[ok2]); out['rmse_cf_linear_weather_same_mask'] = rmse((GL - y)[ok2])
        for p in [1, 2, 3]:
            mm = ok & (ph[sel][evalm][:, None] == p)
            out[f'rmse_cf_linear_weather_phase{15 * p}'] = rmse((GL - y)[mm])
        hm = np.isfinite(cf_h) & np.isfinite(U) & te_h[:, None]
        out['rmse_cf_on_the_hour'] = rmse((g(U) - cf_h)[hm])
        res[f'{src}_spd{lev}'] = out
        save_json('Q2_intrahour.json', res)
    return res


def q3(W, cf15, t15, tr_h, te_h):
    """history/neighbour information beyond Pangu weather at hourly origins, leads 1/3/6 h."""
    S = cf15.shape[1]
    hidx = (t15 - T0) // 3600; ph = ((t15 - T0) % 3600) // 900
    n = (T1 - T0) // 3600
    cf_h = np.full((n, S), np.nan, np.float32); cf_h[hidx[ph == 0]] = cf15[ph == 0]
    cf_m1 = np.full((n, S), np.nan, np.float32); cf_m1[1:] = cf_h[:-1]
    slat, slon, _ = stations()
    d = np.sqrt(((slat[:, None] - slat[None]) * 111) ** 2 + ((slon[:, None] - slon[None]) * 111 * np.cos(np.deg2rad(44.5))) ** 2)
    nbr = np.argsort(d, 1)[:, 1:9]
    res = {}
    # wrong-day: same clock hour, other day in same month window (shift 7..21 days, fixed per origin)
    shift = 24 * rng.integers(7, 22, n)
    wrong = np.clip(np.arange(n) - shift, 0, n - 1)
    perm = rng.permutation(S)
    PST, frame_of = W['PST'], W['frame_of']
    for tau in [1, 3, 6]:
        rows = {}
        o = np.arange(n - tau); tgt = o + tau
        ot = T0 + o * 3600
        iss_o = ((ot - 15 * 3600) // 86400) * 86400          # issue available at the origin (daily protocol)
        f_tgt = frame_of(iss_o, (ot + tau * 3600 - iss_o) // 3600)
        f_org = frame_of(iss_o, (ot - iss_o) // 3600)
        def pv(name, f, s):
            x = np.full(len(f), np.nan, np.float32); g = f >= 0; x[g] = PST[name][f[g], s]; return x
        for s in range(S):
            y = cf_h[tgt, s]
            pw = [pv('spd1000', f_tgt, s), pv('spd925', f_tgt, s), pv('spd10', f_tgt, s), pv('u925', f_tgt, s), pv('v925', f_tgt, s),
                  pv('spd1000', f_org, s), pv('spd925', f_org, s)]
            own = [cf_h[o, s], cf_h[o, s] - cf_m1[o, s]]
            nb = [np.nanmean(cf_h[o][:, nbr[s]], 1), np.nanmean(cf_h[o][:, nbr[s]] - cf_m1[o][:, nbr[s]], 1)]
            nb_wrong = [np.nanmean(cf_h[wrong[o]][:, nbr[s]], 1), np.nanmean(cf_h[wrong[o]][:, nbr[s]] - cf_m1[wrong[o]][:, nbr[s]], 1)]
            nb_perm = [np.nanmean(cf_h[o][:, nbr[perm[s]]], 1), np.nanmean(cf_h[o][:, nbr[perm[s]]] - cf_m1[o][:, nbr[perm[s]]], 1)]
            sets = dict(W=pw, W_own=pw + own, W_own_nbr=pw + own + nb, W_own_nbrWRONGDAY=pw + own + nb_wrong,
                        W_own_nbrOTHERSTATION=pw + own + nb_perm, own_only=own, own_nbr=own + nb)
            trm = tr_h[o] & np.isfinite(y); tem = te_h[o] & te_h[tgt] & np.isfinite(y)
            for name, cols in sets.items():
                X = np.stack(cols, 1)
                a = trm & np.isfinite(X).all(1); b = tem & np.isfinite(X).all(1)
                m = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, max_leaf_nodes=31, random_state=0).fit(X[a], y[a])
                e = m.predict(X[b]) - y[b]
                hi = cf_h[o][b, s] >= 0.75
                rows.setdefault(name, []).append((rmse(e), float(np.mean(np.abs(e))), rmse(e[hi]) if hi.sum() > 30 else np.nan))
        res[f'lead{tau}h'] = {k: dict(rmse=float(np.mean([r[0] for r in v])), mae=float(np.mean([r[1] for r in v])),
                                     high_power_rmse=float(np.nanmean([r[2] for r in v]))) for k, v in rows.items()}
        save_json('Q3_history_neighbors.json', res)
        print('q3 lead', tau, flush=True)
    return res


if __name__ == '__main__':
    import sys
    t15, cf15 = load_power()
    W = station_weather()
    n = (T1 - T0) // 3600
    hours = T0 + np.arange(n) * 3600
    tr_h, te_h = split_masks(hours)
    hidx = (t15 - T0) // 3600; ph = ((t15 - T0) % 3600) // 900
    cf_h = np.full((n, cf15.shape[1]), np.nan, np.float32); cf_h[hidx[ph == 0]] = cf15[ph == 0]
    which = sys.argv[1:] or ['q1', 'q2', 'q3']
    if 'q1' in which: q1(W, cf_h, tr_h, te_h); print('q1 done', flush=True)
    if 'q2' in which: q2(W, t15, cf15, tr_h, te_h); print('q2 done', flush=True)
    if 'q3' in which: q3(W, cf15, t15, tr_h, te_h); print('q3 done', flush=True)
