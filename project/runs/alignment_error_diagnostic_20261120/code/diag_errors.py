"""D1-D4: how large are Pangu (daily 00 UTC, leads 9-45 h) errors relative to ERA5 on the Jilin crop?
D1 point error by lead and variable (Murphy decomposition); D2 smoothing (gradient energy, spectra, hourly tendency);
D3 spatial displacement error (in-sample and out-of-sample); D4 timing error at farms (in-sample and out-of-sample).
ERA5 is the reference analysis, not hub-height truth. 2022-2023 only."""
import numpy as np
from common import *

LEADS = np.arange(9, 46)


def build():
    raw, issue, lead = load_pangu()
    era, eidx, _ = load_era5()
    valid = issue + lead * 3600
    keep = (valid >= T0) & (valid < T1)
    hidx = np.where(keep, (valid - T0) // 3600, 0)
    keep &= eidx[hidx] >= 0
    issues = np.unique(issue[keep])
    grid = np.full((len(issues), len(LEADS)), -1, np.int64)
    ii = np.searchsorted(issues, issue[keep]); ll = lead[keep] - 9
    grid[ii, ll] = np.flatnonzero(keep)
    full = (grid >= 0).all(1)
    grid = grid[full]; issues = issues[full]
    sel = grid.ravel()
    chans = [CI[c] for c in ['u10', 'v10', 'u1000', 'v1000', 'u925', 'v925', 't2m', 'msl']]
    u, inv = np.unique(sel, return_inverse=True)
    P = np.asarray(raw[u])[:, chans][inv].reshape(len(issues), len(LEADS), len(chans), 24, 25)
    vh = (issues[:, None] + LEADS[None, :] * 3600 - T0) // 3600
    u, inv = np.unique(eidx[vh.ravel()], return_inverse=True)
    E = np.asarray(era[u])[:, chans][inv].reshape(P.shape)
    names = ['u10', 'v10', 'u1000', 'v1000', 'u925', 'v925', 't2m', 'msl']
    return P, E, issues, names


def fields(X, names):
    d = {n: X[:, :, k] for k, n in enumerate(names)}
    d['spd10'] = speed(d['u10'], d['v10']); d['spd1000'] = speed(d['u1000'], d['v1000']); d['spd925'] = speed(d['u925'], d['v925'])
    return d


def murphy(p, e, axis=0):
    b = (p - e).mean(axis); sp = p.std(axis); se = e.std(axis)
    r = ((p - p.mean(axis)) * (e - e.mean(axis))).mean(axis) / (sp * se)
    mse = ((p - e) ** 2).mean(axis)
    return dict(bias=b, rmse=np.sqrt(mse), mae=np.abs(p - e).mean(axis), corr=r, std_p=sp, std_e=se,
                mse=mse, bias2=b ** 2, amp2=(sp - se) ** 2, phase=2 * sp * se * (1 - r))


def d1(Pf, Ef, sr, sc):
    res = {}
    for v in ['spd10', 'spd1000', 'spd925', 'u10', 'v10', 'u925', 'v925', 't2m', 'msl']:
        p, _ = bilinear(Pf[v], np.broadcast_to(sr, Pf[v].shape[:2] + sr.shape), np.broadcast_to(sc, Pf[v].shape[:2] + sc.shape))
        e, _ = bilinear(Ef[v], np.broadcast_to(sr, Ef[v].shape[:2] + sr.shape), np.broadcast_to(sc, Ef[v].shape[:2] + sc.shape))
        # p,e: [issue, lead, station]
        m = murphy(p, e, axis=0)  # per lead, station
        out = {k: np.nanmean(x, axis=-1).tolist() for k, x in m.items()}  # station-mean per lead
        res[v] = out
    # wind direction error at 925 when both speeds > 3 m/s
    for lev in ['10', '925']:
        pu, _ = bilinear(Pf['u' + lev], *[np.broadcast_to(a, Pf['u' + lev].shape[:2] + a.shape) for a in (sr, sc)])
        pv, _ = bilinear(Pf['v' + lev], *[np.broadcast_to(a, Pf['v' + lev].shape[:2] + a.shape) for a in (sr, sc)])
        eu, _ = bilinear(Ef['u' + lev], *[np.broadcast_to(a, Ef['u' + lev].shape[:2] + a.shape) for a in (sr, sc)])
        ev, _ = bilinear(Ef['v' + lev], *[np.broadcast_to(a, Ef['v' + lev].shape[:2] + a.shape) for a in (sr, sc)])
        ang = np.rad2deg(np.angle(np.exp(1j * (np.arctan2(pv, pu) - np.arctan2(ev, eu)))))
        ok = (speed(pu, pv) > 3) & (speed(eu, ev) > 3)
        res['dir_abs_err_deg_' + lev] = [float(np.median(np.abs(ang[:, l][ok[:, l]]))) for l in range(len(LEADS))]
        res['vector_rmse_' + lev] = np.sqrt(((pu - eu) ** 2 + (pv - ev) ** 2).mean(axis=(0, 2))).tolist()
    return res


def grad_energy(F, lat):
    gy = (F[..., 2:, 1:-1] - F[..., :-2, 1:-1]) / (2 * DY_KM)
    dx = dx_km(lat[1:-1])[:, None]
    gx = (F[..., 1:-1, 2:] - F[..., 1:-1, :-2]) / (2 * dx)
    return (gx ** 2 + gy ** 2).mean(axis=(-1, -2))


def radial_spectrum(F):
    X = F[..., :24, :24]
    yy, xx = np.mgrid[0:24, 0:24]
    A = np.stack([np.ones(576), yy.ravel(), xx.ravel()], 1)
    flat = X.reshape(-1, 576)
    coef, *_ = np.linalg.lstsq(A, flat.T, rcond=None)
    X = (flat - (A @ coef).T).reshape(X.shape)
    w = np.hanning(24)
    S = np.abs(np.fft.fft2(X * w[:, None] * w[None, :])) ** 2
    ky = np.fft.fftfreq(24); kx = np.fft.fftfreq(24)
    k = np.sqrt(ky[:, None] ** 2 + kx[None, :] ** 2)  # cycles per grid step
    bins = np.array([0, 1 / 12, 1 / 6, 1 / 4, 1 / 3, 0.75])  # wavelength >12, 6-12, 4-6, 3-4, <3 cells
    out = []
    for a, b in zip(bins[:-1], bins[1:]):
        m = (k >= a) & (k < b) & (k > 0)
        out.append(S[..., m].mean(-1))
    return np.stack(out, -1), ['>12 cells', '6-12', '4-6', '3-4', '<3']


def d2(Pf, Ef, sr, sc):
    res = {}
    for v in ['spd10', 'spd925']:
        gp = grad_energy(Pf[v], LAT); ge = grad_energy(Ef[v], LAT)
        res[f'{v}_gradient_energy_ratio_by_lead'] = (gp.mean(0) / ge.mean(0)).tolist()
        sp, labels = radial_spectrum(Pf[v]); se, _ = radial_spectrum(Ef[v])
        res[f'{v}_spectrum_ratio_by_lead'] = (sp.mean(0) / se.mean(0)).tolist()
        res['spectrum_bins'] = labels
        p, _ = bilinear(Pf[v], *[np.broadcast_to(a, Pf[v].shape[:2] + a.shape) for a in (sr, sc)])
        e, _ = bilinear(Ef[v], *[np.broadcast_to(a, Ef[v].shape[:2] + a.shape) for a in (sr, sc)])
        dp = np.diff(p, axis=1); de = np.diff(e, axis=1)
        res[f'{v}_hourly_tendency_std_ratio_by_lead'] = (dp.std(axis=(0, 2)) / de.std(axis=(0, 2))).tolist()
        res[f'{v}_hourly_tendency_std_era5'] = float(de.std())
        res[f'{v}_hourly_tendency_std_pangu'] = float(dp.std())
    return res


def d3(Pf, Ef):
    res = {}
    S = 3
    r0, r1, c0, c1 = 3, 21, 3, 22
    shifts = [(dy, dx) for dy in range(-S, S + 1) for dx in range(-S, S + 1)]
    for v in ['spd10', 'spd925']:
        P, E = Pf[v], Ef[v]
        Et = E[..., r0:r1, c0:c1]
        mse = np.stack([((P[..., r0 + dy:r1 + dy, c0 + dx:c1 + dx] - Et) ** 2).mean((-1, -2)) for dy, dx in shifts], -1)
        z = shifts.index((0, 0))
        best = mse.argmin(-1)
        gain = 1 - mse.min(-1) / mse[..., z]
        sh = np.array(shifts)[best]  # [issue, lead, 2]
        km = np.sqrt((sh[..., 0] * DY_KM) ** 2 + (sh[..., 1] * dx_km(44.5)) ** 2)
        res[v] = dict(
            insample_mse_gain_by_lead=gain.mean(0).tolist(),
            insample_frac_zero_shift_by_lead=(best == z).mean(0).tolist(),
            insample_median_shift_km_by_lead=np.median(km, 0).tolist(),
            insample_mean_shift_km_by_lead=km.mean(0).tolist(),
            consecutive_lead_same_shift_frac=float((best[:, 1:] == best[:, :-1]).mean()),
        )
        # out-of-sample: choose per-issue shift on leads 15-26, apply to leads 27-38
        a = (LEADS >= 15) & (LEADS <= 26); b = (LEADS >= 27) & (LEADS <= 38)
        choose = mse[:, a].mean(1).argmin(-1)
        app = mse[:, b].mean(1)
        oos = 1 - app[np.arange(len(app)), choose] / app[:, z]
        res[v]['oos_issue_shift_mse_gain_mean'] = float(oos.mean())
        res[v]['oos_issue_shift_mse_gain_median'] = float(np.median(oos))
        res[v]['oos_frac_issues_improved'] = float((oos > 0).mean())
        res[v]['oos_frac_issues_zero_shift'] = float((choose == z).mean())
        # same-issue mean MSE by shift (systematic displacement?)
        res[v]['mean_mse_by_shift'] = {f'{dy},{dx}': float(mse[..., k].mean()) for k, (dy, dx) in enumerate(shifts)}
        res[v]['zero_shift_mse'] = float(mse[..., z].mean())
    return res


def d4(Pf, Ef, sr, sc):
    res = {}
    lags = np.arange(-3, 4)
    for v in ['spd10', 'spd925']:
        p, _ = bilinear(Pf[v], *[np.broadcast_to(a, Pf[v].shape[:2] + a.shape) for a in (sr, sc)])
        e, _ = bilinear(Ef[v], *[np.broadcast_to(a, Ef[v].shape[:2] + a.shape) for a in (sr, sc)])
        # p[:, l] vs e[:, l+L]: forecast feature at lead l verifies at l+L (positive L => forecast too early)
        base = np.arange(3, len(LEADS) - 3)  # leads 12..42
        err = np.stack([((p[:, base] - e[:, base + L]) ** 2) for L in lags], -1)  # [issue, t, station, lag]
        z = 3
        first = LEADS[base] <= 26; second = ~first
        ins = err.mean(1)  # [issue, station, lag]
        best = ins.argmin(-1)
        res[v] = dict(
            insample_lag_hist={int(L): float((best == k).mean()) for k, L in enumerate(lags)},
            insample_mse_gain_mean=float((1 - ins.min(-1) / ins[..., z]).mean()),
        )
        choose = err[:, first].mean(1).argmin(-1)
        app = err[:, second].mean(1)
        g = 1 - np.take_along_axis(app, choose[..., None], -1)[..., 0] / app[..., z]
        res[v]['oos_mse_gain_mean'] = float(g.mean())
        res[v]['oos_mse_gain_median'] = float(np.median(g))
        res[v]['oos_frac_improved'] = float((g > 0).mean())
        res[v]['oos_chosen_lag_hist'] = {int(L): float((choose == k).mean()) for k, L in enumerate(lags)}
        res[v]['mean_mse_by_lag'] = {int(L): float(err[..., k].mean()) for k, L in enumerate(lags)}
    return res


if __name__ == '__main__':
    P, E, issues, names = build()
    Pf = fields(P, names); Ef = fields(E, names)
    slat, slon, cap = stations()
    sr, sc = frac_index(slat, slon)
    out = dict(n_issues=int(len(issues)), leads=LEADS.tolist(), first_issue=int(issues[0]), last_issue=int(issues[-1]),
               note='ERA5 is a 0.25 deg reanalysis reference, not hub-height truth; Pangu was initialised from ERA5 (lead-0 identical).')
    out['D1_point_error'] = d1(Pf, Ef, sr, sc); save_json('D1_D4_errors.json', out)
    out['D2_smoothing'] = d2(Pf, Ef, sr, sc); save_json('D1_D4_errors.json', out)
    out['D3_spatial_shift'] = d3(Pf, Ef); save_json('D1_D4_errors.json', out)
    out['D4_timing'] = d4(Pf, Ef, sr, sc); save_json('D1_D4_errors.json', out)
    # ERA5 own variability references
    e10, _ = bilinear(Ef['spd10'], *[np.broadcast_to(a, Ef['spd10'].shape[:2] + a.shape) for a in (sr, sc)])
    e925, _ = bilinear(Ef['spd925'], *[np.broadcast_to(a, Ef['spd925'].shape[:2] + a.shape) for a in (sr, sc)])
    out['reference'] = dict(era5_spd10_mean=float(e10.mean()), era5_spd10_std=float(e10.std()),
                            era5_spd925_mean=float(e925.mean()), era5_spd925_std=float(e925.std()))
    save_json('D1_D4_errors.json', out)
    print('done')
