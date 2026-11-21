"""Q9: treating CLDAS-2.0 (0.0625 deg hourly analysis) as PERFECT future weather (oracle, also at test time),
can it supply information to recover intra-hour (15-min) power detail?  Cross spatio-temporal alignment view:
hourly fine-grid fields -> 15-min farm power, via (Taylor) space-to-time conversion along the flow.
Steering wind = ERA5 100 m (u,v) at the farm (CLDAS has no direction).

T1  field level (no power): does CLDAS sub-0.25deg structure persist 1 h and move with the flow?
    (a) corr of dynamic sub-grid anomaly between t and t+1 at zero shift vs advected by alpha*V*1h;
    (b) drop-middle-hour reconstruction (gap 2 h): linear vs semi-Lagrangian advection, alpha in {0.25,0.5,1}.
T2  intra-hour residual with ORACLE hourly power anchors: rP = p(h0+s) - [(1-s)P(h0)+sP(h0+1)], s=1/4,2/4,3/4.
    Models (pooled GBDT over farms): power-only (anchors h0-1..h0+2) vs + CLDAS intra-hour features
    (advected/upstream/downstream samples, CLDAS temporal curvature, patch std, precip, radiation) vs
    + same CLDAS features from 7 days later (control) vs + ERA5-100m analogues (0.25 deg).
T3  intra-hour fluctuation intensity log(mean rP^2 over the hour): power level only vs + CLDAS vs control.
T4  end-to-end, CLDAS only (no power): hourly GBDT + linear interpolation vs 15-min GBDT with intra-hour features.
Fit 2022, evaluate 2023 excl. Nov; T2 headline also within-2022 (odd/even months, single product)."""
import numpy as np, datetime as dt
from sklearn.isotonic import IsotonicRegression
from sklearn.ensemble import HistGradientBoostingRegressor
from common import *
from power_tests import load_power, split_masks
from fluctuation_info import bands

D = 0.0625
rng = np.random.default_rng(0)


def gb(it=300): return HistGradientBoostingRegressor(max_iter=it, learning_rate=0.05, max_leaf_nodes=63, random_state=0)


def corr(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    return float(np.corrcoef(a[m], b[m])[0, 1]) if m.sum() > 50 else np.nan


class Grid:
    """flat bilinear sampler for a [n, H, W] field on a regular lat/lon grid."""
    def __init__(self, F, lat0, dlat, lon0, dlon):
        self.F = np.ascontiguousarray(F, np.float32).ravel(); self.n, self.H, self.W = F.shape
        self.lat0, self.dlat, self.lon0, self.dlon = lat0, dlat, lon0, dlon

    def __call__(self, h, lat, lon):
        h = np.clip(np.asarray(h), 0, self.n - 1)
        r = np.clip((lat - self.lat0) / self.dlat, 0, self.H - 1 - 1e-6); c = np.clip((lon - self.lon0) / self.dlon, 0, self.W - 1 - 1e-6)
        r0 = np.floor(r).astype(np.int64); c0 = np.floor(c).astype(np.int64); fr = r - r0; fc = c - c0
        i = h * (self.H * self.W) + r0 * self.W + c0
        F = self.F
        return F[i] * (1 - fr) * (1 - fc) + F[i + self.W] * fr * (1 - fc) + F[i + 1] * (1 - fr) * fc + F[i + self.W + 1] * fr * fc


def disp(u, v, tau_h, lat):
    """lat/lon displacement (deg) travelled in tau_h hours with wind (u,v) m/s."""
    km = 3.6 * tau_h
    return v * km / 111.2, u * km / (111.2 * np.cos(np.deg2rad(lat)))


if __name__ == '__main__':
    res = {}
    C = np.load(OUT / 'cldas_crop.npy', mmap_mode='r')
    clat = np.load(OUT / 'cldas_lat.npy').astype(float); clon = np.load(OUT / 'cldas_lon.npy').astype(float)
    n = (T1 - T0) // 3600; hours = T0 + np.arange(n) * 3600
    WIN = np.asarray(C[:, 0]); PRE = np.asarray(C[:, 5]); SRA = np.asarray(C[:, 4])
    GW = Grid(WIN, clat[0], D, clon[0], D)
    era, eidx, w100 = load_era5()
    W100 = np.asarray(w100)                                     # [n,2,24,25], LAT descending
    GU = Grid(W100[:, 0], LAT[0], -DLAT, LON[0], DLAT); GV = Grid(W100[:, 1], LAT[0], -DLAT, LON[0], DLAT)
    GE = Grid(speed(W100[:, 0], W100[:, 1]), LAT[0], -DLAT, LON[0], DLAT)
    slat, slon, _ = stations(); S = len(slat)
    tr, te = split_masks(hours)
    odd = np.array([dt.datetime.utcfromtimestamp(int(x) + 8 * 3600).month % 2 == 1 for x in hours])
    print('loaded', flush=True)

    # ================= T1 field level =================
    Hc, Wc = WIN.shape[1] // 4, WIN.shape[2] // 4
    hs = np.flatnonzero(te[:-2] & te[1:-1] & te[2:] & np.isfinite(WIN[:-2, 54, 57]) & np.isfinite(WIN[1:-1, 54, 57]) & np.isfinite(WIN[2:, 54, 57]))[::7]
    def sub_anom(F):
        Fc = F[:, :Hc * 4, :Wc * 4]
        return Fc - np.repeat(np.repeat(np.nanmean(Fc.reshape(len(F), Hc, 4, Wc, 4), (2, 4)), 4, 1), 4, 2)
    A0 = sub_anom(WIN[hs]); A1 = sub_anom(WIN[hs + 1]); A2 = sub_anom(WIN[hs + 2])
    static = np.nanmean(sub_anom(WIN[np.flatnonzero(tr)[::7]]), 0)
    A0 -= static; A1 -= static; A2 -= static
    m = 16   # interior margin (cells) for displacement
    gl, gk = np.meshgrid(clat[:Hc * 4][m:-m], clon[:Wc * 4][m:-m], indexing='ij')
    P = gl.size; LA = np.broadcast_to(gl.ravel(), (len(hs), P)); LO = np.broadcast_to(gk.ravel(), (len(hs), P))
    hh = np.broadcast_to(hs[:, None], (len(hs), P))
    u = 0.5 * (GU(hh, LA, LO) + GU(hh + 1, LA, LO)); v = 0.5 * (GV(hh, LA, LO) + GV(hh + 1, LA, LO))
    GA0 = Grid(A0, clat[0], D, clon[0], D); idx = np.arange(len(hs))[:, None] * np.ones((1, P), int)
    t1 = {'n_frames': int(len(hs)), 'dyn_subgrid_rms_ms': float(np.sqrt(np.nanmean(A1 ** 2)))}
    a1 = A1[:, m:-m, m:-m].reshape(len(hs), -1)
    t1['corr_dyn_subgrid_t_t+1'] = {'zero_shift': float(np.nanmean([corr(A0[i, m:-m, m:-m].ravel(), a1[i]) for i in range(len(hs))]))}
    for al in [0.25, 0.5, 1.0]:
        dla, dlo = disp(u, v, al, LA)
        x = GA0(idx, LA - dla, LO - dlo)
        t1['corr_dyn_subgrid_t_t+1'][f'advected_alpha{al}'] = float(np.nanmean([corr(x[i], a1[i]) for i in range(len(hs))]))
    t1['corr_dyn_subgrid_t_t+2_zero_shift'] = float(np.nanmean([corr(A0[i, m:-m, m:-m].ravel(), A2[i, m:-m, m:-m].ravel()) for i in range(len(hs))]))
    # drop-middle-hour reconstruction on the full field (not anomaly)
    truth = GW(hh + 1, LA, LO); lin = 0.5 * (GW(hh, LA, LO) + GW(hh + 2, LA, LO))
    u2 = GU(hh + 1, LA, LO); v2 = GV(hh + 1, LA, LO)
    rec = {'linear': float(np.sqrt(np.nanmean((lin - truth) ** 2))), 'persistence_from_t': float(np.sqrt(np.nanmean((GW(hh, LA, LO) - truth) ** 2)))}
    spd = np.hypot(u2, v2)
    for al in [0.25, 0.5, 1.0]:
        dla, dlo = disp(u2, v2, al, LA)
        adv = 0.5 * (GW(hh, LA - dla, LO - dlo) + GW(hh + 2, LA + dla, LO + dlo))
        rec[f'advect_alpha{al}'] = float(np.sqrt(np.nanmean((adv - truth) ** 2)))
        fast = spd > 10
        rec[f'advect_alpha{al}_fast>10'] = float(np.sqrt(np.nanmean((adv - truth)[fast] ** 2)))
    rec['linear_fast>10'] = float(np.sqrt(np.nanmean((lin - truth)[spd > 10] ** 2)))
    t1['gap2h_reconstruction_rmse_ms'] = rec
    res['T1'] = t1; save_json('Q9_cldas_intrahour.json', res); print('T1', t1, flush=True)
    del A0, A1, A2, LA, LO, hh, u, v, truth, lin, u2, v2, spd

    # ================= power & per-farm curves =================
    t15, cf15 = load_power()
    k15 = (t15 - T0) // 900
    Q = np.full((n * 4, S), np.nan); Q[k15] = cf15
    Ph = Q[0::4]                                                  # oracle hourly anchors (:00 values)
    LAs = np.broadcast_to(slat, (n, S)); LOs = np.broadcast_to(slon, (n, S)); HH = np.broadcast_to(np.arange(n)[:, None], (n, S))
    pt = GW(HH, LAs, LOs)
    ri = np.round((slat - clat[0]) / D).astype(int); ci = np.round((slon - clon[0]) / D).astype(int)
    P9 = np.stack([np.nanmean(WIN[:, ri[s] - 4:ri[s] + 5, ci[s] - 4:ci[s] + 5], (1, 2)) for s in range(S)], 1)
    SD9 = np.stack([np.nanstd(WIN[:, ri[s] - 4:ri[s] + 5, ci[s] - 4:ci[s] + 5], (1, 2)) for s in range(S)], 1)
    prf = GW.__class__(PRE, clat[0], D, clon[0], D)(HH, LAs, LOs); srf = Grid(SRA, clat[0], D, clon[0], D)(HH, LAs, LOs)
    uf = GU(HH, LAs, LOs); vf = GV(HH, LAs, LOs); ef = GE(HH, LAs, LOs)
    curves = []
    for s in range(S):
        a = tr & np.isfinite(pt[:, s]) & np.isfinite(Ph[:, s])
        curves.append(IsotonicRegression(out_of_bounds='clip').fit(pt[a, s], Ph[a, s]))
    curvesE = []
    for s in range(S):
        a = tr & np.isfinite(ef[:, s]) & np.isfinite(Ph[:, s])
        curvesE.append(IsotonicRegression(out_of_bounds='clip').fit(ef[a, s], Ph[a, s]))
    def G(X, cv):   # X [N,S] speed -> CF via per-farm curve
        out = np.full(X.shape, np.nan)
        for s in range(S):
            ok = np.isfinite(X[:, s]); out[ok, s] = cv[s].predict(X[ok, s])
        return out

    # ================= intra-hour samples =================
    h0 = np.repeat(np.arange(1, n - 2), 3); sfr = np.tile(np.array([0.25, 0.5, 0.75]), n - 3)
    N = len(h0)
    H0 = np.broadcast_to(h0[:, None], (N, S)); SF = np.broadcast_to(sfr[:, None], (N, S))
    LAn = np.broadcast_to(slat, (N, S)); LOn = np.broadcast_to(slon, (N, S))
    y = Q[4 * h0 + (sfr * 4).astype(int)]
    lin = (1 - SF) * Ph[h0] + SF * Ph[h0 + 1]
    rP = y - lin
    pm1, p0, p1, p2 = Ph[h0 - 1], Ph[h0], Ph[h0 + 1], Ph[h0 + 2]
    def catmull(a, b, c_, d, t):
        return 0.5 * (2 * b + (-a + c_) * t + (2 * a - 5 * b + 4 * c_ - d) * t ** 2 + (-a + 3 * b - 3 * c_ + d) * t ** 3)
    cr = catmull(pm1, p0, p1, p2, SF) - lin
    trS = tr[h0 - 1] & tr[h0 + 2]; teS = te[h0 - 1] & te[h0 + 2]

    def cldas_feats(off):
        """CLDAS intra-hour features for sample hours h0+off (off=0 real, off=168 wrong-week control)."""
        a = np.clip(h0 + off, 1, n - 3); A = np.broadcast_to(a[:, None], (N, S))
        u = (1 - SF) * uf[a] + SF * uf[a + 1]; v = (1 - SF) * vf[a] + SF * vf[a + 1]
        f0 = GW(A, LAn, LOn); f1 = GW(A + 1, LAn, LOn)
        gl0, gl1 = G(f0, curves), G(f1, curves)
        ling = (1 - SF) * gl0 + SF * gl1
        out = {'lin_g': ling, 'spd0': f0, 'spd1': f1, 'dspd': f1 - f0}
        cm1 = G(GW(A - 1, LAn, LOn), curves); c2 = G(GW(A + 2, LAn, LOn), curves)
        out['cldas_cubic_minus_lin'] = catmull(cm1, gl0, gl1, c2, SF) - ling
        for al in [0.5, 1.0, 2.0]:
            dla0, dlo0 = disp(u, v, al * SF, LAn); dla1, dlo1 = disp(u, v, al * (1 - SF), LAn)
            up = G(GW(A, LAn - dla0, LOn - dlo0), curves); dn = G(GW(A + 1, LAn + dla1, LOn + dlo1), curves)
            out[f'adv{al}_minus_lin'] = (1 - SF) * up + SF * dn - ling
            out[f'up{al}_minus_pt0'] = up - gl0; out[f'dn{al}_minus_pt1'] = dn - gl1
        line = np.stack([GW(A, LAn - disp(u, v, t, LAn)[0], LOn - disp(u, v, t, LAn)[1]) for t in [0, 0.25, 0.5, 0.75, 1.0]], -1)
        out['upstream_line_std'] = np.nanstd(line, -1)
        out['upstream_line_std_cf'] = np.nanstd(G(line.reshape(N, -1), curves) if False else np.stack([G(line[..., k], curves) for k in range(5)], -1), -1)
        out['patch9_std'] = SD9[a]; out['patch9_mean0'] = P9[a]; out['patch9_d'] = P9[a + 1] - P9[a]
        out['prcp'] = prf[a]; out['swdn'] = srf[a]; out['wspd_e100'] = np.hypot(u, v)
        return out

    def era_feats():
        a = h0; A = H0
        u = (1 - SF) * uf[a] + SF * uf[a + 1]; v = (1 - SF) * vf[a] + SF * vf[a + 1]
        g0_, g1_ = G(GE(A, LAn, LOn), curvesE), G(GE(A + 1, LAn, LOn), curvesE)
        ling = (1 - SF) * g0_ + SF * g1_
        out = {'e_lin_g': ling, 'e_cubic_minus_lin': catmull(G(GE(A - 1, LAn, LOn), curvesE), g0_, g1_, G(GE(A + 2, LAn, LOn), curvesE), SF) - ling}
        for al in [0.5, 1.0, 2.0]:
            dla0, dlo0 = disp(u, v, al * SF, LAn); dla1, dlo1 = disp(u, v, al * (1 - SF), LAn)
            out[f'e_adv{al}_minus_lin'] = (1 - SF) * G(GE(A, LAn - dla0, LOn - dlo0), curvesE) + SF * G(GE(A + 1, LAn + dla1, LOn + dlo1), curvesE) - ling
        return out

    print('building features', flush=True)
    base = {'s': SF, 'p_m1': pm1, 'p0': p0, 'p1': p1, 'p2': p2, 'cr_minus_lin': cr, 'hod': np.broadcast_to(((hours[h0] // 3600) % 24)[:, None], (N, S))}
    FC = cldas_feats(0); FW = cldas_feats(168); FE = era_feats()
    res['T2_univariate_corr_with_rP_2023'] = {k: corr(v[teS], rP[teS]) for k, v in {**FC, **FE, 'cr_minus_lin': cr}.items()}
    save_json('Q9_cldas_intrahour.json', res); print(res['T2_univariate_corr_with_rP_2023'], flush=True)

    def stack(dicts):
        cols = [v for d in dicts for v in d.values()]
        return np.stack([np.asarray(c, np.float32).ravel() for c in cols], 1)
    yv = rP.ravel(); fin = np.isfinite(yv)
    TR = np.broadcast_to(trS[:, None], (N, S)).ravel() & fin; TE = np.broadcast_to(teS[:, None], (N, S)).ravel() & fin
    ODD = np.broadcast_to((odd[h0])[:, None], (N, S)).ravel(); TRo = TR & ODD; TEo = np.broadcast_to(trS[:, None], (N, S)).ravel() & fin & ~ODD
    ramp = np.broadcast_to(np.abs(p1 - p0), (N, S)).ravel() if False else np.abs(p1 - p0).ravel()

    def fit_eval(X, trm, tem, cap=900000):
        a = np.flatnonzero(trm); a = rng.choice(a, min(cap, len(a)), replace=False)
        mdl = gb().fit(X[a], yv[a]); p = mdl.predict(X[tem]); return p

    sets = {'power_only': [base], 'power+CLDAS': [base, FC], 'power+CLDAS_wrongweek': [base, FW], 'power+ERA5_100m': [base, FE],
            'power+CLDAS+ERA5': [base, FC, FE]}
    t2 = {'rP_var_share_of_15min_var_2023': float(np.nanvar(yv[TE]) / np.nanvar(y.ravel()[TE])),
          'rmse_linear_2023': float(np.sqrt(np.mean(yv[TE] ** 2))),
          'catmull_rom_from_power': dict(R2=float(1 - np.mean((yv[TE] - cr.ravel()[TE]) ** 2) / np.mean(yv[TE] ** 2)))}
    rmask = ramp[TE] > 0.2
    for name, ds in sets.items():
        X = stack(ds); p = fit_eval(X, TR, TE); e = yv[TE] - p
        t2[name] = dict(R2_vs_linear=float(1 - np.mean(e ** 2) / np.mean(yv[TE] ** 2)), rmse_15min=float(np.sqrt(np.mean(e ** 2))),
                        R2_on_ramp_hours=float(1 - np.mean(e[rmask] ** 2) / np.mean(yv[TE][rmask] ** 2)))
        if name in ('power_only', 'power+CLDAS'):
            po = fit_eval(X, TRo, TEo); eo = yv[TEo] - po
            t2[name]['R2_within2022_RE01'] = float(1 - np.mean(eo ** 2) / np.mean(yv[TEo] ** 2))
        print('T2', name, t2[name], flush=True)
        res['T2'] = t2; save_json('Q9_cldas_intrahour.json', res)

    # ================= T3 intensity =================
    hrs = np.arange(1, n - 2)
    R2h = rP.reshape(n - 3, 3, S)
    z = np.log(np.nanmean(R2h ** 2, 1) + 1e-4); z[np.isnan(R2h).any(1)] = np.nan
    def hourly(d): return {k: np.asarray(v).reshape(n - 3, 3, S)[:, 1] for k, v in d.items()}
    lvl = {'p0': Ph[hrs], 'p1': Ph[hrs + 1], 'dp': np.abs(Ph[hrs + 1] - Ph[hrs]), 'pm1': Ph[hrs - 1], 'p2': Ph[hrs + 2],
           'slope': np.abs(Ph[hrs + 2] - Ph[hrs - 1])}
    keepC = ['upstream_line_std', 'upstream_line_std_cf', 'patch9_std', 'prcp', 'swdn', 'wspd_e100', 'dspd', 'spd0', 'patch9_mean0', 'lin_g',
             'adv1.0_minus_lin', 'cldas_cubic_minus_lin']
    hc = hourly({k: FC[k] for k in keepC}); hw = hourly({k: FW[k] for k in keepC})
    zz = z.ravel(); fz = np.isfinite(zz)
    trh = np.broadcast_to((tr[hrs - 1] & tr[hrs + 2])[:, None], (n - 3, S)).ravel() & fz
    teh = np.broadcast_to((te[hrs - 1] & te[hrs + 2])[:, None], (n - 3, S)).ravel() & fz
    t3 = {}
    for name, ds in {'level_only': [lvl], 'level+CLDAS': [lvl, hc], 'level+CLDAS_wrongweek': [lvl, hw], 'CLDAS_only': [hc]}.items():
        X = stack(ds)
        a = np.flatnonzero(trh); a = rng.choice(a, min(600000, len(a)), replace=False)
        p = gb().fit(X[a], zz[a]).predict(X[teh])
        t3[name] = float(1 - np.mean((p - zz[teh]) ** 2) / np.var(zz[teh]))
    res['T3_intensity_R2'] = t3; save_json('Q9_cldas_intrahour.json', res); print('T3', t3, flush=True)

    # ================= T4 end-to-end, CLDAS only =================
    # hourly model on :00 targets, then linear interpolation; vs 15-min model with intra-hour features
    gpt = G(pt, curves)
    Xh = np.stack([gpt.ravel(), pt.ravel(), P9.ravel(), np.broadcast_to(((hours // 3600) % 24)[:, None], (n, S)).ravel(),
                   np.hypot(uf, vf).ravel()], 1).astype(np.float32)
    yh = Ph.ravel(); okh = np.isfinite(yh) & np.isfinite(Xh).all(1)
    trH = np.broadcast_to(tr[:, None], (n, S)).ravel() & okh
    a = rng.choice(np.flatnonzero(trH), min(600000, trH.sum()), replace=False)
    mh = gb().fit(Xh[a], yh[a])
    ph_ = np.full(n * S, np.nan); ok = np.isfinite(Xh).all(1); ph_[ok] = mh.predict(Xh[ok]); ph_ = ph_.reshape(n, S)
    # 15-min predictions: ph=0 from hourly model in both; intra-hour: A = linear, B = 15-min model
    predA = np.full((n * 4, S), np.nan); predB = np.full((n * 4, S), np.nan)
    predA[0::4] = ph_; predB[0::4] = ph_
    linA = (1 - SF) * ph_[h0] + SF * ph_[h0 + 1]
    kk = 4 * h0 + (sfr * 4).astype(int)
    predA[kk] = linA
    # B: learn the intra-hour deviation from linA using CLDAS-only features (no power anchors)
    Xb = stack([{'s': SF, 'linA': linA, 'ph0': ph_[h0], 'ph1': ph_[h0 + 1], 'phm1': ph_[h0 - 1], 'ph2': ph_[h0 + 2]}, FC])
    yb = (y - linA).ravel(); fb = np.isfinite(yb) & np.isfinite(linA.ravel())
    trb = np.broadcast_to(trS[:, None], (N, S)).ravel() & fb
    a = rng.choice(np.flatnonzero(trb), min(900000, trb.sum()), replace=False)
    mb = gb().fit(Xb[a], yb[a])
    okb = np.isfinite(linA.ravel())
    pb = np.full(N * S, np.nan); pb[okb] = mb.predict(Xb[okb]); predB[kk] = linA + pb.reshape(N, S)
    te15 = te[np.arange(n * 4) // 4][:, None] & np.isfinite(Q)
    intra = (np.arange(n * 4) % 4 != 0)[:, None] & te15
    t4 = {}
    for name, Pp in [('hourly+linear', predA), ('hourly+CLDAS_intrahour_model', predB)]:
        e = (Pp - Q)
        t4[name] = dict(rmse_all15=float(np.sqrt(np.nanmean(e[te15] ** 2))), rmse_intrahour_points=float(np.sqrt(np.nanmean(e[intra] ** 2))))
        Bo = bands(np.where(np.isfinite(Q), Q, np.nan)); Bp = bands(Pp)
        tm = te[np.arange(n * 4) // 4]
        for b in Bo:
            rr = []
            for s in range(S):
                o, p = Bo[b][tm, s], Bp[b][tm, s]; okk = np.isfinite(o) & np.isfinite(p)
                if okk.sum() > 500: rr.append(np.corrcoef(o[okk], p[okk])[0, 1])
            t4[name][f'band_corr_{b}'] = float(np.mean(rr))
    res['T4_cldas_only_end_to_end'] = t4; save_json('Q9_cldas_intrahour.json', res); print('T4', t4, flush=True)
