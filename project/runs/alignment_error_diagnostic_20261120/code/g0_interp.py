"""G0: does advection-corrected (semi-Lagrangian, knot-preserving) temporal interpolation beat linear interpolation
on this 0.25 deg / hourly grid?  Withhold intermediate hours (gap 2/3/6 h) and reconstruct them.
Part A: ERA5 truth (hourly analysis).  Part B: Pangu same-issue frames, verified against (i) Pangu's own hidden
frames and (ii) ERA5 at the same valid times.  Also: how much does Pangu's own intermediate hour add over linear
interpolation when verified against ERA5?  2022-2023 only."""
import numpy as np
from common import *

rng = np.random.default_rng(0)


def interp(F0, F1, U0, U1, V0, V1, r, c, s, dt_h, alpha=1.0):
    """F0,F1 [N,H,W] scalar fields at anchors; U/V steering fields [N,H,W]; r,c [N,S] target points; s fraction."""
    lin0, _ = bilinear(F0, r, c); lin1, _ = bilinear(F1, r, c)
    lin = (1 - s) * lin0 + s * lin1
    u = (1 - s) * bilinear(U0, r, c)[0] + s * bilinear(U1, r, c)[0]
    v = (1 - s) * bilinear(V0, r, c)[0] + s * bilinear(V1, r, c)[0]
    T = dt_h * 3600.0 / 1000.0 * alpha  # km per (m/s) over the whole gap
    lat = LAT[0] - r * DLAT
    rb = r + v * s * T / DY_KM; cb = c - u * s * T / dx_km(lat)
    rf = r - v * (1 - s) * T / DY_KM; cf = c + u * (1 - s) * T / dx_km(lat)
    a0, in0 = bilinear(F0, rb, cb); a1, in1 = bilinear(F1, rf, cf)
    adv = (1 - s) * a0 + s * a1
    return lin, adv, in0 & in1


def boot_gain(sse_l, sse_a, day, n=500):
    d = np.unique(day, return_inverse=True)[1]
    L = np.bincount(d, sse_l); A = np.bincount(d, sse_a); D = len(L)
    g = []
    for _ in range(n):
        k = rng.integers(0, D, D)
        g.append(1 - np.sqrt(A[k].sum() / L[k].sum()))
    return [float(np.percentile(g, 2.5)), float(np.percentile(g, 97.5))]


def summarize(truth, lin, adv, ok, day, steer):
    m = ok & np.isfinite(truth) & np.isfinite(lin) & np.isfinite(adv)
    el = (lin - truth) ** 2; ea = (adv - truth) ** 2
    out = dict(n=int(m.sum()), in_domain_frac=float(ok.mean()),
               rmse_linear=float(np.sqrt(el[m].mean())), rmse_advect=float(np.sqrt(ea[m].mean())))
    out['rmse_gain_pct'] = 100 * (1 - out['rmse_advect'] / out['rmse_linear'])
    dd = np.broadcast_to(day[:, None], truth.shape)[m]
    out['gain_ci95_pct'] = [100 * x for x in boot_gain(el[m], ea[m], dd)]
    fast = m & (steer > 10)
    if fast.sum() > 100:
        out['fast_flow_n'] = int(fast.sum())
        out['fast_flow_rmse_gain_pct'] = float(100 * (1 - np.sqrt(ea[fast].mean() / el[fast].mean())))
    return out


def era5_part():
    era, eidx, w100 = load_era5()
    slat, slon, _ = stations()
    sr, sc = frac_index(slat, slon)
    gr, gc = np.meshgrid(np.arange(6, 18), np.arange(6, 19), indexing='ij')
    pts = dict(stations=(sr, sc), interior_grid=(gr.ravel().astype(float), gc.ravel().astype(float)))
    n = len(eidx)
    ok_h = (eidx >= 0) & np.isfinite(w100[:, 0, 0, 0])
    res = {}
    for gap in [2, 3, 6]:
        anchors = np.array([h for h in range(0, n - gap, gap) if ok_h[h:h + gap + 1].all()])
        if len(anchors) > 3000:
            anchors = np.sort(rng.choice(anchors, 3000, replace=False))
        def grab(hours):
            e = np.asarray(era[eidx[hours]]); w = np.asarray(w100[hours])
            return dict(u10=e[:, 1], v10=e[:, 2], u925=e[:, CI['u925']], v925=e[:, CI['v925']],
                        u1000=e[:, CI['u1000']], v1000=e[:, CI['v1000']], u100=w[:, 0], v100=w[:, 1])
        A0 = grab(anchors); A1 = grab(anchors + gap)
        for j in range(1, gap):
            s = j / gap
            T = grab(anchors + j)
            day = (anchors + j) // 24
            for var, (fu, fv) in dict(spd100=('u100', 'v100'), spd10=('u10', 'v10'), spd925=('u925', 'v925')).items():
                F0 = speed(A0[fu], A0[fv]); F1 = speed(A1[fu], A1[fv]); Ft = speed(T[fu], T[fv])
                for steer_name, (su, sv) in dict(own=(fu, fv), w925=('u925', 'v925')).items():
                    for alpha in ([0.5, 1.0, 1.5] if steer_name == 'own' else [1.0]):
                        for pname, (r, c) in pts.items():
                            R = np.broadcast_to(r, (len(anchors), len(r))); C = np.broadcast_to(c, R.shape)
                            lin, adv, ok = interp(F0, F1, A0[su], A1[su], A0[sv], A1[sv], R, C, s, gap, alpha)
                            truth, _ = bilinear(Ft, R, C)
                            st = speed(*(bilinear(x, R, C)[0] for x in ((1 - s) * A0[su] + s * A1[su], (1 - s) * A0[sv] + s * A1[sv])))
                            key = f'gap{gap}h/s={j}of{gap}/{var}/steer={steer_name}/alpha={alpha}/{pname}'
                            res[key] = summarize(truth, lin, adv, ok, day, st)
        save_json('G0_era5.json', res)
        print('era5 gap', gap, flush=True)
    return res


def pangu_part():
    raw, issue, lead = load_pangu()
    era, eidx, _ = load_era5()
    valid = issue + lead * 3600
    keep = (valid >= T0) & (valid < T1)
    issues = np.unique(issue[keep])
    issues = np.sort(rng.choice(issues, min(len(issues), 400), replace=False))
    slat, slon, _ = stations()
    sr, sc = frac_index(slat, slon)
    look = {(int(i), int(l)): k for k, (i, l) in enumerate(zip(issue, lead))}
    res = {}
    for gap in [2, 3, 6]:
        rows = []
        for it in issues:
            for L0 in range(9, 46 - gap, gap):
                ks = [look.get((int(it), L)) for L in range(L0, L0 + gap + 1)]
                hs = [(int(it) + L * 3600 - T0) // 3600 for L in range(L0, L0 + gap + 1)]
                if None in ks or any(h < 0 or h >= len(eidx) or eidx[h] < 0 for h in hs):
                    continue
                rows.append((ks, hs, (int(it) - T0) // 86400))
        K = np.array([r[0] for r in rows]); Hh = np.array([r[1] for r in rows]); day = np.array([r[2] for r in rows])
        uk, inv = np.unique(K.ravel(), return_inverse=True)
        Pk = np.asarray(raw[uk])[:, [1, 2, CI['u925'], CI['v925']]][inv].reshape(*K.shape, 4, 24, 25)
        uh, invh = np.unique(eidx[Hh.ravel()], return_inverse=True)
        Ek = np.asarray(era[uh])[:, [1, 2, CI['u925'], CI['v925']]][invh].reshape(Pk.shape)
        R = np.broadcast_to(sr, (len(rows), len(sr))); C = np.broadcast_to(sc, R.shape)
        for j in range(1, gap):
            s = j / gap
            for var, (iu, iv) in dict(spd10=(0, 1), spd925=(2, 3)).items():
                F0 = speed(Pk[:, 0, iu], Pk[:, 0, iv]); F1 = speed(Pk[:, -1, iu], Pk[:, -1, iv])
                lin, adv, ok = interp(F0, F1, Pk[:, 0, iu], Pk[:, -1, iu], Pk[:, 0, iv], Pk[:, -1, iv], R, C, s, gap)
                st = speed(bilinear(Pk[:, j, iu], R, C)[0], bilinear(Pk[:, j, iv], R, C)[0])
                p_mid, _ = bilinear(speed(Pk[:, j, iu], Pk[:, j, iv]), R, C)
                e_mid, _ = bilinear(speed(Ek[:, j, iu], Ek[:, j, iv]), R, C)
                base = f'gap{gap}h/s={j}of{gap}/{var}'
                res[base + '/truth=pangu_hidden'] = summarize(p_mid, lin, adv, ok, day, st)
                res[base + '/truth=era5'] = summarize(e_mid, lin, adv, ok, day, st)
                m = np.isfinite(e_mid)
                res[base + '/era5_rmse_of_pangu_actual_mid_frame'] = float(np.sqrt(((p_mid - e_mid) ** 2)[m].mean()))
        save_json('G0_pangu.json', res)
        print('pangu gap', gap, flush=True)
    return res


if __name__ == '__main__':
    import sys
    if 'pangu' in sys.argv:
        pangu_part()
    else:
        era5_part()
