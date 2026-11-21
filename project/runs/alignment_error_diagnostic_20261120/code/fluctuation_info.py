"""Q6: what does Pangu (daily protocol, leads 15-38 h) tell us about power FLUCTUATIONS, by time scale?
(1) Band decomposition of 15-min CF with centred moving averages (exactly additive):
      >24h = MA96 ; 6-24h = MA24-MA96 ; 1-6h = MA4-MA24 ; <1h = p-MA4
    Same decomposition on weather-based predictions (hourly -> 15 min linear):
      ERA5  : g(ERA5 100 m)                           (perfect-forecast ceiling)
      PANGU : decoupled g(f(Pangu))  (best from Q5)
    Per band: variance share, corr, R2, std ratio.
(2) Ramps of hourly-mean power over 1/3/6 h: corr of changes; events |dP|>0.2: POD/FAR/CSI exact and +-1/2 h tolerance.
(3) Fluctuation intensity: std of the <1h band in 3-h blocks and of the 1-6h band in 6-h blocks;
    GBDT from Pangu block features vs calendar-only climatology vs ERA5 features (R2).
Fit 2022, evaluate 2023 excl. Nov."""
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.ensemble import HistGradientBoostingRegressor
from common import *
from power_tests import load_power, split_masks, station_weather


def ma(x, w):
    """centred moving average along axis 0 with NaN if window incomplete."""
    c = np.cumsum(np.vstack([np.zeros((1,) + x.shape[1:]), np.nan_to_num(x)]), 0)
    k = np.cumsum(np.vstack([np.zeros((1,) + x.shape[1:]), np.isfinite(x).astype(float)]), 0)
    out = np.full(x.shape, np.nan)
    h = w // 2
    s = (c[w:] - c[:-w]); cnt = (k[w:] - k[:-w])
    v = np.where(cnt == w, s / w, np.nan)
    out[h:h + len(v)] = v
    return out


def bands(p):
    m96, m24, m4 = ma(p, 96), ma(p, 24), ma(p, 4)
    return {'>24h': m96, '6-24h': m24 - m96, '1-6h': m4 - m24, '<1h': p - m4}


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
    feats = ['P_spd10', 'P_spd1000', 'P_spd925', 'P_u10', 'P_v10', 'P_u925', 'P_v925', 'P_t2m', 'P_t925', 'P_msl']
    hod = (hours // 3600) % 24
    # hourly predictions
    predE = np.full((n, S), np.nan); predP = np.full((n, S), np.nan)
    for s in range(S):
        u = W['E_spd100'][:, s]; y = cf_h[:, s]
        a = tr & np.isfinite(u) & np.isfinite(y)
        g = IsotonicRegression(out_of_bounds='clip').fit(u[a], y[a])
        ok = np.isfinite(u); predE[ok, s] = g.predict(u[ok])
        X = np.stack([W[f][:, s] for f in feats] + [np.sin(2 * np.pi * hod / 24), np.cos(2 * np.pi * hod / 24)], 1)
        fx = np.isfinite(X).all(1)
        a1 = tr & fx & np.isfinite(u)
        uh = gb().fit(X[a1], u[a1]).predict(X[fx])
        predP[fx, s] = g.predict(uh)
    # to 15-min grid (time index k = (t15-T0)/900)
    K = len(t15); k = (t15 - T0) // 900
    def to15(H):
        out = np.full((K, S), np.nan)
        h0 = k // 4; f = (k % 4) / 4.0
        nxt = np.minimum(h0 + 1, n - 1)
        out[:] = (1 - f)[:, None] * H[h0] + f[:, None] * H[nxt]
        return out
    E15, P15 = to15(predE), to15(predP)
    te15 = te[k // 4]
    obsB = bands(cf15.astype(float)); eB = bands(E15); pB = bands(P15)
    res = {'bands': {}}
    tot = np.nanvar(cf15[te15])
    for b in obsB:
        o = obsB[b][te15]
        row = dict(var_share=float(np.nanvar(o) / tot))
        for name, B in [('ERA5', eB), ('PANGU', pB)]:
            p = B[b][te15]
            m = np.isfinite(o) & np.isfinite(p)
            r = [];  r2 = []; sr = []
            for s in range(S):
                ms = m[:, s]
                if ms.sum() < 500: continue
                oo, pp = o[ms, s], p[ms, s]
                r.append(np.corrcoef(oo, pp)[0, 1]); r2.append(1 - np.mean((oo - pp) ** 2) / np.var(oo)); sr.append(pp.std() / oo.std())
            row[name] = dict(corr=float(np.mean(r)), r2=float(np.mean(r2)), std_ratio=float(np.mean(sr)))
        res['bands'][b] = row
    save_json('Q6_fluctuation.json', res)
    if 'intensity_only' in __import__('sys').argv: res = {}
    # ramps on hourly-mean power (MA4 aligned to hour) -> use hourly cf_h smoothed: hourly mean over [h, h+1)
    hm = np.full((n, S), np.nan)
    blk = np.full((n * 4, S), np.nan); blk[k] = cf15
    hm = np.nanmean(blk.reshape(n, 4, S), 1) if True else hm
    cnt = np.isfinite(blk.reshape(n, 4, S)).sum(1); hm[cnt < 4] = np.nan
    res['ramps'] = {}
    for D in [1, 3, 6]:
        dO = hm[D:] - hm[:-D]; dE = predE[D:] - predE[:-D]; dP = predP[D:] - predP[:-D]
        mask = te[:-D] & te[D:]
        row = {}
        for name, dd in [('ERA5', dE), ('PANGU', dP)]:
            m = mask[:, None] & np.isfinite(dO) & np.isfinite(dd)
            row[name + '_corr_of_changes'] = float(np.mean([np.corrcoef(dO[m[:, s], s], dd[m[:, s], s])[0, 1] for s in range(S)]))
            row[name + '_std_ratio_of_changes'] = float(np.mean([dd[m[:, s], s].std() / dO[m[:, s], s].std() for s in range(S)]))
            for thr in [0.2]:
                for tol in [0, 1, 2]:
                    hits = fa = miss = 0
                    for sgn in [1, -1]:
                        ob = m & (sgn * dO > thr); pr = m & (sgn * dd > thr)
                        # tolerance: dilate along time
                        def dil(x):
                            y = x.copy()
                            for t in range(1, tol + 1):
                                y[t:] |= x[:-t]; y[:-t] |= x[t:]
                            return y
                        hits += (ob & dil(pr)).sum(); miss += (ob & ~dil(pr)).sum(); fa += (pr & ~dil(ob)).sum()
                    row[f'{name}_thr{thr}_tol{tol}h'] = dict(POD=float(hits / max(hits + miss, 1)), FAR=float(fa / max(fa + hits, 1)),
                                                           CSI=float(hits / max(hits + miss + fa, 1)), n_obs_events=int(hits + miss))
        res['ramps'][f'{D}h'] = row
    save_json('Q6_fluctuation.json', res)
    # fluctuation intensity
    res['intensity'] = {}
    for bname, L in [('<1h', 3), ('1-6h', 6)]:
        o = np.full((n * 4, S), np.nan); o[k] = obsB[bname]
        nb = n // L
        tgt = np.nanstd(o[:nb * L * 4].reshape(nb, L * 4, S), 1)
        cnt = np.isfinite(o[:nb * L * 4].reshape(nb, L * 4, S)).sum(1); tgt[cnt < L * 4 * 0.9] = np.nan
        bh = np.arange(nb) * L
        trb, teb = tr[bh], te[bh]
        month = ((hours[bh] + 8 * 3600) // 86400 % 365) / 365.0
        cal = [np.sin(2 * np.pi * hod[bh] / 24), np.cos(2 * np.pi * hod[bh] / 24), np.sin(2 * np.pi * month), np.cos(2 * np.pi * month)]
        rows = {'calendar': [], 'pangu': [], 'era5': [], 'pangu+calendar': [], 'level_only': [], 'level_plus_slope': []}
        for s in range(S):
            def blockfeat(pref, names):
                fs = []
                for nm in names:
                    x = W[pref + nm][:nb * L, s].reshape(nb, L)
                    fs += [np.nanmean(x, 1), np.nanmax(x, 1) - np.nanmin(x, 1)]
                return fs
            pf = blockfeat('P_', ['spd10', 'spd925', 'spd1000', 't2m', 't925']) + [np.nanmean((W['P_t925'] - W['P_t2m'])[:nb * L, s].reshape(nb, L), 1)]
            ef = blockfeat('E_', ['spd10', 'spd925', 'spd100', 't2m', 't925'])
            y = np.log(tgt[:, s] + 1e-3)
            lv = predP[:nb * L, s].reshape(nb, L)
            lev = [np.nanmean(lv, 1)]
            lev2 = lev + [np.nanmean(lv * (1 - lv), 1), np.nanmax(lv, 1) - np.nanmin(lv, 1)]
            for name, cols in [('calendar', cal), ('pangu', pf), ('era5', ef), ('pangu+calendar', pf + cal), ('level_only', lev), ('level_plus_slope', lev2)]:
                X = np.stack(cols, 1)
                a = trb & np.isfinite(X).all(1) & np.isfinite(y); b = teb & np.isfinite(X).all(1) & np.isfinite(y)
                if a.sum() < 200 or b.sum() < 100: continue
                p = gb().fit(X[a], y[a]).predict(X[b])
                rows[name].append(1 - np.mean((p - y[b]) ** 2) / np.var(y[b]))
        res['intensity'][f'{bname}_std_in_{L}h_blocks_log'] = {k2: float(np.mean(v)) for k2, v in rows.items()}
    save_json('Q6_fluctuation.json', res)
    import json; print(json.dumps(res, indent=1, ensure_ascii=False))
