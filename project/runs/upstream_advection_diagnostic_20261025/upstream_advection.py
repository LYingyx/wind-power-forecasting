"""Zero-training diagnostic: is upstream farm power history a leading signal for a downstream farm?

Uses observed 15-min power (2022-2023 only; 2024 is never read) and Pangu 10 m wind.
For every farm pair 20-150 km apart and every 15-min time t, the pair is classified by the
forecast wind at the pair midpoint (wind speed >= 4 m/s):
  A_UP  : wind blows from A towards B  (cos >= 0.7)
  B_UP  : wind blows from B towards A  (cos <= -0.7)
  CROSS : |cos| <= 0.3
Two analyses per pair and class:
  1. Lagged cross-correlation between A's 1 h power change at t and B's 1 h power change at t+tau,
     tau in [-4 h, +4 h]. Advection predicts a positive-lag peak near distance / along-wind speed for A_UP,
     a negative-lag peak for B_UP and no lead for CROSS.
  2. Incremental predictability: B's future change CF_B(t+h) - CF_B(t), h = 1/2/3 h, regressed on
     B's own recent changes, current level and the NWP-forecast wind change at B (base), then adding
     A's recent power history with a 15 min data delay (full). Fit on 2022, evaluate on 2023.
"""
import csv
from pathlib import Path
import h5py
import numpy as np

ROOT = Path('/shared_data_ssd2/liying/wind_cluster_project_20261018/project')
H5 = ROOT / 'data/jilin_56farm_pad8.h5'
CACHE = ROOT / 'data/model_cache_pad5_f8/static'
OUT = Path(__file__).resolve().parent
END_2023 = 1704067200          # 2024-01-01 00:00 UTC; nothing at or after this is used
START_2023 = 1672531200        # 2023-01-01 00:00 UTC
STEP = 900
LAGS = np.arange(-16, 17)      # 15-min steps, -4 h .. +4 h
HORIZONS = {'1h': 4, '2h': 8, '3h': 12}


def haversine(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2
    return 2 * 6371 * np.arcsin(np.sqrt(a))


def corr(x, y):
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 200:
        return np.nan
    x, y = x[ok] - x[ok].mean(), y[ok] - y[ok].mean()
    d = np.sqrt((x * x).sum() * (y * y).sum())
    return float((x * y).sum() / d) if d > 0 else np.nan


def ols_r2(xtr, ytr, xte, yte):
    ok_tr = np.isfinite(xtr).all(1) & np.isfinite(ytr)
    ok_te = np.isfinite(xte).all(1) & np.isfinite(yte)
    if ok_tr.sum() < 300 or ok_te.sum() < 300:
        return np.nan, np.nan, 0
    a = np.hstack([xtr[ok_tr], np.ones((ok_tr.sum(), 1))])
    b = np.hstack([xte[ok_te], np.ones((ok_te.sum(), 1))])
    w = np.linalg.lstsq(a, ytr[ok_tr], rcond=None)[0]
    res = yte[ok_te] - b @ w
    mse = float((res ** 2).mean())
    return mse, float(np.abs(res).mean()), int(ok_te.sum())


def main():
    f = h5py.File(H5, 'r')
    ts = f['power/timestamp_unix'][:]
    keep = ts < END_2023
    ts = ts[keep]
    mw = f['power/station_mw'][:][keep].astype(float)
    ids_h5 = [x.decode() if isinstance(x, bytes) else str(x) for x in f['power'].attrs.get('station_ids', [])] if 'station_ids' in f['power'].attrs else None
    sid = np.load(CACHE / 'station_id.npy').astype(str)
    cap = np.load(CACHE / 'station_capacity_mw.npy').astype(float)
    lat = np.load(CACHE / 'station_latitude.npy').astype(float)
    lon = np.load(CACHE / 'station_longitude.npy').astype(float)
    cf = mw / cap[None, :]
    cf[(cf < -0.05) | (cf > 1.2)] = np.nan
    n_t, n_s = cf.shape
    assert np.all(np.diff(ts) > 0)
    # regular 15-min index (fill gaps with NaN)
    t0 = ts[0]
    idx = ((ts - t0) // STEP).astype(int)
    T = idx.max() + 1
    grid_cf = np.full((T, n_s), np.nan)
    grid_cf[idx] = cf
    times = t0 + np.arange(T) * STEP
    is_train = times < START_2023

    # wind: latest run issued at least 15 min before t, lead index for t
    glat, glon = f['crop/latitude'][:], f['crop/longitude'][:]
    issue = f['nwp/issue_unix'][:]
    rl = f['nwp/run_lead_to_frame'][:]
    mean = f['normalization/mean'][:]; std = f['normalization/std'][:]

    pairs = []
    for a in range(n_s):
        for b in range(a + 1, n_s):
            d = haversine(lat[a], lon[a], lat[b], lon[b])
            if 20 <= d <= 150:
                pairs.append((a, b, d))
    # grid points needed: each station and each pair midpoint
    def nearest(la, lo):
        return int(np.abs(glat - la).argmin()), int(np.abs(glon - lo).argmin())
    st_pt = [nearest(lat[s], lon[s]) for s in range(n_s)]
    mid_pt = [nearest((lat[a] + lat[b]) / 2, (lon[a] + lon[b]) / 2) for a, b, _ in pairs]
    pts = sorted(set(st_pt) | set(mid_pt))
    pid = {p: i for i, p in enumerate(pts)}
    pi_ = np.array([p[0] for p in pts]); pj = np.array([p[1] for p in pts])

    n_frames = f['nwp/values'].shape[0]
    u = np.empty((n_frames, len(pts)), 'f4'); v = np.empty_like(u)
    block = 8000
    for s0 in range(0, n_frames, block):
        x = f['nwp/values'][s0:s0 + block, 1:3]            # u10, v10 (normalized)
        u[s0:s0 + block] = x[:, 0][:, pi_, pj] * std[1] + mean[1]
        v[s0:s0 + block] = x[:, 1][:, pi_, pj] * std[2] + mean[2]

    run_of_issue = {int(t): i for i, t in enumerate(issue)}
    def frame_for(valid_times, origin_times):
        """frame index of the latest run issued <= origin-15min, for each valid time (or -1)."""
        H = ((origin_times - STEP) // 3600) * 3600
        r = np.array([run_of_issue.get(int(h), -1) for h in H])
        k = np.clip(np.round((valid_times - H) / 3600).astype(int) - 1, 0, rl.shape[1] - 1)
        fr = np.where(r >= 0, rl[np.maximum(r, 0), k], -1)
        return fr
    fr_now = frame_for(times, times)
    def wind_at(frames, point):
        ok = frames >= 0
        uu = np.where(ok, u[np.maximum(frames, 0), point], np.nan)
        vv = np.where(ok, v[np.maximum(frames, 0), point], np.nan)
        return uu, vv
    fr_future = {h: frame_for(times + steps * STEP, times) for h, steps in HORIZONS.items()}

    def change(x, steps):
        out = np.full_like(x, np.nan); out[steps:] = x[steps:] - x[:-steps]; return out
    def shift(x, steps):   # value at t+steps
        out = np.full_like(x, np.nan)
        if steps > 0: out[:-steps] = x[steps:]
        elif steps < 0: out[-steps:] = x[:steps]
        else: out = x.copy()
        return out

    rows_cc, rows_pred = [], []
    for (a, b, d), mp in zip(pairs, mid_pt):
        um, vm = wind_at(fr_now, pid[mp])
        speed = np.hypot(um, vm)
        # unit vector A->B in east/north km
        dx = (lon[b] - lon[a]) * np.pi / 180 * 6371 * np.cos(np.radians((lat[a] + lat[b]) / 2))
        dy = (lat[b] - lat[a]) * np.pi / 180 * 6371
        ex, ey = dx / d, dy / d
        cos = (um * ex + vm * ey) / np.maximum(speed, 1e-6)
        valid_wind = speed >= 4
        classes = {'A_UP': valid_wind & (cos >= 0.7), 'B_UP': valid_wind & (cos <= -0.7), 'CROSS': valid_wind & (np.abs(cos) <= 0.3)}
        ca, cb = grid_cf[:, a], grid_cf[:, b]
        da, db = change(ca, 4), change(cb, 4)
        for cname, m in classes.items():
            if m.sum() < 500:
                continue
            along = speed * np.abs(cos)
            exp_lag_h = float(np.nanmedian(d / (along[m] * 3.6)))
            cc = [corr(da[m], shift(db, L)[m]) for L in LAGS]
            cc = np.array(cc)
            k = int(np.nanargmax(cc)) if np.isfinite(cc).any() else 0
            el = int(np.clip(np.round(exp_lag_h * 4), 1, 16))
            rows_cc.append(dict(a=a, b=b, distance_km=d, cls=cname, n=int(m.sum()), mean_speed=float(np.nanmean(speed[m])),
                                expected_lag_h=exp_lag_h, peak_lag_h=float(LAGS[k] / 4), peak_corr=float(cc[k]),
                                corr_lag0=float(cc[16]), corr_plus_expected=float(cc[16 + el]), corr_minus_expected=float(cc[16 - el])))
        # incremental predictability of B's future change; A delayed by 15 min
        ub_now, vb_now = wind_at(fr_now, pid[st_pt[b]])
        sp_now = np.hypot(ub_now, vb_now)
        a_hist = [shift(ca, -1), shift(change(ca, 4), -1), shift(change(ca, 8), -1)]
        for h, steps in HORIZONS.items():
            uf, vf = wind_at(fr_future[h], pid[st_pt[b]])
            sp_f = np.hypot(uf, vf)
            y = shift(cb, steps) - cb
            base = np.column_stack([cb, change(cb, 1), change(cb, 4), sp_now, sp_f, sp_f - sp_now, (sp_f - sp_now) * (1 - cb)])
            full = np.column_stack([base] + a_hist)
            for cname, m in classes.items():
                tr, te = m & is_train, m & ~is_train
                if tr.sum() < 500 or te.sum() < 300:
                    continue
                mse0, mae0, n0 = ols_r2(base[tr], y[tr], base[te], y[te])
                mse1, mae1, n1 = ols_r2(full[tr], y[tr], full[te], y[te])
                rows_pred.append(dict(a=a, b=b, distance_km=d, cls=cname, horizon=h, n_test=n1, mse_base=mse0, mse_full=mse1,
                                      mae_base=mae0, mae_full=mae1))
    for name, rows in [('lagged_crosscorrelation.csv', rows_cc), ('incremental_predictability.csv', rows_pred)]:
        with open(OUT / name, 'w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print('pairs', len(pairs), 'cc rows', len(rows_cc), 'pred rows', len(rows_pred))


if __name__ == '__main__':
    main()
