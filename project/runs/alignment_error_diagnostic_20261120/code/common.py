"""Shared paths/helpers for the Pangu-vs-ERA5 alignment error diagnostics (2022-2023 only; no 2024)."""
from __future__ import annotations
import json, os
from pathlib import Path
import numpy as np

PROJECT = Path('/shared_data_ssd2/liying/wind_cluster_project_20261018/project')
RUN = PROJECT / 'runs/alignment_error_diagnostic_20261120'
OUT = RUN / 'output'
BUNDLE = Path('/shared_data_ssd3/liying/operational_pangu_daily_20261114_v1')
ERA5_DIAG = PROJECT / 'runs/era5_privileged_diagnostic_20261101_v1/output'
STATIC = PROJECT / 'data/operational_bundle_v1/cache/static'
POWER_H5 = PROJECT / 'data/jilin_56farm_pad8.h5'
ERA5_SURF = Path('/shared_data_hdd1/All_Datasets/ERA5/surface_new')
CH = ['msl', 'u10', 'v10', 't2m', 'z1000', 'z925', 'q1000', 'q925', 't1000', 't925', 'u1000', 'u925', 'v1000', 'v925']
CI = {c: i for i, c in enumerate(CH)}
T0 = 1640995200  # 2022-01-01 00 UTC
T1 = 1704067200  # 2024-01-01 00 UTC (exclusive)
R_EARTH = 6371.0
LAT = np.load(STATIC / 'weather_latitude.npy')   # 24, descending 47.25..41.5
LON = np.load(STATIC / 'weather_longitude.npy')  # 25, ascending 120.75..126.75
DLAT = 0.25
DY_KM = np.deg2rad(DLAT) * R_EARTH                       # ~27.8 km
def dx_km(lat): return np.deg2rad(DLAT) * R_EARTH * np.cos(np.deg2rad(lat))


def save_json(name, obj):
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / name
    tmp = p.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=True, default=float) + '\n')
    tmp.replace(p)


def stations():
    import h5py
    with h5py.File(POWER_H5, 'r') as f:
        lat = f['stations/latitude'][:].astype(float)
        lon = f['stations/longitude'][:].astype(float)
        cap = f['stations/capacity_mw'][:].astype(float)
    return lat, lon, cap


def frac_index(lat, lon):
    """fractional (row, col) on the 24x25 crop grid."""
    return (LAT[0] - np.asarray(lat)) / DLAT, (np.asarray(lon) - LON[0]) / DLAT


def bilinear(field, r, c):
    """field [..., H, W]; r, c arrays of identical shape broadcastable to field[...,0,0] leading dims + point dims.
    Returns field sampled at (r, c) with edge clipping; also returns in-domain mask."""
    H, W = field.shape[-2:]
    inside = (r >= 0) & (r <= H - 1) & (c >= 0) & (c <= W - 1)
    r = np.clip(r, 0, H - 1 - 1e-6); c = np.clip(c, 0, W - 1 - 1e-6)
    r0 = np.floor(r).astype(int); c0 = np.floor(c).astype(int)
    fr = r - r0; fc = c - c0
    def g(rr, cc):
        return np.take_along_axis(field.reshape(*field.shape[:-2], H * W), (rr * W + cc), axis=-1)
    v = (g(r0, c0) * (1 - fr) * (1 - fc) + g(r0 + 1, c0) * fr * (1 - fc)
         + g(r0, c0 + 1) * (1 - fr) * fc + g(r0 + 1, c0 + 1) * fr * fc)
    return v, inside


def load_pangu():
    raw = np.load(BUNDLE / 'raw_frames.npy', mmap_mode='r')
    issue = np.load(BUNDLE / 'frame_issue_unix.npy')
    lead = np.load(BUNDLE / 'frame_lead_hours.npy')
    return raw, issue, lead


def load_era5():
    """hourly ERA5 crop (14 channels) for 2022-2023 + 100 m wind if extracted; returns dict by hour index."""
    era = np.load(ERA5_DIAG / 'era5_crop.npy', mmap_mode='r')
    valid = np.load(ERA5_DIAG / 'valid_unix.npy')
    done = np.load(ERA5_DIAG / 'surface_done.npy') & np.load(ERA5_DIAG / 'pressure_done.npy')
    n = (T1 - T0) // 3600
    idx = np.full(n, -1, dtype=np.int64)
    ok = done & (valid >= T0) & (valid < T1)
    idx[(valid[ok] - T0) // 3600] = np.flatnonzero(ok)
    w100 = None
    p = OUT / 'era5_100m.npy'
    if p.exists():
        w100 = np.load(p, mmap_mode='r')  # [n, 2, 24, 25] by hour index
    return era, idx, w100


def speed(u, v):
    return np.sqrt(u * u + v * v)
