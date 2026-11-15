"""Rebuild availability-correct train/val/test caches without altering old runs.

Reads only the required 24x25 regional crop of each global, hourly Pangu file.
All power timestamps and targets retain their original UTC Unix alignment.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import time
import h5py
import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / 'src'))
from wind_forcing.data import operational
from wind_forcing.data.operational import source_path, LOCAL
from wind_forcing.models.components import build_station_patch_metadata

DEFAULT_ROOT = Path('/hj102_data_large/ERA5_forecasting/Pangu/Forecast_data')
DEFAULT_OUT = Path('/shared_data_ssd3/liying/operational_pangu_20261027_v1')
OLD = PROJECT / 'data/model_cache_pad5_f8'
H5 = PROJECT / 'data/jilin_56farm_pad8.h5'
PHYSICS = PROJECT / 'runs/regional_reliability_validation_20261027_v1/code/physics.py'
PROTOCOL = operational.TWICE_DAILY
REQUIRE_SAME_SAMPLES = False


def plan_times(origin):
    return operational.plan_times(origin, PROTOCOL)


def november_exclusions(origin):
    return operational.november_exclusions(origin, PROTOCOL)


def now():
    return datetime.now().astimezone().isoformat()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    os.replace(temp, path)


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    with temp.open('wb') as f:
        np.save(f, value, allow_pickle=False)
    os.replace(temp, path)


def mmap_output(path, shape, dtype='f4'):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return np.lib.format.open_memmap(path, mode='w+', dtype=dtype, shape=shape)


def load_physics():
    spec = importlib.util.spec_from_file_location('operational_frozen_physics', PHYSICS)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def timestamp(t, local=False):
    return datetime.fromtimestamp(int(t), tz=LOCAL if local else timezone.utc).isoformat()


def build_plans(out, raw_root):
    plans = {}
    all_keys = set()
    exclusions = {}
    for split in ['train', 'val', 'test']:
        original = np.load(OLD / split / 'origin_unix.npy')
        reasons = november_exclusions(original) if split == 'test' else {}
        excluded = np.logical_or.reduce(list(reasons.values())) if reasons else np.zeros(len(original), bool)
        if REQUIRE_SAME_SAMPLES and excluded.any():
            raise ValueError(f'{split}: new protocol excludes {excluded.sum()} existing samples; sample changes require an explicit decision')
        keep = np.flatnonzero(~excluded)
        origin = original[keep]
        times = plan_times(origin)
        base, first, inverse = np.unique(times['base'], return_index=True, return_inverse=True)
        issue = times['issue'][first]
        future = times['future'][first]
        past = times['past'][first]
        assert np.array_equal(issue[inverse], times['issue'])
        assert np.all((origin - times['issue'] >= PROTOCOL.latency) &
                      (origin - times['issue'] < PROTOCOL.latency + PROTOCOL.period))
        for t, valid in zip(issue, np.concatenate([past, future], 1)):
            all_keys.update((int(t), int((v - t) // 3600)) for v in valid)
        plans[split] = dict(keep=keep, origin=origin, base=base, cycle=inverse, issue=issue, future=future, past=past)
        exclusions[split] = dict(original=len(original), retained=len(origin), removed=int(excluded.sum()),
            reason_counts_overlapping={k: int(v.sum()) for k, v in reasons.items()})
        if excluded.any():
            with (out / f'excluded_{split}.csv').open('w') as f:
                writer = csv.writer(f)
                writer.writerow(['source_row', 'origin_utc', 'origin_beijing', 'reasons'])
                for i in np.flatnonzero(excluded):
                    writer.writerow([int(i), timestamp(original[i]), timestamp(original[i], True),
                                     '|'.join(k for k, v in reasons.items() if v[i])])
        with (out / f'provenance_{split}.csv').open('w') as f:
            writer = csv.writer(f)
            writer.writerow(['row', 'source_row', 'origin_utc', 'origin_beijing', 'issue_utc', 'issue_beijing',
                             'available_beijing', 'nwp_age_hours', 'target_end_beijing', 'last_weather_lead'])
            for i, t in enumerate(origin):
                it = int(times['issue'][i])
                writer.writerow([i, int(keep[i]), timestamp(t), timestamp(t, True), timestamp(it), timestamp(it, True),
                                 timestamp(it + PROTOCOL.latency, True), (int(t) - it) / 3600,
                                 timestamp(t + 21600, True), int((times['future'][i, -1] - it) // 3600)])
    keys = sorted(all_keys)
    lookup = {key: i for i, key in enumerate(keys)}
    for split, p in plans.items():
        for name in ['future', 'past']:
            p[name + '_frames'] = np.array([[lookup[(int(t), int((v - t) // 3600))] for v in vs]
                                           for t, vs in zip(p['issue'], p[name])], dtype=np.int64)
        np.savez(out / f'plan_{split}.npz', **p)
    atomic_json(out / 'SAMPLE_SELECTION.json', exclusions)
    save(out / 'frame_issue_unix.npy', np.array([k[0] for k in keys], dtype=np.int64))
    save(out / 'frame_lead_hours.npy', np.array([k[1] for k in keys], dtype=np.int16))
    return plans, keys, exclusions


def source_coverage(out, raw_root, keys):
    """Check exact issue/lead identities before allocating a large frame store."""
    paths = [source_path(raw_root, *key) for key in keys]
    missing = []
    with (out / 'required_source_files.csv').open('w') as f:
        writer = csv.writer(f)
        writer.writerow(['frame', 'issue_unix', 'issue_utc', 'lead_hours', 'path', 'exists'])
        for i, (key, path) in enumerate(zip(keys, paths)):
            exists = path.is_file()
            if not exists:
                missing.append(str(path))
            writer.writerow([i, key[0], timestamp(key[0]), key[1], str(path), int(exists)])
    coverage = dict(timestamp=now(), root=str(raw_root), root_accessible=raw_root.is_dir(),
        required_files=len(keys), missing_count=len(missing), missing=missing,
        minimum_lead=min(k[1] for k in keys), maximum_lead=max(k[1] for k in keys),
        issues=len(set(k[0] for k in keys)), global_shape=[1, 14, 721, 1440],
        complete=not missing)
    atomic_json(out / 'SOURCE_COVERAGE.json', coverage)
    return paths, coverage


def extract_raw(out, raw_root, keys, workers):
    paths, coverage = source_coverage(out, raw_root, keys)
    if not coverage['complete']:
        raise FileNotFoundError(f"{coverage['missing_count']} required same-issue Pangu files unavailable; see SOURCE_COVERAGE.json")
    lat = np.load(OLD / 'static/weather_latitude.npy')
    lon = np.load(OLD / 'static/weather_longitude.npy')
    rows = np.rint((90 - lat) / .25).astype(int)
    cols = np.rint(lon / .25).astype(int)
    assert np.array_equal(np.diff(rows), np.ones(len(rows) - 1))
    assert np.array_equal(np.diff(cols), np.ones(len(cols) - 1))
    ys = slice(int(rows[0]), int(rows[-1]) + 1)
    xs = slice(int(cols[0]), int(cols[-1]) + 1)
    shape = (len(keys), 14, len(rows), len(cols))
    raw_path = out / 'raw_frames.npy'
    done_path = out / 'raw_done.npy'
    if raw_path.exists() and done_path.exists():
        raw = np.lib.format.open_memmap(raw_path, mode='r+')
        done = np.lib.format.open_memmap(done_path, mode='r+')
        assert raw.shape == shape and done.shape == (len(keys),)
    else:
        raw = mmap_output(raw_path, shape)
        done = mmap_output(done_path, (len(keys),), '?')
        done[:] = False
        done.flush()
    coverage.update(global_rows=[ys.start, ys.stop], global_columns=[xs.start, xs.stop])
    atomic_json(out / 'SOURCE_COVERAGE.json', coverage)
    def read(i):
        path = paths[i]
        before = path.stat()
        data = np.load(path, mmap_mode='r', allow_pickle=False)
        if data.shape != (1, 14, 721, 1440) or data.dtype != np.dtype('float32'):
            raise ValueError(f'Unexpected archive format: {path}: {data.shape}/{data.dtype}')
        block = np.array(data[0, :, ys, xs], dtype=np.float32, copy=True)
        del data
        if not np.isfinite(block).all():
            raise ValueError(f'Nonfinite required crop: {path}')
        negative = [int((block[c] < 0).sum()) for c in [6, 7]]
        block[[6, 7]] = np.maximum(block[[6, 7]], 0)
        after = path.stat()
        assert (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns)
        return i, block, [i, *keys[i], str(path), before.st_size, before.st_mtime_ns,
                          hashlib.sha256(block.tobytes()).hexdigest(), *negative]
    pending = np.flatnonzero(~done)
    receipt = out / 'source_crop_receipts.csv'
    with receipt.open('a') as f, ThreadPoolExecutor(max_workers=workers) as pool:
        writer = csv.writer(f)
        if f.tell() == 0:
            writer.writerow(['frame', 'issue_unix', 'lead_hours', 'path', 'size', 'mtime_ns', 'clean_crop_sha256', 'q1000_clipped', 'q925_clipped'])
        start = time.monotonic()
        for first in range(0, len(pending), 128):
            ids = pending[first:first + 128]
            for i, block, row in pool.map(read, ids):
                raw[i] = block
                writer.writerow(row)
            raw.flush()
            f.flush()
            os.fsync(f.fileno())
            done[ids] = True
            done.flush()
            atomic_json(out / 'STATUS.json', dict(timestamp=now(), state='extracting_same_issue_crops',
                done=int(done.sum()), total=len(keys), workers=workers, elapsed_seconds=time.monotonic() - start))
            if first % 1024 == 0:
                print(f'raw {int(done.sum())}/{len(keys)}', flush=True)
    assert done.all()
    return raw


def fit_normalization(out, raw, train):
    ids = np.unique(np.concatenate([train['future_frames'].ravel(), train['past_frames'].ravel()]))
    total = np.zeros(14, np.float64)
    square = np.zeros(14, np.float64)
    count = 0
    for start in range(0, len(ids), 128):
        x = np.asarray(raw[ids[start:start + 128]], dtype=np.float64)
        total += x.sum((0, 2, 3))
        square += np.square(x).sum((0, 2, 3))
        count += x.shape[0] * x.shape[2] * x.shape[3]
    mean = total / count
    std = np.sqrt(np.maximum(square / count - mean ** 2, 0))
    assert np.all(std > 1e-6)
    static = out / 'cache/static'
    static.mkdir(parents=True, exist_ok=True)
    for path in (OLD / 'static').glob('*.npy'):
        if path.stem not in ['nwp_mean', 'nwp_std']:
            shutil.copy2(path, static / path.name)
    mean, std = mean.astype('f4'), std.astype('f4')
    save(static / 'nwp_mean.npy', mean)
    save(static / 'nwp_std.npy', std)
    save(out / 'normalization_frame_index.npy', ids)
    atomic_json(out / 'NORMALIZATION.json', dict(fitted_split='train', frame_count=len(ids),
        value_count_per_channel=count, method='unique train-referenced physical crops, six past plus eight weather nodes',
        mean=mean.tolist(), std=std.tolist(), validation_or_test_used=False))
    return mean, std


def bilinear(values, ids, weights):
    # [..., channel, latitude, longitude] -> [..., station, channel]
    flat = values.reshape(*values.shape[:-2], -1)
    v = flat[..., ids]
    return (v * weights).sum(-1).swapaxes(-1, -2)


def dynamics(raw, ids, weights, station_lat):
    lat = np.load(OLD / 'static/weather_latitude.npy')
    lon = np.load(OLD / 'static/weather_longitude.npy')
    dphi = np.deg2rad(float(lat[1] - lat[0])) * 6371000
    dlam = np.deg2rad(float(lon[1] - lon[0])) * 6371000
    w = np.asarray(raw, dtype=np.float64)
    def point(field):
        return (field.reshape(*field.shape[:-2], -1)[..., ids] * weights).sum(-1)
    phi = np.deg2rad(station_lat)
    f = 2 * 7.2921159e-5 * np.sin(phi)
    ug = -point(np.gradient(w[:, :, 5], dphi, axis=-2, edge_order=2)) / f
    vg = point(np.gradient(w[:, :, 5], dlam, axis=-1, edge_order=2)) / np.cos(phi) / f
    dtdy = point(np.gradient(w[:, :, 9], dphi, axis=-2, edge_order=2))
    dtdx = point(np.gradient(w[:, :, 9], dlam, axis=-1, edge_order=2)) / np.cos(phi)
    p = bilinear(w, ids, weights)
    adv = -(p[..., 11] * dtdx + p[..., 13] * dtdy) * 3600
    dpdt = np.gradient(p[..., 0], 1., axis=1) / 100
    dvdt = np.gradient(np.hypot(p[..., 1], p[..., 2]), 1., axis=1)
    return np.stack([ug / 10, vg / 10, (p[..., 1] - ug) / 10, (p[..., 2] - vg) / 10,
                     adv, dpdt, dvdt], axis=-1).transpose(0, 2, 1, 3).astype('f4')


def prepare_split(out, split, p, raw, mean, std, ids, weights, station_lat):
    root = out / 'cache' / split
    physical = out / 'physical' / split
    root.mkdir(parents=True, exist_ok=True)
    physical.mkdir(parents=True, exist_ok=True)
    n, c = len(p['origin']), len(p['base'])
    copy_names = ['station_history', 'station_future', 'cluster_history', 'cluster_future',
                  'future_capacity_warning', 'sample_index', 'origin_phase']
    for name in copy_names:
        old = np.load(OLD / split / f'{name}.npy', mmap_mode='r')
        dest = mmap_output(root / f'{name}.npy', (n, *old.shape[1:]), old.dtype)
        for start in range(0, n, 512):
            dest[start:start + 512] = old[p['keep'][start:start + 512]]
        dest.flush()
        del dest
    issue = p['issue'][p['cycle']]
    arrays = dict(origin_unix=p['origin'], source_row_index=p['keep'], weather_cycle_index=p['cycle'],
        origin_issue_unix=issue, origin_available_unix=issue + PROTOCOL.latency,
        nwp_age_minutes=(p['origin'] - issue) // 60,
        weather_time_offset_hours=((p['future'][p['cycle']] - p['origin'][:, None]) / 3600).astype('f4'),
        weather_valid_unix=p['future'], weather_issue_unix=np.broadcast_to(p['issue'][:, None], (c, 8)),
        weather_available_unix=np.broadcast_to(p['issue'][:, None] + PROTOCOL.latency, (c, 8)),
        cycle_sample_index=np.arange(c, dtype=np.int64), cycle_origin_hour_unix=p['base'])
    for name, array in arrays.items():
        save(root / f'{name}.npy', array)
    with h5py.File(H5, 'r') as f:
        time_axis = f['power/timestamp_unix'][:]
        index = np.searchsorted(time_axis, p['future'])
        assert np.all(index < len(time_axis)) and np.array_equal(time_axis[index], p['future'])
        station_power = f['power/station_mw'][:]
        cap = f['stations/capacity_mw'][:]
        warning = f['power/capacity_warning'][:]
        target = (station_power[index] / cap).transpose(0, 2, 1)
        assert np.isfinite(target).all()
        save(root / 'weather_station_target.npy', target.astype('f4'))
        save(root / 'weather_station_capacity_warning.npy', warning[index].transpose(0, 2, 1))
    future = mmap_output(root / 'weather.npy', (c, 8, *raw.shape[1:]))
    local = mmap_output(out / 'local_weather' / f'{split}.npy', (c, 56, 8, 14))
    past = mmap_output(physical / 'past_weather.npy', (c, 56, 6, 14))
    dyn = mmap_output(physical / 'dynamics.npy', (c, 56, 8, 7))
    for start in range(0, c, 48):
        end = min(start + 48, c)
        x = np.asarray(raw[p['future_frames'][start:end]])
        normalized = (x - mean[None, None, :, None, None]) / std[None, None, :, None, None]
        future[start:end] = normalized
        local[start:end] = bilinear(normalized, ids, weights).transpose(0, 2, 1, 3)
        history = np.asarray(raw[p['past_frames'][start:end]])
        history = (history - mean[None, None, :, None, None]) / std[None, None, :, None, None]
        past[start:end] = bilinear(history, ids, weights).transpose(0, 2, 1, 3)
        dyn[start:end] = dynamics(x, ids, weights, station_lat)
        if start % 480 == 0:
            atomic_json(out / 'STATUS.json', dict(timestamp=now(), state='building_cache_and_physics', split=split, done=end, total=c))
            print(split, end, '/', c, flush=True)
    for array in [future, local, past, dyn]:
        array.flush()
    del future, local, past, dyn
    save(physical / 'past_valid_unix.npy', p['past'])
    save(physical / 'past_issue_unix.npy', np.broadcast_to(p['issue'][:, None], (c, 6)))
    save(physical / 'future_issue_unix.npy', np.broadcast_to(p['issue'][:, None], (c, 8)))
    # A previous-cycle disagreement cache is deliberately absent, not stale data.
    atomic_json(physical / 'READY.json', dict(timestamp=now(), n_samples=n, n_cycles=c,
        protocol=PROTOCOL.name, all_nodes_from_same_issue=True,
        historical_nodes=6, future_nodes=8, reference_nwp_is_available=True,
        prior_cycle_disagreement_available=False))
    return dict(sample_count=n, cycle_count=c, weather_shape=[8, *raw.shape[1:]],
        station_history_shape=[56, 24], station_future_shape=[56, 24])


def validate(out, plans, raw, mean, std, ids, weights):
    from wind_forcing.data.dataset import ForecastDataset
    from wind_forcing.data.operational import interpolate_nodes
    result = {}
    for split, p in plans.items():
        ds = ForecastDataset(out / 'cache', split, preload_weather=False, verbose=False)
        assert len(ds) == len(p['origin'])
        age = ds.arrays['nwp_age_minutes']
        assert age.min() >= PROTOCOL.latency_hours * 60
        assert age.max() <= (PROTOCOL.latency_hours + PROTOCOL.period_hours) * 60 - 15
        issue = p['issue'][p['cycle']]
        assert np.all(issue + PROTOCOL.latency <= p['origin'])
        assert np.all(issue + PROTOCOL.latency + PROTOCOL.period > p['origin'])
        if split == 'test':
            assert not any(mask.any() for mask in november_exclusions(p['origin']).values())
        local = np.load(out / 'local_weather' / f'{split}.npy', mmap_mode='r')
        past = np.load(out / 'physical' / split / 'past_weather.npy', mmap_mode='r')
        dyn = np.load(out / 'physical' / split / 'dynamics.npy', mmap_mode='r')
        for array in [local, past, dyn, ds.arrays['station_history'], ds.arrays['station_future']]:
            for a in range(0, len(array), 128):
                assert np.isfinite(array[a:a + 128]).all()
        probe = np.unique([0, len(p['base']) // 2, len(p['base']) - 1])
        physical = np.asarray(raw[p['future_frames'][probe]])
        expected = (physical - mean[None, None, :, None, None]) / std[None, None, :, None, None]
        np.testing.assert_allclose(ds.weather.values[probe], expected, atol=1e-6, rtol=1e-6)
        point = bilinear(expected, ids, weights).transpose(0, 2, 1, 3)
        np.testing.assert_allclose(local[probe], point, atol=1e-6, rtol=1e-6)
        chosen = np.unique(np.linspace(0, len(p['origin']) - 1, 100).astype(int))
        query = np.arange(1, 25) / 4
        values = interpolate_nodes(local[p['cycle'][chosen], 0], ds.arrays['weather_time_offset_hours'][chosen], query)
        assert np.isfinite(values).all()
        source_h = np.load(OLD / split / 'station_history.npy', mmap_mode='r')
        source_y = np.load(OLD / split / 'station_future.npy', mmap_mode='r')
        np.testing.assert_array_equal(ds.arrays['station_history'], source_h[p['keep']])
        np.testing.assert_array_equal(ds.arrays['station_future'], source_y[p['keep']])
        result[split] = dict(samples=len(ds), cycles=len(p['base']), min_age_hours=float(age.min() / 60),
            max_age_hours=float(age.max() / 60), no_future_issue=True, all_nodes_same_issue=True,
            original_power_exactly_preserved=True, all_values_finite=True, no_endpoint_extrapolation=True,
            first_origin_beijing=timestamp(p['origin'][0], True), last_origin_beijing=timestamp(p['origin'][-1], True))
    return result


def activate_bundle(out, config):
    manifest = json.loads((out / 'cache/manifest.json').read_text())
    identity = manifest['operational_protocol']
    if identity['protocol'] != PROTOCOL.name:
        raise ValueError('Completed bundle does not match requested protocol')
    fingerprint = manifest['preprocess_fingerprint']
    success = json.loads((out / 'SUCCESS.json').read_text())
    if success['preprocess_fingerprint'] != fingerprint:
        raise ValueError('Completion marker and cache fingerprint disagree')
    audit = json.loads((out / 'AUDIT.json').read_text())
    if not audit.get('passed') or audit['manifest_sha256'] != digest(out / 'cache/manifest.json'):
        raise ValueError('Bundle audit does not match the completed cache')
    independent = out / 'INDEPENDENT_AUDIT.json'
    if PROTOCOL == operational.DAILY:
        if not independent.is_file():
            raise ValueError('Run audit_operational_pangu.py --bundle before activating the daily protocol')
        independent_audit = json.loads(independent.read_text())
        if (not independent_audit.get('passed') or
                independent_audit.get('preprocess_fingerprint') != fingerprint):
            raise ValueError('Independent audit is absent or belongs to another dataset')
    link = Path(config.get('cache_dir', PROJECT / 'data/model_cache_operational_08_20_lag12h_v1'))
    if not link.exists():
        link.symlink_to(out / 'cache', target_is_directory=True)
    if link.resolve() != (out / 'cache').resolve():
        raise ValueError(f'Cache link points to a different dataset: {link}')
    atomic_json(PROJECT / 'data/ACTIVE_DATASET.json', dict(timestamp=now(), protocol=identity['protocol'],
        cache_dir=str(link), bundle_dir=str(out), preprocess_fingerprint=fingerprint,
        local_weather_dir=str(out / 'local_weather'), physics_dir=str(out / 'physical'),
        source_forecast_root=identity['root'], existing_frozen_training_runs_retargeted=False,
        require_new_training_and_normalization=True, audit=str(out / 'AUDIT.json'),
        independent_audit=str(independent) if independent.is_file() else None))


def main():
    global PROTOCOL, OLD, H5, REQUIRE_SAME_SAMPLES
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--root', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--activate', action='store_true')
    parser.add_argument('--plan-only', action='store_true', help='Write sample/source requirements without reading weather arrays or activating data')
    args = parser.parse_args()
    if args.plan_only and args.activate:
        parser.error('--plan-only cannot activate an unbuilt dataset')
    config = json.loads(args.config.read_text()) if args.config else {}
    PROTOCOL = operational.PROTOCOLS[config.get('protocol', operational.TWICE_DAILY.name)]
    args.root = args.root or Path(config.get('source_forecast_root', DEFAULT_ROOT))
    args.output = args.output or Path(config.get('bundle_dir', DEFAULT_OUT))
    OLD = Path(config.get('sample_cache', OLD)).resolve()
    H5 = Path(config.get('source_power_h5', H5)).resolve()
    REQUIRE_SAME_SAMPLES = config.get('require_same_samples', False)
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    lock = (out / 'builder.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (out / 'SUCCESS.json').exists():
        if args.activate:
            activate_bundle(out, config)
            print('Activated:', out)
        print('Already complete:', out)
        return
    identity = dict(protocol=PROTOCOL.name, root=str(args.root.resolve()),
        power_source=str(H5), old_cache=str(OLD), source_manifest_sha256=digest(OLD / 'manifest.json'),
        source_power_h5_size=H5.stat().st_size, source_power_h5_mtime_ns=H5.stat().st_mtime_ns,
        power_timezone='Asia/Shanghai', forecast_filename_timezone='UTC',
        issue_hours_utc=list(range(0, 24, PROTOCOL.period_hours)),
        issue_hours_beijing=[(x + 8) % 24 for x in range(0, 24, PROTOCOL.period_hours)],
        availability_delay_hours=PROTOCOL.latency_hours,
        max_required_hourly_forecast_lead=PROTOCOL.maximum_lead_hours, past_nodes=6, future_nodes=8,
        future_node_alignment='floor(origin UTC hour) + [0..7] hours',
        power_origin_and_targets='exactly preserve selected samples' if REQUIRE_SAME_SAMPLES else 'preserve old samples except explicit November and boundary exclusions',
        require_same_samples=REQUIRE_SAME_SAMPLES,
        regional_power_delay_minutes=config.get('regional_power_delay_minutes', 15),
        excluded_month='2024-11', exclusion='local history/target/NWP valid window; UTC or local November issue',
        forecast_power_steps=24, power_frequency_minutes=15,
        full_test_evaluation_performed=False, data_preparation_only=True,
        code_hashes={str(x.relative_to(PROJECT)): digest(x) for x in [Path(__file__), PROJECT / 'src/wind_forcing/data/operational.py', PHYSICS]})
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    if (out / 'PROTOCOL.json').exists():
        assert json.loads((out / 'PROTOCOL.json').read_text())['preprocess_fingerprint'] == fingerprint, 'Cannot resume with altered source/protocol/code'
    atomic_json(out / 'PROTOCOL.json', dict(**identity, preprocess_fingerprint=fingerprint))
    atomic_json(out / 'STATUS.json', dict(timestamp=now(), state='planning', pid=os.getpid()))
    plans, keys, exclusions = build_plans(out, args.root)
    if args.plan_only:
        _, coverage = source_coverage(out, args.root, keys)
        atomic_json(out / 'STATUS.json', dict(timestamp=now(), state='planned' if coverage['complete'] else 'awaiting_source',
            required_files=len(keys), missing_files=coverage['missing_count'], sample_selection=exclusions,
            training_ready=False, training_started=False))
        print(json.dumps(dict(output=str(out), source_complete=coverage['complete'],
            required_files=len(keys), missing_files=coverage['missing_count'], samples=exclusions), ensure_ascii=False))
        return
    raw = extract_raw(out, args.root, keys, max(1, min(args.workers, 8)))
    mean, std = fit_normalization(out, raw, plans['train'])
    static = out / 'cache/static'
    lat = np.load(static / 'weather_latitude.npy')
    lon = np.load(static / 'weather_longitude.npy')
    slat = np.load(static / 'station_latitude.npy')
    slon = np.load(static / 'station_longitude.npy')
    ids, _, weights = build_station_patch_metadata(lat, lon, slat, slon, 'bilinear', 1)
    assert ids.shape == weights.shape == (56, 4)
    save(static / 'bilinear_patch_index.npy', ids)
    save(static / 'bilinear_weight.npy', weights)
    physics = load_physics()
    near, geometry, mask = physics.neighbor_geometry(slat, slon)
    for name, value in [('neighbor_ids', near), ('neighbor_geometry', geometry), ('neighbor_mask', mask)]:
        save(out / 'physical' / f'{name}.npy', value)
    dem = PROJECT / 'artifacts/column_weather_20260928_v1/data/station_dem_srtm30m.json'
    elevations = json.loads(dem.read_text())['records']
    save(out / 'physical/elevation.npy', np.array([x['terrain_elevation_m'] for x in elevations], dtype='f4'))
    summaries = {}
    for split, plan in plans.items():
        summaries[split] = prepare_split(out, split, plan, raw, mean, std, ids, weights, slat)
    old_manifest = json.loads((OLD / 'manifest.json').read_text())
    arrays = []
    for path in sorted((out / 'cache').glob('*/*.npy')):
        a = np.load(path, mmap_mode='r')
        arrays.append(dict(relative_path=str(path.relative_to(out / 'cache')), file=path.name, shape=list(a.shape), dtype=str(a.dtype)))
    manifest = dict(schema_version=1, preprocess_fingerprint=fingerprint, source_h5=str(H5),
        raw_forecast_root=str(args.root), operational_protocol=identity,
        weather_values_are_prestandardized=True, no_spatial_interpolation=True, no_temporal_interpolation=True,
        crop=old_manifest['crop'], station_count=56, total_capacity_mw=old_manifest['total_capacity_mw'],
        history_power_steps=24, forecast_power_steps=24, future_nwp_nodes=8,
        target_lead_hours=(np.arange(1, 25) / 4).tolist(), splits=summaries, arrays=arrays,
        local_weather_dir=str(out / 'local_weather'), physics_dir=str(out / 'physical'))
    atomic_json(out / 'cache/manifest.json', manifest)
    atomic_json(out / 'local_weather/manifest.json', dict(preprocess_fingerprint=fingerprint,
        source_cache=str(out / 'cache'), method='bilinear same-issue standardized Pangu', splits=summaries))
    audit = validate(out, plans, raw, mean, std, ids, weights)
    atomic_json(out / 'AUDIT.json', dict(timestamp=now(), passed=True, splits=audit,
        source_frames=len(keys), sample_selection=exclusions, normalization_train_only=True,
        test_data_prepared=True, test_model_evaluation_performed=False,
        old_scientific_runs_unchanged=True, manifest_sha256=digest(out / 'cache/manifest.json')))
    atomic_json(out / 'SUCCESS.json', dict(timestamp=now(), preprocess_fingerprint=fingerprint, audit_passed=True))
    atomic_json(out / 'STATUS.json', dict(timestamp=now(), state='complete', splits=summaries))
    if args.activate:
        activate_bundle(out, config)
    print(json.dumps(dict(output=str(out), audit=audit), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        # Keep partial extraction resumable; never publish a ready dataset on failure.
        print(f'FAILED: {type(error).__name__}: {error}', file=sys.stderr, flush=True)
        raise
