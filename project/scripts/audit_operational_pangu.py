"""Independent timestamp, source-value and BASE/SEL loader checks; no model eval."""
from pathlib import Path
import argparse
import csv
import hashlib
import importlib.util
import json
import sys
from datetime import datetime
import h5py
import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / 'src'))
from wind_forcing.data.operational import plan_times, source_path, november_exclusions, unix, PROTOCOLS
from wind_forcing.data.operational_farm import OperationalFarmData


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, default=Path('/shared_data_ssd3/liying/operational_pangu_20261027_v1'))
    args = parser.parse_args()
    torch.set_num_threads(1)
    bundle = args.bundle.resolve()
    assert json.loads((bundle / 'AUDIT.json').read_text())['passed']
    assert (bundle / 'SUCCESS.json').exists()
    cache = bundle / 'cache'
    protocol = json.loads((bundle / 'PROTOCOL.json').read_text())
    availability = PROTOCOLS[protocol['protocol']]
    manifest = json.loads((cache / 'manifest.json').read_text())
    assert manifest['preprocess_fingerprint'] == protocol['preprocess_fingerprint']
    for name, expected in protocol['code_hashes'].items():
        assert hashlib.sha256((PROJECT / name).read_bytes()).hexdigest() == expected
    frame_issue = np.load(bundle / 'frame_issue_unix.npy')
    frame_lead = np.load(bundle / 'frame_lead_hours.npy')
    raw = np.load(bundle / 'raw_frames.npy', mmap_mode='r')
    receipts = {}
    with (bundle / 'source_crop_receipts.csv').open() as f:
        for row in csv.DictReader(f):
            receipts[int(row['frame'])] = row
    assert len(receipts) == len(raw)
    for i in range(len(raw)):
        assert hashlib.sha256(np.asarray(raw[i]).tobytes()).hexdigest() == receipts[i]['clean_crop_sha256']
    checks = []
    # Compare fresh external crops at stratified positions with the extracted store.
    for i in np.unique(np.linspace(0, len(raw) - 1, 18).astype(int)):
        p = source_path(protocol['root'], int(frame_issue[i]), int(frame_lead[i]))
        data = np.load(p, mmap_mode='r')
        crop = np.array(data[0, :, 171:195, 483:508])
        crop[[6, 7]] = np.maximum(crop[[6, 7]], 0)
        np.testing.assert_array_equal(raw[i], crop)
        checks.append(str(p))
    h5 = PROJECT / 'data/jilin_56farm_pad8.h5'
    with h5py.File(h5, 'r') as f:
        times = f['power/timestamp_unix'][:]
        power = f['power/station_mw'][:]
        capacity = f['stations/capacity_mw'][:]
        for split in ['train', 'val', 'test']:
            origin = np.load(cache / split / 'origin_unix.npy')
            p = plan_times(origin, availability)
            np.testing.assert_array_equal(np.load(cache / split / 'origin_issue_unix.npy'), p['issue'])
            np.testing.assert_array_equal(np.load(cache / split / 'origin_available_unix.npy'), p['available'])
            selected = np.unique(np.linspace(0, len(origin) - 1, 40).astype(int))
            positions = np.searchsorted(times, origin[selected])
            np.testing.assert_array_equal(times[positions], origin[selected])
            hpos = positions[:, None] + np.arange(-23, 1)
            ypos = positions[:, None] + np.arange(1, 25)
            np.testing.assert_array_equal(times[hpos], origin[selected, None] + np.arange(-23, 1)[None] * 900)
            np.testing.assert_array_equal(times[ypos], origin[selected, None] + np.arange(1, 25)[None] * 900)
            h = np.load(cache / split / 'station_history.npy', mmap_mode='r')
            y = np.load(cache / split / 'station_future.npy', mmap_mode='r')
            np.testing.assert_array_equal(h[selected], (power[hpos] / capacity).transpose(0, 2, 1))
            np.testing.assert_array_equal(y[selected], (power[ypos] / capacity).transpose(0, 2, 1))
            local_year = (origin + 8 * 3600).astype('datetime64[s]').astype('datetime64[Y]').astype(int) + 1970
            assert (local_year == {'train': 2022, 'val': 2023, 'test': 2024}[split]).all()
            if split == 'test':
                assert not any(x.any() for x in november_exclusions(origin, availability).values())
    # Exhaustive per-day switch invariants on 2023, including month/year rollovers.
    origins = unix('2023-01-01 00:00:00') + np.arange(365 * 96) * 900
    issue = plan_times(origins, availability)['issue']
    changes = np.flatnonzero(np.diff(issue)) + 1
    local_hours = ((origins[changes] + 28800) % 86400) / 3600
    expected_hours = {(x + availability.latency_hours + 8) % 24
                      for x in range(0, 24, availability.period_hours)}
    assert set(local_hours) == expected_hours
    assert len(changes) == 365 * (24 // availability.period_hours)
    loaders = []
    for features in ['BASE', 'SEL']:
        train = OperationalFarmData(cache, 'train', 0, features, limit=128)
        for split in ['val', 'test']:
            data = OperationalFarmData(cache, split, 0, features, stats=train.stats, limit=128)
            batch = data.batch(slice(0, 16))
            assert data.h.shape == data.y.shape == (128, 24)
            assert data.x.shape[-1] == (22 if features == 'SEL' else 0)
            assert all(torch.isfinite(x).all() for x in batch.values())
            assert np.all(data.available_unix <= data.origin)
            loaders.append(dict(feature_set=features, split=split, samples=data.n,
                                weather_shape=list(data.weather.shape), feature_shape=list(data.x.shape)))
        invalid = dict(train.stats, data_fingerprint='legacy')
        try:
            OperationalFarmData(cache, 'val', 0, features, stats=invalid, limit=4)
        except ValueError:
            pass
        else:
            raise AssertionError('Legacy statistics incorrectly accepted')
    result = dict(timestamp=datetime.now().astimezone().isoformat(), passed=True,
        protocol=availability.name, preprocess_fingerprint=manifest['preprocess_fingerprint'],
        raw_crop_hashes_verified=len(raw), fresh_archive_frames_verified=checks,
        original_h5_power_time_and_values_verified=True, local_year_splits_verified=True,
        exhaustive_2023_switches_verified=len(changes), loader_checks=loaders,
        stale_normalization_rejected=True, test_model_evaluation_performed=False)
    (bundle / 'INDEPENDENT_AUDIT.json').write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
