"""Separate historical protocols; preserve farm/seed values without averaging."""
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

P = Path('/shared_data_ssd2/liying/wind_cluster_project_20261018/project')
H = Path('/shared_data_hdd1/liying/hj102/huawei/storage/liying/new_try/wind_cluster_forced_evolution')
O = P / 'runs/adaptive_weather_single_farm_20261022_v1/analysis/rmse_review_20261024/remote_inventory'

def read(p):
    with p.open() as f:
        return list(csv.DictReader(f))

def write(name, rows):
    with (O / name).open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

def station_metrics(pred, target, capacity):
    e = (pred.astype(np.float64) - target.astype(np.float64)) / capacity[None, :, None]
    return np.abs(e).mean((0, 2)), np.sqrt(np.square(e).mean((0, 2)))

def main():
    O.mkdir(exist_ok=True)
    clean_path = H / 'artifacts/station_clean_20260912/clean_per_station_metrics.csv'
    clean = read(clean_path)
    index = {(r['model'], r['split'], int(r['seed']), int(r['station_index'])): r for r in clean}
    assert len(index) == len(clean) == 2240
    pairs = []
    def add(protocol, candidate, reference, split, seed, station, a, b, source):
        pairs.append(dict(protocol=protocol, candidate=candidate, reference=reference, split=split, seed=seed, station=station, candidate_mae=float(a[0]), reference_mae=float(b[0]), delta_mae=float(a[0])-float(b[0]), candidate_rmse=float(a[1]), reference_rmse=float(b[1]), delta_rmse=float(a[1])-float(b[1]), candidate_source=source))
    for (m, sp, seed, station), a in index.items():
        if m == 'v1_direct_mae':
            b = index[('transformer_mae', sp, seed, station)]
            add('historical_shared_clean_20260912', m, 'transformer_mae', sp, seed, station, (a['cf_mae'], a['cf_rmse']), (b['cf_mae'], b['cf_rmse']), str(clean_path))
    actual = []
    checked = []
    for seed in range(42, 45):
        rp = H / f'artifacts/station_clean_20260912/runs/transformer_mae_seed{seed}/predictions_test.npz'
        if rp.is_symlink() and not rp.exists():
            rp = H / rp.readlink().relative_to('/storage/liying/new_try/wind_cluster_forced_evolution')
        cap = np.array([float(index[('transformer_mae', 'test', seed, s)]['station_capacity_mw']) for s in range(56)])
        with np.load(rp) as z:
            target = z['station_target_mw']; origin = z['origin_unix']; samples = z['sample_index']
            ref_mae, ref_rmse = station_metrics(z['station_prediction_mw'], target, cap)
        for s in range(56):
            row = index[('transformer_mae', 'test', seed, s)]
            np.testing.assert_allclose([ref_mae[s], ref_rmse[s]], [float(row['cf_mae']), float(row['cf_rmse'])], rtol=1e-7, atol=1e-9)
        for family in ['physics_gate', 'level_shape']:
            fs = list((H / f'artifacts/{family}_20260912/runs/{family}_seed{seed}/attempts').glob('test*/predictions_test.npz'))
            assert len(fs) == 1, fs
            f = fs[0]
            with np.load(f) as z:
                np.testing.assert_array_equal(z['origin_unix'], origin)
                np.testing.assert_array_equal(z['sample_index'], samples)
                np.testing.assert_array_equal(z['station_target_mw'], target)
                pred = z['station_prediction_mw']; assert np.isfinite(pred).all()
                mae, rmse = station_metrics(pred, target, cap)
                del pred
            metrics = json.loads((f.parent / 'metrics_test.json').read_text())
            np.testing.assert_allclose(mae, metrics['per_station_mae_cf'], rtol=1e-6, atol=1e-8)
            for s in range(56):
                actual.append(dict(protocol='historical_shared_mse_20260912', model=family, split='test', year=2024, seed=seed, station_index=s, station_capacity_mw=cap[s], mae=mae[s], rmse=rmse[s], samples=len(origin), source_file=str(f)))
                add('historical_shared_architecture_and_loss_differ', family, 'transformer_mae_clean', 'test', seed, s, (mae[s], rmse[s]), (ref_mae[s], ref_rmse[s]), str(f))
            checked.append(dict(model=family, seed=seed, path=str(f), aligned_origins=True, aligned_sample_index=True, identical_targets=True, finite_predictions=True, original_station_mae_matches=True))
            print(f'Validated {family} seed{seed}', flush=True)
    arows = read(P / 'runs/single_farm_review_20261018/tables/prediction_station_seed.csv')
    brows = read(P / 'runs/single_farm_review_20261018/tables/shared_baselines_station_seed.csv')
    baseline = {(r['model'],r['split'],r['seed'],r['station']):r for r in brows}
    for a in arows:
        if a['model'] != 'E2C':continue
        for model in ['TRANSFORMER_FC','TIDE_FC']:
            b = baseline[(model,a['split'],a['seed'],a['station'])]
            add('historical_shared_E2C_additional_region_history_FP32_vs_BF16', 'E2C', model, a['split'], a['seed'], a['station'], (a['mae'],a['rmse']), (b['mae'],b['rmse']), str(P / 'runs/single_farm_review_20261018/tables/prediction_station_seed.csv'))
    counts = defaultdict(Counter)
    for row in pairs:
        c = counts[(row['protocol'],row['candidate'],row['reference'],row['split'],int(row['seed']))]
        c['n'] += 1
        c['mae_better'] += row['delta_mae'] < 0
        c['rmse_better'] += row['delta_rmse'] < 0
        c['mae_better_rmse_worse'] += row['delta_mae'] < 0 and row['delta_rmse'] > 0
        c['mae_worse_rmse_better'] += row['delta_mae'] > 0 and row['delta_rmse'] < 0
    totals = defaultdict(Counter)
    count_rows = []
    for (protocol,candidate,reference,split,seed), c in sorted(counts.items()):
        count_rows.append(dict(protocol=protocol,candidate=candidate,reference=reference,split=split,seed=seed,**c))
        totals[(candidate,reference,split)].update(c)
    write('historical_physical_actual_per_farm_seed.csv', actual)
    write('historical_paired_actual_values.csv', pairs)
    write('historical_direction_counts_per_seed.csv', count_rows)
    check = dict(passed=True, raw_archives_recomputed=9, physical_candidate_archives=6, reference_archives=3, physical_actual_rows=len(actual), pair_rows=len(pairs), across_farm_or_seed_means=False, checks=checked)
    (O / 'HISTORICAL_CHECK.json').write_text(json.dumps(check, ensure_ascii=False, indent=2)+'\n')
    for k,c in totals.items():print(k,dict(c))

if __name__ == '__main__':
    main()
