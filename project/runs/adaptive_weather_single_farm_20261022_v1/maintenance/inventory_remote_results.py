"""Read-only inventory of historical metrics; never copy predictions/checkpoints."""
import csv
import gzip
import hashlib
import json
import os
import subprocess
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

PROJECT = Path('/shared_data_ssd2/liying/wind_cluster_project_20261018/project')
OUT = PROJECT / 'runs/adaptive_weather_single_farm_20261022_v1/analysis/rmse_review_20261024/remote_inventory'
ROOTS = [
    '/shared_data_ssd1/liying/new_try_v2/wind_cluster_forced_evolution',
    '/shared_data_ssd1/liying/new_try_v2/wind_cluster_migration_20260921/project',
    '/shared_data_ssd2/liying/new_try/wind_cluster_forced_evolution',
    '/shared_data_ssd2/liying/new_try/wind_cluster_migration_20260921/project',
    '/shared_data_hdd1/liying/hj102/data/liying/new_try/wind_cluster_forced_evolution',
    '/shared_data_hdd1/liying/hj102/data/liying/new_try/wind_cluster_migration_20260921/project',
    '/shared_data_hdd1/liying/hj102/huawei/storage/liying/new_try/wind_cluster_forced_evolution',
    '/shared_data_hdd1/liying/hj102/huawei/storage/liying/new_try/wind_cluster_migration_20260921/project',
]
SKIP = {'data', 'datasets', 'cache', 'caches', 'env', 'venv', 'node_modules', '__pycache__', 'site-packages', 'predictions', 'prediction_cache', 'checkpoints'}

def scan(root):
    if not root.is_dir():
        return
    command = ['rg', '--files', '--hidden', '--no-ignore', '--threads', '16', '-g', '*.json', '-g', '!**/.*/**']
    for name in sorted(SKIP):
        command.extend(['-g', '!**/' + name + '/**'])
    # Apply ignore patterns below the scan root, not to /.../data/... ancestors.
    result = subprocess.run(command + ['.'], cwd=root, capture_output=True, text=True)
    if result.returncode not in (0, 1):
        raise RuntimeError(result.stderr)
    for name in sorted(result.stdout.splitlines()):
        path = root / name
        if 'metric' in path.name.lower() or 'score' in path.name.lower() or path.name == 'results.json':
            yield path

def read_metric(path):
    if path.is_symlink() and not path.exists():
        # Migration snapshots retain absolute links to the original machine.
        # The actual attempt files are inventoried separately; never rewrite links.
        return path, None, 0
    data = path.read_bytes()
    return path, hashlib.sha256(data).hexdigest(), len(data)

def metric_values(obj, prefix=''):
    result = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            key = prefix + str(k)
            if isinstance(v, dict):
                result.update(metric_values(v, key + '.'))
            elif isinstance(v, (int, float)) and any(s in key.lower() for s in ('mae', 'mse', 'rmse', 'r2', 'loss', 'samples')):
                result[key] = v
    return result

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    canon = {}
    # Inventory all paths, but avoid rereading tens of thousands of known copies.
    # Content-hash new paths and deterministic samples of existing paths. Never
    # label an un-hashed path as a byte-identical duplicate.
    for p in scan(PROJECT / 'artifacts'):
        canon[str(p.relative_to(PROJECT))] = (None, p)
    hist_paths = {}
    for p in scan(PROJECT / 'historical_reports'):
        hist_paths['artifacts/' + str(p.relative_to(PROJECT / 'historical_reports'))] = p
    print('Canonical metric files: ' + str(len(canon)), flush=True)
    counts = defaultdict(Counter)
    unique = {}
    rows = []
    broken_links = []
    for root_s in ROOTS:
        root = Path(root_s)
        before = len(rows)
        samples = Counter()
        def inspect(p):
            rel = str(p.relative_to(root))
            exp = p.relative_to(root).parts[1]
            samples[exp] += 1
            existing = rel in canon or rel in hist_paths
            if existing and samples[exp] > 2:
                return p, '', ''
            return read_metric(p)
        with ThreadPoolExecutor(max_workers=32) as pool:
            records = list(pool.map(inspect, scan(root / 'artifacts')))
        for p, h, size in records:
            if h is None:
                broken_links.append(str(p))
                continue
            rel = str(p.relative_to(root))
            experiment = p.relative_to(root).parts[1]
            target = canon.get(rel)
            equivalent = ''
            if h == '':
                status = 'existing_path_not_content_hashed'
                equivalent = str(target[1] if target else hist_paths[rel])
            elif target:
                if target[0] is None:
                    _, target_hash, _ = read_metric(target[1])
                    canon[rel] = target = (target_hash, target[1])
                if h == target[0]:
                    status = 'same_path_same_bytes'
                else:
                    try:
                        a, b = metric_values(json.loads(p.read_text())), metric_values(json.loads(target[1].read_text()))
                        status = 'same_scalar_metrics_metadata_changed' if a and a == b else 'same_path_different_metrics'
                    except (ValueError, OSError):
                        status = 'same_path_different_unparsed'
                equivalent = str(target[1])
            elif rel in hist_paths and h == read_metric(hist_paths[rel])[1]:
                status = 'same_bytes_historical_report'
                equivalent = str(hist_paths[rel])
            else:
                status = 'absent_from_canonical_metrics'
            counts[(root_s, experiment)][status] += 1
            row = dict(root=root_s, experiment=experiment, relative_path=rel, bytes=size, sha256=h, status=status, canonical_file=equivalent)
            rows.append(row)
            if status in ('absent_from_canonical_metrics', 'same_path_different_metrics', 'same_path_different_unparsed'):
                key = (rel, h)
                if key not in unique:
                    try:
                        values = metric_values(json.loads(p.read_text()))
                    except (ValueError, OSError):
                        values = {}
                    unique[key] = dict(row, source_file=str(p), metric_values=json.dumps(values, ensure_ascii=False, separators=(',', ':')))
        print(json.dumps({'root': root_s, 'metric_files': len(rows) - before}, ensure_ascii=False), flush=True)
    fields = ['root', 'experiment', 'relative_path', 'bytes', 'sha256', 'status', 'canonical_file']
    with gzip.open(OUT / 'metric_file_inventory.csv.gz', 'wt', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
    with gzip.open(OUT / 'additional_metric_files.csv.gz', 'wt', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields + ['source_file', 'metric_values']); w.writeheader(); w.writerows(unique.values())
    statuses = sorted({r['status'] for r in rows})
    with (OUT / 'experiment_inventory.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['root', 'experiment', 'metric_files'] + statuses)
        w.writeheader()
        for (root, exp), c in sorted(counts.items()):
            w.writerow(dict(root=root, experiment=exp, metric_files=sum(c.values()), **c))
    summary = dict(canonical_metric_files=len(canon), scanned_metric_files=len(rows), status_counts=dict(Counter(r['status'] for r in rows)), additional_unique_path_hash_pairs=len(unique), additional_experiments=sorted({r['experiment'] for r in unique.values()}), broken_metric_links=len(broken_links), roots=ROOTS, method='All matching metric JSON paths inventoried; new paths fully content-hashed; up to first two existing-path candidates per experiment/root content-hashed. Other existing paths are explicitly not claimed byte-identical.')
    (OUT / 'broken_metric_links.txt').write_text('\n'.join(broken_links)+'\n')
    (OUT / 'INVENTORY.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    reconcile_historical_reports()

def reconcile_historical_reports():
    """Also recognize lightweight historical reports already in the main project."""
    historical = {}
    historical_hashes = {}
    for p in scan(PROJECT / 'historical_reports'):
        _, h, size = read_metric(p)
        if h:
            historical[str(p.relative_to(PROJECT / 'historical_reports'))] = (h, p)
            historical_hashes[h] = str(p)
    with gzip.open(OUT / 'metric_file_inventory.csv.gz', 'rt') as f:
        rows = list(csv.DictReader(f))
    with gzip.open(OUT / 'additional_metric_files.csv.gz', 'rt') as f:
        extra_reader = csv.DictReader(f); extra_fields = extra_reader.fieldnames; extra = list(extra_reader)
    for row in rows:
        if row['status'] != 'absent_from_canonical_metrics':
            continue
        target = historical.get(row['relative_path'].removeprefix('artifacts/'))
        if target and row['sha256'] == target[0]:
            row['status'] = 'same_bytes_historical_report'
            row['canonical_file'] = str(target[1])
        elif row['sha256'] in historical_hashes:
            row['status'] = 'same_bytes_historical_report'
            row['canonical_file'] = historical_hashes[row['sha256']]
    remaining = {(r['relative_path'],r['sha256']) for r in rows if r['status'] in ('absent_from_canonical_metrics', 'same_path_different_metrics', 'same_path_different_unparsed')}
    extra = [r for r in extra if (r['relative_path'],r['sha256']) in remaining]
    with gzip.open(OUT / 'metric_file_inventory.csv.gz', 'wt', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    with gzip.open(OUT / 'additional_metric_files.csv.gz', 'wt', newline='') as f:
        w = csv.DictWriter(f, fieldnames=extra_fields); w.writeheader(); w.writerows(extra)
    counts = defaultdict(Counter)
    for r in rows:
        counts[(r['root'],r['experiment'])][r['status']] += 1
    statuses = sorted({r['status'] for r in rows})
    with (OUT / 'experiment_inventory.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['root','experiment','metric_files'] + statuses); w.writeheader()
        for (root, exp), c in sorted(counts.items()):
            w.writerow(dict(root=root,experiment=exp,metric_files=sum(c.values()),**c))
    summary = json.loads((OUT / 'INVENTORY.json').read_text())
    summary.update(canonical_scope=['artifacts','historical_reports'], historical_report_metric_files=len(historical), status_counts=dict(Counter(r['status'] for r in rows)), additional_unique_path_hash_pairs=len(extra), additional_experiments=sorted({r['experiment'] for r in extra}))
    (OUT / 'INVENTORY.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'reconciled_status_counts':summary['status_counts'],'additional_unique_files':len(extra)},ensure_ascii=False),flush=True)

if __name__ == '__main__':
    main()
