"""Estimate the independent-training tail from verified new-protocol throughput."""
from pathlib import Path
import collections
import concurrent.futures
import datetime
import json
import math
import statistics
import subprocess

P = Path(__file__).resolve().parents[1]
NAMES = ['daily_protocol_restart_20261114_v1', 'daily_site_finetune_20261114_v1',
         'daily_ablation_restart_20261114_v1']


def atomic(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def host_snapshot(host):
    code = '''from pathlib import Path
import json,time,subprocess
r=Path('/home/liying/wind_physics_project/project/runs/daily_ablation_restart_20261114_v1')
p=r/'schedulers/status.json';s=json.loads(p.read_text());progress={}
for a in s.get('active',[]):
 q=r/'jobs'/a['id']/'progress.json'
 if q.exists():progress[a['id']]=json.loads(q.read_text())
print(json.dumps(dict(scheduler=s,age_seconds=time.time()-p.stat().st_mtime,progress=progress,
gpu_csv=subprocess.check_output(['nvidia-smi','--query-gpu=index,utilization.gpu,memory.free','--format=csv,noheader,nounits'],text=True))))'''
    result = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8', host, 'python3 -'],
                            input=code, text=True, capture_output=True, timeout=45, check=True)
    return host, json.loads(result.stdout)


def main():
    if all((P/'runs'/n/'COMPLETE.json').exists() for n in NAMES):
        runs = {n: json.loads((P/'runs'/n/'COMPLETE.json').read_text()) for n in NAMES}
        result = dict(timestamp=datetime.datetime.now().astimezone().isoformat(), phase='complete',
                      total_planned=2571, total_completed=sum(r['completed'] for r in runs.values()),
                      remaining=0, total_active=0, central_remaining_hours=0, planning_range_hours=[0, 0],
                      finished_at=max(r['timestamp'] for r in runs.values()), runs=runs,
                      result_report='DAILY_PROTOCOL_RESULTS_REVIEW_20261115_CN.md',
                      completion_audit='analysis/daily_protocol_review_20261115/COMPLETION_AUDIT.json',
                      scope='2571 retraining jobs and 2023 validation; excludes 2024/LM extensions/unseen-farm transfer')
        atomic(P/'RETRAIN_ETA_20261115.json', result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    assert all((P/'runs'/n/'COMPLETE.json').exists() for n in NAMES[:2]), 'Use the earlier phase ETA before shared/personalization complete'
    with concurrent.futures.ThreadPoolExecutor(2) as pool:
        hosts = dict(pool.map(host_snapshot, ['v1006001', 'v1006002']))
    now = datetime.datetime.now().astimezone()
    runs = {}; stages = collections.defaultdict(lambda: dict(completed=0, planned=0))
    completions = []; measures = collections.defaultdict(list); remaining = []
    for name in NAMES:
        r = P/'runs'/name; manifest = json.loads((r/'manifest.json').read_text())
        done = 0; failed = []
        for spec in manifest:
            job = r/'jobs'/spec['id']; success = (job/'SUCCESS.json').exists()
            done += success
            if (job/'FAILURE.json').exists():failed.append(spec['id'])
            stage = spec['stage']; stages[stage]['planned'] += 1; stages[stage]['completed'] += success
            if not success:remaining.append(spec)
            if name != NAMES[2] or not success:continue
            metric = json.loads((job/'metrics.json').read_text())
            history = json.loads((job/'history.json').read_text())
            if stage == 'independent':
                completions.append(datetime.datetime.fromisoformat(metric['timestamp']))
                measures[spec['group']].append(dict(epochs=len(history), epoch_seconds=statistics.median(x['seconds'] for x in history)))
        status = json.loads((r/'STATUS.json').read_text())
        complete = json.loads((r/'COMPLETE.json').read_text()) if (r/'COMPLETE.json').exists() else None
        runs[name] = dict(completed=done, planned=len(manifest), remaining=len(manifest)-done,
                         failed=failed, errors=status.get('errors', {}), complete=complete,
                         status_timestamp=status['timestamp'])
    assert all(s['stage'] == 'independent' for s in remaining), 'Tail estimate applies only to independent jobs'
    windows = []
    for minutes in [15, 30, 60, 120]:
        count = sum(0 <= (now-t).total_seconds() < minutes*60 for t in completions)
        rate = count*60/minutes
        windows.append(dict(minutes=minutes, completed=count, jobs_per_hour=rate,
                            remaining_hours=len(remaining)/rate if rate else None))
    empirical = [w['remaining_hours'] for w in windows if w['remaining_hours'] is not None]
    bounds = [math.floor(min(empirical)), math.ceil(max(empirical)+1/6)] if empirical and remaining else [0, 0]
    central = windows[2]['remaining_hours']
    if central is not None and remaining:central += 1/6  # final collection allowance
    active = [a for h in hosts.values() if h['age_seconds'] < 120 for a in h['scheduler'].get('active', [])]
    result = dict(timestamp=now.isoformat(), phase='independent_training_tail', total_planned=2571,
                  total_completed=sum(r['completed'] for r in runs.values()), remaining=len(remaining),
                  runs=runs, stages=dict(stages), hosts=hosts, active=active, total_active=len(active),
                  throughput_windows=windows, central_remaining_hours=central, planning_range_hours=bounds,
                  planning_finish_range=[(now+datetime.timedelta(hours=h)).isoformat() for h in bounds],
                  remaining_by_family=dict(collections.Counter(s['family'] for s in remaining)),
                  independent_measurements={g:dict(completed=len(v), median_epochs=statistics.median(x['epochs'] for x in v),
                      median_epoch_seconds=statistics.median(x['epoch_seconds'] for x in v)) for g,v in measures.items()},
                  limitations=['Based on locally verified successful independent jobs, timestamped by model evaluation completion.',
                               'Measured throughput includes scheduling, resource contention and return delays; future farm convergence and other users may change it.',
                               'No capacity added and no scientific settings changed during this check.',
                               '2571 retraining jobs and their 2023 validation only; excludes 2024 reevaluation and conditional LM extensions.'])
    atomic(P/'RETRAIN_ETA_20261115.json', result)
    atomic(P/'operations/eta_recovery_20261115/TAIL_STATUS.json', result)
    print(json.dumps({k:result[k] for k in ['timestamp','total_completed','remaining','total_active','throughput_windows','central_remaining_hours','planning_range_hours','remaining_by_family']}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
