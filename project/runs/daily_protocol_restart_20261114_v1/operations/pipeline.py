"""Resume weather preparation, audit the merged dataset, then deploy fresh training."""
from pathlib import Path
import fcntl
import json
import os
import subprocess
import sys
import time
import traceback

RUN=Path(__file__).resolve().parents[1]
PROJECT=RUN.parents[1]
sys.path.insert(0,str(RUN/'code'))
from settings import SOURCE,DATA,atomic_json,now,sha


def state(stage,**extra):
    atomic_json(RUN/'operations/PIPELINE_STATUS.json',dict(timestamp=now(),pid=os.getpid(),stage=stage,**extra))
    if not (RUN/'LAUNCH_STATUS.json').exists():
        prior=json.loads((RUN/'STATUS.json').read_text()) if (RUN/'STATUS.json').exists() else {}
        prior.update(timestamp=now(),state=stage,training_started=False,training_completed=0,**extra)
        if 'weather' in extra:
            prior.update(weather_raw_ready=extra['weather'].get('done'),weather_raw_total=extra['weather'].get('total'))
        prior['weather_data_rebuilt']=(SOURCE/'SUCCESS.json').exists()
        prior['training_pipeline_migration_complete']=(DATA/'READY.json').exists()
        atomic_json(RUN/'STATUS.json',prior)
    print(now(),stage,extra,flush=True)


def execute(label,script,*args):
    state(label)
    subprocess.run([sys.executable,'-u',str(script),*map(str,args)],cwd=PROJECT,check=True)


def weather():
    config=PROJECT/'configs/operational_pangu_d1_08_lag15h.json'
    while not (SOURCE/'SUCCESS.json').exists():
        with (SOURCE/'builder.lock').open('a+') as lock:
            try:
                fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                running=False
            except BlockingIOError:running=True
        if not running:
            execute('building_weather',PROJECT/'scripts/prepare_operational_pangu.py','--config',config,'--workers','8')
            break
        p=SOURCE/'STATUS.json'
        state('waiting_for_weather_builder',weather=json.loads(p.read_text()) if p.exists() else {})
        time.sleep(30)
    if not (SOURCE/'INDEPENDENT_AUDIT.json').exists():
        execute('independent_weather_audit',PROJECT/'scripts/audit_operational_pangu.py','--bundle',SOURCE)
    execute('activating_daily_dataset',PROJECT/'scripts/prepare_operational_pangu.py','--config',config,'--activate')


def freeze():
    paths=[*RUN.glob('code/**/*.py'),*RUN.glob('operations/*.py'),*RUN.glob('selection_inputs/*')]
    paths += [RUN/x for x in ['PROTOCOL.json','QC_RULES.json','manifest.json','DATA_FREEZE.json','EXECUTION_PLAN_CN.md','data/READY.json','operations/LOCAL_ADMISSION.json','operations/REMOTE_ADMISSION.json']]
    files={str(p.relative_to(RUN)):sha(p) for p in paths if p.is_file() and '__pycache__' not in p.parts}
    atomic_json(RUN/'SCIENCE_FREEZE.json',dict(timestamp=now(),files=files))


def run():
    lock=(RUN/'operations/pipeline.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    weather()
    ready=DATA/'READY.json'
    if not ready.exists():execute('preparing_fixed_cohort_training_data',RUN/'code/prepare.py')
    assert json.loads(ready.read_text())['preprocess_fingerprint']==json.loads((SOURCE/'PROTOCOL.json').read_text())['preprocess_fingerprint']
    execute('generating_fresh_job_manifest',RUN/'code/plan.py')
    execute('checking_models_and_regional_time_alignment',RUN/'operations/checks.py')
    freeze()
    if not (RUN/'lm_archive/AUDIT.json').exists():
        with (RUN/'operations/prepare_lm.log').open('a') as log:
            archive=subprocess.Popen([sys.executable,'-u',str(RUN/'code/prepare_lm.py')],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        atomic_json(RUN/'operations/LM_PID.json',dict(pid=archive.pid))
    if not (RUN/'LAUNCH_STATUS.json').exists():
        execute('deploying_two_v100_hosts_and_local_4090',RUN/'operations/launch.py')
    # Bootstrap verifies every transferred data hash and tests GPU arithmetic before training.
    state('waiting_for_all_host_bootstrap')
    from launch import remote,REMOTE
    while True:
        statuses={}
        for host in ['v1006001','v1006002']:
            code=f"from pathlib import Path;import json,time;p=Path({(REMOTE+'/operations/GPU_FIXTURE.json')!r});q=Path({(REMOTE+'/schedulers/status.json')!r});print(json.dumps(dict(fixture=p.exists(),scheduler=q.exists() and time.time()-q.stat().st_mtime<120)))"
            statuses[host]=json.loads(remote(host,code))
        local_status=RUN/'schedulers/status.json'
        statuses['local4090']=dict(fixture=(RUN/'operations/GPU_FIXTURE.json').exists(),scheduler=local_status.exists() and time.time()-local_status.stat().st_mtime<120)
        state('bootstrap_progress',hosts=statuses)
        if all(x['fixture'] and x['scheduler'] for x in statuses.values()):break
        time.sleep(30)
    log=(RUN/'operations/supervise.log').open('a')
    supervisor=subprocess.Popen([sys.executable,'-u',str(RUN/'operations/supervise.py')],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    state('training_supervised',supervisor_pid=supervisor.pid,planned=180,old_weights_reused=False)
    atomic_json(RUN/'MIGRATION_READY.json',dict(timestamp=now(),passed=True,preprocess_fingerprint=json.loads(ready.read_text())['preprocess_fingerprint'],training_hosts=['v1006001','v1006002','local4090']))
    with (RUN/'operations/followup.log').open('a') as log:
        child=subprocess.Popen([sys.executable,'-u',str(RUN/'operations/followup.py')],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    atomic_json(RUN/'operations/FOLLOWUP_PID.json',dict(pid=child.pid))


if __name__=='__main__':
    try:run()
    except Exception:
        state('failed',error=traceback.format_exc())
        raise
