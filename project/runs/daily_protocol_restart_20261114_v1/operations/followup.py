"""Run the authorized daily personalization rerun after shared jobs finish."""
from pathlib import Path
import fcntl
import json
import os
import subprocess
import sys
import time
import traceback

RUN=Path(__file__).resolve().parents[1];PROJECT=RUN.parents[1]
sys.path.insert(0,str(RUN/'code'))
from settings import atomic_json,now
NEXT=PROJECT/'runs/daily_site_finetune_20261114_v1'


def status(stage,**x):
    atomic_json(RUN/'operations/FOLLOWUP_STATUS.json',dict(timestamp=now(),pid=os.getpid(),stage=stage,**x))


def call(script,*args,env=None):
    subprocess.run([sys.executable,'-u',str(script),*args],cwd=PROJECT,env=env,check=True)


def main():
    lock=(RUN/'operations/followup.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    while not (RUN/'COMPLETE.json').exists():
        status('waiting_for_shared_180_jobs');time.sleep(45)
    if (NEXT/'LAUNCH_STATUS.json').exists():status('personalization_already_launched');return
    status('preparing_daily_personalization')
    call(NEXT/'code/prepare_daily.py')
    call(NEXT/'operations/checks.py')
    status('waiting_for_local_gpu_fixture')
    from probe import probe
    while True:
        cards=probe();choices=[(int(g),x) for g,x in cards.items() if x['free_mib']>8192 and x['utilization']<80]
        if choices:break
        time.sleep(30)
    gpu=min(choices,key=lambda x:x[1]['utilization'])[0]
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu))
    call(NEXT/'operations/checks.py','--fixture',env=env)
    status('deploying_daily_personalization')
    call(NEXT/'operations/deploy.py')
    status('personalization_supervised',planned=1100)


if __name__=='__main__':
    try:main()
    except Exception:
        status('failed',error=traceback.format_exc());raise
