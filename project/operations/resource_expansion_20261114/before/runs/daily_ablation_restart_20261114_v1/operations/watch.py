"""Persistently queue all missing ablations after the already authorized stages."""
from pathlib import Path
import sys,os,fcntl,time,subprocess,traceback
RUN=Path(__file__).resolve().parents[1];sys.path.insert(0,str(RUN/'code'))
from settings import *

def main():
    logroot=Path('/shared_data_ssd3/liying')/RUN.name/'control_health';logroot.mkdir(parents=True,exist_ok=True)
    lock=(logroot/'watch.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    dependencies=[PROJECT/'runs'/name for name in ['daily_protocol_restart_20261114_v1','daily_site_finetune_20261114_v1']]
    while not (RUN/'COMPLETE.json').exists():
        try:
            waiting=[p.name for p in dependencies if not (p/'COMPLETE.json').exists()]
            if waiting:stage='waiting_for_prior_stages'
            elif not (RUN/'LAUNCH_STATUS.json').exists():
                stage='deploying';atomic_json(RUN/'operations/QUEUE_STATUS.json',dict(timestamp=now(),pid=os.getpid(),stage=stage,planned=1291))
                subprocess.run([sys.executable,'-u',str(RUN/'operations/deploy.py')],cwd=PROJECT,check=True)
                stage='training_supervised'
            else:
                stage='training_supervised';launch=json.loads((RUN/'LAUNCH_STATUS.json').read_text());pid=launch['supervisor_pid']
                parts=Path(f'/proc/{pid}/cmdline')
                alive=parts.exists() and str(RUN/'operations/supervise.py').encode() in parts.read_bytes().split(b'\0')
                if not alive:
                    with (logroot/'supervise.log').open('a') as log:p=subprocess.Popen([sys.executable,'-u',str(RUN/'operations/supervise.py')],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    launch['supervisor_pid']=p.pid;atomic_json(RUN/'LAUNCH_STATUS.json',launch)
            atomic_json(RUN/'operations/QUEUE_STATUS.json',dict(timestamp=now(),pid=os.getpid(),stage=stage,waiting_for=waiting,planned=1291,completed=0 if not (RUN/'STATUS.json').exists() else json.loads((RUN/'STATUS.json').read_text())['completed']))
        except Exception:
            atomic_json(RUN/'operations/QUEUE_STATUS.json',dict(timestamp=now(),pid=os.getpid(),stage='retrying_after_error',error=traceback.format_exc(),planned=1291));traceback.print_exc()
        time.sleep(45)
    atomic_json(RUN/'operations/QUEUE_STATUS.json',dict(timestamp=now(),pid=os.getpid(),stage='complete',planned=1291,completed=1291))
if __name__=='__main__':main()
