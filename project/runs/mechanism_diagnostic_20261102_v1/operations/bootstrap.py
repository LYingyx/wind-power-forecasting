from pathlib import Path
import sys,os,json,time,subprocess,argparse
RUN=Path(__file__).resolve().parents[1];sys.path.insert(0,str(RUN/'code'))
from settings import *
from probe import probe
from gpu_policy import Policy,decide

def main(gpus,remote):
    for rel,digest in json.loads((RUN/'SCIENCE_FREEZE.json').read_text())['files'].items():assert sha(RUN/rel)==digest,rel
    for rel,digest in json.loads((RUN/'DATA_FREEZE.json').read_text())['files'].items():assert sha(DATA/rel)==digest,rel
    assert json.loads((RUN/'operations/CHECKS.json').read_text())['passed']
    samples={g:[] for g in gpus.split(',')};policy=Policy(allow_shared_compute=False)
    while True:
        if (RUN/'COMPLETE.json').exists() or (RUN/'operations/STOP.json').exists():return
        snapshot=probe();chosen=None;decisions={}
        for g in samples:
            samples[g].append(snapshot[g]);samples[g]=samples[g][-3:]
            d=decide(samples[g],0,0,1000,2048,policy);decisions[g]=d
            if chosen is None and d['launch']:chosen=g
        atomic_json(RUN/'operations/BOOTSTRAP_STATUS.json',dict(timestamp=now(),stage='waiting_for_spare_compute_and_memory',chosen_gpu=chosen,decisions=decisions,pid=os.getpid()))
        if chosen is not None:break
        time.sleep(10)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=chosen)
    subprocess.run([sys.executable,str(RUN/'operations/checks.py'),'--device','cuda:0'],env=env,check=True)
    for jid in ['C_REAL_s00_r42','I_MUL_s00_r42']:
        subprocess.run([sys.executable,str(RUN/'code/train.py'),'--job-id',jid,'--smoke'],env=env,check=True)
    atomic_json(RUN/'operations/BOOTSTRAP_STATUS.json',dict(timestamp=now(),stage='scheduler_started',gpu=chosen,pid=os.getpid()))
    os.execv(sys.executable,[sys.executable,'-u',str(RUN/'operations/scheduler.py'),'--gpus',gpus]+(['--remote'] if remote else []))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--gpus',required=True);p.add_argument('--remote',action='store_true');a=p.parse_args();main(a.gpus,a.remote)
