from pathlib import Path
import sys,json,os,time,subprocess,argparse
RUN=Path(__file__).resolve().parents[1];sys.path.insert(0,str(RUN/'code'))
from settings import atomic_json,now,sha
from probe import probe
from gpu_policy import Policy,decide
def main(gpus,local=False):
    for rel,digest in json.loads((RUN/'SCIENCE_FREEZE.json').read_text())['files'].items():assert sha(RUN/rel)==digest,rel
    for rel,digest in json.loads((RUN/'DATA_FREEZE.json').read_text())['files'].items():assert sha(RUN/'data'/rel)==digest,rel
    samples={str(g):[] for g in gpus.split(',')};policy=Policy(allow_shared_compute=local)
    while True:
        snapshots=probe();chosen=None
        for gpu in samples:
            samples[gpu].append(snapshots[gpu]);samples[gpu]=samples[gpu][-3:]
            decision=decide(samples[gpu],0,0,1000,2048,policy)
            memory_ready=snapshots[gpu]['free_mib']>=max(4096,int(snapshots[gpu]['total_mib']*.2))+2048
            if chosen is None and (decision['launch'] or (local and memory_ready)):chosen=gpu
        atomic_json(RUN/'operations/BOOTSTRAP_STATUS.json',dict(timestamp=now(),stage='waiting_for_gpu_checks',gpu=chosen))
        if chosen is not None:break
        time.sleep(10)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=chosen)
    subprocess.run([sys.executable,str(RUN/'operations/checks.py'),'--device','cuda:0'],check=True,env=env)
    # Local CPU smoke has its own directory; force a distinct GPU smoke id.
    jid='age12_E1_s00_r43' if local else 'age12_E1_s00_r42'
    subprocess.run([sys.executable,str(RUN/'code/train.py'),'--job-id',jid,'--smoke'],check=True,env=env)
    atomic_json(RUN/'operations/BOOTSTRAP_STATUS.json',dict(timestamp=now(),stage='checks_passed_starting_scheduler',gpu=chosen))
    os.execv(sys.executable,[sys.executable,'-u',str(RUN/'operations/scheduler.py'),'--gpus',gpus]+([] if local else ['--remote']))
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--gpus',default='0,1,2,3,4,5,6,7');ap.add_argument('--local',action='store_true');a=ap.parse_args();main(a.gpus,a.local)
