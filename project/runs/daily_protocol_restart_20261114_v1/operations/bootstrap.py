from pathlib import Path
import sys,json,time,os,subprocess,socket,argparse
RUN=Path(__file__).resolve().parents[1];sys.path.insert(0,str(RUN/'code'))
from settings import *
from probe import probe
from gpu_policy import Policy,decide

def main(gpu_ids,remote=False):
    for rel,digest in json.loads((RUN/'SCIENCE_FREEZE.json').read_text())['files'].items():assert sha(RUN/rel)==digest,rel
    for rel,digest in json.loads((RUN/'DATA_FREEZE.json').read_text())['files'].items():assert sha(RUN/rel)==digest,rel
    atomic_json(RUN/'operations/DATA_AUDIT.json',dict(timestamp=now(),passed=True,host=socket.gethostname()))
    admission_path=RUN/('operations/REMOTE_ADMISSION.json' if remote else 'operations/LOCAL_ADMISSION.json')
    admission=json.loads(admission_path.read_text()) if admission_path.exists() else {}
    samples={g:[] for g in gpu_ids};policy=Policy(initial_worker_mib=3072,allow_shared_compute=bool(admission.get('allow_shared_compute',False)),total_compute_target=int(admission.get('total_compute_target',92)))
    while True:
        if (RUN/'operations/STOP.json').exists():return
        gpu=probe();chosen=None
        for g in gpu_ids:
            samples[g]=(samples[g]+[gpu[g]])[-3:];decision=decide(samples[g],0,0,1000,3072,policy)
            if decision['launch']:chosen=g;break
        atomic_json(RUN/'operations/BOOTSTRAP_STATUS.json',dict(timestamp=now(),host=socket.gethostname(),stage='GPU_fixture' if chosen else 'waiting_for_free_compute',gpu=chosen))
        if chosen is not None:
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=chosen,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONNOUSERSITE='1')
            subprocess.run([sys.executable,str(RUN/'operations/checks.py'),'--fixture'],env=env,check=True)
            # One training/validation epoch on a tiny sample exercises checkpoint writing.
            subprocess.run([sys.executable,str(RUN/'code/train.py'),'--job-id','TUNE_W0_R3_c0_r40','--smoke'],env=env,check=True)
            atomic_json(RUN/'operations/BOOTSTRAP_STATUS.json',dict(timestamp=now(),host=socket.gethostname(),stage='scheduler',gpu=chosen))
            args=[sys.executable,'-u',str(RUN/'operations/scheduler.py'),'--gpus',','.join(gpu_ids)]+(['--remote'] if remote else [])
            os.execv(sys.executable,args)
        time.sleep(10)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--gpus',required=True);p.add_argument('--remote',action='store_true');a=p.parse_args();main(a.gpus.split(','),a.remote)
