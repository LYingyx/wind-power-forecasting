"""Wait for measured GPU headroom, verify the runtime, then enter the scheduler."""
from pathlib import Path
import argparse,collections,fcntl,json,os,subprocess,sys,time
from resource_scheduler import RUN,probe,validate
from gpu_policy import Policy,decide
from settings import atomic_json,now


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--allow-shared-compute',action='store_true');parser.add_argument('--warmup-seconds',type=int,default=40);args=parser.parse_args()
    folder=RUN/'operations'
    lock=(folder/'bootstrap.lock').open('a+')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise SystemExit('Bootstrap already active')
    samples=collections.defaultdict(list);policy=Policy(allow_shared_compute=args.allow_shared_compute,warmup_seconds=args.warmup_seconds)
    check=folder/'GPU_COMPATIBILITY.json'
    while not check.exists() or not json.loads(check.read_text()).get('passed'):
        from settings import JOBS
        if (folder/'GLOBAL_COMPLETE.json').exists() or all((JOBS/s['id']/'SUCCESS.json').exists() for s in json.loads((RUN/'manifest.json').read_text())):return
        decisions={};chosen=None
        for gpu,snapshot in probe().items():
            samples[gpu].append(snapshot);samples[gpu]=samples[gpu][-policy.idle_samples:]
            decisions[gpu]=decide(samples[gpu],0,0,1000,policy.initial_worker_mib,policy)
            if chosen is None and decisions[gpu]['launch']:chosen=gpu
        atomic_json(folder/'BOOTSTRAP.json',dict(timestamp=now(),pid=os.getpid(),state='waiting_for_gpu_headroom',decisions=decisions,resource_policy=policy.__dict__))
        if chosen is not None:
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=chosen,PYTHONNOUSERSITE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
            subprocess.run([sys.executable,'-s',str(folder/'gpu_compatibility.py')],env=env,check=True)
        else:time.sleep(policy.poll_seconds)
    validate()
    atomic_json(folder/'BOOTSTRAP.json',dict(timestamp=now(),pid=os.getpid(),state='verified_starting_scheduler',resource_policy=policy.__dict__))
    command=[sys.executable,'-s',str(folder/'resource_scheduler.py'),'--warmup-seconds',str(args.warmup_seconds)]
    if args.allow_shared_compute:command.append('--allow-shared-compute')
    os.execv(sys.executable,command)


if __name__=='__main__':main()
