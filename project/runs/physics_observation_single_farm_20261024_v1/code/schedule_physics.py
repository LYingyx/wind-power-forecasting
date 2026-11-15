"""One coordinator per host, shared advisory locks, never crowd another user's GPU."""
from settings import *
import argparse,subprocess,time,fcntl,socket,sys


def gpu_state():
    output=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    states={}
    for line in output.splitlines():
        index,uuid,free,util=[x.strip() for x in line.split(',')]
        states[index]=dict(uuid=uuid,free_mib=int(free),utilization=int(util),foreign_pids=[])
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True)
    byuuid={s['uuid']:s for s in states.values()}
    for line in apps.splitlines():
        if ',' not in line:continue
        uuid,pid=[x.strip() for x in line.split(',')]
        try:owner=Path('/proc',pid).stat().st_uid
        except OSError:owner=-1
        if owner!=os.getuid() and uuid in byuuid:byuuid[uuid]['foreign_pids'].append(int(pid))
    return states


def validate():
    checks=json.loads((RUN/'CHECKS.json').read_text());assert checks['passed']
    smoke=json.loads((RUN/'SMOKE_CHECK.json').read_text());assert smoke['passed']
    freeze=json.loads((RUN/'SOURCE_FREEZE.json').read_text())
    for path,expected in freeze['files'].items():assert sha(Path(path))==expected, 'Source changed: '+path
    assert (RUN/'analysis/DIAGNOSTICS_COMPLETE.json').exists()


def run(max_workers=4):
    setup();host=socket.gethostname();folder=RUN/'schedulers';folder.mkdir(exist_ok=True)
    lock=(folder/(host+'.lock')).open('a+')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise SystemExit('Scheduler already running for '+host)
    validate();manifest=json.loads((RUN/'manifest.json').read_text());running={};idle={};failed_on_host=[]
    while True:
        for jid,item in list(running.items()):
            if item['process'].poll() is not None:
                if item['process'].returncode!=0:failed_on_host.append(jid)
                item['lock'].close();item['log'].close();del running[jid]
        states=gpu_state();done=failed=0
        for spec in manifest:
            out=JOBS/spec['id'];done+=(out/'SUCCESS.json').exists();failed+=(out/'FAILURE.json').exists() and not (out/'SUCCESS.json').exists()
        for gpu,state in states.items():
            if len(failed_on_host)>=2:break
            own=sum(x['gpu']==gpu for x in running.values())
            clear=not state['foreign_pids'] and state['free_mib']>=8192 and (state['utilization']<=35 or own>0)
            idle[gpu]=idle.get(gpu,0)+1 if clear else 0
            if idle[gpu]<2:continue
            for spec in manifest:
                if own>=max_workers or state['free_mib']-own*1024<4096:break
                jid=spec['id'];out=JOBS/jid
                if jid in running or (out/'SUCCESS.json').exists() or (out/'FAILURE.json').exists():continue
                out.mkdir(exist_ok=True);claim=(out/'worker.lock').open('a+')
                try:fcntl.flock(claim,fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError:claim.close();continue
                if (out/'SUCCESS.json').exists():claim.close();continue
                log=(out/'worker.log').open('a');env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
                process=subprocess.Popen([sys.executable,str(RUN/'code/train_physics.py'),'--job-id',jid],env=env,stdout=log,stderr=subprocess.STDOUT,pass_fds=(claim.fileno(),))
                running[jid]=dict(process=process,lock=claim,log=log,gpu=gpu);own+=1
                print(now(),'started',jid,'gpu',gpu,flush=True)
        active=[dict(id=jid,pid=x['process'].pid,gpu=x['gpu'],host=host) for jid,x in running.items()]
        atomic_json(folder/(host+'.json'),dict(timestamp=now(),pid=os.getpid(),host=host,active=active,gpus=states,max_workers_per_gpu=max_workers,
            policy='two scans free>=8GiB, no foreign compute process; idle utilization<=35% before first worker',waiting_for_resources=not active,failed_on_host=failed_on_host,circuit_breaker=len(failed_on_host)>=2))
        from collect_physics import collect
        status=collect()
        if (done+failed>=len(manifest) or len(failed_on_host)>=2) and not running:break
        time.sleep(30)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--max-workers-per-gpu',type=int,default=4);ap.add_argument('--probe',action='store_true');a=ap.parse_args()
    if a.probe:print(json.dumps(gpu_state()))
    else:run(a.max_workers_per_gpu)
