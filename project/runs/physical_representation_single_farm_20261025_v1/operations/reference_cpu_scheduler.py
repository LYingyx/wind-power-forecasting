"""Small CPU-only replay pool; shares per-job locks with the GPU dispatcher."""
from pathlib import Path
import os,sys,json,time,fcntl,subprocess,socket,argparse,traceback
RUN=Path(__file__).resolve().parents[1];sys.path.insert(0,str(RUN/'code'))
from settings import JOBS,atomic_json,now

def worker(jid):
    from diagnose_references import run
    spec=next(s for s in json.loads((RUN/'manifest.json').read_text()) if s['id']==jid)
    assert spec['task']=='diagnose_reference'
    out=JOBS/jid
    atomic_json(out/'runtime.json',dict(host=socket.gethostname(),pid=os.getpid(),device='cpu',started=now(),wrapper='reference_cpu',threads=1))
    os.nice(10)
    try:run(spec,'cpu')
    except BaseException:
        atomic_json(out/'FAILURE.json',dict(timestamp=now(),host=socket.gethostname(),traceback=traceback.format_exc()));raise

def schedule():
    from resource_scheduler import validate,cpu_memory_available,status_collect
    validate();folder=RUN/'schedulers';folder.mkdir(exist_ok=True)
    lock=(folder/'reference_cpu.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    specs=[s for s in json.loads((RUN/'manifest.json').read_text()) if s['task']=='diagnose_reference']
    running={};failed=0
    while True:
        for jid,item in list(running.items()):
            if item[0].poll() is not None:
                if item[0].returncode:failed+=1
                item[1].close();item[2].close();del running[jid]
        free,total=cpu_memory_available()
        room=os.getloadavg()[0]<os.cpu_count()*.75 and free>max(8192,total*.15)+3072
        if room and failed<2:
            for spec in specs:
                if len(running)>=4:break
                jid=spec['id'];out=JOBS/jid
                if jid in running or any((out/n).exists() for n in ['SUCCESS.json','FAILURE.json']):continue
                out.mkdir(exist_ok=True);claim=(out/'worker.lock').open('a+')
                try:fcntl.flock(claim,fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError:claim.close();continue
                if any((out/n).exists() for n in ['SUCCESS.json','FAILURE.json']):claim.close();continue
                log=(out/'worker.log').open('a')
                env=dict(os.environ,PYTHONNOUSERSITE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
                child=subprocess.Popen([sys.executable,'-s',__file__,'--job-id',jid],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,pass_fds=(claim.fileno(),))
                running[jid]=(child,claim,log)
        active=[dict(id=j,pid=t[0].pid,gpu='cpu',host=socket.gethostname()) for j,t in running.items()]
        atomic_json(folder/'reference_cpu.json',dict(timestamp=now(),pid=os.getpid(),enabled=True,active=active,maximum_workers=4,failed=failed,room=room))
        status_collect()
        done=all(any((JOBS/s['id']/n).exists() for n in ['SUCCESS.json','FAILURE.json']) for s in specs)
        if not running and (done or failed>=2):break
        time.sleep(20)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--job-id');a=p.parse_args()
    worker(a.job_id) if a.job_id else schedule()
