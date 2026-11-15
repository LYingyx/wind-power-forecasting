"""Operational restart with adoption; worker model fits remain in memory."""
from pathlib import Path
import json,os,sys,signal,subprocess,time
RUN=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(RUN/'code'));sys.path.insert(0,str(RUN/'operations'))
from relocate import activate
activate(RUN)
from settings import atomic_json,now
from cpu_scheduler_v2 import proc

def main(maximum,refs,replace_gpu=False,replace_boot=False):
    old=json.loads((RUN/'schedulers/cpu_scheduler_v2.json').read_text());pid=old['pid'];cmd=(Path('/proc')/str(pid)/'cmdline').read_bytes().decode();assert str(RUN/'operations/cpu_scheduler_v2.py') in cmd
    os.kill(pid,signal.SIGSTOP);os.kill(pid,signal.SIGKILL)
    for _ in range(40):
        p=proc(pid)
        if p is None or p['state'] in ['Z','X']:break
        time.sleep(.05)
    active=[]
    for line in subprocess.check_output(['ps','-eo','pid,args'],text=True).splitlines():
        parts=line.strip().split(None,1)
        if len(parts)!=2 or str(RUN/'operations') not in parts[1] or '--job-id ' not in parts[1]:continue
        if not any(name in parts[1] for name in ['cpu_scheduler_v2.py','reference_cpu_scheduler.py']):continue
        child=int(parts[0]);jid=parts[1].split('--job-id ',1)[1].split()[0];st=proc(child)
        if st and st['state'] not in ['Z','X']:active.append(dict(id=jid,pid=child,start=st['start']))
    prior=RUN/'operations/CPU_ADOPT.json'
    if prior.exists():(RUN/'operations/CPU_ADOPT_BEFORE_LIMITS.json').write_bytes(prior.read_bytes())
    atomic_json(prior,dict(timestamp=now(),previous_dispatcher_pid=pid,active=active,restarted_workers=False))
    env=dict(os.environ,PYTHONNOUSERSITE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1');launched={}
    args=[sys.executable,'-s',str(RUN/'operations/cpu_scheduler_v2.py'),'--max-workers',str(maximum),'--reference-slots',str(refs)]
    with (RUN/'operations/cpu_scheduler_v2.log').open('a') as f:p=subprocess.Popen(args,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True,env=dict(env,CUDA_VISIBLE_DEVICES=''))
    launched['cpu']=p.pid
    if replace_gpu:
        for path in (RUN/'schedulers').glob('*.json'):
            d=json.loads(path.read_text())
            if 'gpus' not in d:continue
            oldpid=d['pid'];st=proc(oldpid)
            if not st or st['state'] in ['Z','X']:continue
            cmd=(Path('/proc')/str(oldpid)/'cmdline').read_bytes().decode();assert str(RUN) in cmd and 'resource_scheduler.py' in cmd
            os.kill(oldpid,signal.SIGKILL)
        args=[sys.executable,'-s',str(RUN/'operations/gpu_scheduler_limits.py'),'--warmup-seconds','20']
        with (RUN/'operations/gpu_scheduler_limits.log').open('a') as f:p=subprocess.Popen(args,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True,env=env)
        launched['gpu']=p.pid
    if replace_boot:
        boot=json.loads((RUN/'operations/BOOTSTRAP.json').read_text());oldpid=boot['pid'];cmd=(Path('/proc')/str(oldpid)/'cmdline').read_bytes().decode();assert str(RUN) in cmd and 'bootstrap_remote.py' in cmd and boot['state']=='waiting_for_gpu_headroom'
        os.kill(oldpid,signal.SIGTERM)
        args=[sys.executable,'-s',str(RUN/'operations/bootstrap_limits.py'),'--warmup-seconds','20']
        with (RUN/'operations/bootstrap_limits.log').open('a') as f:p=subprocess.Popen(args,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True,env=env)
        launched['bootstrap']=p.pid
    atomic_json(RUN/'operations/CGROUP_RESTART.json',dict(timestamp=now(),launched=launched,adopted=active))
    print(json.dumps(dict(launched=launched,adopted=len(active))))
if __name__=='__main__':main(int(sys.argv[1]),int(sys.argv[2]),'--gpu' in sys.argv,'--bootstrap' in sys.argv)
