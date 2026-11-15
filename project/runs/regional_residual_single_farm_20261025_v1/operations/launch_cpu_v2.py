"""Deploy the new CPU routing; adopt in-flight local work without restarting it."""
from pathlib import Path
import sys,os,json,signal,subprocess,shlex,time
RUN=Path(__file__).resolve().parents[1];sys.path.insert(0,str(RUN/'code'))
from settings import atomic_json,now,sha
from cpu_scheduler_v2 import proc
REMOTE='/home/liying/wind_physics_project/project/runs/'+RUN.name
PYTHON='/home/liying/wind_physics_project/project/.runtime/chronos/bin/python3.11'

def main():
    assert not (RUN/'operations/CPU_V2_LAUNCH.json').exists(),'Already launched; inspect before restarting'
    checks=json.loads((RUN/'operations/CPU_SCHEDULING_CHECKS.json').read_text());assert checks['passed']
    for path,h in json.loads((RUN/'SOURCE_FREEZE.json').read_text())['files'].items():assert sha(path)==h
    remotes={}
    for host,maximum,refs in [('v1006001',56,6),('v1006002',20,4)]:
        args=[PYTHON,'-s',REMOTE+'/operations/cpu_scheduler_v2.py','--max-workers',str(maximum),'--reference-slots',str(refs)]
        script=f'''from pathlib import Path
import subprocess,os,json,datetime
r=Path({REMOTE!r});assert json.loads((r/'operations/CPU_RUNTIME_READY.json').read_text())['passed']
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONNOUSERSITE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
with (r/'operations/cpu_scheduler_v2.log').open('a') as f:p=subprocess.Popen({args!r},env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps(dict(pid=p.pid,args={args!r},timestamp=datetime.datetime.now().astimezone().isoformat())))'''
        reply=subprocess.check_output(['ssh','-o','BatchMode=yes',host,'python3 -c '+shlex.quote(script)],text=True,timeout=30);remotes[host]=json.loads(reply)
    # Stop only the old dispatcher; its workers retain their inherited job locks.
    old=json.loads((RUN/'schedulers/reference_cpu.json').read_text());pid=old['pid'];cmd=Path(f'/proc/{pid}/cmdline').read_bytes().decode().replace('\0',' ')
    assert str(RUN/'operations/reference_cpu_scheduler.py') in cmd and '--job-id' not in cmd
    os.kill(pid,signal.SIGTERM)
    deadline=time.monotonic()+5
    while proc(pid) is not None and proc(pid)['state'] not in ['Z','X'] and time.monotonic()<deadline:time.sleep(.05)
    state=proc(pid);assert state is None or state['state'] in ['Z','X']
    active=[]
    for line in subprocess.check_output(['ps','-eo','pid,args'],text=True).splitlines():
        fields=line.strip().split(None,1)
        if len(fields)!=2:continue
        if str(RUN/'operations/reference_cpu_scheduler.py') not in fields[1] or '--job-id ' not in fields[1]:continue
        child=int(fields[0]);jid=fields[1].split('--job-id ',1)[1].split()[0];st=proc(child)
        if not st or st['state'] in ['Z','X']:continue
        os.setpriority(os.PRIO_PROCESS,child,15);active.append(dict(id=jid,pid=child,start=st['start']))
    atomic_json(RUN/'operations/CPU_ADOPT.json',dict(timestamp=now(),previous_dispatcher_pid=pid,active=active,restarted_workers=False))
    atomic_json(RUN/'schedulers/reference_cpu.json',dict(timestamp=now(),pid=pid,enabled=False,active=[],superseded_by='cpu_scheduler_v2',adopted_workers=active))
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONNOUSERSITE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1');local={}
    for name,extra in [('cpu_scheduler_v2',['--max-workers','16','--reference-slots','4']),('cpu_remote_controller',[])]:
        args=[sys.executable,'-s',str(RUN/'operations'/(name+'.py')),*extra]
        with (RUN/'operations'/(name+'.log')).open('a') as f:p=subprocess.Popen(args,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        local[name]=dict(pid=p.pid,args=args,timestamp=now())
    atomic_json(RUN/'operations/CPU_V2_LAUNCH.json',dict(timestamp=now(),remote=remotes,local=local,adopted=active,gpu_dispatchers_unchanged=True))
    print(json.dumps(dict(remote=remotes,local=local,adopted=active),indent=2))
if __name__=='__main__':main()
