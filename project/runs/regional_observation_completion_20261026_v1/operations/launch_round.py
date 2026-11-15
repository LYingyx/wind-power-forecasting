"""Start explicitly authorized hosts and the central coordinator once."""
from pathlib import Path
import os,sys,subprocess,json,shlex
RUN=Path(__file__).resolve().parents[1];sys.path.insert(0,str(RUN/'code'))
from settings import atomic_json,now
assert (RUN/'operations/DEPLOYED.json').exists()
assert not (RUN/'operations/LAUNCH_REMOTES.json').exists(),'Already launched; inspect before restarting'
remote='/home/liying/wind_physics_project/project/runs/'+RUN.name
python='/home/liying/wind_physics_project/project/.runtime/chronos/bin/python3.11'
record={}
for host in ['v1006001','v1006002']:
    args=[python,'-s',remote+'/operations/bootstrap_limits.py','--warmup-seconds','20']
    # Latest request asks all hosts to yield to other workloads; use conservative admission.
    script=f'''import subprocess,os,json,datetime
from pathlib import Path
r=Path({remote!r});env=dict(os.environ,PYTHONNOUSERSITE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
with (r/'operations/bootstrap_remote.log').open('a') as f:
 p=subprocess.Popen({args!r},stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True,env=env)
print(json.dumps(dict(pid=p.pid,host={host!r},arguments={args!r},started=datetime.datetime.now().astimezone().isoformat())))'''
    response=subprocess.run(['ssh','-o','BatchMode=yes',host,'python3 -c '+shlex.quote(script)],capture_output=True,text=True,check=True,timeout=30)
    record[host]=json.loads(response.stdout)
atomic_json(RUN/'operations/LAUNCH_REMOTES.json',record)
local={};env=dict(os.environ,PYTHONNOUSERSITE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
for name,args in [('remote_controller',[]),('bootstrap_limits',['--warmup-seconds','20']),('finalize_when_done',[])]:
    cmd=[sys.executable,'-s',str(RUN/'operations'/(name+'.py')),*args]
    with (RUN/'operations'/(name+'.log')).open('a') as f:
        child=subprocess.Popen(cmd,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True,env=env)
    local[name]=dict(pid=child.pid,args=cmd,started=now())
atomic_json(RUN/'operations/LAUNCH_LOCAL.json',local)
print(json.dumps(dict(remote=record,local=local),indent=2))
