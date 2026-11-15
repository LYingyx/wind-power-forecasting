"""Restart this run's control processes; log independently of SSD2's quota."""
from pathlib import Path
import sys,os,json,time,subprocess,shlex,fcntl,traceback
RUN=Path(__file__).resolve().parents[1];sys.path.insert(0,str(RUN/'code'))
from settings import *
REMOTE='/home/liying/wind_physics_project/project/runs/'+RUN.name
LOGROOT=Path('/shared_data_ssd3/liying')/RUN.name/'control_health'
RPY='/home/liying/wind_physics_project/project/.runtime/chronos/bin/python'

def ensure_script(root,python,name,args):
    return f'''from pathlib import Path
import subprocess,os,json
r=Path({root!r});script=str(r/'operations'/{(name+'.py')!r});pids=[]
for p in Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try:
  parts=(p/'cmdline').read_bytes().split(b'\\0')
  if script.encode() in parts:pids.append(int(p.name))
 except (OSError,PermissionError):pass
result=dict(process={name!r},pids=pids,restarted=False)
if not pids and not (r/'operations/STOP.json').exists() and not (r/'COMPLETE.json').exists():
 status=r/'schedulers/status.json'
 blocked=False
 if {name!r}=='scheduler' and status.exists():
  try:blocked=bool(json.loads(status.read_text()).get('circuit_breaker'))
  except (OSError,ValueError):pass
 if blocked:result['blocked']='worker_failure_circuit_breaker'
 else:
  log=(r/'operations'/{(name+'.log')!r}).open('a');env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONNOUSERSITE='1')
  child=subprocess.Popen([{python!r},'-u',script,*{args!r}],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  result.update(pids=[child.pid],restarted=True)
print(json.dumps(result))'''

def run():
    LOGROOT.mkdir(parents=True,exist_ok=True);lock=(LOGROOT/'supervisor.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    while not (RUN/'COMPLETE.json').exists() and not (RUN/'operations/STOP.json').exists():
        records=[]
        for host in ['v1006001','v1006002','local']:
            todo=['coordinator','scheduler'] if host=='local' else ['scheduler']
            for name in todo:
                pools=json.loads((RUN/'operations/RESOURCE_POOLS.json').read_text())
                ids=pools[host]
                if name=='scheduler' and not ids:continue
                args=['--gpus',','.join(map(str,ids))]+([] if host=='local' else ['--remote']) if name=='scheduler' else []
                script=ensure_script(str(RUN) if host=='local' else REMOTE,sys.executable if host=='local' else RPY,name,args)
                cmd=[sys.executable,'-c',script] if host=='local' else ['ssh','-o','BatchMode=yes','-o','ConnectTimeout=8',host,'python3 -c '+shlex.quote(script)]
                try:
                    result=subprocess.check_output(cmd,text=True,timeout=40,stderr=subprocess.STDOUT);records.append(dict(host=host,**json.loads(result)))
                except Exception as exc:records.append(dict(host=host,process=name,error=repr(exc)))
        status=dict(timestamp=now(),supervisor_pid=os.getpid(),processes=records)
        atomic_json(LOGROOT/'HEALTH.json',status)
        try:atomic_json(RUN/'operations/CONTROL_HEALTH.json',status)
        except OSError as exc:print(now(),'metadata write error',repr(exc),flush=True)
        if any(x.get('restarted') or x.get('error') for x in records):print(json.dumps(status),flush=True)
        time.sleep(45)
    atomic_json(LOGROOT/'SUPERVISOR_EXIT.json',dict(timestamp=now(),reason='complete_or_requested_stop'))
if __name__=='__main__':run()
