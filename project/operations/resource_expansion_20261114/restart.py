"""Cooperatively reload this run's schedulers; never signal unrelated workers."""
from pathlib import Path
import json,subprocess,concurrent.futures,os,signal,time,hashlib
P=Path(__file__).resolve().parents[2];R=P/'runs/daily_protocol_restart_20261114_v1';X=P/'operations/resource_expansion_20261114';remote_root='/home/liying/wind_physics_project/project/runs/'+R.name

def remote(h,c):
 p=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=8',h,'python3 -'],input=c,text=True,capture_output=True,timeout=60)
 if p.returncode:raise RuntimeError(h+p.stderr)
 return json.loads(p.stdout)
def local_signal(pid,script):
 path=Path(f'/proc/{pid}/cmdline')
 if path.exists() and str(script).encode() in path.read_bytes().split(b'\0'):os.kill(pid,signal.SIGTERM);return True
 return False
# Stop only the controller being revised; current workers checkpoint via their scheduler.
health=json.loads((R/'operations/CONTROL_HEALTH.json').read_text());local_signal(health['supervisor_pid'],R/'operations/supervise.py')
A=P/'runs/daily_ablation_restart_20261114_v1';watch=json.loads((A/'operations/WATCH_PID.json').read_text());local_signal(watch['pid'],A/'operations/watch.py')
status=json.loads((R/'STATUS.json').read_text());(X/'BEFORE_STATUS.json').write_text(json.dumps(status,indent=2)+'\n')
state=json.loads((R/'schedulers/status.json').read_text());local_signal(state['pid'],R/'operations/scheduler.py')
for host in ['v1006001','v1006002']:
 print(host,remote(host,f'''from pathlib import Path
import json,os,signal
r=Path({remote_root!r});s=json.loads((r/'schedulers/status.json').read_text());pid=s['pid'];script=str(r/'operations/scheduler.py').encode();p=Path(f'/proc/{{pid}}/cmdline')
assert p.exists() and script in p.read_bytes().split(b'\\0')
os.kill(pid,signal.SIGTERM)
print(json.dumps(dict(scheduler=pid,active=len(s['active']),cooperative_drain=True)))'''),flush=True)
# Signal handlers save checkpoints, including optimizer/RNG, before releasing locks.
def alive(root):
 return f'''from pathlib import Path
import json
root={str(root)!r};result=[]
for p in Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try:
  parts=(p/'cmdline').read_bytes().split(b'\\0')
  if (root+'/operations/scheduler.py').encode() in parts or (root+'/code/train.py').encode() in parts:result.append(int(p.name))
 except OSError:pass
print(json.dumps(result))'''
for attempt in range(30):
 local=json.loads(subprocess.check_output(['python3','-c',alive(R)],text=True));remaining=dict(local=local)
 for h in ['v1006001','v1006002']:remaining[h]=remote(h,alive(remote_root))
 print('draining',remaining,flush=True)
 if not any(remaining.values()):break
 time.sleep(5)
else:raise RuntimeError('Workers still checkpointing; no forced termination')
files=['operations/gpu_policy.py','operations/efficiency_v3.py','operations/scheduler.py','operations/supervise.py','operations/LOCAL_ADMISSION.json','operations/REMOTE_ADMISSION.json','operations/RESOURCE_POOLS.json','SCIENCE_FREEZE.json']
for h in ['v1006001','v1006002']:
 for f in files:
  subprocess.run(['rsync','-a',str(R/f),h+':'+remote_root+'/'+str(Path(f).parent)+'/'],check=True,timeout=60,stdout=subprocess.DEVNULL)
 print(h,remote(h,f'''from pathlib import Path
import hashlib,json
r=Path({remote_root!r});f=json.loads((r/'SCIENCE_FREEZE.json').read_text())['files']
for rel,digest in f.items():assert hashlib.sha256((r/rel).read_bytes()).hexdigest()==digest,rel
print(json.dumps(dict(frozen_files_verified=len(f))))'''),flush=True)
logroot=Path('/shared_data_ssd3/liying')/R.name/'control_health';env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONNOUSERSITE='1');python=str(P/'.runtime/chronos/bin/python')
with (logroot/'supervise.log').open('a') as log:p=subprocess.Popen([python,'-u',str(R/'operations/supervise.py')],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
print('new shared supervisor',p.pid,flush=True)
# The supplemental watcher now starts whenever verified data is available.
logs=Path('/shared_data_ssd3/liying')/A.name/'control_health'
with (logs/'watch.log').open('a') as log:w=subprocess.Popen([python,'-u',str(A/'operations/watch.py')],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
(A/'operations/WATCH_PID.json').write_text(json.dumps(dict(pid=w.pid,script=str(A/'operations/watch.py')),indent=2)+'\n')
(X/'RESTART.json').write_text(json.dumps(dict(shared_supervisor_pid=p.pid,supplementary_watcher_pid=w.pid,cooperative_checkpoint_resume=True,science_unchanged=True),indent=2)+'\n')
print('new supplemental watcher',w.pid,flush=True)
