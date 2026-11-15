"""Migrate paused leases with checkpoint hashes; change operations only."""
from pathlib import Path
import subprocess,json,time,os,signal,hashlib,datetime
P=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
NAMES=['daily_protocol_restart_20261114_v1','daily_ablation_restart_20261114_v1']
RBASE='/home/liying/wind_physics_project/project/runs';events=[]
def remote(host,code):
 p=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=8',host,'python3 -'],input=code,text=True,capture_output=True,timeout=60)
 if p.returncode:raise RuntimeError(host+':'+p.stderr)
 return json.loads(p.stdout)
def signal_script(script,sig=signal.SIGTERM):
 pids=[]
 for p in Path('/proc').iterdir():
  if not p.name.isdigit():continue
  try:
   if str(script).encode() in (p/'cmdline').read_bytes().split(b'\0'):os.kill(int(p.name),sig);pids.append(int(p.name))
  except (OSError,PermissionError):pass
 return pids
# Stop watchdog temporarily so it cannot recreate a supervisor mid-migration.
ab=P/'runs'/NAMES[1];print('watch',signal_script(ab/'operations/watch.py'),flush=True)
for name in NAMES:
 r=P/'runs'/name
 print('supervisor',name,signal_script(r/'operations/supervise.py'),flush=True)
 print('coordinator',name,signal_script(r/'operations/coordinator.py'),flush=True)
# All 6002 workers are paused. Verify, then stop scheduler and clear its lease.
for name in NAMES:
 print(remote('v1006002',f'''from pathlib import Path
import os,signal,json
r=Path({(RBASE+'/'+name)!r});matches=[]
for p in Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try:
  parts=(p/'cmdline').read_bytes().split(b'\\0')
  if str(r/'code/train.py').encode() in parts:matches.append(int(p.name))
  if str(r/'operations/scheduler.py').encode() in parts:os.kill(int(p.name),signal.SIGTERM)
 except OSError:pass
assert not matches,('6002 has resumed workers; stop and retry safely',matches)
(r/'operations/LEASE.json').write_text(json.dumps(dict(jobs=[])))
print(json.dumps(dict(run=r.name,paused_verified=True)))'''),flush=True)
# Drain only the 6001 supplementary scheduler to change its GPU pool; it saves RNG/optimizer.
for name in NAMES:
 print(remote('v1006001',f'''from pathlib import Path
import os,signal,json
r=Path({(RBASE+'/'+name)!r});pids=[]
for p in Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try:
  if str(r/'operations/scheduler.py').encode() in (p/'cmdline').read_bytes().split(b'\\0'):os.kill(int(p.name),signal.SIGTERM);pids.append(int(p.name))
 except OSError:pass
print(json.dumps(dict(run=r.name,draining=pids)))'''),flush=True)
for i in range(24):
 active=remote('v1006001',f'''from pathlib import Path
import json
roots={[RBASE+'/'+n for n in NAMES]!r};ids=[]
for p in Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try:
  parts=(p/'cmdline').read_bytes().split(b'\\0')
  if any((r+'/code/train.py').encode() in parts or (r+'/operations/scheduler.py').encode() in parts for r in roots):ids.append(int(p.name))
 except OSError:pass
print(json.dumps(ids))''')
 if not active:break
 print('waiting for checkpoints',active,flush=True);time.sleep(5)
else:raise RuntimeError('6001 draining did not finish')
for name in NAMES:
 r=P/'runs'/name;rr=RBASE+'/'+name
 # Existing coordinator file descriptors are gone; migrate exact job state.
 markers=[p for p in (r/'jobs').glob('*/REMOTE.json') if json.loads(p.read_text())['host']=='v1006002']
 for marker in markers:
  jid=marker.parent.name;local=marker.parent
  meta=remote('v1006002',f'''from pathlib import Path
import hashlib,json,fcntl
p=Path({(rr+'/jobs/'+jid)!r});p.mkdir(exist_ok=True);lock=(p/'worker.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert not (p/'SUCCESS.json').exists() and not (p/'FAILURE.json').exists()
files={{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in p.iterdir() if f.is_file() and f.name in ['latest.pt','best.pt','spec.json','history.json']}}
print(json.dumps(files))''')
  subprocess.run(['rsync','-a','--exclude=worker.lock','--exclude=REMOTE.json','--exclude=YIELD_REQUEST.json','v1006002:'+rr+'/jobs/'+jid+'/',str(local)+'/'],check=True,timeout=120,stdout=subprocess.DEVNULL)
  for f,h in meta.items():assert hashlib.sha256((local/f).read_bytes()).hexdigest()==h
  remote('v1006001',f"from pathlib import Path;import json;p=Path({(rr+'/jobs/'+jid)!r});p.mkdir(parents=True,exist_ok=True);print('true')")
  subprocess.run(['rsync','-a','--exclude=worker.lock','--exclude=REMOTE.json','--exclude=YIELD_REQUEST.json',str(local)+'/', 'v1006001:'+rr+'/jobs/'+jid+'/'],check=True,timeout=120,stdout=subprocess.DEVNULL)
  remote('v1006001',f'''from pathlib import Path
import hashlib,json
p=Path({(rr+'/jobs/'+jid)!r});files={meta!r}
for f,h in files.items():assert hashlib.sha256((p/f).read_bytes()).hexdigest()==h
print('true')''')
  marker.write_text(json.dumps(dict(host='v1006001',timestamp=datetime.datetime.now().astimezone().isoformat(),migrated_from='v1006002',checkpoint_hashes=meta),indent=2)+'\n')
  events.append(dict(run=name,job=jid,source='v1006002',target='v1006001',files=meta));print('migrated',name,jid,flush=True)
 # Only assign new remote leases when this host has running work or an eligible GPU.
 cp=r/'operations/coordinator.py';s=cp.read_text();backup=OUT/name;backup.mkdir(exist_ok=True)
 (backup/'coordinator.py.before').write_text(s)
 old="target=min(96,len(active)+max(8,len(active)//2)) if state['fresh'] else 0"
 new="can_admit=any(d.get('launch') for d in state['status'].get('decisions',{}).values())\n                target=min(96,len(active)+max(8,len(active)//2)) if state['fresh'] and (active or can_admit) else 0"
 assert old in s,name;cp.write_text(s.replace(old,new))
 # Prioritize finishing shared jobs on2cards; use5 idle6001cards for large ablations.
 pools=json.loads((r/'operations/RESOURCE_POOLS.json').read_text());pools['v1006001']=[1,2,3] if name==NAMES[0] else [0,4,5,6,7]
 (backup/'RESOURCE_POOLS.before.json').write_bytes((r/'operations/RESOURCE_POOLS.json').read_bytes())
 (r/'operations/RESOURCE_POOLS.json').write_text(json.dumps(pools,indent=2)+'\n')
 for fn in ['SCIENCE_FREEZE.json','PLAN_FREEZE.json']:
  fp=r/fn
  if not fp.exists():continue
  f=json.loads(fp.read_text());(backup/(fn+'.before')).write_bytes(fp.read_bytes())
  for rel in ['operations/coordinator.py','operations/RESOURCE_POOLS.json']:
   f['files'][rel]=hashlib.sha256((r/rel).read_bytes()).hexdigest()
  f['resource_recovery_20261115']=dict(timestamp=datetime.datetime.now().astimezone().isoformat(),science_unchanged=True,reason='migrate paused checkpoints off unavailable6002; lend idle6001 GPUs to ablations')
  fp.write_text(json.dumps(f,ensure_ascii=False,indent=2)+'\n')
 for host in ['v1006001','v1006002']:
  for f in ['operations/coordinator.py','operations/RESOURCE_POOLS.json','SCIENCE_FREEZE.json']:
   subprocess.run(['rsync','-a',str(r/f),host+':'+rr+'/'+str(Path(f).parent)+'/'],check=True,timeout=60,stdout=subprocess.DEVNULL)
# Personalization will inherit the same nonoverlapping pool as its shared sources.
r=P/'runs/daily_site_finetune_20261114_v1';p=r/'operations/RESOURCE_POOLS.json';d=json.loads(p.read_text());d['v1006001']=[1,2,3];p.write_text(json.dumps(d,indent=2)+'\n')
env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONNOUSERSITE='1');py=str(P/'.runtime/chronos/bin/python');controllers=[]
for name in NAMES:
 r=P/'runs'/name;logroot=Path('/shared_data_ssd3/liying')/name/'control_health'
 with (logroot/'supervise.log').open('a') as log:c=subprocess.Popen([py,'-u',str(r/'operations/supervise.py')],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 launch=r/'LAUNCH_STATUS.json';v=json.loads(launch.read_text());v['supervisor_pid']=c.pid;launch.write_text(json.dumps(v,indent=2)+'\n');controllers.append(dict(run=name,pid=c.pid))
with (Path('/shared_data_ssd3/liying')/NAMES[1]/'control_health/watch.log').open('a') as log:w=subprocess.Popen([py,'-u',str(ab/'operations/watch.py')],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
(ab/'operations/WATCH_PID.json').write_text(json.dumps(dict(pid=w.pid,script=str(ab/'operations/watch.py')),indent=2)+'\n')
(OUT/'RECOVERY.json').write_text(json.dumps(dict(timestamp=datetime.datetime.now().astimezone().isoformat(),passed=True,migrations=events,controllers=controllers,watcher=w.pid,scientific_spec_and_training_code_unchanged=True),indent=2)+'\n')
print('RECOVERED',len(events),'paused leases',controllers,flush=True)
