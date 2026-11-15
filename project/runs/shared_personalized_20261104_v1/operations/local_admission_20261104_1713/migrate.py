from pathlib import Path
import json,subprocess,shlex,time,hashlib,os,signal,datetime,py_compile
R=Path('/shared_data_ssd2/liying/wind_cluster_project_20261018/project/runs/shared_personalized_20261104_v1'); A=R/'operations/local_admission_20261104_1713'; RR='/home/liying/wind_physics_project/project/runs/'+R.name
H='v1006002'
def stamp():return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
def atomic(p,obj):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(obj,indent=2));os.replace(t,p)
def remote(code):
 p=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=8',H,'python3 -'],input=code,text=True,capture_output=True,timeout=45)
 if p.returncode:raise RuntimeError(p.stderr)
 return json.loads(p.stdout)
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
ids=[p.parent.name for p in (R/'jobs').glob('*/REMOTE.json') if json.loads(p.read_text())['host']==H]
rows=remote(f'''from pathlib import Path
import json
r=Path({RR!r});out=[]
for j in {ids!r}:
 p=r/'jobs'/j
 if (p/'SUCCESS.json').exists() or (p/'FAILURE.json').exists():continue
 h=json.loads((p/'history.json').read_text()) if (p/'history.json').exists() else []
 v=json.loads((p/'progress.json').read_text()) if (p/'progress.json').exists() else {{}}
 best=min(h,key=lambda x:x['validation_mae'])['epoch'] if h else 2
 rem=best+10-(v.get('epoch',1)-1)-v.get('step',0)/max(1,v.get('total_steps',3351))*.75
 out.append(dict(id=j,remaining_epochs=rem))
print(json.dumps(out))''')
selected=[x['id'] for x in sorted(rows,key=lambda x:x['remaining_epochs'],reverse=True) if x['remaining_epochs']>1.5][:16]
atomic(A/'SELECTION.json',dict(timestamp=stamp(),jobs=selected,all_candidates=rows));print('Selected',len(selected),selected,flush=True)
# Durable remote reservation prevents relaunch even if a stale lease is published.
remote(f'''from pathlib import Path
import json,os,time
r=Path({RR!r})
for j in {selected!r}:
 p=r/'jobs'/j;p.mkdir(exist_ok=True)
 for name,data in [('REMOTE.json',dict(host='local',reason='user_requested_4090_migration')),('YIELD_REQUEST.json',dict(reason='migrate_to_local_4090'))]:
  t=p/(name+'.tmp');t.write_text(json.dumps(data));os.replace(t,p/name)
p=r/'operations/LEASE.json';d=json.loads(p.read_text());d['jobs']=[j for j in d['jobs'] if j not in {selected!r}];t=p.with_suffix('.tmp');t.write_text(json.dumps(d));os.replace(t,p)
print('true')''')
ready=[]
for attempt in range(24):
 ready=remote(f'''from pathlib import Path
import fcntl,json
r=Path({RR!r});ready=[]
for j in {selected!r}:
 p=r/'jobs'/j;f=(p/'worker.lock').open('a+')
 try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);ready.append(j)
 except BlockingIOError:pass
 finally:f.close()
print(json.dumps(ready))''')
 if len(ready)==len(selected):break
 time.sleep(2)
assert set(ready)==set(selected),('workers did not yield',set(selected)-set(ready))
records=[]
for j in selected:
 src=remote(f'''from pathlib import Path
import json,hashlib
p=Path({(RR+'/jobs/'+j)!r})
print(json.dumps({{x.name:hashlib.sha256(x.read_bytes()).hexdigest() for x in p.iterdir() if x.name in ['latest.pt','best.pt','spec.json','history.json','metrics.json','SUCCESS.json']}}))''')
 out=R/'jobs'/j
 subprocess.run(['rsync','-a','--exclude=worker.lock','--exclude=REMOTE.json','--exclude=YIELD_REQUEST.json','--exclude=SUCCESS.json',H+':'+RR+'/jobs/'+j+'/',str(out)+'/'],check=True,timeout=60)
 for name,sha in src.items():
  if name!='SUCCESS.json':assert digest(out/name)==sha,(j,name)
 atomic(out/'LOCAL_ONLY.json',dict(timestamp=stamp(),source_host=H,checkpoint_sha256=src.get('latest.pt'),reason='user_requested_available_4090_compute'))
 if 'SUCCESS.json' in src:
  m=json.loads((out/'metrics.json').read_text());assert digest(out/'best.pt')==m['checkpoint_sha256'];atomic(out/'SUCCESS.json',dict(timestamp=stamp(),passed=True,verified_return=True,source_host=H))
 else: assert (out/'latest.pt').exists() or not src,(j,src)
 (out/'REMOTE.json').unlink();(out/'YIELD_REQUEST.json').unlink(missing_ok=True)
 records.append(dict(id=j,sha256=src));print('verified transfer',j,flush=True)
atomic(A/'TRANSFERS.json',dict(timestamp=stamp(),source_host=H,jobs=records))
atomic(R/'operations/LOCAL_ADMISSION.json',dict(timestamp=stamp(),allow_shared_compute=True,max_workers_per_gpu=4,total_compute_target=90,worker_nice=10,minimum_reserve='max(4096 MiB, 20% of VRAM)',max_own_worker_slowdown=1.5,authorization='User requested direct launch on available 4090 VRAM, multiple processes per GPU; preserve other users processes'))
frozen=json.loads((R/'SCIENCE_FREEZE.json').read_text());changed=[]
for rel,old in frozen['files'].items():
 new=digest(R/rel)
 if old!=new:
  assert rel in ['operations/scheduler.py','operations/coordinator.py'],rel
  changed.append(rel);frozen['files'][rel]=new
for rel in changed:py_compile.compile(str(R/rel),doraise=True)
frozen['timestamp']=stamp();atomic(R/'SCIENCE_FREEZE.json',frozen)
atomic(A/'VERIFIED.json',dict(timestamp=stamp(),changed_operations=changed,scientific_files_unchanged=True,transferred=len(records)))
health=json.loads((A/'CONTROL_BEFORE.json').read_text())
for p in health['processes']:
 if p['host']=='local':
  pid=p['pids'][0];os.kill(pid,signal.SIGTERM);os.kill(pid,signal.SIGCONT)
os.kill(health['supervisor_pid'],signal.SIGCONT)
print('Resumed supervisor; local scheduler will launch transferred jobs',flush=True)
