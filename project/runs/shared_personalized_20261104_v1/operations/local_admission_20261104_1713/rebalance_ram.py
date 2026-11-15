from pathlib import Path
import json,subprocess,hashlib,os,signal,datetime,fcntl,sys
R=Path('/shared_data_ssd2/liying/wind_cluster_project_20261018/project/runs/shared_personalized_20261104_v1');A=R/'operations/local_admission_20261104_1713';RR='/home/liying/wind_physics_project/project/runs/'+R.name;H='v1006001'
def atomic(p,d):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(d,indent=2));os.replace(t,p)
def remote(code):
 p=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=8',H,'python3 -'],input=code,text=True,capture_output=True,timeout=45);assert p.returncode==0,p.stderr;return json.loads(p.stdout)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
h=json.loads((R/'operations/CONTROL_HEALTH.json').read_text());sup=h['supervisor_pid'];coord=next(p['pids'][0] for p in h['processes'] if p['host']=='local' and p['process']=='coordinator')
for pid,name in [(sup,'supervise.py'),(coord,'coordinator.py')]:
 assert str(R/'operations'/name).encode() in Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0');os.kill(pid,signal.SIGSTOP)
locks={};records=[]
try:
 for marker in (R/'jobs').glob('*/LOCAL_ONLY.json'):
  out=marker.parent
  if (out/'SUCCESS.json').exists() or (out/'FAILURE.json').exists() or (out/'REMOTE.json').exists():continue
  f=(out/'worker.lock').open('a+')
  try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError:f.close();continue
  locks[out.name]=f
 print('Pending handoff',list(locks),flush=True)
 for j in locks:
  out=R/'jobs'/j
  remote(f'from pathlib import Path\np=Path({(RR+"/jobs/"+j)!r});p.mkdir(exist_ok=True);assert not (p/"SUCCESS.json").exists();print("true")')
  subprocess.run(['rsync','-a','--exclude=worker.lock','--exclude=LOCAL_ONLY.json','--exclude=REMOTE.json','--exclude=YIELD_REQUEST.json',str(out)+'/',H+':'+RR+'/jobs/'+j+'/'],check=True,timeout=60)
  hashes=remote(f'from pathlib import Path\nimport hashlib,json\np=Path({(RR+"/jobs/"+j)!r});print(json.dumps({{n:hashlib.sha256((p/n).read_bytes()).hexdigest() for n in ["latest.pt","best.pt","spec.json"] if (p/n).exists()}}))')
  for n,d in hashes.items():assert sha(out/n)==d,(j,n)
  assert hashes.get('latest.pt'),j
  atomic(out/'REMOTE.json',dict(host=H,timestamp=datetime.datetime.now().astimezone().isoformat(),reason='4090_system_RAM_admission_limit',checkpoint_verified=hashes['latest.pt']))
  (out/'LOCAL_ONLY.json').unlink();records.append(dict(id=j,host=H,sha256=hashes));print('Rebalanced',j,flush=True)
 atomic(A/'RAM_REBALANCE.json',dict(timestamp=datetime.datetime.now().astimezone().isoformat(),jobs=records,local_running_kept=True))
finally:
 # Restart only coordinator so it adopts the verified new remote reservations.
 os.kill(coord,signal.SIGTERM);os.kill(coord,signal.SIGCONT)
 for f in locks.values():f.close()
 os.kill(sup,signal.SIGCONT)
print('Handoff done; existing 4090 workers unaffected',flush=True)
