"""Move the last ten idle remote reservations with exact checkpoint verification."""
from pathlib import Path
import subprocess, json, os, signal, time, hashlib, datetime, fcntl
P=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
R=P/'runs/daily_ablation_restart_20261114_v1'
RR='/home/liying/wind_physics_project/project/runs/'+R.name
def remote(host,code):
    p=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=8',host,'python3 -'],input=code,text=True,capture_output=True,timeout=60,check=True)
    return json.loads(p.stdout)
def signal_script(script):
    ids=[]
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():continue
        try:
            if str(script).encode() in (p/'cmdline').read_bytes().split(b'\0'):
                os.kill(int(p.name),signal.SIGTERM);ids.append(int(p.name))
        except OSError:pass
    return ids
for n in ['watch','supervise','coordinator']:
    print('stop',n,signal_script(R/'operations'/f'{n}.py'),flush=True)
time.sleep(2)
lock=(R/'operations/coordinator.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
for host in ['v1006002','v1006001']:
    print(host,remote(host,f'''from pathlib import Path
import os,signal,json
r=Path({RR!r});workers=[];schedulers=[]
for p in Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try:
  parts=(p/'cmdline').read_bytes().split(b'\\0')
  if str(r/'code/train.py').encode() in parts:workers.append(int(p.name))
  if str(r/'operations/scheduler.py').encode() in parts:schedulers.append(int(p.name))
 except OSError:pass
assert not workers,workers
for pid in schedulers:os.kill(pid,signal.SIGTERM)
(r/'operations/LEASE.json').write_text(json.dumps(dict(jobs=[])))
print(json.dumps(dict(idle=True,schedulers=schedulers)))'''),flush=True)
events=[]
for marker in (R/'jobs').glob('*/REMOTE.json'):
    meta=json.loads(marker.read_text());assert meta['host']=='v1006002'
    jid=marker.parent.name;local=marker.parent
    hashes=remote('v1006002',f'''from pathlib import Path
import hashlib,fcntl,json
p=Path({(RR+'/jobs/'+jid)!r});p.mkdir(exist_ok=True)
f=(p/'worker.lock').open('a+');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert not (p/'SUCCESS.json').exists() and not (p/'FAILURE.json').exists()
print(json.dumps({{q.name:hashlib.sha256(q.read_bytes()).hexdigest() for q in p.iterdir() if q.is_file() and q.name in ['latest.pt','best.pt','spec.json','history.json']}}))''')
    args=['rsync','-a','--exclude=worker.lock','--exclude=REMOTE.json','--exclude=YIELD_REQUEST.json']
    subprocess.run([*args,'v1006002:'+RR+'/jobs/'+jid+'/',str(local)+'/'],check=True,timeout=90)
    for name,h in hashes.items():assert hashlib.sha256((local/name).read_bytes()).hexdigest()==h
    remote('v1006001',f"from pathlib import Path;Path({(RR+'/jobs/'+jid)!r}).mkdir(exist_ok=True);print('true')")
    subprocess.run([*args,str(local)+'/', 'v1006001:'+RR+'/jobs/'+jid+'/'],check=True,timeout=90)
    remote('v1006001',f'''from pathlib import Path
import hashlib
p=Path({(RR+'/jobs/'+jid)!r})
for name,h in {hashes!r}.items():assert hashlib.sha256((p/name).read_bytes()).hexdigest()==h
print('true')''')
    marker.write_text(json.dumps(dict(host='v1006001',timestamp=datetime.datetime.now().astimezone().isoformat(),migrated_from='v1006002',checkpoint_hashes=hashes),indent=2)+'\n')
    events.append(dict(job=jid,files=hashes));print('migrated',jid,flush=True)
# All remaining work is now held on 6001. No model or scheduling settings change.
lock.close()
env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONNOUSERSITE='1')
logroot=Path('/shared_data_ssd3/liying')/R.name/'control_health';py=str(P/'.runtime/chronos/bin/python')
with (logroot/'supervise.log').open('a') as f:
    supervisor=subprocess.Popen([py,'-u',str(R/'operations/supervise.py')],env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
v=json.loads((R/'LAUNCH_STATUS.json').read_text());v['supervisor_pid']=supervisor.pid;(R/'LAUNCH_STATUS.json').write_text(json.dumps(v,indent=2)+'\n')
with (logroot/'watch.log').open('a') as f:
    watcher=subprocess.Popen([py,'-u',str(R/'operations/watch.py')],env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
(OUT/'TAIL_RECOVERY.json').write_text(json.dumps(dict(timestamp=datetime.datetime.now().astimezone().isoformat(),passed=True,jobs=events,supervisor_pid=supervisor.pid,watch_pid=watcher.pid,scientific_files_unchanged=True),indent=2)+'\n')
print('restarted',supervisor.pid,watcher.pid,flush=True)
