"""Distribute per-site adaptation jobs; workers resume from exclusive checkpoints."""
from pathlib import Path
import sys,os,json,time,subprocess,shlex,fcntl,threading
RUN=Path(__file__).resolve().parents[1];sys.path.insert(0,str(RUN/'code'))
from settings import *
from collect import collect
REMOTE='/home/liying/wind_physics_project/project/runs/'+RUN.name
HOSTS=['v1006001','v1006002']
def remote_python(host,script):
    p=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=8',host,'python3 -c '+shlex.quote(script)],capture_output=True,text=True,timeout=45)
    if p.returncode:raise RuntimeError(p.stderr[-1000:])
    return json.loads(p.stdout)
def lease(host,jids):
    payload=json.dumps(dict(timestamp=now(),jobs=jids))
    return remote_python(host,f"from pathlib import Path;import os,json;p=Path({(REMOTE+'/operations/LEASE.json')!r});t=p.with_suffix('.tmp');t.write_text({payload!r});os.replace(t,p);print('true')")
def query(host,jids):
    return remote_python(host,f'''from pathlib import Path
import json,time
r=Path({REMOTE!r});done=[];failed=[]
for j in {jids!r}:
 if (r/'jobs'/j/'SUCCESS.json').exists():done.append(j)
 elif (r/'jobs'/j/'FAILURE.json').exists():failed.append(j)
p=r/'schedulers/status.json';s=json.loads(p.read_text()) if p.exists() else {{}}
print(json.dumps(dict(done=done,failed=failed,status=s,fresh=p.exists() and time.time()-p.stat().st_mtime<120)))''')
def run():
    lock=(RUN/'operations/coordinator.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    manifest=json.loads((RUN/'manifest.json').read_text());specs={s['id']:s for s in manifest};held={h:{} for h in HOSTS}
    for marker in JOBS.glob('*/REMOTE.json'):
        m=json.loads(marker.read_text());f=(marker.parent/'worker.lock').open('a+');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);held[m['host']][marker.parent.name]=f
    lease_lock=threading.Lock();lease_jobs={h:list(held[h]) for h in HOSTS};stop=threading.Event()
    def heartbeat():
        while not stop.is_set():
            for host in HOSTS:
                try:
                    with lease_lock:lease(host,list(lease_jobs[host]))
                except Exception as exc:print(now(),'heartbeat',host,repr(exc),flush=True)
            stop.wait(30)
    threading.Thread(target=heartbeat,daemon=True).start();last_report=0
    while True:
        states={};errors={}
        for host in HOSTS:
            try:
                state=query(host,list(held[host]));states[host]=state
                for jid in state['done']+state['failed']:
                    out=JOBS/jid
                    subprocess.run(['rsync','-a','--exclude=SUCCESS.json','--exclude=worker.lock','--exclude=YIELD_REQUEST.json',host+':'+REMOTE+'/jobs/'+jid+'/',str(out)+'/'],check=True,timeout=90,stdout=subprocess.DEVNULL)
                    if jid in state['done']:
                        d=json.loads((out/'metrics.json').read_text());assert d['requested_spec']==specs[jid] and not d['smoke_only'] and d['evaluation_year']==2023
                        assert sha(out/'best.pt')==d['checkpoint_sha256'];atomic_json(out/'SUCCESS.json',dict(timestamp=now(),source_host=host,verified_return=True,passed=True))
                    (out/'REMOTE.json').unlink();held[host].pop(jid).close()
                active=state['status'].get('active',[])
                # Only reserve on hosts whose policy admits work, or already running work.
                slots=sum(bool(d.get('launch')) for d in state['status'].get('decisions',{}).values())
                target=min(64,len(active)+max(4,2*slots)) if state['fresh'] and (active or slots) else len(held[host])
                if state['status'].get('circuit_breaker'):target=0
                for spec in manifest:
                    if len(held[host])>=target:break
                    jid=spec['id'];out=JOBS/jid
                    if any((out/f).exists() for f in ['SUCCESS.json','FAILURE.json','REMOTE.json','LOCAL_ONLY.json']):continue
                    out.mkdir(parents=True,exist_ok=True);f=(out/'worker.lock').open('a+')
                    try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
                    except BlockingIOError:f.close();continue
                    if (out/'SUCCESS.json').exists():f.close();continue
                    atomic_json(out/'REMOTE.json',dict(host=host,timestamp=now()));held[host][jid]=f
                with lease_lock:lease_jobs[host]=list(held[host]);lease(host,lease_jobs[host])
            except Exception as exc:errors[host]=repr(exc)
        lp=RUN/'schedulers/status.json';local=json.loads(lp.read_text()) if lp.exists() else {}
        completed=sum((JOBS/s['id']/'SUCCESS.json').exists() for s in manifest);failed=[s['id'] for s in manifest if (JOBS/s['id']/'FAILURE.json').exists()]
        active=(local.get('active',[]) if lp.exists() and time.time()-lp.stat().st_mtime<120 else [])+[a for st in states.values() if st['fresh'] for a in st['status'].get('active',[])]
        active=list({a['id']:a for a in active if not (JOBS/a['id']/'SUCCESS.json').exists()}.values())
        atomic_json(RUN/'STATUS.json',dict(timestamp=now(),planned=len(manifest),completed=completed,failed=failed,active=active,pending=len(manifest)-completed-len(failed)-len(active),hosts=states,errors=errors))
        if time.monotonic()-last_report>60 or completed==len(manifest):
            try:gate=collect();last_report=time.monotonic()
            except Exception as exc:atomic_json(RUN/'operations/COLLECT_ERROR.json',dict(timestamp=now(),error=repr(exc)))
        if completed==len(manifest):
            gate=collect();stop.set()
            for host in HOSTS:remote_python(host,f"from pathlib import Path;p=Path({(REMOTE+'/operations/STOP.json')!r});p.write_text('{{}}');print('true')")
            atomic_json(RUN/'COMPLETE.json',dict(timestamp=now(),completed=completed,failed=0,scope='Z1_and_RM_GPU_jobs',gate=gate['stage']));break
        time.sleep(10)
if __name__=='__main__':run()
