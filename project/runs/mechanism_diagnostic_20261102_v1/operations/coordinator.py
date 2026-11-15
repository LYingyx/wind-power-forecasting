"""Lease small queues to independent hosts; verify completed results before counting."""
from pathlib import Path
import sys,os,json,time,subprocess,shlex,fcntl,socket
RUN=Path(__file__).resolve().parents[1];sys.path.insert(0,str(RUN/'code'))
from settings import JOBS,atomic_json,now,sha
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
r=Path({REMOTE!r}); ids={jids!r}; done=[];failed=[]
for j in ids:
 if (r/'jobs'/j/'SUCCESS.json').exists():done.append(j)
 elif (r/'jobs'/j/'FAILURE.json').exists():failed.append(j)
p=r/'schedulers/status.json';s=json.loads(p.read_text()) if p.exists() else {{}}
print(json.dumps(dict(done=done,failed=failed,status=s,fresh=p.exists() and time.time()-p.stat().st_mtime<120)))''')
def run():
    lock=(RUN/'operations/coordinator.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    manifest=json.loads((RUN/'manifest.json').read_text());specs={s['id']:s for s in manifest};held={h:{} for h in HOSTS}
    # Re-adopt reservations after coordinator restart; never duplicate a remote job.
    for marker in JOBS.glob('*/REMOTE.json'):
        m=json.loads(marker.read_text());f=(marker.parent/'worker.lock').open('a+')
        fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);held[m['host']][marker.parent.name]=f
    last_report=0
    while True:
        states={};errors={}
        for host in HOSTS:
            try:
                state=query(host,list(held[host]));states[host]=state
                for jid in state['done']+state['failed']:
                    out=JOBS/jid
                    subprocess.run(['rsync','-a','--exclude=SUCCESS.json','--exclude=worker.lock','--exclude=YIELD_REQUEST.json',host+':'+REMOTE+'/jobs/'+jid+'/',str(out)+'/'],check=True,timeout=120,stdout=subprocess.DEVNULL)
                    if jid in state['done']:
                        d=json.loads((out/'metrics.json').read_text());assert d['spec']==specs[jid] and not d['smoke_only'] and d['evaluation_year']==2023
                        assert sha(out/'best.pt')==d['checkpoint_sha256']
                        atomic_json(out/'SUCCESS.json',dict(timestamp=now(),source_host=host,verified_return=True,passed=True))
                    (out/'REMOTE.json').unlink();held[host].pop(jid).close()
                active=state['status'].get('active',[])
                target=min(96,len(active)+max(8,len(active)//2)) if state['fresh'] else 0
                if state['status'].get('circuit_breaker'):target=0
                for spec in manifest:
                    if len(held[host])>=target:break
                    jid=spec['id'];out=JOBS/jid
                    if any((out/f).exists() for f in ['SUCCESS.json','FAILURE.json','REMOTE.json']):continue
                    out.mkdir(parents=True,exist_ok=True);f=(out/'worker.lock').open('a+')
                    try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
                    except BlockingIOError:f.close();continue
                    if (out/'SUCCESS.json').exists():f.close();continue
                    atomic_json(out/'REMOTE.json',dict(host=host,timestamp=now()));held[host][jid]=f
                lease(host,list(held[host]))
            except Exception as exc:errors[host]=str(exc)
        localp=RUN/'schedulers/status.json';local=json.loads(localp.read_text()) if localp.exists() else {}
        completed=sum((JOBS/s['id']/'SUCCESS.json').exists() for s in manifest)
        failed=[s['id'] for s in manifest if (JOBS/s['id']/'FAILURE.json').exists()]
        local_active=local.get('active',[]) if localp.exists() and time.time()-localp.stat().st_mtime<120 else []
        active=local_active+[a for s in states.values() if s['fresh'] for a in s['status'].get('active',[])]
        active=list({a['id']:a for a in active if not (JOBS/a['id']/'SUCCESS.json').exists()}.values())
        status=dict(timestamp=now(),planned=len(manifest),completed=completed,failed=failed,active=active,pending=len(manifest)-completed-len(failed)-len(active),hosts=states,errors=errors)
        atomic_json(RUN/'STATUS.json',status)
        if time.monotonic()-last_report>300 or completed==len(manifest):
            subprocess.run([sys.executable,str(RUN/'operations/collect.py')],check=completed==len(manifest));last_report=time.monotonic()
        if completed==len(manifest):
            for host in HOSTS:remote_python(host,f"from pathlib import Path;p=Path({(REMOTE+'/operations/STOP.json')!r});p.write_text('{{}}');print('true')")
            atomic_json(RUN/'COMPLETE.json',dict(timestamp=now(),completed=completed,failed=0));break
        time.sleep(15)
if __name__=='__main__':run()
