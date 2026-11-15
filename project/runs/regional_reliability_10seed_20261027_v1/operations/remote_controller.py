"""Reserve central jobs, lease small queues to non-shared V100 hosts, verify returned artifacts."""
from pathlib import Path
import os,sys,json,time,subprocess,fcntl,shlex,socket,signal
RUN=Path(__file__).resolve().parents[1];sys.path.insert(0,str(RUN/'code'))
from settings import JOBS,PROJECT,atomic_json,now,sha
from resource_scheduler import status_collect
REMOTE_PROJECT='/home/liying/wind_physics_project/project'
REMOTE_RUN=REMOTE_PROJECT+'/runs/'+RUN.name
HOSTS=['v1006001','v1006002'];BATCH=64


def ssh(host,script):
    result=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=8',host,'python3 -c '+shlex.quote(script)],capture_output=True,text=True,timeout=45)
    if result.returncode:raise RuntimeError(host+': '+result.stderr[-1000:])
    return json.loads(result.stdout)


def write_lease(host,jobs):
    jobs=[j for j in jobs if json.loads((JOBS/j/'REMOTE_RESERVATION.json').read_text()).get('ready',False)]
    value=dict(timestamp=now(),jobs=jobs,coordinator=socket.gethostname())
    payload=json.dumps(value)+chr(10)
    script=f"from pathlib import Path; import json,os; p=Path({(REMOTE_RUN+'/operations/LEASED_JOBS.json')!r}); t=p.with_suffix('.tmp'); t.write_text({payload!r}); os.replace(t,p); print(json.dumps(dict(ok=True)))"
    return ssh(host,script)


def query(host):
    script=f'''from pathlib import Path
import json,os
r=Path({REMOTE_RUN!r}); jobs=json.loads((r/'operations/LEASED_JOBS.json').read_text()).get('jobs',[])
done=[];failed=[];active=[]
for jid in jobs:
 p=r/'jobs'/jid
 if (p/'SUCCESS.json').exists():done.append(jid)
 elif (p/'FAILURE.json').exists():failed.append(jid)
for p in (r/'schedulers').glob('*.json'):
 try:
  d=json.loads(p.read_text())
  if d.get('enabled',True):active.extend(d.get('active',[]))
 except (OSError,ValueError):pass
compat=r/'operations/GPU_COMPATIBILITY.json'
ready=compat.exists() and json.loads(compat.read_text()).get('passed',False)
print(json.dumps(dict(done=done,failed=failed,active=active,ready=ready)))'''
    return ssh(host,script)


def stage(host,spec,marker):
    if spec['task']=='diagnose_reference':
        source=PROJECT/'runs'/spec['source_run']/'jobs'/spec['source_job']
        result=json.loads((source/'metrics.json').read_text());sel=result['selections']['mae']
        cp=source/sel['checkpoint'];assert sha(cp)==sel['sha256']
        remote=REMOTE_PROJECT+'/runs/'+spec['source_run']+'/jobs/'+spec['source_job'];remote_cp=remote+'/'+cp.name
        prior=ssh(host,f"from pathlib import Path; import json; p=Path({remote!r}); p.mkdir(parents=True,exist_ok=True); print(json.dumps(dict(exists=Path({remote_cp!r}).exists())))")
        marker.update(source_checkpoint=str(cp),source_sha256=sel['sha256'],remote_checkpoint=remote_cp,
                      remove_staged_checkpoint=marker.get('remove_staged_checkpoint',not prior['exists']))
        atomic_json(JOBS/spec['id']/'REMOTE_RESERVATION.json',marker)
        subprocess.run(['rsync','-a',str(cp),str(source/'metrics.json'),str(source/'normalization.json'),host+':'+remote+'/'],check=True,stdout=subprocess.DEVNULL,timeout=90)
    marker['ready']=True;atomic_json(JOBS/spec['id']/'REMOTE_RESERVATION.json',marker)


def receive(host,jid,expected,success=True):
    out=JOBS/jid;out.mkdir(exist_ok=True)
    marker=json.loads((out/'REMOTE_RESERVATION.json').read_text())
    args=['rsync','-a','--exclude=SUCCESS.json','--exclude=worker.lock','--exclude=worker.log',host+':'+REMOTE_RUN+'/jobs/'+jid+'/',str(out)+'/']
    subprocess.run(args,check=True,timeout=120,stdout=subprocess.DEVNULL)
    if success:
        result=json.loads((out/'metrics.json').read_text());spec=json.loads((out/'spec.json').read_text());assert spec==expected
        if expected['task']=='train':
            for selected in result['selections'].values():assert sha(out/selected['checkpoint'])==selected['sha256']
        else:
            assert result['source_sha256']==marker['source_sha256'] and sha(marker['source_checkpoint'])==marker['source_sha256']
            result['execution_source_checkpoint']=result['source_checkpoint'];result['source_checkpoint']=marker['source_checkpoint']
            atomic_json(out/'metrics.json',result)
            d=json.loads((out/'diagnostics.json').read_text());d['execution_source_checkpoint']=d['source_checkpoint'];d['source_checkpoint']=marker['source_checkpoint']
            atomic_json(out/'diagnostics.json',d)
        atomic_json(out/'SUCCESS.json',dict(timestamp=now(),source_host=host,verified_return=True,smoke_only=False))
        # Retain small remote receipts; selected weights have one authoritative central copy.
        ssh(host,f'''from pathlib import Path
import json
p=Path({(REMOTE_RUN+'/jobs/'+jid)!r})
for f in p.glob('epoch_*.pt'):f.unlink()
(p/'ARCHIVED.json').write_text(json.dumps(dict(central_path={str(out)!r},verified=True)))
print(json.dumps(dict(ok=True)))''')
        if marker.get('remove_staged_checkpoint'):
            ssh(host,f"from pathlib import Path; import hashlib,json; p=Path({marker['remote_checkpoint']!r}); assert not p.exists() or hashlib.sha256(p.read_bytes()).hexdigest()=={marker['source_sha256']!r}; p.unlink(missing_ok=True); print(json.dumps(dict(ok=True)))")
    else:assert (out/'FAILURE.json').exists()
    (out/'REMOTE_RESERVATION.json').unlink(missing_ok=True)


def run():
    lock=(RUN/'operations/remote_controller.lock').open('a+')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise SystemExit('Remote controller already active')
    manifest=json.loads((RUN/'manifest.json').read_text());specs={s['id']:s for s in manifest}
    held={h:{} for h in HOSTS};draining=False
    def stop(sig,frame):
        nonlocal draining
        draining=True
    signal.signal(signal.SIGTERM,stop)
    for marker in JOBS.glob('*/REMOTE_RESERVATION.json'):
        info=json.loads(marker.read_text());host=info['host'];jid=marker.parent.name
        if host not in held:continue
        f=(marker.parent/'worker.lock').open('a+')
        try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:f.close();continue
        held[host][jid]=f
    while True:
        states={}
        for host in HOSTS:
            try:
                for jid in held[host]:
                    marker=json.loads((JOBS/jid/'REMOTE_RESERVATION.json').read_text())
                    if not marker.get('ready'):stage(host,specs[jid],marker)
                write_lease(host,list(held[host]))
                state=query(host)
                for jid in state['done']:
                    if jid in held[host]:receive(host,jid,specs[jid]);held[host].pop(jid).close()
                for jid in state['failed']:
                    if jid in held[host]:receive(host,jid,specs[jid],False);held[host].pop(jid).close()
                if not draining and state['ready']:
                    for spec in manifest:
                        if spec.get('local_only'):continue
                        if len(held[host])>=BATCH:break
                        jid=spec['id'];out=JOBS/jid
                        # Already-started local jobs retain their device/RNG resume path;
                        # remote hosts receive only untouched jobs, avoiding duplicate fitting.
                        if any((out/name).exists() for name in ['SUCCESS.json','FAILURE.json','REMOTE_RESERVATION.json','spec.json','latest.pt']):continue
                        out.mkdir(exist_ok=True);claim=(out/'worker.lock').open('a+')
                        try:fcntl.flock(claim,fcntl.LOCK_EX|fcntl.LOCK_NB)
                        except BlockingIOError:claim.close();continue
                        if any((out/name).exists() for name in ['SUCCESS.json','FAILURE.json','spec.json','latest.pt']):claim.close();continue
                        marker=dict(host=host,id=jid,timestamp=now(),controller_pid=os.getpid(),ready=False)
                        atomic_json(out/'REMOTE_RESERVATION.json',marker)
                        held[host][jid]=claim;stage(host,spec,marker)
                write_lease(host,list(held[host]))
                states[host]=dict(timestamp=now(),leased=len(held[host]),active=state['active'],connected=True,ready=state['ready'])
                atomic_json(RUN/'schedulers'/('remote_'+host+'.json'),dict(timestamp=now(),host=host,enabled=True,active=state['active'],leased=len(held[host])))
            except Exception as e:
                states[host]=dict(timestamp=now(),leased=len(held[host]),connected=False,error=str(e))
        atomic_json(RUN/'operations/REMOTE_STATUS.json',dict(timestamp=now(),pid=os.getpid(),hosts=states,draining=draining))
        status=status_collect()
        if status['completed']+status['failed']>=len(manifest) and not any(held.values()):
            for host in HOSTS:
                ssh(host,f"from pathlib import Path; import json; Path({(REMOTE_RUN+'/operations/GLOBAL_COMPLETE.json')!r}).write_text(json.dumps(dict(complete=True))); print(json.dumps(dict(ok=True)))")
            break
        if draining and not any(held.values()):break
        time.sleep(10)

if __name__=='__main__':run()
