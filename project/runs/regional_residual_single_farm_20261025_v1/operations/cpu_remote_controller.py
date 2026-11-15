"""Lease CPU jobs separately from GPU jobs; stage only selected replay weights."""
from pathlib import Path
import os,sys,json,time,subprocess,fcntl,shlex,socket,signal
RUN=Path(__file__).resolve().parents[1];sys.path.insert(0,str(RUN/'code'))
from settings import JOBS,PROJECT,atomic_json,now,sha
from resource_scheduler import status_collect
REMOTE_PROJECT='/home/liying/wind_physics_project/project';REMOTE_RUN=REMOTE_PROJECT+'/runs/'+RUN.name
HOSTS=['v1006001','v1006002']

def ssh(host,script):
    result=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=8',host,'python3 -c '+shlex.quote(script)],capture_output=True,text=True,timeout=45)
    if result.returncode:raise RuntimeError(host+': '+result.stderr[-1500:])
    return json.loads(result.stdout)

def lease(host,held):
    jobs=[jid for jid in held if json.loads((JOBS/jid/'CPU_REMOTE_RESERVATION.json').read_text()).get('ready')]
    payload=json.dumps(dict(timestamp=now(),jobs=jobs,coordinator=socket.gethostname()))+'\n'
    ssh(host,f"from pathlib import Path; import json,os; p=Path({(REMOTE_RUN+'/operations/CPU_LEASED_JOBS.json')!r}); t=p.with_suffix('.tmp'); t.write_text({payload!r}); os.replace(t,p); print(json.dumps(dict(ok=True)))")

def query(host):
    return ssh(host,f'''from pathlib import Path
import json,time
r=Path({REMOTE_RUN!r});p=r/'schedulers/cpu_scheduler_v2.json';d=json.loads(p.read_text()) if p.exists() and time.time()-p.stat().st_mtime<60 else dict(capacity=0,active=[])
leases=json.loads((r/'operations/CPU_LEASED_JOBS.json').read_text()).get('jobs',[])
d['done']=[j for j in leases if (r/'jobs'/j/'SUCCESS.json').exists()]
d['failed_jobs']=[j for j in leases if (r/'jobs'/j/'FAILURE.json').exists() and not (r/'jobs'/j/'SUCCESS.json').exists()]
d['ready']=(r/'operations/CPU_RUNTIME_READY.json').exists()
print(json.dumps(d))''')

def stage(host,spec,marker):
    if spec['task']=='diagnose_reference':
        source=PROJECT/'runs'/spec['source_run']/'jobs'/spec['source_job'];m=json.loads((source/'metrics.json').read_text());sel=m['selections']['mae'];cp=source/sel['checkpoint'];assert sha(cp)==sel['sha256']
        remote=REMOTE_PROJECT+'/runs/'+spec['source_run']+'/jobs/'+spec['source_job'];remote_cp=remote+'/'+cp.name
        prior=ssh(host,f"from pathlib import Path; import json; p=Path({remote!r}); p.mkdir(parents=True,exist_ok=True); print(json.dumps(dict(preexisting_checkpoint=Path({remote_cp!r}).exists())))")
        # Persist staging ownership before transfer; recovery repeats this transfer safely.
        marker.update(source_checkpoint=str(cp),source_sha256=sel['sha256'],remote_checkpoint=remote_cp,remove_staged_checkpoint=marker.get('remove_staged_checkpoint',not prior['preexisting_checkpoint']))
        atomic_json(JOBS/spec['id']/'CPU_REMOTE_RESERVATION.json',marker)
        subprocess.run(['rsync','-a',str(cp),str(source/'metrics.json'),str(source/'normalization.json'),host+':'+remote+'/'],check=True,stdout=subprocess.DEVNULL,timeout=90)
    ssh(host,f"from pathlib import Path; import json; (Path({(REMOTE_RUN+'/jobs/'+spec['id'])!r})/'CPU_RETRY_HOLD.json').unlink(missing_ok=True); print(json.dumps(dict(ok=True)))")
    marker['ready']=True;atomic_json(JOBS/spec['id']/'CPU_REMOTE_RESERVATION.json',marker)

def receive(host,spec,marker,success):
    jid=spec['id'];out=JOBS/jid
    subprocess.run(['rsync','-a','--exclude=SUCCESS.json','--exclude=worker.lock','--exclude=worker.log',host+':'+REMOTE_RUN+'/jobs/'+jid+'/',str(out)+'/'],check=True,stdout=subprocess.DEVNULL,timeout=120)
    if success:
        result=json.loads((out/'metrics.json').read_text());assert json.loads((out/'spec.json').read_text())==spec and not result['smoke_only']
        if spec['task']=='arx':assert sha(out/'coefficients.npz')==result['artifact_sha256']
        else:
            assert result['source_sha256']==marker['source_sha256'] and sha(marker['source_checkpoint'])==marker['source_sha256']
            result['execution_source_checkpoint']=result['source_checkpoint'];result['source_checkpoint']=marker['source_checkpoint'];atomic_json(out/'metrics.json',result)
            d=json.loads((out/'diagnostics.json').read_text());d['execution_source_checkpoint']=d['source_checkpoint'];d['source_checkpoint']=marker['source_checkpoint'];atomic_json(out/'diagnostics.json',d)
        atomic_json(out/'SUCCESS.json',dict(timestamp=now(),source_host=host,verified_return=True,smoke_only=False,cpu_task=True))
    else:assert (out/'FAILURE.json').exists()
    if success:
        cp=marker.get('remote_checkpoint') if marker.get('remove_staged_checkpoint') else None
        ssh(host,f'''from pathlib import Path
import json,hashlib
p=Path({(REMOTE_RUN+'/jobs/'+jid)!r});cp={cp!r}
if cp:
 q=Path(cp)
 if q.exists():
  assert hashlib.sha256(q.read_bytes()).hexdigest()=={marker.get('source_sha256')!r};q.unlink()
(p/'coefficients.npz').unlink(missing_ok=True)
(p/'CPU_ARCHIVED.json').write_text(json.dumps(dict(central_path={str(out)!r},verified=True,staged_checkpoint_removed=bool(cp))))
print(json.dumps(dict(ok=True)))''')
    (out/'CPU_REMOTE_RESERVATION.json').unlink(missing_ok=True)

def main():
    lock=(RUN/'operations/cpu_remote_controller.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    specs=[s for s in json.loads((RUN/'manifest.json').read_text()) if s['task'] in ['arx','diagnose_reference']];byid={s['id']:s for s in specs};held={h:{} for h in HOSTS};draining=False
    def stop(sig,frame):
        nonlocal draining
        draining=True
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    for p in JOBS.glob('*/CPU_REMOTE_RESERVATION.json'):
        marker=json.loads(p.read_text());host=marker['host'];f=(p.parent/'worker.lock').open('a+')
        try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:f.close();continue
        held[host][p.parent.name]=f
    while True:
        states={}
        for host in HOSTS:
            try:
                for jid in held[host]:
                    marker=json.loads((JOBS/jid/'CPU_REMOTE_RESERVATION.json').read_text())
                    if not marker.get('ready'):stage(host,byid[jid],marker)
                lease(host,held[host]);state=query(host)
                for success,key in [(True,'done'),(False,'failed_jobs')]:
                    for jid in state[key]:
                        if jid in held[host]:
                            marker=json.loads((JOBS/jid/'CPU_REMOTE_RESERVATION.json').read_text());receive(host,byid[jid],marker,success);held[host].pop(jid).close()
                cap=int(state.get('capacity',0));reserve=int(state.get('reference_slots',4));reference_left=any(s['task']=='diagnose_reference' and not (JOBS/s['id']/'SUCCESS.json').exists() for s in specs)
                targets={'diagnose_reference':min(16,2*reserve) if cap else 0,'arx':max(0,cap-(reserve if reference_left else 0))+2 if cap else 0}
                if not draining and state['ready'] and not state.get('failed'):
                    added={'arx':0,'diagnose_reference':0}
                    for spec in sorted(specs,key=lambda s:(s['task']!='arx',s['site'],s['id'])):
                        kind=spec['task']
                        if added[kind]>=(6 if kind=='arx' else 8) or sum(byid[j]['task']==kind for j in held[host])>=targets[kind]:continue
                        jid=spec['id'];out=JOBS/jid
                        if any((out/n).exists() for n in ['SUCCESS.json','FAILURE.json','CPU_REMOTE_RESERVATION.json','spec.json']):continue
                        out.mkdir(exist_ok=True);claim=(out/'worker.lock').open('a+')
                        try:fcntl.flock(claim,fcntl.LOCK_EX|fcntl.LOCK_NB)
                        except BlockingIOError:claim.close();continue
                        if any((out/n).exists() for n in ['SUCCESS.json','FAILURE.json','CPU_REMOTE_RESERVATION.json','spec.json']):claim.close();continue
                        marker=dict(host=host,id=jid,timestamp=now(),controller_pid=os.getpid(),ready=False,authorized_routing_override='CPU tasks may use both V100 hosts per user; original local_only remains a GPU-controller exclusion')
                        atomic_json(out/'CPU_REMOTE_RESERVATION.json',marker);held[host][jid]=claim;stage(host,spec,marker);added[kind]+=1
                lease(host,held[host]);states[host]=dict(timestamp=now(),connected=True,leased=len(held[host]),capacity=cap,active=state.get('active',[]),targets=targets,failures=state.get('failed',[]))
                atomic_json(RUN/'schedulers'/('cpu_remote_'+host+'.json'),dict(timestamp=now(),host=host,enabled=True,active=state.get('active',[])))
            except Exception as e:states[host]=dict(timestamp=now(),connected=False,error=str(e),leased=len(held[host]))
        atomic_json(RUN/'operations/CPU_REMOTE_STATUS.json',dict(timestamp=now(),pid=os.getpid(),hosts=states,draining=draining));status_collect()
        done=all(any((JOBS/s['id']/n).exists() for n in ['SUCCESS.json','FAILURE.json']) for s in specs)
        if done and not any(held.values()):
            atomic_json(RUN/'operations/CPU_GLOBAL_COMPLETE.json',dict(timestamp=now(),complete=True))
            for host in HOSTS:ssh(host,f"from pathlib import Path; import json; Path({(REMOTE_RUN+'/operations/CPU_GLOBAL_COMPLETE.json')!r}).write_text(json.dumps(dict(complete=True))); print(json.dumps(dict(ok=True)))")
            break
        if draining and not any(held.values()):break
        time.sleep(5)
if __name__=='__main__':main()
