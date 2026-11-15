"""Reserve central jobs, lease small queues to non-shared V100 hosts, verify returned artifacts."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
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
    if spec['task']=='evaluate':
        source=PROJECT/'runs/readout_factorial_20261028_v1/jobs'/spec['source_job']
        result=json.loads((source/'metrics.json').read_text());sel=result['selections']['mae'];cp=source/sel['checkpoint'];assert sha(cp)==sel['sha256']
        frozen=json.loads((RUN/'TEST_FREEZE.json').read_text())['parent_checkpoints'][spec['source_job']]
        assert sel['sha256']==frozen['sha256'] and sha(source/'spec.json')==frozen['spec_sha256'] and sha(source/'normalization.json')==frozen['normalization_sha256']
        remote=REMOTE_RUN+'/sources/'+spec['source_job']
        ssh(host,f"from pathlib import Path; import json;Path({remote!r}).mkdir(parents=True,exist_ok=True);print(json.dumps(True))")
        subprocess.run(['rsync','-a',str(cp),str(source/'metrics.json'),str(source/'spec.json'),str(source/'normalization.json'),host+':'+remote+'/'],check=True,timeout=90,stdout=subprocess.DEVNULL)
        marker.update(source_sha256=sel['sha256'],staged_checkpoint=remote+'/'+cp.name)
    marker['ready']=True;atomic_json(JOBS/spec['id']/'REMOTE_RESERVATION.json',marker)

def receive(host,jid,expected,success=True):
    out=JOBS/jid;marker=json.loads((out/'REMOTE_RESERVATION.json').read_text())
    subprocess.run(['rsync','-a','--exclude=SUCCESS.json','--exclude=worker.lock','--exclude=worker.log',host+':'+REMOTE_RUN+'/jobs/'+jid+'/',str(out)+'/'],check=True,timeout=120,stdout=subprocess.DEVNULL)
    if success:
        result=json.loads((out/'metrics.json').read_text());spec=json.loads((out/'spec.json').read_text());assert spec==expected
        test=json.loads((out/'test.json').read_text());assert test['year']==2024 and test['selection_year']==2023
        if expected['task']=='train':
            for selected in result['selections'].values():assert sha(out/selected['checkpoint'])==selected['sha256']
            assert test['source_sha256']==result['selections']['mae']['sha256']
        else:assert result['source_sha256']==test['source_sha256']==marker['source_sha256']
        atomic_json(out/'SUCCESS.json',dict(timestamp=now(),source_host=host,verified_return=True,smoke_only=False,test_evaluated=True))
        ssh(host,f"from pathlib import Path; import json;p=Path({(REMOTE_RUN+'/jobs/'+jid)!r});[f.unlink() for f in p.glob('epoch_*.pt')];print(json.dumps(True))")
        if marker.get('staged_checkpoint'):ssh(host,f"from pathlib import Path;import json;Path({marker['staged_checkpoint']!r}).unlink(missing_ok=True);print(json.dumps(True))")
    else:assert (out/'FAILURE.json').exists()
    (out/'REMOTE_RESERVATION.json').unlink(missing_ok=True)


def stage_many(host,jids,specs):
    def one(jid):
        marker=json.loads((JOBS/jid/'REMOTE_RESERVATION.json').read_text())
        if not marker.get('ready'):stage(host,specs[jid],marker)
    errors=[]
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures=[pool.submit(one,jid) for jid in jids]
        for f in as_completed(futures):
            try:f.result()
            except Exception as e:errors.append(str(e))
    if errors:raise RuntimeError('; '.join(errors))

def receive_many(host,state,held,specs):
    jobs=[(jid,success) for key,success in [('done',True),('failed',False)] for jid in state[key] if jid in held]
    errors=[]
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures={pool.submit(receive,host,jid,specs[jid],success):jid for jid,success in jobs}
        for f in as_completed(futures):
            jid=futures[f]
            try:f.result();held.pop(jid).close()
            except Exception as e:errors.append(str(e))
    if errors:raise RuntimeError('; '.join(errors))

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
        manifest=json.loads((RUN/'manifest.json').read_text());specs={s['id']:s for s in manifest}
        states={}
        for host in HOSTS:
            try:
                stage_many(host,list(held[host]),specs)
                write_lease(host,list(held[host]))
                state=query(host)
                receive_many(host,state,held[host],specs)
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
                        held[host][jid]=claim
                    stage_many(host,list(held[host]),specs)
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
