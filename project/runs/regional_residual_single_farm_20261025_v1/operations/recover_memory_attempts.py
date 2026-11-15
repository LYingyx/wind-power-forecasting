"""Archive real resource failures and retry under corrected RAM admission."""
from pathlib import Path
import os,sys,json,signal,subprocess,shlex,fcntl,datetime,time
RUN=Path(__file__).resolve().parents[1];sys.path.insert(0,str(RUN/'code'))
from settings import JOBS,atomic_json,now

pid=json.loads((RUN/'operations/CPU_REMOTE_STATUS.json').read_text())['pid']
cmd=(Path('/proc')/str(pid)/'cmdline').read_bytes().decode();assert str(RUN/'operations/cpu_remote_controller.py') in cmd
os.kill(pid,signal.SIGSTOP);os.kill(pid,signal.SIGKILL)
for _ in range(30):
 p=Path('/proc')/str(pid)/'stat'
 if not p.exists() or p.read_text().rsplit(')',1)[1].split()[0] in ['Z','X']:break
 time.sleep(.05)
recovered=[];skipped=[];stamp=datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
for failure in sorted(JOBS.glob('ARX*/FAILURE.json')):
 out=failure.parent;detail=json.loads(failure.read_text());jid=out.name
 assert detail.get('return_code') in [-9,-15],(jid,detail)
 assert not (out/'SUCCESS.json').exists()
 lock=(out/'worker.lock').open('a+')
 try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 except BlockingIOError:lock.close();skipped.append(jid);continue
 # The only affected host in this incident is 6001; source runtime proves it.
 runtime=json.loads((out/'runtime.json').read_text());assert runtime['host']=='iipserver1'
 remote='/home/liying/wind_physics_project/project/runs/'+RUN.name+'/jobs/'+jid
 script=f'''from pathlib import Path
import json,os
p=Path({remote!r});runtime=json.loads((p/'runtime.json').read_text());pid=runtime['pid'];stat=Path('/proc')/str(pid)/'stat'
if stat.exists():assert stat.read_text().rsplit(')',1)[1].split()[0] in ['Z','X']
assert not (p/'SUCCESS.json').exists()
(p/'CPU_RETRY_HOLD.json').write_text(json.dumps(dict(reason='archived resource interruption; awaiting a fresh CPU lease')))
a=p/'attempts'/{stamp!r};a.mkdir(parents=True,exist_ok=True)
for f in list(p.iterdir()):
 if f.is_file() and f.name not in ['worker.lock','CPU_RETRY_HOLD.json']:os.replace(f,a/f.name)
print(json.dumps(dict(archived=True)))'''
 subprocess.run(['ssh','-o','BatchMode=yes','v1006001','python3 -c '+shlex.quote(script)],check=True,capture_output=True,text=True,timeout=25)
 archive=out/'attempts'/stamp;archive.mkdir(parents=True,exist_ok=True)
 for p in list(out.iterdir()):
  if p.is_file() and p.name!='worker.lock':os.replace(p,archive/p.name)
 atomic_json(out/'RETRY_AUTHORIZATION.json',dict(timestamp=now(),reason='account cgroup memory admission fixed; same scientific spec',previous_attempt=str(archive),previous_return_code=detail['return_code']))
 recovered.append(dict(id=jid,previous_return_code=detail['return_code'],archive=str(archive)))
 lock.close()
atomic_json(RUN/'operations/RESOURCE_RECOVERY.json',dict(timestamp=now(),cause='v1006001 user cgroup memory.max 101403095040 bytes was missed by host-only admission',recovered=recovered,skipped_locked=skipped,model_or_protocol_changed=False))
env=dict(os.environ,PYTHONNOUSERSITE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='')
with (RUN/'operations/cpu_remote_controller.log').open('a') as f:child=subprocess.Popen([sys.executable,'-s',str(RUN/'operations/cpu_remote_controller.py')],env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps(dict(recovered=len(recovered),skipped=skipped,controller_pid=child.pid)))
