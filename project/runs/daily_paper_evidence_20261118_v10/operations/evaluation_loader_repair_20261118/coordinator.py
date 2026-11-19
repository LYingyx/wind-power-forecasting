"""Collect verified outcomes; select linear hyperparameters; keep all three queues alive."""
from pathlib import Path
import sys,json,time,os,subprocess,shlex,fcntl,traceback
RUN=Path(__file__).resolve().parents[1];sys.path.insert(0,str(RUN/'code'))
from settings import atomic_json,now,sha
from collect import collect
from uncertainty import report as uncertainty_report
HOSTS=['v1006001','v1006002'];REMOTE='/home/liying/wind_physics_project/project/runs/'+RUN.name
RPY='/home/liying/wind_physics_project/project/.runtime/chronos/bin/python'

def remote(h,code,timeout=40):
 p=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=8',h,'python3 -c '+shlex.quote(code)],capture_output=True,text=True,check=True,timeout=timeout)
 return json.loads(p.stdout)

def select(specs):
 p=RUN/'assets/selected/LINEAR.json'
 if p.exists():return
 ts=[s for s in specs if s.get('stage')=='tune']
 if not all((RUN/'jobs'/s['id']/'DONE.json').exists() for s in ts):return
 scores=[]
 for c in range(8):
  jobs=[s for s in ts if s['candidate']==c]
  score=sum(json.loads((RUN/'jobs'/s['id']/'metrics.json').read_text())['metrics']['mae'] for s in jobs)/len(jobs)
  scores.append(dict(candidate=c,mae=score,config=jobs[0]['config']))
 winner=min(scores,key=lambda x:(x['mae'],x['candidate']))
 atomic_json(p,dict(timestamp=now(),config=winner['config'],candidate=winner['candidate'],scores=scores,selection='mean 2023 macroMAE over seeds40,41; no 2024 access'))

def main():
 lock=(RUN/'operations/coordinator.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 specs=json.loads((RUN/'manifest.json').read_text());byid={s['id']:s for s in specs};cycles=0;last_hour=0.
 while True:
  errors=[];states={};assign=json.loads((RUN/'operations/ASSIGNMENTS.json').read_text())
  for host in HOSTS:
   ids=[j for j,h in assign.items() if h==host]
   try:
    x=remote(host,f'''from pathlib import Path
import json,time
r=Path({REMOTE!r});p=r/'schedulers/status.json'
print(json.dumps(dict(done=[j for j in {ids!r} if (r/'jobs'/j/'DONE.json').exists()],status=json.loads(p.read_text()) if p.exists() else {{}},fresh=p.exists() and time.time()-p.stat().st_mtime<180)))''')
    states[host]=x
    subprocess.run(['rsync','-a','--exclude=DONE.json','--include=*/','--include=*.json','--exclude=*',host+':'+REMOTE+'/jobs/',str(RUN/'jobs')+'/'],check=True,timeout=50,stdout=subprocess.DEVNULL)
    for jid in x['done']:
     out=RUN/'jobs'/jid
     if (out/'DONE.json').exists():continue
     subprocess.run(['rsync','-a','--exclude=DONE.json','--exclude=worker.lock',host+':'+REMOTE+'/jobs/'+jid+'/',str(out)+'/'],check=True,timeout=90,stdout=subprocess.DEVNULL)
     s=byid[jid]
     if s.get('task')!='evaluate':
      m=json.loads((out/'metrics.json').read_text());assert m['requested_spec']==s and not m['smoke_only'];assert sha(out/'best.pt')==m['checkpoint_sha256']
     if s.get('stage')!='tune':
      for split in s.get('splits',['val','test']):
       m=json.loads((out/f'metrics_{split}.json').read_text());assert m['requested_spec']==s and not m['smoke_only']
       expected=s.get('checkpoint_sha256') if s.get('task')=='evaluate' else sha(out/'best.pt');assert m['checkpoint_sha256']==expected
     atomic_json(out/'DONE.json',dict(timestamp=now(),passed=True,source_host=host,verified_return=True))
   except Exception as e:errors.append(dict(host=host,error=repr(e)))
  select(specs)
  for host in HOSTS:
   try:
    subprocess.run(['rsync','-a',str(RUN/'assets/selected')+'/',host+':'+REMOTE+'/assets/selected/'],check=True,timeout=30,stdout=subprocess.DEVNULL)
   except Exception as e:errors.append(dict(host=host,selection_sync=repr(e)))
  lp=RUN/'schedulers/status.json';states['local4090']=dict(status=json.loads(lp.read_text()) if lp.exists() else {},fresh=lp.exists() and time.time()-lp.stat().st_mtime<180)
  done=[s['id'] for s in specs if (RUN/'jobs'/s['id']/'DONE.json').exists()];fail=[s['id'] for s in specs if (RUN/'jobs'/s['id']/'FAILURE.json').exists()]
  active=[a for h in states.values() if h.get('fresh') for a in h['status'].get('active',[]) if a['id'] not in done]
  evidence=collect()
  try:uncertainty_report()
  except Exception as e:errors.append(dict(uncertainty=repr(e)))
  status=dict(timestamp=now(),total=len(specs),completed=len(done),active=active,failures=fail,errors=errors,hosts=states,evidence=evidence)
  atomic_json(RUN/'STATUS.json',status)
  if time.monotonic()-last_hour>=3600:
   atomic_json(RUN/'HOURLY_STATUS.json',status);last_hour=time.monotonic()
  if len(done)==len(specs):atomic_json(RUN/'COMPLETE.json',status);break
  # Report errors, never silently call a stalled queue complete.
  cycles+=1;time.sleep(60)
if __name__=='__main__':main()
