from pathlib import Path
import sys,time,json,fcntl,subprocess,traceback,datetime,os
RUN=Path(__file__).resolve().parents[1];sys.path.insert(0,str(RUN/'code'))
from settings import *
from collect_physics import collect
from reevaluation_report import run as report,parent_ledger,choose_parent
from manage import HOSTS,REMOTE_RUN,ssh,start

def healthy():
 # A resource-waiting bootstrap counts as healthy: never bypass its GPU policy.
 names={'bootstrap_efficient_v3.py','resource_scheduler_efficient_v3.py'}
 def local_has(wanted):
  for p in Path('/proc').iterdir():
   if not p.name.isdigit():continue
   try:
    args=[x.decode() for x in (p/'cmdline').read_bytes().split(b'\0') if x]
    if args and 'python' in args[0] and any(str(RUN) in a and Path(a).name in wanted for a in args):return True
   except (OSError,UnicodeError):pass
  return False
 if not local_has({'remote_controller.py'}) or not local_has(names):return False
 for host in HOSTS:
  script=f'''from pathlib import Path
import json
found=False
for p in Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try:
  a=[x.decode() for x in (p/'cmdline').read_bytes().split(b'\\0') if x]
  if a and 'python' in a[0] and any({REMOTE_RUN!r} in s and Path(s).name in {names!r} for s in a):found=True
 except (OSError,UnicodeError):pass
print(json.dumps(found))'''
  if not json.loads(ssh(host,script)):return False
 return True


def wait_state(state,**extra):
 atomic_json(RUN/'operations/FOLLOWUP_STATUS.json',dict(timestamp=now(),pid=os.getpid(),state=state,**extra))

def activate():
 marker=PARENT/'operations/COMPLETE_AUDIT.json'
 if not marker.exists():return False
 audit=json.loads(marker.read_text());assert audit['passed']
 parent=json.loads((PARENT/'manifest.json').read_text());assert audit['jobs']==len(parent)
 if not all((PARENT/'jobs'/s['id']/'SUCCESS.json').exists() for s in parent):return False
 freeze=RUN/'TEST_FREEZE.json'
 if not freeze.exists():
  data,checkpoints=parent_ledger();selected=choose_parent(data)
  atomic_json(freeze,dict(timestamp=now(),parent_completion_passed=True,parent_run=PARENT.name,parent_audit_sha256=sha(marker),parent_jobs=len(parent),data_fingerprint=FINGERPRINT,parent_checkpoints=checkpoints,selection=selected,comparison_manifest_sha256=sha(RUN/'manifest.json'),baseline_configs_sha256=sha(RUN/'baseline_configs.json'),source_freeze_sha256=sha(RUN/'SOURCE_FREEZE.json'),user_authorized='2026-10-28: evaluate2024 after current round; complete per-farm baseline training/evaluation',test_data_not_used_to_select=True))
 for host in HOSTS:
  subprocess.run(['rsync','-a',str(freeze),host+':'+REMOTE_RUN+'/'],check=True,timeout=60)
  assert ssh(host,f"from pathlib import Path;import hashlib;print(hashlib.sha256(Path({(REMOTE_RUN+'/TEST_FREEZE.json')!r}).read_bytes()).hexdigest())").strip()==sha(freeze)
 wait_state('FROZEN_STARTING',main=require_test_freeze()['selection']['choices'])
 if not (RUN/'analysis/PERSISTENCE.json').exists():
  subprocess.run([sys.executable,'-s',str(RUN/'code/persistence.py')],check=True,timeout=600)
 start(False);atomic_json(RUN/'operations/ACTIVATED.json',dict(timestamp=now(),test_freeze_sha256=sha(freeze)))
 return True

def audit_complete():
 manifest=json.loads((RUN/'manifest.json').read_text());freeze=require_test_freeze();trained=0;replayed=0
 for s in manifest:
  p=JOBS/s['id'];assert json.loads((p/'spec.json').read_text())==s
  test=json.loads((p/'test.json').read_text());assert test['year']==2024 and test['selection_year']==2023 and test['data_fingerprint']==FINGERPRINT
  assert test['n_origins']==27982 and test['selection_metric']=='mae'
  if s['task']=='train':
   m=json.loads((p/'metrics.json').read_text());assert not m['smoke_only'] and m['data_fingerprint']==FINGERPRINT
   for sel in m['selections'].values():assert sha(p/sel['checkpoint'])==sel['sha256']
   assert test['source_sha256']==m['selections']['mae']['sha256'];trained+=1
  else:
   assert test['source_sha256']==freeze['parent_checkpoints'][s['source_job']]['sha256'];replayed+=1
 assert len(list((RUN/'analysis/stations').glob('*.csv')))==56
 persist=json.loads((RUN/'analysis/PERSISTENCE.json').read_text());assert len(persist['rows'])==112
 atomic_json(RUN/'operations/COMPLETE_AUDIT.json',dict(timestamp=now(),passed=True,jobs=len(manifest),new_training=trained,replayed=replayed,farms=56,persistence_rows=112,test_year=2024,excluded_month=11))

def main():
 setup();lock=(RUN/'operations/finalizer.lock').open('a+')
 try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 except BlockingIOError:return
 last_export=0.;last_healthy=0.;active=(RUN/'operations/ACTIVATED.json').exists()
 while True:
  try:
   if not active:
    if not activate():
     parent=json.loads((PARENT/'STATUS.json').read_text());wait_state('WAIT_PARENT',parent_completed=parent['completed'],parent_planned=parent['planned'],parent_failed=parent['failed'],test_evaluation_started=False)
     if time.monotonic()-last_export>300:collect();last_export=time.monotonic()
     time.sleep(30);continue
    active=True
   state=collect(export=False)
   if time.monotonic()-last_export>300 or state['completed']==state['planned']:
    collect();report();last_export=time.monotonic()
   wait_state('RUNNING',completed=state['completed'],planned=state['planned'],failed=state['failed'],test_evaluation_started=state['test_year_evaluated'])
   if state['failed']:atomic_json(RUN/'operations/ATTENTION_REQUIRED.json',dict(timestamp=now(),failed_jobs=state['failed_jobs'],reason='failure remains visible and never counts as completion'))
   elif state['completed']==state['planned']:
    audit_complete();atomic_json(RUN/'operations/GLOBAL_COMPLETE.json',dict(timestamp=now(),complete=True));wait_state('COMPLETE',completed=state['completed'],planned=state['planned']);return
   elif time.monotonic()-last_healthy>180:
    if not healthy():start(False)
    last_healthy=time.monotonic()
   (RUN/'operations/FINALIZER_ERROR.json').unlink(missing_ok=True)
  except Exception:
   atomic_json(RUN/'operations/FINALIZER_ERROR.json',dict(timestamp=now(),traceback=traceback.format_exc()));traceback.print_exc();wait_state('RETRY_AFTER_ERROR',error_file='FINALIZER_ERROR.json')
  time.sleep(30)
if __name__=='__main__':main()
