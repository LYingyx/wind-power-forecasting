from pathlib import Path
import sys,time,json,fcntl,traceback,subprocess,os
RUN=Path(__file__).resolve().parents[1];sys.path.insert(0,str(RUN/'code'))
from settings import *
from collect_physics import collect
from followup_report import run as report
from manage import freeze,deploy,start,HOSTS,REMOTE_RUN,ssh


def wait_state(state,**kwargs):
 atomic_json(RUN/'operations/FOLLOWUP_STATUS.json',dict(timestamp=now(),pid=os.getpid(),state=state,**kwargs))


def parent_stopped():
 names={'scheduled_worker.py','resource_scheduler_efficient_v3.py','bootstrap_efficient_v3.py','remote_controller.py'}
 def script(run):return f'''from pathlib import Path
import json
found=[]
for p in Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try:
  args=[s.decode() for s in (p/'cmdline').read_bytes().split(b'\\0') if s]
  if args and 'python' in args[0] and any({run!r} in a and Path(a).name in {names!r} for a in args):found.append(p.name)
 except (OSError,UnicodeError):pass
print(json.dumps(found))'''
 local=subprocess.check_output([sys.executable,'-c',script(str(PARENT))],text=True)
 if json.loads(local):return False
 for host in HOSTS:
  if json.loads(ssh(host,script('/home/liying/wind_physics_project/project/runs/'+PARENT.name))):return False
 return True


def activate():
 audit=PARENT/'operations/COMPLETE_AUDIT.json';selection=PARENT/'SELECTION_FREEZE.json'
 if not audit.exists() or not selection.exists():return False
 a=json.loads(audit.read_text());f=json.loads(selection.read_text())
 assert a['passed'] and a['jobs']==2800 and not f['test_year_evaluated']
 selected=f['selected']
 if selected is None:
  wait_state('SKIPPED_NO_2023_CANDIDATE_PASSED',parent_audit_sha256=sha(audit));return 'skipped'
 if not parent_stopped():
  wait_state('WAIT_PARENT_WORKERS_EXIT',selected=selected);return False
 if not (RUN/'TEST_FREEZE.json').exists():
  candidates=json.loads((RUN/'candidate_manifest.json').read_text())
  specs=[s for s in candidates if s['configuration'] in [selected,'W0_H32','TF_R3_RAW']]
  assert len(specs)==1680
  for s in specs:
   p=PARENT/'jobs'/s['source_job'];record=f['checkpoints'][s['source_job']]
   assert sha(p/record['checkpoint'])==record['sha256'] and sha(p/'spec.json')==record['spec_sha256'] and sha(p/'normalization.json')==record['normalization_sha256']
  atomic_json(RUN/'manifest.json',specs)
  freeze()
  atomic_json(RUN/'TEST_FREEZE.json',dict(timestamp=now(),selected=selected,parent_completion_passed=True,parent_audit_sha256=sha(audit),parent_selection_freeze_sha256=sha(selection),parent_checkpoints=f['checkpoints'],comparison_manifest_sha256=sha(RUN/'manifest.json'),source_freeze_sha256=sha(RUN/'SOURCE_FREEZE.json'),data_fingerprint=FINGERPRINT,interpretation='Exploratory2024 after prior test inspection; selection only2023',test_data_used_for_this_selection=False))
 for host in HOSTS:
  deploy(host)
  subprocess.run(['rsync','-a',str(RUN/'TEST_FREEZE.json'),host+':'+REMOTE_RUN+'/'],check=True,timeout=60)
  assert ssh(host,f"from pathlib import Path;import hashlib;print(hashlib.sha256(Path({(REMOTE_RUN+'/TEST_FREEZE.json')!r}).read_bytes()).hexdigest())").strip()==sha(RUN/'TEST_FREEZE.json')
 start(False)
 atomic_json(RUN/'operations/ACTIVATED.json',dict(timestamp=now(),selected=selected,jobs=1680))
 return True


def audit_complete():
 f=require_test_freeze();specs=json.loads((RUN/'manifest.json').read_text());assert len(specs)==1680
 for s in specs:
  p=JOBS/s['id'];assert (p/'SUCCESS.json').exists() and json.loads((p/'spec.json').read_text())==s
  t=json.loads((p/'test.json').read_text());assert t['year']==2024 and t['n_origins']==27982 and t['selection_year']==2023 and t['selection_metric']=='mae'
  assert t['source_sha256']==f['parent_checkpoints'][s['source_job']]['sha256'] and t['data_fingerprint']==FINGERPRINT
 assert len(list((RUN/'analysis/stations').glob('*.csv')))==56
 atomic_json(RUN/'operations/COMPLETE_AUDIT.json',dict(timestamp=now(),passed=True,jobs=1680,farms=56,seeds=10,selected=f['selected'],evaluation_year=2024,exploratory=True))


def main():
 setup();lock=(RUN/'operations/finalizer.lock').open('a+')
 try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 except BlockingIOError:return
 last=0.;active=(RUN/'operations/ACTIVATED.json').exists()
 while True:
  try:
   if not active:
    activated=activate()
    if activated=='skipped':return
    if not activated:
     s=json.loads((PARENT/'STATUS.json').read_text());wait_state('WAIT_2023_COMPLETE_AND_SELECTION_FREEZE',completed=s['completed'],planned=s['planned'],failed=s['failed'],test_evaluated=False)
     time.sleep(30);continue
    active=True
   state=collect(False);complete=state['completed']==1680 and not state['failed']
   if time.monotonic()-last>300 or complete:report();last=time.monotonic()
   wait_state('RUNNING',completed=state['completed'],planned=1680,failed=state['failed'])
   if state['failed']:atomic_json(RUN/'operations/ATTENTION_REQUIRED.json',dict(timestamp=now(),failed_jobs=state['failed_jobs']))
   if complete:
    audit_complete();atomic_json(RUN/'operations/GLOBAL_COMPLETE.json',dict(timestamp=now(),complete=True));wait_state('COMPLETE',completed=1680,planned=1680);return
   (RUN/'operations/FINALIZER_ERROR.json').unlink(missing_ok=True)
  except Exception:
   atomic_json(RUN/'operations/FINALIZER_ERROR.json',dict(timestamp=now(),traceback=traceback.format_exc()));traceback.print_exc()
  time.sleep(30)

if __name__=='__main__':main()
