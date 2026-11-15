from pathlib import Path
import sys,time,json,fcntl,subprocess,traceback
RUN=Path(__file__).resolve().parents[1];sys.path.insert(0,str(RUN/'code'))
from settings import *
from collect_physics import collect
from report import run as report
from manage import HOSTS,REMOTE_RUN,ssh,start

def activate_extension(chosen):
 screen=json.loads((RUN/'screen_manifest.json').read_text());candidates=json.loads((RUN/'candidate_manifest.json').read_text())
 extension=[s for s in candidates if s['seed']>=44 and s['configuration'] in chosen]
 manifest=screen+extension
 atomic_json(RUN/'operations/REPLICATION_ACTIVATION.json',dict(timestamp=now(),configurations=chosen,additional_jobs=len(extension),selection_year=2023,rule='frozen paired both-metric threshold and high-power guard',report_sha256=sha(RUN/'analysis/REPORT.json')))
 atomic_json(RUN/'manifest.json',manifest)
 for host in HOSTS:
  subprocess.run(['rsync','-a',str(RUN/'manifest.json'),host+':'+REMOTE_RUN+'/'],check=True,timeout=60)
  ssh(host,f"from pathlib import Path; Path({REMOTE_RUN+'/operations/GLOBAL_COMPLETE.json'!r}).unlink(missing_ok=True)")
 (RUN/'operations/GLOBAL_COMPLETE.json').unlink(missing_ok=True)
 if extension:start(False)
 return len(extension)

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

def audit_complete():
 manifest=json.loads((RUN/'manifest.json').read_text());verified=0
 for spec in manifest:
  folder=JOBS/spec['id'];m=json.loads((folder/'metrics.json').read_text())
  assert m['data_fingerprint']==FINGERPRINT and not m['smoke_only']
  assert json.loads((folder/'spec.json').read_text())==spec
  for selection in m['selections'].values():assert sha(folder/selection['checkpoint'])==selection['sha256'];verified+=1
 atomic_json(RUN/'operations/COMPLETE_AUDIT.json',dict(timestamp=now(),passed=True,jobs=len(manifest),verified_selected_weights=verified,test_2024_evaluated=False))

def main():
 setup();lock=(RUN/'operations/finalizer.lock').open('a+')
 try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 except BlockingIOError:return
 while True:
  try:
   state=collect();r=report()
   atomic_json(RUN/'operations/FINALIZER_STATUS.json',dict(timestamp=now(),state=state,automatic_replication=True))
   if state['failed']:
    atomic_json(RUN/'operations/ATTENTION_REQUIRED.json',dict(timestamp=now(),failed_jobs=state['failed_jobs'],reason='scientific/runtime failure; not silently marked completed'))
   elif state['completed']==state['planned']:
    marker=RUN/'operations/REPLICATION_ACTIVATION.json'
    if not marker.exists() and activate_extension(r['replicate_configurations']):time.sleep(20);continue
    audit_complete();report();atomic_json(RUN/'operations/GLOBAL_COMPLETE.json',dict(timestamp=now(),complete=True))
    print('All activated experiments complete and verified',flush=True);return
   elif not healthy():start(False)
  except Exception:
   atomic_json(RUN/'operations/FINALIZER_ERROR.json',dict(timestamp=now(),traceback=traceback.format_exc()));traceback.print_exc()
  time.sleep(60)
if __name__=='__main__':main()
