"""Audit incomplete output and reconcile markers after the quota interruption."""
from pathlib import Path
import sys,json,sqlite3,hashlib,os,datetime
import numpy as np
RUN=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(RUN/'code'))
from config import atomic_json,sha,now,CACHE
assert (RUN/'evaluation').is_symlink() and (RUN/'STORAGE_RELOCATION.json').exists()
for proc in Path('/proc').iterdir():
 if not proc.name.isdigit() or int(proc.name)==os.getpid():continue
 try:cmd=(proc/'cmdline').read_bytes().replace(b'\0',b' ').decode()
 except (FileNotFoundError,PermissionError,ProcessLookupError):continue
 assert not (str(RUN) in cmd and any(x in cmd for x in ['/code/evaluate.py --spec','/maintenance/evaluate_compact.py --spec','/code/scheduler.py'])),cmd
freeze=json.loads((RUN/'TRAINING_CODE_FROZEN.json').read_text());assert all(sha(RUN/k)==v for k,v in freeze.items())
db=sqlite3.connect(RUN/'queue.sqlite');db.row_factory=sqlite3.Row
assert db.execute('pragma integrity_check').fetchall()[0][0]=='ok'
rows=db.execute('select * from jobs').fetchall();train=0;complete=0;changed=[];removed=[]
shape=(len(np.load(CACHE/'test/origin_unix.npy')),24)
for row in rows:
 if row['kind']=='train':
  assert row['status']=='done' and (RUN/'jobs'/row['id']/'TRAIN_SUCCESS.json').exists()
  train+=1;continue
 folder=RUN/'evaluation'/row['id'];success=folder/'SUCCESS.json'
 if success.exists():
  m=json.loads((folder/'metrics.json').read_text());assert m['selection_freeze_sha256']==sha(RUN/'ALL_SELECTIONS_FROZEN.json')
  for label,metrics in m['interventions'].items():
   f=folder/f'{label}_prediction.npy';assert f.exists() and sha(f)==metrics['prediction_sha256'],str(f)
   pred=np.load(f,mmap_mode='r');assert pred.shape==shape and np.isfinite(pred).all()
  if row['status']!='done':
   db.execute("update jobs set status='done',finished=?,error=NULL where id=?",(m['completed_at'],row['id']));changed.append(dict(id=row['id'],status='done',reason='valid completion marker'))
  complete+=1
 else:
  for metrics_path in folder.glob('*_metrics.json'):
   pred_path=folder/(metrics_path.name.removesuffix('_metrics.json')+'_prediction.npy')
   try:
    m=json.loads(metrics_path.read_text());pred=np.load(pred_path,mmap_mode='r')
    assert pred.shape==shape and np.isfinite(pred).all() and sha(pred_path)==m['prediction_sha256']
   except (ValueError,OSError,KeyError,AssertionError,EOFError):
    # Removing only the invalid cached metric forces the frozen evaluator to recompute this variant.
    removed.append(dict(path=str(metrics_path.relative_to(RUN)),reason='invalid incomplete cached variant',bytes=metrics_path.stat().st_size));metrics_path.unlink()
  if row['status']!='pending':
   db.execute("update jobs set status='pending',pid=NULL,error=? where id=?",('Storage quota interruption; recovered without changing scientific code',row['id']));changed.append(dict(id=row['id'],status='pending',reason='resume incomplete evaluation'))
assert train==8016
for parent in [RUN/'evaluation',RUN/'specs',RUN/'jobs',RUN/'analysis']:
 for f in parent.rglob('*.tmp'):
  removed.append(dict(path=str(f.relative_to(RUN)),bytes=f.stat().st_size,reason='orphan temporary file; no active workers'));f.unlink()
db.commit()
assert db.execute('pragma integrity_check').fetchall()[0][0]=='ok'
counts=[dict(r) for r in db.execute('select kind,status,count(*) count from jobs group by kind,status')];db.close()
atomic_json(RUN/'QUOTA_RECOVERY.json',dict(timestamp=now(),passed=True,training_jobs_complete=train,evaluations_complete=complete,queue_integrity='ok',frozen_code_unchanged=True,changes=changed,removed_intermediate_files=removed,by_kind=counts))
print('RECOVERED',counts,'removed_temporary_or_invalid_files',len(removed),flush=True)
