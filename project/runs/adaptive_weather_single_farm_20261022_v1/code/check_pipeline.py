from config import *
import tempfile,sqlite3
import scheduler
from data import Data
from evaluate import donors

def main():
 checks={};saved=scheduler.RUN
 with tempfile.TemporaryDirectory() as td:
  scheduler.RUN=Path(td);(scheduler.RUN/'selected').mkdir();atomic_json(scheduler.RUN/'PROTOCOL.json',PROTOCOL)
  db=scheduler.connect();scheduler.initialize(db)
  assert db.execute('SELECT COUNT(*) FROM jobs').fetchone()[0]==8016
  assert scheduler.resolve(spec('S3',6,42,'formal')) is None
  assert not (scheduler.RUN/'ALL_SELECTIONS_FROZEN.json').exists()
  # Synthetic scalar fixtures exercise selection dependencies, not model results.
  for group in ['ST','SPACE','S1','TIME','TRANSFORMER_FC','TIDE_FC']:
   for r in db.execute("SELECT * FROM jobs WHERE id LIKE 'tune_%' AND grp=?",(group,)).fetchall():
    s=scheduler.resolve(json.loads(r['spec']));assert s is not None
    m={**s,'val':{'mae':.1+.001*s['k']+.01*int(s['front_lr']>.0002)}}
    atomic_json(scheduler.RUN/'jobs'/r['id']/'metrics.json',m)
    db.execute("UPDATE jobs SET status='done',spec=? WHERE id=?",(json.dumps(s),r['id']))
   db.commit();scheduler.select(db)
  assert (scheduler.RUN/'ALL_SELECTIONS_FROZEN.json').exists()
  frozen=json.loads((scheduler.RUN/'ALL_SELECTIONS_FROZEN.json').read_text());assert frozen['groups']['ST']['k']==3
  assert all(x['front_lr']==2e-4 for x in frozen['groups'].values())
  assert frozen['gstar'] in TRANSFERS
  assert all(scheduler.resolve(json.loads(r[0])) is not None for r in db.execute("SELECT spec FROM jobs WHERE status='pending'"))
  db.close();checks['8016_job_dag_and_selection_dependencies']=True
 scheduler.RUN=saved
 try:Data('test',6,3,'cpu',limit=1)
 except AssertionError:checks['2024_guard_before_freeze']=True
 else:raise AssertionError('Test loader opened before freeze')
 # Smoke evaluation includes all five weather and five weight permutations.
 path=RUN/'smoke_evaluation/eval_smoke_ST11_s06_r42';m=json.loads((path/'metrics.json').read_text())
 assert m['evaluation_split']=='val' and m['smoke_only'] and 'test' not in m
 assert len(m['interventions'])==11
 y=np.load(OLD/'prepared/val/06/target.npy',mmap_mode='r')[:2048]
 for label,value in m['interventions'].items():
  p=np.load(path/f'{label}_prediction.npy');e=p.astype(np.float64)-y.astype(np.float64)
  assert abs(np.abs(e).mean()-value['mae'])<1e-12 and abs(np.sqrt((e*e).mean())-value['rmse'])<1e-12
 checks['11_validation_interventions_recomputed']=True
 for seed in PROTOCOL['intervention_seeds']:
  d=donors('val',seed);assert np.array_equal(np.sort(d),np.arange(len(d)))
 checks['full_validation_derangements']=True
 atomic_json(RUN/'PIPELINE_CHECK.json',dict(passed=True,checks=checks,completed_at=now()))
 print(json.dumps(checks,indent=2),flush=True)

if __name__=='__main__':main()
