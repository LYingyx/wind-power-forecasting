"""Durable staged DAG: validation-only tuning -> freeze -> train/evaluate/audit."""
from config import *
import argparse,collections,fcntl,sqlite3,subprocess,time,sys

def connect():
 db=sqlite3.connect(RUN/'queue.sqlite',timeout=60);db.row_factory=sqlite3.Row;db.execute('PRAGMA journal_mode=WAL')
 db.execute('CREATE TABLE IF NOT EXISTS jobs(id TEXT PRIMARY KEY, spec TEXT, kind TEXT, grp TEXT, status TEXT, attempts INTEGER DEFAULT 0,pid INTEGER,gpu INTEGER,started TEXT,finished TEXT,error TEXT)')
 return db
def insert(db,s,kind):db.execute('INSERT OR IGNORE INTO jobs(id,spec,kind,grp,status) VALUES(?,?,?,?,?)',(s['id'],json.dumps(s),kind,s['group'],'pending'))
def initialize(db):
 for arm in ['ST00','ST11']:
  for k in [3,5,7]:
   for lr in [2e-4,1e-3]:
    for site in REPS:
     for seed in [40,41]:insert(db,spec(arm,site,seed,'tune',k,lr),'train')
 for group,arms in [('SPACE',['S2','S3']),('S1',['S1']),('TIME',['T1_7','T2_7','T1_8','T2_8']),('TRANSFORMER_FC',[a for a in TRANSFERS if a.startswith('TRANSFORMER')]),('TIDE_FC',[a for a in TRANSFERS if a.startswith('TIDE')])]:
  for arm in arms:
   for lr in [2e-4,1e-3]:
    for site in REPS:
     for seed in [40,41]:
      s=spec(arm,site,seed,'tune',3,lr,group);s['id']=s['id'].replace('_k3_','_kselected_')
      if group!='TIME':s['k']=None
      insert(db,s,'train')
 for arm in FORMAL:
  for site in range(56):
   for seed in range(42,47):
    s=spec(arm,site,seed,'formal');s['k']=None;s['front_lr']=None;insert(db,s,'train')
 for arm in ['W0','ST00','ST11','GSTAR']:
  for site in range(56):
   for seed in range(47,52):
    s=spec(arm,site,seed,'extension');s['k']=None;s['front_lr']=None;insert(db,s,'train')
 db.commit();assert db.execute("SELECT COUNT(*) FROM jobs WHERE kind='train'").fetchone()[0]==8016

def select(db):
 for group,expected in [('ST',144),('SPACE',48),('S1',24),('TIME',96),('TRANSFORMER_FC',72),('TIDE_FC',72)]:
  out=RUN/'selected'/f'{group}.json'
  if out.exists():continue
  rows=db.execute("SELECT * FROM jobs WHERE kind='train' AND grp=? AND id LIKE 'tune_%'",(group,)).fetchall()
  if len(rows)!=expected or any(r['status']!='done' for r in rows):continue
  scores=collections.defaultdict(list);arm_scores=collections.defaultdict(list);hashes={}
  for r in rows:
   m=json.loads((RUN/'jobs'/r['id']/'metrics.json').read_text());assert 'test' not in m
   key=(m['k'],m['front_lr']);scores[key].append(m['val']['mae']);arm_scores[m['arm'],*key].append(m['val']['mae']);hashes[r['id']]=sha(RUN/'jobs'/r['id']/'metrics.json')
  chosen=min(scores,key=lambda k:(float(np.mean(scores[k])),k[0],k[1]))
  atomic_json(out,dict(group=group,k=chosen[0],front_lr=chosen[1],selection_split='2023_only',selection_rule='equal farm/seed/paired-arm MAE, then smaller k/lr',
    scores=[dict(k=k,front_lr=lr,selection_mae=float(np.mean(v))) for (k,lr),v in scores.items()],
    arm_scores=[dict(arm=arm,k=k,front_lr=lr,selection_mae=float(np.mean(v))) for (arm,k,lr),v in arm_scores.items()],trial_metrics_sha256=hashes,selected_at=now()))
  print('SELECT',group,chosen,flush=True)
 frozen=RUN/'ALL_SELECTIONS_FROZEN.json'
 groups=['ST','SPACE','S1','TIME','TRANSFORMER_FC','TIDE_FC']
 if not frozen.exists() and all((RUN/'selected'/f'{g}.json').exists() for g in groups):
  choices={g:json.loads((RUN/'selected'/f'{g}.json').read_text()) for g in groups};candidates=[]
  for g in ['TRANSFORMER_FC','TIDE_FC']:
   cfg=choices[g]
   candidates.extend(x for x in cfg['arm_scores'] if x['arm'].endswith(('__S1','__ST00')) and x['k']==cfg['k'] and x['front_lr']==cfg['front_lr'])
  best=min(candidates,key=lambda x:(x['selection_mae'],x['arm']))
  atomic_json(frozen,dict(timestamp=now(),selection_year=2023,groups=choices,gstar=best['arm'],gstar_candidates=candidates,protocol_sha256=sha(RUN/'PROTOCOL.json')))
  print('ALL SELECTIONS FROZEN',best['arm'],flush=True)

def resolve(s):
 s=dict(s)
 if s['arm']=='GSTAR':
  p=RUN/'ALL_SELECTIONS_FROZEN.json'
  if not p.exists():return None
  arm=json.loads(p.read_text())['gstar'];b,f=decode_arm(arm);s.update(arm=arm,family=arm,backbone=b,front=f,group=b,backbone_config=backbone_config(b))
 if s['stage']=='tune':
  if s['k'] is None:
   p=RUN/'selected/ST.json'
   if not p.exists():return None
   s['k']=json.loads(p.read_text())['k']
  return s
 group=s['group']
 if group=='FIXED':s['k']=3;s['front_lr']=2e-4;return s
 p=RUN/'selected'/f'{group}.json'
 if not p.exists():return None
 cfg=json.loads(p.read_text());s['k']=cfg['k'];s['front_lr']=cfg['front_lr'];return s

def gpu_info():
 out=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.total,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
 return [dict(zip(['index','total','used','util'],map(int,line.split(',')))) for line in out.strip().splitlines()]
def alive(pid,jid):
 try:return jid in Path(f'/proc/{pid}/cmdline').read_bytes().decode().replace('\0',' ')
 except (FileNotFoundError,PermissionError,TypeError):return False
def marker(row):return RUN/('jobs' if row['kind']=='train' else 'evaluation')/row['id']/('TRAIN_SUCCESS.json' if row['kind']=='train' else 'SUCCESS.json')
def add_evaluations(db):
 if not (RUN/'ALL_SELECTIONS_FROZEN.json').exists() or not (RUN/'prepared/test/SUCCESS.json').exists():return
 for r in db.execute("SELECT * FROM jobs WHERE kind='train' AND status='done' AND id NOT LIKE 'tune_%'").fetchall():
  s=json.loads(r['spec']);s['source_job_id']=s['id'];s['id']='eval_'+s['id'];insert(db,s,'evaluate')
 # Original five-seed W0 checkpoints remain immutable and receive new intervention evaluation.
 for site in range(56):
  for seed in range(42,47):
   s=spec('W0',site,seed,'reuse');s['source_job_id']=f'ablation_NEUTRAL_LAST_s{site:02d}_r{seed}';s['legacy_source']=True;s['id']='eval_'+s['id'];insert(db,s,'evaluate')
 db.commit()

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--initialize-only',action='store_true');a=ap.parse_args()
 lock=open(RUN/'scheduler.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 db=connect();initialize(db)
 if a.initialize_only:return
 assert json.loads((RUN/'PREFLIGHT.json').read_text())['passed']
 assert json.loads((RUN/'CONCURRENCY_BENCHMARK.json').read_text())['passed']
 assert json.loads((RUN/'RESUME_CHECK.json').read_text())['passed']
 assert json.loads((RUN/'PIPELINE_CHECK.json').read_text())['passed']
 benchmark=json.loads((RUN/'CONCURRENCY_BENCHMARK.json').read_text())
 hashes={str(p.relative_to(RUN)):sha(p) for p in sorted((RUN/'code').rglob('*.py')) if p.name not in ['scheduler.py','supervise.py','analyze.py','verify_completion.py','benchmark.py','preflight.py']}
 frozen=RUN/'TRAINING_CODE_FROZEN.json'
 if frozen.exists():assert json.loads(frozen.read_text())==hashes
 else:atomic_json(frozen,hashes)
 children={};prep=None;analysis=None;last_analysis=0;caps={i:int(benchmark['recommended_workers']) for i in range(8)}
 if (RUN/'WORKER_CAPS.json').exists():caps={int(k):v for k,v in json.loads((RUN/'WORKER_CAPS.json').read_text()).items()}
 budget=max(2048,int(benchmark['worker_budget_mib']))
 while not (RUN/'STOP').exists():
  for row in db.execute("SELECT * FROM jobs WHERE status='running'").fetchall():
   proc=children.get(row['id']);running=proc.poll() is None if proc else alive(row['pid'],row['id'])
   if running:continue
   if marker(row).exists():db.execute("UPDATE jobs SET status='done',finished=?,error=NULL WHERE id=?",(now(),row['id']))
   else:
    fail=marker(row).parent/'FAILURE.json';err=fail.read_text()[-6000:] if fail.exists() else 'Exited without completion marker; resume checkpoint'
    if 'out of memory' in err.lower():caps[row['gpu']]=max(1,caps[row['gpu']]//2);atomic_json(RUN/'WORKER_CAPS.json',caps)
    status='pending' if row['attempts']<3 else 'failed';db.execute('UPDATE jobs SET status=?,pid=NULL,error=? WHERE id=?',(status,err,row['id']))
   children.pop(row['id'],None)
  db.commit();select(db)
  if (RUN/'ALL_SELECTIONS_FROZEN.json').exists() and not (RUN/'prepared/test/SUCCESS.json').exists():
   if prep is None:
    log=open(RUN/'prepare_test.log','a');prep=subprocess.Popen([sys.executable,str(RUN/'code/prepare.py'),'--splits','test'],stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'));log.close()
   elif prep.poll() is not None:
    if prep.returncode:raise RuntimeError('Test preparation failed; see prepare_test.log')
  add_evaluations(db)
  active=list(db.execute("SELECT * FROM jobs WHERE status='running'"));counts=collections.Counter(r['gpu'] for r in active)
  pending=list(db.execute("SELECT * FROM jobs WHERE status='pending' ORDER BY CASE WHEN id LIKE 'tune_%' THEN 0 WHEN kind='train' THEN 1 ELSE 2 END,id"))
  ready=[]
  for row in pending:
   s=resolve(json.loads(row['spec']))
   if s is not None:ready.append((row,s))
  gpus=gpu_info()
  for gpu in sorted(gpus,key=lambda x:(counts[x['index']],x['util'],-x['total']+x['used'])):
   if not ready:break
   gi=gpu['index'];free=gpu['total']-gpu['used']
   if counts[gi]>=caps[gi] or free<4096+budget:continue
   # Saturated external cards are reconsidered continuously, not oversubscribed blindly.
   if counts[gi]==0 and gpu['util']>=95:continue
   row,s=ready.pop(0);sp=RUN/'specs'/f"{row['id']}.json";atomic_json(sp,s)
   log=open(RUN/'logs'/f"{row['id']}.log",'a');env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gi),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONUNBUFFERED='1',PYTHONHASHSEED=str(s['seed']))
   worker=RUN/'code/train.py' if row['kind']=='train' else RUN/'maintenance/evaluate_compact.py'
   proc=subprocess.Popen([sys.executable,str(worker),'--spec',str(sp)],stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,env=env,cwd=ROOT,start_new_session=True);log.close();children[row['id']]=proc
   db.execute("UPDATE jobs SET spec=?,status='running',attempts=attempts+1,pid=?,gpu=?,started=? WHERE id=?",(json.dumps(s),proc.pid,gi,now(),row['id']));db.commit();counts[gi]+=1
   print('LAUNCH',gi,row['id'],proc.pid,flush=True)
  bykind=[dict(r) for r in db.execute('SELECT kind,status,COUNT(*) count FROM jobs GROUP BY kind,status')]
  running=[]
  for r in db.execute("SELECT id,kind,gpu,pid,started FROM jobs WHERE status='running'"):
   x=dict(r);p=RUN/('jobs' if r['kind']=='train' else 'evaluation')/r['id']/'progress.json'
   if p.exists():
    try:x['progress']=json.loads(p.read_text())
    except json.JSONDecodeError:pass
   running.append(x)
  atomic_json(RUN/'STATUS.json',dict(timestamp=now(),scheduler_pid=os.getpid(),by_kind=bykind,running=running,gpus=gpus,worker_caps=caps,
    selections_frozen=(RUN/'ALL_SELECTIONS_FROZEN.json').exists(),planned_training_jobs=8016,selected_groups=[p.stem for p in (RUN/'selected').glob('*.json')],failures=[dict(x) for x in db.execute("SELECT id,error FROM jobs WHERE status='failed'")]))
  if analysis is not None and analysis.poll() is not None:analysis=None
  if analysis is None and time.time()-last_analysis>600:
   log=open(RUN/'analysis.log','a');analysis=subprocess.Popen([sys.executable,str(RUN/'code/analyze.py')],stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OPENBLAS_NUM_THREADS='1'));log.close();last_analysis=time.time()
  train_done=db.execute("SELECT COUNT(*) FROM jobs WHERE kind='train' AND status='done'").fetchone()[0]
  eval_done=db.execute("SELECT COUNT(*) FROM jobs WHERE kind='evaluate' AND status='done'").fetchone()[0]
  failed=db.execute("SELECT COUNT(*) FROM jobs WHERE status='failed'").fetchone()[0]
  if train_done==8016 and eval_done==7840:
   if analysis is not None:analysis.wait()
   subprocess.run([sys.executable,str(RUN/'maintenance/storage.py')],check=True)
   subprocess.run([sys.executable,str(RUN/'code/analyze.py')],check=True)
   subprocess.run([sys.executable,str(RUN/'code/verify_completion.py')],check=True)
   atomic_json(RUN/'FINISHED.json',dict(timestamp=now(),all_experiments_complete=True,train_jobs=train_done,evaluation_jobs=eval_done));return
  if failed and not running and not ready:
   atomic_json(RUN/'BLOCKED.json',dict(timestamp=now(),reason='Exhausted retries; unresolved selection or evaluation dependency',failed_jobs=failed))
   raise RuntimeError('Queue blocked by failed dependencies; see STATUS.json')
  time.sleep(3)

if __name__=='__main__':main()
