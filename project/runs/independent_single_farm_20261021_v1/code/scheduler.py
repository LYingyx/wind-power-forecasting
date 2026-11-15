"""Durable SQLite DAG, resource-aware CUDA workers, validation-only selection."""
from common import *
import argparse, collections, datetime, fcntl, signal, sqlite3, subprocess, time, traceback

def now():return datetime.datetime.now().astimezone().isoformat()

def connect():
    db=sqlite3.connect(RUN/'queue.sqlite',timeout=30)
    db.row_factory=sqlite3.Row;db.execute('PRAGMA journal_mode=WAL')
    db.execute('CREATE TABLE IF NOT EXISTS jobs(id TEXT PRIMARY KEY, spec TEXT, family TEXT, site INTEGER, seed INTEGER, stage TEXT, priority INTEGER, dependency TEXT, selection TEXT, status TEXT, attempts INTEGER DEFAULT 0, pid INTEGER, gpu INTEGER, started TEXT, finished TEXT, error TEXT)')
    return db

def insert(db,spec,priority,dependency=None,selection=None):
    db.execute('INSERT OR IGNORE INTO jobs(id,spec,family,site,seed,stage,priority,dependency,selection,status) VALUES(?,?,?,?,?,?,?,?,?,?)',
        (spec['id'],json.dumps(spec),spec['family'],spec['site'],spec['seed'],spec['stage'],priority,dependency,selection,'pending'))

def initialize(db):
    ids=list(map(str,static()['station_id'])); reps=[ids.index(s) for s in REPRESENTATIVES]
    for site in range(56):
        for seed in SEEDS+(TUNE_SEEDS if site in reps else []):
            insert(db,dict(id=ref_id(site,seed),family='B0',site=site,seed=seed,stage='reference',config=dict(width=64,lr=2e-4,weight_decay=1e-4)),0 if seed in TUNE_SEEDS else 30)
    for family in ['E2C_CBH']+FAMILIES:
        for ci in range(4):
            for site in reps:
                for seed in TUNE_SEEDS:
                    spec=dict(id=f'tune_{family}_c{ci}_s{site:02d}_r{seed}',family=family,site=site,seed=seed,stage='tune',config_index=ci,config=config_for(family,ci))
                    insert(db,spec,10,ref_id(site,seed) if family=='E2C_CBH' else None)
        for site in range(56):
            for seed in SEEDS:
                spec=dict(id=f'formal_{family}_s{site:02d}_r{seed}',family=family,site=site,seed=seed,stage='formal',config=None)
                insert(db,spec,20 if family=='E2C_CBH' else 30,ref_id(site,seed) if family=='E2C_CBH' else None,family)
    for arm in ARMS:
        for site in range(56):
            for seed in SEEDS if arm=='E0' else SEEDS[:5]:
                spec=dict(id=f'ablation_{arm}_s{site:02d}_r{seed}',family=arm,site=site,seed=seed,stage='ablation',config=None)
                insert(db,spec,40,ref_id(site,seed),'E2C_CBH')
    db.commit();assert db.execute('SELECT COUNT(*) FROM jobs').fetchone()[0]==9500

def gpu_info():
    out=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.total,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    return [dict(zip(['index','total','used','utilization'],map(int,line.split(',')))) for line in out.strip().splitlines()]

def pid_is_job(pid,jid):
    try:return jid in Path(f'/proc/{pid}/cmdline').read_bytes().decode().replace('\0',' ')
    except (FileNotFoundError,PermissionError):return False

def select_completed(db):
    directory=RUN/'selected';directory.mkdir(exist_ok=True)
    for family in ['E2C_CBH']+FAMILIES:
        dest=directory/f'{family}.json'
        if dest.exists():continue
        rows=db.execute("SELECT * FROM jobs WHERE family=? AND stage='tune'",(family,)).fetchall()
        if len(rows)!=48 or any(r['status']!='done' for r in rows):continue
        results=collections.defaultdict(list);hashes={}
        for r in rows:
            spec=json.loads(r['spec']);path=RUN/'jobs'/r['id']/'metrics.json';m=json.loads(path.read_text())
            assert 'test' not in m
            results[spec['config_index']].append(m['validation_selection_mae']);hashes[r['id']]=sha(path)
        scores={k:float(np.mean(v)) for k,v in results.items()};assert all(len(v)==12 for v in results.values())
        chosen=min(scores,key=lambda k:(scores[k],k))
        atomic_json(dest,dict(family=family,selected_config_index=chosen,config=config_for(family,chosen),validation_macro_mae=scores,
                            trial_metrics_sha256=hashes,selected_at=now(),selection_split='2023_validation_only'))
        print('SELECTED',family,chosen,scores,flush=True)

def status(db,gpus):
    counts=dict(db.execute('SELECT status,COUNT(*) FROM jobs GROUP BY status').fetchall())
    families=[dict(r) for r in db.execute('SELECT family,stage,status,COUNT(*) AS count FROM jobs GROUP BY family,stage,status')]
    running=[dict(r) for r in db.execute("SELECT id,pid,gpu,started FROM jobs WHERE status='running'")]
    for r in running:
        p=RUN/'jobs'/r['id']/'progress.json'
        if p.exists():
            try:r['progress']=json.loads(p.read_text())
            except json.JSONDecodeError:pass
    failures=[dict(r) for r in db.execute("SELECT id,status,attempts,error FROM jobs WHERE status IN ('failed','blocked')")]
    obj=dict(timestamp=now(),scheduler_pid=os.getpid(),counts=counts,total_jobs=9500,by_family=families,running=running,gpus=gpus,failures=failures)
    atomic_json(RUN/'STATUS.json',obj);return counts

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--initialize-only',action='store_true');ap.add_argument('--retry-failed',action='store_true');a=ap.parse_args()
    RUN.mkdir(exist_ok=True);lock=open(RUN/'scheduler.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    db=connect();initialize(db)
    if a.initialize_only:return
    assert json.loads((RUN/'PREFLIGHT.json').read_text())['passed']
    assert (RUN/'prepared/AUDIT.json').exists()
    if a.retry_failed:
        db.execute("UPDATE jobs SET status='pending',attempts=0 WHERE status IN ('failed','blocked')")
        db.commit();(RUN/'FINISHED.json').unlink(missing_ok=True)
    atomic_json(RUN/'PROTOCOL.json',PROTOCOL)
    # Hash training code before any launch; scheduler/analysis can evolve separately.
    paths=['common.py','models.py','train.py','prepare.py','reference_original.py','column_original.py','baselines_original.py']
    hashes={p:sha(RUN/'code'/p) for p in paths}
    frozen=RUN/'TRAINING_CODE_FROZEN.json'
    if frozen.exists():assert json.loads(frozen.read_text())==hashes,'Training source changed after launch'
    else:atomic_json(frozen,hashes)
    children={};last_analysis=0;analysis=None
    (RUN/'specs').mkdir(exist_ok=True);(RUN/'logs').mkdir(exist_ok=True)
    while True:
        for row in db.execute("SELECT * FROM jobs WHERE status='running'").fetchall():
            jid=row['id'];p=children.get(jid);alive=p.poll() is None if p else pid_is_job(row['pid'],jid)
            if alive:continue
            if (RUN/'jobs'/jid/'SUCCESS.json').exists():
                db.execute("UPDATE jobs SET status='done',finished=?,error=NULL WHERE id=?",(now(),jid))
            else:
                fail=RUN/'jobs'/jid/'FAILURE.json';err=fail.read_text()[-6000:] if fail.exists() else 'Worker exited without SUCCESS; checkpoint will resume'
                new='pending' if row['attempts']<3 else 'failed'
                db.execute('UPDATE jobs SET status=?,error=?,pid=NULL WHERE id=?',(new,err,jid));print('RETRY/FAIL',jid,new,err[-300:],flush=True)
            children.pop(jid,None)
        db.commit();select_completed(db)
        # Exhausted prerequisites must be visible, rather than leaving an infinite queue.
        db.execute("UPDATE jobs SET status='blocked',error='Prerequisite failed after retries' WHERE status='pending' AND dependency IN (SELECT id FROM jobs WHERE status IN ('failed','blocked'))")
        failed_selection={r[0] for r in db.execute("SELECT DISTINCT family FROM jobs WHERE stage='tune' AND status IN ('failed','blocked')")}
        for family in failed_selection:
            db.execute("UPDATE jobs SET status='blocked',error='Hyperparameter selection has a failed prerequisite' WHERE status='pending' AND selection=?",(family,))
        db.commit()
        # Also reconcile complete output from jobs finishing during scheduler restart.
        completed={r[0] for r in db.execute("SELECT id FROM jobs WHERE status='done'")}
        active=list(db.execute("SELECT * FROM jobs WHERE status='running'"));counts=collections.Counter(r['gpu'] for r in active)
        family_active=collections.Counter(r['family'] for r in active)
        gpus=gpu_info();pending=list(db.execute("SELECT * FROM jobs WHERE status='pending' ORDER BY priority,attempts,site,seed,id"))
        ready=[]
        for row in pending:
            if row['dependency'] and row['dependency'] not in completed:continue
            if row['selection'] and not (RUN/'selected'/f"{row['selection']}.json").exists():continue
            ready.append(row)
        # At most one new process/card/poll avoids oversubscribing before CUDA allocates.
        for gpu in sorted(gpus,key=lambda g:(counts[g['index']],g['utilization'],-(g['total']-g['used']))):
            if not ready:break
            gi=gpu['index'];free=gpu['total']-gpu['used']
            if counts[gi]>=PROTOCOL['max_gpu_workers'] or free<PROTOCOL['gpu_reserve_mib']+PROTOCOL['initial_worker_budget_mib']:continue
            row=min(ready,key=lambda r:(r['priority'],family_active[r['family']],r['site'],r['seed'],r['id']))
            ready.remove(row);spec=json.loads(row['spec'])
            if row['selection']:spec['config']=json.loads((RUN/'selected'/f"{row['selection']}.json").read_text())['config']
            specpath=RUN/'specs'/f"{row['id']}.json";atomic_json(specpath,spec)
            logfile=open(RUN/'logs'/f"{row['id']}.log",'a')
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gi),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONUNBUFFERED='1',PYTHONHASHSEED=str(spec['seed']))
            proc=subprocess.Popen([sys.executable,str(RUN/'code/train.py'),'--spec',str(specpath)],stdout=logfile,stderr=subprocess.STDOUT,env=env,cwd=ROOT,start_new_session=True)
            logfile.close();children[row['id']]=proc
            db.execute("UPDATE jobs SET status='running',attempts=attempts+1,pid=?,gpu=?,started=?,spec=? WHERE id=?",(proc.pid,gi,now(),json.dumps(spec),row['id']))
            db.commit();counts[gi]+=1;family_active[row['family']]+=1;print('LAUNCH',gi,row['id'],proc.pid,flush=True)
        counts=status(db,gpus)
        if analysis is not None and analysis.poll() is not None:
            if analysis.returncode!=0:print('ANALYSIS_FAILED see analysis.log',flush=True)
            analysis=None
        remaining=counts.get('pending',0)+counts.get('running',0)
        if (time.time()-last_analysis>600 or not remaining) and analysis is None and (RUN/'code/analyze.py').exists():
            log=open(RUN/'analysis.log','a')
            analysis=subprocess.Popen([sys.executable,str(RUN/'code/analyze.py')],stdout=log,stderr=subprocess.STDOUT,
                            env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'),cwd=ROOT)
            log.close();last_analysis=time.time()
        if remaining==0:
            if analysis is not None:
                analysis.wait()
                if analysis.returncode!=0:raise RuntimeError('Final analysis failed')
            all_complete=counts.get('done',0)==9500
            if all_complete:
                log=open(RUN/'completion_audit.log','a')
                audit=subprocess.run([sys.executable,str(RUN/'code/verify_completion.py')],stdout=log,stderr=subprocess.STDOUT,
                    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'),cwd=ROOT)
                log.close();all_complete=audit.returncode==0
            atomic_json(RUN/'FINISHED.json',dict(timestamp=now(),counts=counts,all_experiments_complete=all_complete))
            break
        time.sleep(3)

if __name__=='__main__':main()
