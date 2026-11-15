"""Verify and deploy only the new ablation run; leave active runs untouched."""
from pathlib import Path
import sys,os,json,shutil,subprocess,concurrent.futures
RUN=Path(__file__).resolve().parents[1];sys.path.insert(0,str(RUN/'code'))
from settings import *
REMOTE='/home/liying/wind_physics_project/project/runs/'+RUN.name
RPY='/home/liying/wind_physics_project/project/.runtime/chronos/bin/python'

def remote(host,code,timeout=300):
    p=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=8',host,'python3 -'],input=code,text=True,capture_output=True,timeout=timeout)
    if p.returncode:raise RuntimeError(host+':'+p.stderr[-4000:])
    return p.stdout

def deploy(host):
    remote(host,f'''from pathlib import Path
r=Path({REMOTE!r});r.mkdir(exist_ok=True)
for d in ['jobs','analysis','schedulers','operations','smoke_jobs']:(r/d).mkdir(exist_ok=True)
p=r/'data'
if not p.exists():p.symlink_to('../daily_protocol_restart_20261114_v1/data',target_is_directory=True)
assert (p/'READY.json').exists()
''')
    for name in ['code','assets','PROTOCOL.json','DATA_FREEZE.json','QC_RULES.json','manifest.json','SCIENCE_FREEZE.json','EXECUTION_PLAN_CN.md']:
        subprocess.run(['rsync','-a','--exclude=__pycache__',str(RUN/name),host+':'+REMOTE+'/'],check=True,timeout=180,stdout=subprocess.DEVNULL)
    for p in (RUN/'operations').iterdir():
        if p.suffix=='.py' or p.name in ['CHECKS.json','fixture.pt','LOCAL_ADMISSION.json','REMOTE_ADMISSION.json','RESOURCE_POOLS.json']:
            subprocess.run(['rsync','-a',str(p),host+':'+REMOTE+'/operations/'],check=True,timeout=120,stdout=subprocess.DEVNULL)
    result=remote(host,f'''from pathlib import Path
import sys,json,subprocess,os,time
r=Path({REMOTE!r});sys.path.insert(0,str(r/'code'));from settings import sha,atomic_json,now
for kind in ['SCIENCE_FREEZE','DATA_FREEZE']:
 for rel,digest in json.loads((r/(kind+'.json')).read_text())['files'].items():assert sha(r/rel)==digest,rel
for marker in (r/'data').glob('reference_r*/READY.json'):
 p=marker.parent;meta=json.loads(marker.read_text())
 for rel,digest in meta['files'].items():assert sha(p/rel)==digest
lines=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True).splitlines()
choices=[tuple(map(int,x.split(','))) for x in lines];pools=json.loads((r/'operations/RESOURCE_POOLS.json').read_text());allowed=pools[{host!r}]
choices=[x for x in choices if x[0] in allowed and x[1]>4096 and x[2]<50]
assert choices,'No safe GPU fixture slot; watcher will retry'
g=min(choices,key=lambda x:x[2])[0];env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(g),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONNOUSERSITE='1')
subprocess.run([{RPY!r},str(r/'operations/checks.py'),'--fixture'],env=env,check=True,timeout=90)
atomic_json(r/'operations/DEPLOY_VERIFIED.json',dict(timestamp=now(),passed=True,gpu=g,data_and_science_hashes_verified=True))
print(json.dumps(dict(host={host!r},gpu=g,passed=True)))''')
    print(result,flush=True)

def main():
    if (RUN/'LAUNCH_STATUS.json').exists():return
    source=PROJECT/'runs/daily_protocol_restart_20261114_v1'
    assert (source/'data/READY.json').exists()
    assert json.loads((source/'PROTOCOL.json').read_text())['preprocess_fingerprint']==json.loads((RUN/'PROTOCOL.json').read_text())['preprocess_fingerprint']
    assert json.loads((RUN/'operations/CHECKS.json').read_text())['passed']
    for rel,digest in json.loads((RUN/'PLAN_FREEZE.json').read_text())['files'].items():
        assert sha(RUN/rel)==digest, 'Queued code changed after checks: '+rel
    selected=ASSETS/'selected';selected.mkdir(exist_ok=True)
    for family in ['W0_R3','TF_BASE','TF_R3','TIDE_R3','TFT_R3']:
        p=source/f'assets/selected/{family}.json'
        if p.exists():shutil.copy2(p,selected/f'{family}.json')
    # Freeze this queue's code before any formal training; source queue is immutable.
    paths=[*RUN.glob('code/**/*.py'),*RUN.glob('operations/*.py'),*RUN.glob('assets/**/*')]
    paths=[p for p in paths if p.is_file() and '__pycache__' not in p.parts and 'selected' not in p.parts]+[RUN/n for n in ['PROTOCOL.json','manifest.json','DATA_FREEZE.json','EXECUTION_PLAN_CN.md']]+[RUN/'operations'/n for n in ['LOCAL_ADMISSION.json','REMOTE_ADMISSION.json','RESOURCE_POOLS.json']]
    atomic_json(RUN/'SCIENCE_FREEZE.json',dict(timestamp=now(),files={str(p.relative_to(RUN)):sha(p) for p in paths}))
    with concurrent.futures.ThreadPoolExecutor(2) as pool:
        futures=[pool.submit(deploy,h) for h in ['v1006001','v1006002']]
        for f in futures:f.result()
    # Remote fixtures passed; all three schedulers use the same validated math/data.
    env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONNOUSERSITE='1')
    logroot=Path('/shared_data_ssd3/liying')/RUN.name/'control_health';logroot.mkdir(parents=True,exist_ok=True)
    with (logroot/'supervise.log').open('a') as log:
        process=subprocess.Popen([sys.executable,'-u',str(RUN/'operations/supervise.py')],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    atomic_json(RUN/'LAUNCH_STATUS.json',dict(timestamp=now(),supervisor_pid=process.pid,planned=1291,scope='daily_ablation_and_independent'))
    atomic_json(PROJECT/'ACTIVE_EXPERIMENT.json',dict(timestamp=now(),run=str(RUN),scope='daily_ablation_and_independent',status_path=str(RUN/'STATUS.json')))
    print('launched',process.pid,flush=True)
if __name__=='__main__':main()
