from pathlib import Path
import sys,os,json,subprocess,shlex,concurrent.futures,py_compile
RUN=Path(__file__).resolve().parents[1];sys.path.insert(0,str(RUN/'code'))
from settings import *
REMOTE='/home/liying/wind_physics_project/project/runs/'+RUN.name;RPY='/home/liying/wind_physics_project/project/.runtime/chronos/bin/python'
def remote(host,code,timeout=120):
 p=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=8',host,'python3 -'],input=code,text=True,capture_output=True,timeout=timeout)
 if p.returncode:raise RuntimeError(host+':'+p.stderr[-4000:])
 return p.stdout

def deploy(host):
 remote(host,f'''from pathlib import Path
r=Path({REMOTE!r});r.mkdir(exist_ok=True)
for d in ['jobs','feature_cache','analysis','schedulers','operations']:(r/d).mkdir(exist_ok=True)
p=r/'data'
if not p.exists():p.symlink_to('../daily_protocol_restart_20261114_v1/data',target_is_directory=True)
assert (p/'READY.json').exists()
print('directories ready')''')
 for name in ['code','assets','pretrained','PROTOCOL.json','SOURCE_DATA_FREEZE.json','SOURCE_PROTOCOL.json','SOURCE_QC_RULES.json','SOURCE_SCIENCE_FREEZE.json','manifest.json','SCIENCE_FREEZE.json','EXECUTION_PLAN_CN.md']:
  subprocess.run(['rsync','-a','--exclude=__pycache__',str(RUN/name),host+':'+REMOTE+'/'],check=True,timeout=120,stdout=subprocess.DEVNULL)
 for p in (RUN/'operations').iterdir():
  if p.suffix=='.py' or p.name in ['CHECKS.json','fixture.pt','LOCAL_ADMISSION.json','REMOTE_ADMISSION.json','RESOURCE_POOLS.json']:
   subprocess.run(['rsync','-a',str(p),host+':'+REMOTE+'/operations/'],check=True,timeout=120,stdout=subprocess.DEVNULL)
 code=f'''from pathlib import Path
import sys,json,hashlib,os,subprocess
r=Path({REMOTE!r});sys.path.insert(0,str(r/'code'));from settings import sha,atomic_json,now
for rel,h in json.loads((r/'SCIENCE_FREEZE.json').read_text())['files'].items():assert sha(r/rel)==h,rel
for rel,h in json.loads((r/'SOURCE_DATA_FREEZE.json').read_text())['files'].items():assert sha(r/rel)==h,rel
lines=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True).splitlines()
candidates=[tuple(map(int,x.split(','))) for x in lines];allowed=json.loads((r/'operations/RESOURCE_POOLS.json').read_text())[{host!r}];candidates=[x for x in candidates if x[0] in allowed and x[1]>8192];g=min(candidates,key=lambda x:x[2])[0]
env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(g),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONNOUSERSITE='1')
p=subprocess.run([{RPY!r},str(r/'operations/checks.py'),'--fixture'],env=env,text=True,capture_output=True,timeout=60)
assert p.returncode==0,p.stderr
atomic_json(r/'operations/DEPLOY_VERIFIED.json',dict(timestamp=now(),passed=True,gpu_fixture=g,source_data_hashes_verified=True))
print(json.dumps(dict(host={host!r},passed=True,gpu_fixture=g,fixture=p.stdout.strip())))'''
 print(remote(host,code,240),flush=True)

if __name__=='__main__':
 assert json.loads((RUN/'operations/CHECKS.json').read_text())['passed'];assert json.loads((RUN/'operations/GPU_CHECKS.json').read_text())['passed']
 for p in [*RUN.glob('code/*.py'),*RUN.glob('operations/*.py')]:py_compile.compile(str(p),doraise=True)
 paths=[*RUN.glob('code/**/*.py'),*RUN.glob('operations/*.py'),*RUN.glob('assets/**/*'),*RUN.glob('pretrained/*')]
 paths=[p for p in paths if p.is_file() and '__pycache__' not in p.parts]+[RUN/n for n in ['PROTOCOL.json','manifest.json','SOURCE_DATA_FREEZE.json','EXECUTION_PLAN_CN.md']]+[RUN/'operations'/n for n in ['LOCAL_ADMISSION.json','REMOTE_ADMISSION.json','RESOURCE_POOLS.json']]
 atomic_json(RUN/'SCIENCE_FREEZE.json',dict(timestamp=now(),files={str(p.relative_to(RUN)):sha(p) for p in paths}))
 with concurrent.futures.ThreadPoolExecutor(2) as pool:
  futs={pool.submit(deploy,h):h for h in ['v1006001','v1006002']}
  for f in concurrent.futures.as_completed(futs):f.result()
 atomic_json(RUN/'operations/DEPLOY_VERIFIED.json',dict(timestamp=now(),passed=True,hosts=['local4090','v1006001','v1006002']))
 env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONNOUSERSITE='1')
 logroot=Path('/shared_data_ssd3/liying')/RUN.name/'control_health';logroot.mkdir(parents=True,exist_ok=True)
 with (logroot/'supervise.log').open('a') as log:p=subprocess.Popen([sys.executable,'-u',str(RUN/'operations/supervise.py')],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 atomic_json(RUN/'LAUNCH_STATUS.json',dict(timestamp=now(),supervisor_pid=p.pid,scope='shared_baselines_plus_per_station_finetuning',planned=1100))
 atomic_json(PROJECT/'ACTIVE_EXPERIMENT.json',dict(timestamp=now(),run=str(RUN),scope='shared_plus_per_station_finetuning',status_path=str(RUN/'STATUS.json')))
 print('Launched supervisor',p.pid,flush=True)
