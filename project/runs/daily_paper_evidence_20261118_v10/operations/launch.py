from pathlib import Path
import sys,os,json,subprocess,shlex,concurrent.futures
RUN=Path(__file__).resolve().parents[1];sys.path.insert(0,str(RUN/'code'))
from settings import atomic_json,now,sha
HOSTS=['v1006001','v1006002'];REMOTE='/home/liying/wind_physics_project/project/runs/'+RUN.name
RPY='/home/liying/wind_physics_project/project/.runtime/chronos/bin/python'

def remote(h,code,timeout=60):
 p=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=8',h,'python3 -c '+shlex.quote(code)],capture_output=True,text=True,timeout=timeout,check=True)
 return p.stdout.strip()

def deploy(h):
 remote(h,f'from pathlib import Path;Path({REMOTE!r}).mkdir(parents=True,exist_ok=True)')
 for name in ['code','operations','assets','checkpoints','manifest.json','diagnostic_manifest.json','PROTOCOL.json','SCIENCE_FREEZE.json','README_CN.md']:
  subprocess.run(['rsync','-a','--exclude=__pycache__','--exclude=*.log','--exclude=*.lock',str(RUN/name),h+':'+REMOTE+'/'],check=True,timeout=90)
 remote(h,f'''from pathlib import Path
import shutil
r=Path({REMOTE!r});root=r.parent
for name in ['jobs','schedulers','analysis','data']:(r/name).mkdir(exist_ok=True)
for split,source in [('train',root/'daily_protocol_restart_20261114_v1/data/train'),('val',root/'daily_protocol_restart_20261114_v1/data/val'),('test',root/'daily_test2024_20261117_v9/data/test')]:
 assert (source/'h.npy').exists(),str(source)
 p=r/'data'/split
 if not p.exists():p.symlink_to(source,target_is_directory=True)
shutil.copy2(root/'daily_test2024_20261117_v9/data/READY.json',r/'data/READY.json')''')
 gpu='6' if h=='v1006001' else '7'
 print(h,remote(h,f'''import os,subprocess
env=dict(os.environ,CUDA_VISIBLE_DEVICES={gpu!r},OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
subprocess.run([{RPY!r},'-u',{str(Path(REMOTE)/'operations/checks.py')!r}],env=env,check=True)
subprocess.run([{RPY!r},'-u',{str(Path(REMOTE)/'code/run_job.py')!r},'--job-id','EVAL_TFT_R3_r42','--smoke'],env=env,check=True)''',timeout=240),flush=True)
 return h

def main():
 assert not (RUN/'LAUNCH_STATUS.json').exists()
 assert json.loads((RUN/'operations/CHECKS.json').read_text())['passed']
 files=[p for p in (RUN/'code').rglob('*.py')]+[RUN/'manifest.json',RUN/'PROTOCOL.json',RUN/'README_CN.md']
 files+=list((RUN/'assets').glob('*.json'))+list((RUN/'assets').glob('*.npy'))+list((RUN/'assets').glob('*.npz'))
 atomic_json(RUN/'SCIENCE_FREEZE.json',dict(timestamp=now(),files={str(p.relative_to(RUN)):sha(p) for p in files},note='LINEAR selection is a predeclared runtime dependency, not a frozen preselected result'))
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
  for result in pool.map(deploy,HOSTS):print('verified',result,flush=True)
 processes=[]
 for h in HOSTS:
  record=remote(h,f'''from pathlib import Path
import subprocess,os,json
r=Path({REMOTE!r});log=(r/'operations/scheduler.log').open('a')
env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONNOUSERSITE='1')
p=subprocess.Popen([{RPY!r},'-u',str(r/'operations/scheduler.py'),'--host',{h!r}],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps(dict(host={h!r},process='scheduler',pid=p.pid)))''')
  processes.append(json.loads(record))
 env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONNOUSERSITE='1')
 for name,args in [('scheduler',['--host','local4090']),('coordinator',[])]:
  log=(RUN/f'operations/{name}.log').open('a')
  p=subprocess.Popen([sys.executable,'-u',str(RUN/f'operations/{name}.py'),*args],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  processes.append(dict(host='local4090',process=name,pid=p.pid))
 atomic_json(RUN/'LAUNCH_STATUS.json',dict(timestamp=now(),processes=processes))
 atomic_json(RUN.parents[1]/'ACTIVE_PAPER_EVIDENCE.json',dict(timestamp=now(),run=str(RUN),status=str(RUN/'STATUS.json')))
 print(json.dumps(processes),flush=True)
if __name__=='__main__':main()
