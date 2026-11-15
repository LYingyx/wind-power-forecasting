"""Deploy to exactly the current 4090 host and v1006001; reuse verified local caches."""
from pathlib import Path
import sys,os,json,subprocess,shlex
RUN=Path(__file__).resolve().parents[1];sys.path.insert(0,str(RUN/'code'))
from settings import atomic_json,now
HOST='v1006001';REMOTE='/home/liying/wind_physics_project/project/runs/'+RUN.name
RPY='/home/liying/wind_physics_project/project/.runtime/chronos/bin/python'


def remote(code,timeout=60):
    result=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=8',HOST,'python3 -c '+shlex.quote(code)],
                          check=True,capture_output=True,text=True,timeout=timeout)
    return result.stdout.strip()


def main():
    assert (RUN/'SCIENCE_FREEZE.json').exists()
    assert not (RUN/'LAUNCH_STATUS.json').exists(),'Already launched: inspect status rather than duplicating schedulers'
    remote(f"from pathlib import Path;p=Path({REMOTE!r});p.mkdir(parents=True,exist_ok=True)")
    for name in ['code','operations','assets','reference_metrics','manifest.json','diagnostic_manifest.json',
                 'PROTOCOL.json','SCIENCE_FREEZE.json','DATA_FREEZE.json','README_CN.md']:
        subprocess.run(['rsync','-a','--exclude=__pycache__','--exclude=*.log','--exclude=*.lock',
                        str(RUN/name),HOST+':'+REMOTE+'/'],check=True,timeout=180)
    remote(f'''from pathlib import Path
r=Path({REMOTE!r})
for name in ['jobs','schedulers','analysis']:(r/name).mkdir(exist_ok=True)
if not (r/'data').exists():(r/'data').symlink_to('../daily_protocol_restart_20261114_v1/data',target_is_directory=True)
print('ready')''')
    # Independently hash the data on the V100; model smoke was tested on 4090.
    script=f'''from pathlib import Path
import json,hashlib
r=Path({REMOTE!r})
for name,expected in json.loads((r/'DATA_FREEZE.json').read_text())['files'].items():
 h=hashlib.sha256()
 with (r/name).open('rb') as f:
  for chunk in iter(lambda:f.read(4<<20),b''):h.update(chunk)
 assert h.hexdigest()==expected,name
print('remote data hashes passed')'''
    print(remote(script),flush=True)
    check=f'''import subprocess,os
env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
subprocess.run([{RPY!r},'-u',{(REMOTE+'/operations/checks.py')!r}],env=env,check=True)'''
    print(remote(check,timeout=180),flush=True)
    code=f'''from pathlib import Path
import subprocess,os,json
r=Path({REMOTE!r});log=(r/'operations/scheduler.log').open('a')
env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONNOUSERSITE='1')
p=subprocess.Popen([{RPY!r},'-u',str(r/'operations/scheduler.py'),'--host',{HOST!r}],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps(dict(host={HOST!r},pid=p.pid)))'''
    records=[json.loads(remote(code))]
    env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONNOUSERSITE='1')
    for name,args in [('scheduler',['--host','local4090']),('coordinator',[])]:
        log=(RUN/f'operations/{name}.log').open('a')
        p=subprocess.Popen([sys.executable,'-u',str(RUN/f'operations/{name}.py'),*args],env=env,
                           stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        records.append(dict(host='local4090',process=name,pid=p.pid))
    atomic_json(RUN/'LAUNCH_STATUS.json',dict(timestamp=now(),processes=records));print(json.dumps(records))

if __name__=='__main__':main()
