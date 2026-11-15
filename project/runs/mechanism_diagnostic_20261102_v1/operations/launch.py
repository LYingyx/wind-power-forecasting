from pathlib import Path
import sys,json,subprocess,os,shlex
from concurrent.futures import ThreadPoolExecutor
RUN=Path(__file__).resolve().parents[1];sys.path.insert(0,str(RUN/'code'))
from settings import *
REMOTE_PROJECT='/home/liying/wind_physics_project/project';REMOTE=REMOTE_PROJECT+'/runs/'+RUN.name
ENV=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONNOUSERSITE='1')
def cmd(args,**kw):return subprocess.run(args,check=True,env=ENV,**kw)
def ssh(host,script):
    p=cmd(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=8',host,'python3 -c '+shlex.quote(script)],capture_output=True,text=True,timeout=120)
    return json.loads(p.stdout)
def deploy(host):
    ssh(host,f'''from pathlib import Path
import json
r=Path({REMOTE!r});[(r/d).mkdir(parents=True,exist_ok=True) for d in ['code','operations','jobs','analysis','schedulers','assets']]
data=r.parent/'privileged_wind_age_pilot_20261101_v1/data';assert (data/'READY.json').exists()
if not (r/'data').exists():(r/'data').symlink_to(data,target_is_directory=True)
print('true')''')
    for folder in ['code','assets']:
        cmd(['rsync','-a','--exclude=__pycache__',str(RUN/folder)+'/',host+':'+REMOTE+'/'+folder+'/'])
    files=list((RUN/'operations').glob('*.py'))+[RUN/'operations/CHECKS.json',RUN/'operations/LABEL_AUDIT.json',RUN/'operations/fixture.pt']
    cmd(['rsync','-a',*[str(p) for p in files],host+':'+REMOTE+'/operations/'])
    cmd(['rsync','-a',*[str(RUN/f) for f in ['manifest.json','PROTOCOL.json','PLAN_CN.md','SCIENCE_FREEZE.json','DATA_FREEZE.json']],host+':'+REMOTE+'/'])
    return ssh(host,f'''from pathlib import Path
import os,subprocess,json
r=Path({REMOTE!r});env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONNOUSERSITE='1')
with (r/'operations/bootstrap.log').open('a') as log:
 p=subprocess.Popen([{(REMOTE_PROJECT+'/.runtime/chronos/bin/python')!r},'-u',str(r/'operations/bootstrap.py'),'--gpus','0,1,2,3,4,5,6,7','--remote'],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps(dict(pid=p.pid)))''')
def main():
    assert json.loads((RUN/'operations/CHECKS.json').read_text())['passed']
    assert json.loads((RUN/'operations/LABEL_AUDIT.json').read_text())['passed']
    paths=list((RUN/'code').rglob('*.py'))+list((RUN/'assets').rglob('*'))+[RUN/'manifest.json',RUN/'PROTOCOL.json']
    atomic_json(RUN/'SCIENCE_FREEZE.json',dict(timestamp=now(),files={str(p.relative_to(RUN)):sha(p) for p in paths if p.is_file()},label_audit=sha(RUN/'operations/LABEL_AUDIT.json')))
    old=json.loads((RUN.parent/'privileged_wind_age_pilot_20261101_v1/DATA_FREEZE.json').read_text())
    selected={k:v for k,v in old['files'].items() if not k.startswith('age12/')}
    atomic_json(RUN/'DATA_FREEZE.json',dict(timestamp=now(),files=selected,source='previous verified paired cohort'))
    hosts=['v1006001','v1006002']
    with ThreadPoolExecutor(max_workers=2) as pool:remote=dict(zip(hosts,pool.map(deploy,hosts)))
    local={}
    for name,args in [('bootstrap',['--gpus','4,5,6,7']),('coordinator',[])]:
        with (RUN/'operations'/f'{name}.log').open('a') as log:
            p=subprocess.Popen([sys.executable,'-u',str(RUN/'operations'/f'{name}.py'),*args],env=ENV,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        local[name]=p.pid
    atomic_json(RUN/'LAUNCH_STATUS.json',dict(timestamp=now(),remote=remote,local=local))
    atomic_json(PROJECT/'ACTIVE_EXPERIMENT.json',dict(timestamp=now(),run=str(RUN),stage='2023_pure_weather_and_region_conditioned_interaction',planned_jobs=672,status=str(RUN/'STATUS.json'),results=str(RUN/'analysis/REPORT_CN.md'),hosts=['current4090_gpu4_5_6_7',*hosts],test_read=False))
    print(json.dumps(dict(remote=remote,local=local)),flush=True)
if __name__=='__main__':main()
