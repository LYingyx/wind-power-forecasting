from pathlib import Path
import sys,os,subprocess,json,shlex
RUN=Path(__file__).resolve().parents[1];sys.path.insert(0,str(RUN/'code'))
from settings import *
REMOTE='/home/liying/wind_physics_project/project/runs/'+RUN.name
RPY='/home/liying/wind_physics_project/project/.runtime/chronos/bin/python'
def remote(host,code):
    p=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=8',host,'python3 -c '+shlex.quote(code)],check=True,capture_output=True,text=True,timeout=90);return p.stdout.strip()
def launch():
    records=[]
    for host in ['v1006001','v1006002']:
        remote(host,f"from pathlib import Path;p=Path({REMOTE!r});p.mkdir(parents=True,exist_ok=True)")
        for path in ['code','operations','assets','data','PROTOCOL.json','QC_RULES.json','manifest.json','SCIENCE_FREEZE.json','DATA_FREEZE.json','EXECUTION_PLAN_CN.md','selection_inputs']:
            subprocess.run(['rsync','-aL','--exclude=__pycache__','--exclude=*.log','--exclude=*.lock','--exclude=*STATUS.json','--exclude=GPU_FIXTURE.json',str(RUN/path),host+':'+REMOTE+'/'],check=True,timeout=300)
        code=f'''import subprocess,os,json
from pathlib import Path
r=Path({REMOTE!r});(r/'jobs').mkdir(exist_ok=True);(r/'schedulers').mkdir(exist_ok=True);(r/'analysis').mkdir(exist_ok=True)
log=(r/'operations/bootstrap.log').open('a');env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONNOUSERSITE='1')
p=subprocess.Popen([{RPY!r},'-u',str(r/'operations/bootstrap.py'),'--gpus','0,1,2,3,4,5,6,7','--remote'],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps(dict(pid=p.pid)))'''
        records.append(dict(host=host,**json.loads(remote(host,code))))
    env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONNOUSERSITE='1')
    for name,args in [('coordinator',[]),('bootstrap',['--gpus','0,1,2,3,4,5,6,7'])]:
        log=(RUN/f'operations/{name}.log').open('a');p=subprocess.Popen([sys.executable,'-u',str(RUN/f'operations/{name}.py'),*args],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);records.append(dict(host='local4090',process=name,pid=p.pid))
    atomic_json(RUN/'LAUNCH_STATUS.json',dict(timestamp=now(),processes=records));atomic_json(PROJECT/'ACTIVE_EXPERIMENT.json',dict(timestamp=now(),run=str(RUN),scope='daily_00UTC_latency15h_shared_stages1_2',status_path=str(RUN/'STATUS.json')))
    print(json.dumps(records),flush=True)
if __name__=='__main__':launch()
