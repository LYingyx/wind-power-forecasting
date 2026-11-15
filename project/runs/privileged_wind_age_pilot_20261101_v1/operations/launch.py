"""Audit, deploy, verify device numerics, then start all three durable queues."""
from pathlib import Path
import sys,json,os,time,subprocess,shlex
from concurrent.futures import ThreadPoolExecutor
RUN=Path(__file__).resolve().parents[1];sys.path.insert(0,str(RUN/'code'))
from settings import DATA,PROJECT,atomic_json,now,sha
from probe import probe
from gpu_policy import Policy,decide
REMOTE_PROJECT='/home/liying/wind_physics_project/project';REMOTE=REMOTE_PROJECT+'/runs/'+RUN.name
HOSTS=['v1006001','v1006002'];ENV=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONNOUSERSITE='1')
def status(stage,**kw):atomic_json(RUN/'LAUNCH_STATUS.json',dict(timestamp=now(),stage=stage,**kw))
def cmd(args,**kw):return subprocess.run(args,check=True,env=ENV,**kw)
def ssh(host,code):
    p=cmd(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=8',host,'python3 -c '+shlex.quote(code)],capture_output=True,text=True,timeout=120);return json.loads(p.stdout)
def choose_local():
    samples={str(g):[] for g in [4,5,6,7]};policy=Policy()
    while True:
        gpus=probe()
        for g in samples:
            samples[g].append(gpus[g]);samples[g]=samples[g][-3:]
            if decide(samples[g],0,0,1000,2048,policy)['launch']:return g
        status('waiting_for_local_smoke_gpu');time.sleep(10)
def deploy(host):
    ssh(host,f"from pathlib import Path;r=Path({REMOTE!r});[(r/d).mkdir(parents=True,exist_ok=True) for d in ['jobs','analysis','schedulers','operations','data']];print('true')")
    for folder in ['code','assets']:
        cmd(['rsync','-a','--exclude=__pycache__',str(RUN/folder)+'/',host+':'+REMOTE+'/'+folder+'/'])
    files=list((RUN/'operations').glob('*.py'))+[RUN/'operations/fixture.pt',RUN/'operations/DATA_AUDIT.json']
    cmd(['rsync','-a',*[str(p) for p in files],host+':'+REMOTE+'/operations/'])
    cmd(['rsync','-a',*[str(RUN/f) for f in ['manifest.json','PROTOCOL.json','SCIENCE_FREEZE.json','DATA_FREEZE.json']],host+':'+REMOTE+'/'])
    for folder in ['common','cache','age0','age12','labels']:
        cmd(['rsync','-a',str(DATA/folder)+'/',host+':'+REMOTE+'/data/'+folder+'/'])
    cmd(['rsync','-a',*[str(DATA/f) for f in ['elevation.npy','region_mask.npy','READY.json']],host+':'+REMOTE+'/data/'])
    return ssh(host,f'''from pathlib import Path
import subprocess,os,json
r=Path({REMOTE!r}); env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONNOUSERSITE='1')
with (r/'operations/bootstrap.log').open('a') as log:
 p=subprocess.Popen([{(REMOTE_PROJECT+'/.runtime/chronos/bin/python')!r},'-u',str(r/'operations/remote_bootstrap.py')],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env=env)
print(json.dumps(dict(pid=p.pid)))''')
def main():
    while not (DATA/'READY.json').exists():status('waiting_for_data');time.sleep(15)
    status('data_audit');cmd([sys.executable,str(RUN/'operations/audit_data.py')])
    status('cpu_numerical_checks');cmd([sys.executable,str(RUN/'operations/checks.py'),'--device','cpu'])
    env=dict(ENV,CUDA_VISIBLE_DEVICES='')
    status('cpu_all_arm_and_resume_smoke_checks')
    for arm in ['E0','E1','E2','E3','E4','E5']:
        subprocess.run([sys.executable,'-u',str(RUN/'code/train.py'),'--job-id',f'age12_{arm}_s00_r42','--smoke','--device','cpu'],env=env,check=True)
    # Resume must match uninterrupted training; copy only the smoke directory.
    import shutil,torch
    smoke=RUN/'smoke_jobs/age12_E1_s00_r42';reference=RUN/'smoke_jobs/resume_reference'
    shutil.copytree(smoke,reference,dirs_exist_ok=True)
    for p in smoke.iterdir():p.unlink()
    interrupted=subprocess.run([sys.executable,str(RUN/'code/train.py'),'--job-id','age12_E1_s00_r42','--smoke','--stop-after-epoch','1','--device','cpu'],env=env)
    assert interrupted.returncode==75
    subprocess.run([sys.executable,str(RUN/'code/train.py'),'--job-id','age12_E1_s00_r42','--smoke','--device','cpu'],env=env,check=True)
    a=torch.load(reference/'best.pt',map_location='cpu',weights_only=False)['model'];b=torch.load(smoke/'best.pt',map_location='cpu',weights_only=False)['model']
    assert all(torch.equal(a[k],b[k]) for k in a),'Resume changed training trajectory'
    check=json.loads((RUN/'operations/CHECKS.json').read_text());check.update(all_six_arms_smoke_passed=True,resume_exact=True);atomic_json(RUN/'operations/CHECKS.json',check)
    paths=list((RUN/'code').rglob('*.py'))+list((RUN/'assets').glob('*'))+[RUN/'manifest.json',RUN/'PROTOCOL.json']
    atomic_json(RUN/'SCIENCE_FREEZE.json',dict(timestamp=now(),files={str(p.relative_to(RUN)):sha(p) for p in paths if p.is_file()},data_audit=sha(RUN/'operations/DATA_AUDIT.json')))
    # Hash the exact deployment data, excluding source-extraction scratch arrays.
    paths=[p for name in ['common','cache','age0','age12','labels'] for p in (DATA/name).rglob('*.npy')]+[DATA/'elevation.npy',DATA/'region_mask.npy']
    atomic_json(RUN/'DATA_FREEZE.json',dict(timestamp=now(),files={str(p.relative_to(DATA)):sha(p) for p in paths}))
    status('deploying_remote_hosts')
    with ThreadPoolExecutor(max_workers=2) as pool:remote=dict(zip(HOSTS,pool.map(deploy,HOSTS)))
    local={}
    for name,args in [('remote_bootstrap',['--gpus','4,5,6,7','--local']),('coordinator',[])]:
        with (RUN/'operations'/f'{name}.log').open('a') as log:
            p=subprocess.Popen([sys.executable,'-u',str(RUN/'operations'/f'{name}.py'),*args],env=ENV,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        local[name]=p.pid
    status('queues_started',remote=remote,local=local)
    atomic_json(PROJECT/'ACTIVE_EXPERIMENT.json',dict(timestamp=now(),run=str(RUN),stage='2023_privileged_wind_age_pilot',status=str(RUN/'STATUS.json'),
        planned_jobs=1344,hosts=['current4090_gpu4_5_6_7',*HOSTS],test_read=False,results=str(RUN/'analysis/REPORT_CN.md')))
if __name__=='__main__':
    try:main()
    except BaseException:
        import traceback
        status('failed',traceback=traceback.format_exc());raise
