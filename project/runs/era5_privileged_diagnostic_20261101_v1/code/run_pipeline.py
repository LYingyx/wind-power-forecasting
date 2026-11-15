"""Resumable CPU preparation and diagnostics; launched under a detached tmux."""
import os,subprocess,sys,time,traceback
from prepare_era5 import OUT,RUN,atomic,stamp

def main():
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    children={};streams=[];completed=set()
    if (OUT/'D2_SUMMARY.json').exists() and (OUT/'D2_AUDIT.json').exists():completed.add('diagnostic')
    def start(name,args):
        f=(OUT/f'{name}.log').open('a');streams.append(f)
        p=subprocess.Popen([sys.executable,'-u',*args],stdout=f,stderr=subprocess.STDOUT,env=env,cwd=RUN.parents[1])
        children[name]=p
        atomic(OUT/f'PROCESS_{name}.json',dict(timestamp=stamp(),pid=p.pid,args=args))
    prepare=str(RUN/'code/prepare_era5.py')
    for name in ['surface','pressure']:
        if (OUT/f'{name.upper()}_READY.json').exists():completed.add(name)
        else:start(name,[prepare,name,'--workers','8'])
    while True:
        for name,p in list(children.items()):
            rc=p.poll()
            if rc is None:continue
            if rc:raise RuntimeError(f'{name} failed, exit={rc}; see {OUT/name}.log')
            completed.add(name);del children[name]
        if 'surface' in completed and 'diagnostic' not in completed and 'diagnostic' not in children:
            start('diagnostic',[str(RUN/'code/diagnose.py')])
        if {'surface','pressure'}<=completed and 'labels' not in completed and 'labels' not in children:
            start('labels',[prepare,'labels'])
        atomic(OUT/'PIPELINE_STATUS.json',dict(timestamp=stamp(),state='running',pid=os.getpid(),completed=sorted(completed),running={k:p.pid for k,p in children.items()},gpu_jobs=0))
        if {'surface','pressure','labels','diagnostic'}<=completed:break
        time.sleep(10)
    atomic(OUT/'SUCCESS.json',dict(timestamp=stamp(),passed=True,completed=sorted(completed),test_era5_read=False,power_training_started=False))
    atomic(OUT/'PIPELINE_STATUS.json',dict(timestamp=stamp(),state='completed',completed=sorted(completed),running={},gpu_jobs=0))
if __name__=='__main__':
    try:main()
    except Exception:
        atomic(OUT/'PIPELINE_FAILURE.json',dict(timestamp=stamp(),traceback=traceback.format_exc()))
        raise
