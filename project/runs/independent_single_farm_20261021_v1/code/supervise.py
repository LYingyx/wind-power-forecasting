"""Restart only this run's scheduler after unexpected exits. STOP blocks restart."""
from pathlib import Path
import datetime,os,subprocess,sys,time
RUN=Path(__file__).resolve().parents[1]

def alive(pid):
    try:return str(RUN/'code/scheduler.py') in Path(f'/proc/{pid}/cmdline').read_bytes().decode().replace('\0',' ')
    except (FileNotFoundError,PermissionError):return False

def main():
    import fcntl
    lock=open(RUN/'supervisor.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    while not (RUN/'STOP').exists():
        if (RUN/'FINISHED.json').exists():return
        try:pid=int((RUN/'scheduler.pid').read_text())
        except (FileNotFoundError,ValueError):pid=-1
        if not alive(pid):
            log=open(RUN/'scheduler.log','a')
            p=subprocess.Popen([sys.executable,str(RUN/'code/scheduler.py')],stdout=log,stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,start_new_session=True,cwd=RUN.parents[1],
                env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'))
            log.close();(RUN/'scheduler.pid').write_text(str(p.pid)+'\n')
            print(datetime.datetime.now().isoformat(),'restarted scheduler',p.pid,flush=True)
        time.sleep(30)

if __name__=='__main__':main()
