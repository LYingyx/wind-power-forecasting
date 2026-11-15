from config import *
import fcntl,subprocess,sys,time

def main():
 lock=open(RUN/'supervisor.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 while not (RUN/'STOP').exists() and not (RUN/'FINISHED.json').exists() and not (RUN/'BLOCKED.json').exists():
  try:
   pid=int((RUN/'scheduler.pid').read_text());alive=str(RUN/'code/scheduler.py') in Path(f'/proc/{pid}/cmdline').read_bytes().decode().replace('\0',' ')
  except (FileNotFoundError,ValueError,PermissionError):alive=False
  if not alive:
   log=open(RUN/'scheduler.log','a');proc=subprocess.Popen([sys.executable,str(RUN/'code/scheduler.py')],stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True,cwd=ROOT,env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'));log.close();(RUN/'scheduler.pid').write_text(str(proc.pid))
   print(now(),'START scheduler',proc.pid,flush=True)
  time.sleep(30)

if __name__=='__main__':main()
