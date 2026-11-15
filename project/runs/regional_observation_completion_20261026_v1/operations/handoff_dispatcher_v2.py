"""Replace this run's dispatcher only; its workers checkpoint via the existing drain handler."""
from pathlib import Path
import os,sys,time,json,signal,socket,fcntl
RUN=Path(__file__).resolve().parents[1]

def main():
    token=(RUN/'operations/efficiency_handoff.lock').open('a+')
    fcntl.flock(token,fcntl.LOCK_EX|fcntl.LOCK_NB)
    me=os.getpid();old=[]
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit() or int(proc.name)==me:continue
        try:
            if proc.stat().st_uid!=os.getuid():continue
            args=(proc/'cmdline').read_bytes().split(b'\0')
            accepted=[RUN/'operations'/n for n in ['bootstrap_limits.py','gpu_scheduler_limits.py','resource_scheduler.py','bootstrap_efficient.py','resource_scheduler_efficient.py']]
            if not any(str(p).encode() in args for p in accepted):continue
            os.kill(int(proc.name),signal.SIGTERM);old.append(int(proc.name))
        except (OSError,ProcessLookupError):pass
    marker=RUN/'operations/EFFICIENCY_HANDOFF.json'
    marker.write_text(json.dumps(dict(pid=me,previous_dispatchers=old,state='draining_our_workers'),indent=2))
    # Lock inherited by old workers protects against overlapping dispatchers until all checkpointed.
    path=RUN/'schedulers'/(socket.gethostname()+'.lock');path.parent.mkdir(exist_ok=True)
    with path.open('a+') as check:
        while True:
            try:fcntl.flock(check,fcntl.LOCK_EX|fcntl.LOCK_NB);break
            except BlockingIOError:time.sleep(5)
    marker.write_text(json.dumps(dict(pid=me,previous_dispatchers=old,state='starting_efficiency_scheduler'),indent=2))
    os.execv(sys.executable,[sys.executable,'-s',str(RUN/'operations/bootstrap_efficient_v2.py'),'--warmup-seconds','60'])

if __name__=='__main__':main()
