"""Bounded, low-priority CPU replay; no training or changes to frozen runs."""
from pathlib import Path
import os,sys,time,subprocess,json
HERE=Path(__file__).resolve().parent
pending=[(s,r) for r in [44,45] for s in range(56)
         if not (HERE/'jobs'/f's{s:02d}_r{r}'/'SUCCESS.json').exists()]
active={};failed=[];completed=112-len(pending)
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
while pending or active:
    for p,(site,seed,log) in list(active.items()):
        if p.poll() is not None:
            log.close();del active[p]
            if p.returncode:failed.append([site,seed,p.returncode])
            else:completed+=1
            print(json.dumps(dict(done=completed,active=len(active),pending=len(pending),failed=failed)),flush=True)
    mem={line.split(':')[0]:int(line.split()[1]) for line in Path('/proc/meminfo').read_text().splitlines()}
    if pending and len(active)<6 and os.getloadavg()[0]<os.cpu_count()*.70 and mem['MemAvailable']>32*1024**2:
        site,seed=pending.pop(0);out=HERE/'jobs'/f's{site:02d}_r{seed}';out.mkdir(parents=True,exist_ok=True)
        log=(out/'replay.log').open('w')
        p=subprocess.Popen([sys.executable,str(HERE/'replay_high_power.py'),'--site',str(site),'--seed',str(seed)],env=env,stdout=log,stderr=subprocess.STDOUT)
        active[p]=(site,seed,log)
    time.sleep(.5)
(HERE/'REPLAY_COMPLETION.json').write_text(json.dumps(dict(completed=completed,expected=112,failed=failed),indent=2)+'\n')
if failed:sys.exit(1)
