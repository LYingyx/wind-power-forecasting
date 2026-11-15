"""Independent reference/ARX CPU slots, leases and reversible load shedding."""
from pathlib import Path
import os,sys,json,time,fcntl,subprocess,socket,argparse,traceback,signal,math,collections
RUN=Path(__file__).resolve().parents[1];sys.path.insert(0,str(RUN/'code'))
from relocate import activate
DEPLOYMENT=activate(RUN)
from settings import JOBS,atomic_json,now
from resource_limits import memory_status

def proc(pid):
    try:
        fields=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()
        return dict(state=fields[0],ticks=int(fields[11])+int(fields[12]),start=int(fields[19]))
    except (OSError,ValueError,IndexError):return None

def owned(item):
    value=proc(item['pid'])
    return value if value and value['start']==item['start'] and value['state'] not in ['Z','X'] else None

def safe_signal(item,sig):
    if owned(item):os.kill(item['pid'],sig)

def capacity(cores,external_busy,max_workers,free_mib,total_mib,fraction=.85):
    cpu=max(0,math.floor(cores*fraction-external_busy))
    return min(max_workers,cpu) if free_mib>max(8192,total_mib*.20)+3072 else 0

def choose_admitted(running,cap,reference_slots,reference_waiting):
    refs=sorted((j for j in running if running[j]['task']=='diagnose_reference'),key=lambda j:running[j]['start'])[:cap]
    arx=sorted((j for j in running if running[j]['task']=='arx'),key=lambda j:running[j]['start'])
    reserved=min(cap,reference_slots) if reference_waiting else 0
    return set(refs+arx[:max(0,cap-max(reserved,len(refs)))])

class Sampler:
    def __init__(self):self.before=None;self.own={};self.external=collections.deque(maxlen=3)
    def sample(self,running):
        values=list(map(int,Path('/proc/stat').read_text().splitlines()[0].split()[1:9]))
        current=(sum(values),values[3]+values[4]);own={j:proc(x['pid']) for j,x in running.items()};n=os.cpu_count();busy=0.;own_busy=0.
        if self.before:
            delta=current[0]-self.before[0]
            if delta>0:
                busy=n*(delta-(current[1]-self.before[1]))/delta
                ticks=sum(max(0,p['ticks']-self.own[j]['ticks']) for j,p in own.items() if p and j in self.own and self.own[j] and p['start']==self.own[j]['start'])
                own_busy=n*ticks/delta
                self.external.append(max(0,busy-own_busy))
        self.before=current;self.own=own
        return dict(cores=n,busy_cores=busy,own_busy_cores=own_busy,external_busy_cores=max(self.external,default=float(n)),ready=len(self.external)>=3)

def worker(jid):
    spec=next(s for s in json.loads((RUN/'manifest.json').read_text()) if s['id']==jid)
    assert spec['task'] in ['arx','diagnose_reference']
    os.nice(15)
    if spec['task']=='arx':
        from sparse_arx import run
    else:
        from diagnose_references import run
    out=JOBS/jid;out.mkdir(exist_ok=True)
    atomic_json(out/'runtime.json',dict(host=socket.gethostname(),pid=os.getpid(),device='cpu',started=now(),wrapper='cpu_scheduler_v2',threads=1,nice=15))
    try:run(spec,'cpu')
    except BaseException:
        atomic_json(out/'FAILURE.json',dict(timestamp=now(),host=socket.gethostname(),traceback=traceback.format_exc()));raise

def schedule(args):
    from resource_scheduler import validate,cpu_memory_available,status_collect
    validate();folder=RUN/'schedulers';folder.mkdir(exist_ok=True)
    lock=(folder/'cpu_v2.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    specs=[s for s in json.loads((RUN/'manifest.json').read_text()) if s['task'] in ['arx','diagnose_reference']];byid={s['id']:s for s in specs}
    running={};failed=[];sampler=Sampler();draining=False
    def stop(sig,frame):
        nonlocal draining
        draining=True
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    adopt=RUN/'operations/CPU_ADOPT.json'
    if adopt.exists():
        for x in json.loads(adopt.read_text()).get('active',[]):
            pid=x['pid'];state=proc(pid)
            if not state or state['start']!=x['start']:continue
            cmd=Path(f'/proc/{pid}/cmdline').read_bytes().decode().replace('\0',' ')
            assert str(RUN) in cmd and x['id'] in cmd
            running[x['id']]=dict(pid=pid,start=state['start'],task=byid[x['id']]['task'],process=None,lock=None,log=None,paused=state['state']=='T',adopted=True)
    while True:
        for jid,item in list(running.items()):
            rc=item['process'].poll() if item['process'] is not None else None
            if owned(item):continue
            for key in ['lock','log']:
                if item[key] is not None:item[key].close()
            del running[jid]
            if not (JOBS/jid/'SUCCESS.json').exists():
                failed.append(jid)
                if not (JOBS/jid/'FAILURE.json').exists():atomic_json(JOBS/jid/'FAILURE.json',dict(timestamp=now(),reason='CPU worker exited without success',return_code=rc))
        sample=sampler.sample(running);mem=memory_status();free,total=mem['available_mib'],mem['total_mib']
        cap=min(args.max_workers,max(0,math.floor(sample['cores']*.85-sample['external_busy_cores']))) if sample['ready'] else 0
        # Count resident workers separately: pausing a solver does not free its RAM.
        ram_slots=max(0,math.floor((free-max(8192,total*.20)-3072)/3072))
        cap=min(cap,len(running)+ram_slots)
        can_launch=free>=max(8192,total*.20)+3072
        if not sample['ready']:cap=sum(not x['paused'] for x in running.values())
        complete=(RUN/'operations/CPU_GLOBAL_COMPLETE.json').exists() or (RUN/'operations/GLOBAL_COMPLETE.json').exists()
        if complete:draining=True
        leased=None
        if DEPLOYMENT is not None:
            p=RUN/'operations/CPU_LEASED_JOBS.json';leased=set(json.loads(p.read_text()).get('jobs',[])) if p.exists() and time.time()-p.stat().st_mtime<180 else set()
        pending=[]
        for spec in specs:
            jid=spec['id'];out=JOBS/jid
            if jid in running or any((out/n).exists() for n in ['SUCCESS.json','FAILURE.json','CPU_RETRY_HOLD.json']):continue
            if leased is not None and jid not in leased:continue
            if DEPLOYMENT is None and (out/'CPU_REMOTE_RESERVATION.json').exists():continue
            pending.append(spec)
        has_reference=can_launch and any(s['task']=='diagnose_reference' for s in pending)
        ref_capacity=min(args.reference_slots,sum(x['task']=='diagnose_reference' for x in running.values())+ram_slots)
        # Reserve short-job slots even if every existing slot contains long ARX work.
        admitted=choose_admitted(running,cap,ref_capacity,has_reference)
        # Pauses preserve the in-flight solver; identity checks prevent signaling a reused PID.
        for jid,item in running.items():
            pause=jid not in admitted and not draining
            if pause!=item['paused']:
                safe_signal(item,signal.SIGSTOP if pause else signal.SIGCONT);item['paused']=pause
                print(now(),'pause' if pause else 'resume',jid,flush=True)
        active_count=sum(not x['paused'] for x in running.values());refs=sum(not x['paused'] and x['task']=='diagnose_reference' for x in running.values())
        has_reference=any(s['task']=='diagnose_reference' for s in pending)
        launches=0
        if not draining and sample['ready'] and len(failed)<2:
            # Dedicated fast replay quota; ARX cannot consume these slots while replays wait.
            pending.sort(key=lambda s:(0 if s['task']=='diagnose_reference' and refs<args.reference_slots else 1 if s['task']=='arx' else 2,s['site'],s['id']))
            for spec in pending:
                if active_count>=cap or launches>=2:break
                if spec['task']=='arx' and has_reference and active_count-refs>=max(0,cap-args.reference_slots):continue
                mem_now=memory_status()
                if mem_now['available_mib']<max(8192,mem_now['total_mib']*.20)+3072:break
                jid=spec['id'];out=JOBS/jid;out.mkdir(exist_ok=True);claim=(out/'worker.lock').open('a+')
                try:fcntl.flock(claim,fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError:claim.close();continue
                if any((out/n).exists() for n in ['SUCCESS.json','FAILURE.json','CPU_RETRY_HOLD.json']) or (DEPLOYMENT is None and (out/'CPU_REMOTE_RESERVATION.json').exists()):claim.close();continue
                log=(out/'worker.log').open('a');env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONNOUSERSITE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
                child=subprocess.Popen([sys.executable,'-s',__file__,'--job-id',jid],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,pass_fds=(claim.fileno(),))
                state=proc(child.pid);assert state is not None
                running[jid]=dict(pid=child.pid,start=state['start'],task=spec['task'],process=child,lock=claim,log=log,paused=False,adopted=False)
                active_count+=1;refs+=spec['task']=='diagnose_reference';launches+=1
                print(now(),'launch',jid,flush=True)
        active=[dict(id=j,pid=x['pid'],gpu='cpu',host=socket.gethostname(),task=x['task'],paused=x['paused'],adopted=x['adopted']) for j,x in running.items()]
        state=dict(timestamp=now(),pid=os.getpid(),enabled=True,active=active,capacity=cap,reference_slots=args.reference_slots,max_workers=args.max_workers,resource_sample=sample,memory_limits=mem,free_ram_mib=free,reserve_ram_mib=max(8192,total*.20),failed=failed,draining=draining)
        atomic_json(folder/'cpu_scheduler_v2.json',state)
        if DEPLOYMENT is None:status_collect()
        if not running and (draining or len(failed)>=2 or (DEPLOYMENT is None and all(any((JOBS/s['id']/n).exists() for n in ['SUCCESS.json','FAILURE.json']) for s in specs))):break
        time.sleep(5)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--job-id');p.add_argument('--max-workers',type=int,default=16);p.add_argument('--reference-slots',type=int,default=4);a=p.parse_args()
    worker(a.job_id) if a.job_id else schedule(a)
