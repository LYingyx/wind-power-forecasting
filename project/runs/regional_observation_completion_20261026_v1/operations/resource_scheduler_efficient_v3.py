"""Resource-aware queue for explicitly enabled hosts; immutable scientific worker."""
from pathlib import Path
import os,sys,subprocess,json,time,socket,hashlib,fcntl,collections,statistics,argparse,signal
RUN=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(RUN/'code'))
from relocate import activate,source_path
DEPLOYMENT=activate(RUN)
from settings import JOBS,atomic_json,now,sha,setup
from gpu_policy import Policy,decide
from efficiency_v3 import EfficiencyGovernor


def machine_id():
    raw=Path('/etc/machine-id').read_bytes() if Path('/etc/machine-id').exists() else socket.gethostname().encode()
    return hashlib.sha256(raw.strip()).hexdigest()


def enabled():
    p=json.loads((RUN/'operations/HOST_POLICY.json').read_text())
    return machine_id() in p['allowed_machine_ids']


def command(args):
    return subprocess.check_output(args,text=True,timeout=15,stderr=subprocess.DEVNULL)


def probe(own_pids=()):
    own=set(own_pids);gpus={}
    for line in command(['nvidia-smi','--query-gpu=index,uuid,name,memory.total,memory.free,utilization.gpu','--format=csv,noheader,nounits']).splitlines():
        idx,uuid,name,total,free,util=[s.strip() for s in line.split(',')]
        gpus[idx]=dict(uuid=uuid,name=name,total_mib=int(total),free_mib=int(free),utilization=int(util),other_pids=[],own_pids=[],process_memory_mib={},external_sm=0,process_utilization_known=True)
    uuids={g['uuid']:g for g in gpus.values()}
    for line in command(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader,nounits']).splitlines():
        fields=[s.strip() for s in line.split(',')]
        if len(fields)!=3 or fields[0] not in uuids:continue
        uuid,pid,memory=fields;pid=int(pid);g=uuids[uuid]
        g['own_pids' if pid in own else 'other_pids'].append(pid)
        if memory.isdigit():g['process_memory_mib'][str(pid)]=int(memory)
    process_sm={}
    try:
        for line in command(['nvidia-smi','pmon','-c','1','-s','um']).splitlines():
            cols=line.split()
            if not cols or cols[0].startswith('#') or len(cols)<5:continue
            if cols[1].isdigit() and cols[3].isdigit():process_sm[(cols[0],int(cols[1]))]=int(cols[3])
    except (subprocess.SubprocessError,OSError):pass
    for idx,g in gpus.items():
        g['external_sm']=sum(process_sm.get((idx,pid),0) for pid in g['other_pids'])
        g['process_utilization_known']=all((idx,pid) in process_sm for pid in g['other_pids'])
    return gpus


def cpu_memory_available():
    from resource_limits import effective_memory_available
    return effective_memory_available()

_cpu_previous=None
def cpu_room():
    global _cpu_previous
    row=next(x for x in Path('/proc/stat').read_text().splitlines() if x.startswith('cpu '))
    values=list(map(int,row.split()[1:]));total=sum(values[:8]);idle=values[3]+values[4]
    previous=_cpu_previous;_cpu_previous=(total,idle)
    if previous is None:return False
    dt=total-previous[0];di=idle-previous[1]
    return dt>0 and (1-di/dt)<.85


def validate():
    frozen=json.loads((RUN/'SOURCE_FREEZE.json').read_text())
    for path,expected in frozen['files'].items():assert sha(source_path(RUN,path))==expected,'Frozen science code/config changed: '+path
    for path,expected in json.loads((RUN/'operations/OPERATIONAL_OVERLAY_V2.json').read_text())['files'].items():assert sha(source_path(RUN,path))==expected,path
    for path,expected in json.loads((RUN/'operations/OPERATIONAL_OVERLAY_V3.json').read_text())['files'].items():assert sha(source_path(RUN,path))==expected,path
    assert json.loads((RUN/'operations/CHECKS.json').read_text())['passed']
    assert enabled(),'This host is not enabled in HOST_POLICY.json'
    if DEPLOYMENT is not None:
        assert json.loads((RUN/'operations/GPU_COMPATIBILITY.json').read_text())['passed']


def status_collect():
    from collect_physics import collect
    state=collect();completed={p.parent.name for p in JOBS.glob('*/SUCCESS.json')};active=[]
    for p in (RUN/'schedulers').glob('*.json'):
        try:
            d=json.loads(p.read_text())
            if not d.get('enabled',True) or time.time()-p.stat().st_mtime>120:continue
            active.extend(x for x in d.get('active',[]) if x['id'] not in completed)
        except (OSError,ValueError):continue
    active=list({x['id']:x for x in active}.values())
    state['active']=active;state['pending']=state['planned']-state['completed']-state['failed']-len(active)
    state['enabled_hosts']=json.loads((RUN/'operations/HOST_POLICY.json').read_text())['host_labels']
    atomic_json(RUN/'STATUS.json',state)
    if state['completed']==state['planned']:
        atomic_json(RUN/'SCREENING_COMPLETE.json',state)
    return state


def run(policy=Policy()):
    setup();validate();host=socket.gethostname();identity=machine_id();tag=host+'-'+identity[:8]
    folder=RUN/'schedulers';folder.mkdir(exist_ok=True)
    # Same lock name as old dispatcher so both cannot launch together on this host.
    lock=(folder/(host+'.lock')).open('a+')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise SystemExit('A dispatcher is already active for '+host)
    manifest=json.loads((RUN/'manifest.json').read_text());running={};samples=collections.defaultdict(list)
    efficiency=EfficiencyGovernor(JOBS,policy.max_workers_per_gpu)
    last_launch=collections.defaultdict(lambda:0.);cooldown=collections.defaultdict(lambda:0.);failed=[];draining=False
    def drain(signum,frame):
        nonlocal draining
        draining=True
    signal.signal(signal.SIGTERM,drain);signal.signal(signal.SIGINT,drain)
    finished=set();quarantined=set();decisions={};budget=collections.defaultdict(lambda:policy.initial_worker_mib)
    while True:
        if not enabled() or (RUN/'operations/GLOBAL_COMPLETE.json').exists():draining=True
        for jid,item in list(running.items()):
            rc=item['process'].poll()
            if rc is None:continue
            item['lock'].close();item['log'].close();del running[jid]
            if rc==0:finished.add(jid)
            elif rc==75:
                reason=item.get('yield_reason','')
                cooldown[item['gpu']]=time.monotonic()+(policy.poll_seconds if reason in ['matched_job_epoch_slowdown','efficiency_cap'] else policy.yield_cooldown_seconds)
            else:failed.append(jid);quarantined.add(jid)
        try:gpus=probe(x['process'].pid for x in running.values())
        except (OSError,subprocess.SubprocessError) as e:
            atomic_json(folder/(tag+'.json'),dict(timestamp=now(),pid=os.getpid(),enabled=True,host=host,probe_error=str(e),active=[dict(id=jid,pid=x['process'].pid,gpu=x['gpu'],host=host) for jid,x in running.items()]))
            time.sleep(policy.poll_seconds);continue
        lease_file=RUN/'operations/LEASED_JOBS.json'
        available_cpu=cpu_room()
        leased=None
        if DEPLOYMENT is not None:
            leased=set(json.loads(lease_file.read_text()).get('jobs',[])) if lease_file.exists() and time.time()-lease_file.stat().st_mtime<180 else set()
        for gpu,snapshot in gpus.items():
            samples[gpu].append(snapshot);samples[gpu]=samples[gpu][-6:]
            items=[(jid,x) for jid,x in running.items() if x['gpu']==gpu]
            observed=[snapshot['process_memory_mib'].get(str(x['process'].pid),0) for _,x in items]
            if any(observed):budget[gpu]=max(budget[gpu],int(max(observed)*1.5+256))
            starting=sum(x['process'].pid not in snapshot['own_pids'] for _,x in items)
            decision=decide(samples[gpu],len(items),starting,(time.monotonic()-last_launch[gpu] if items else 1e9),budget[gpu],policy)
            decision=efficiency.control(gpu,items,decision)
            decisions[gpu]=decision
            if draining:
                decision.update(launch=False,yield_count=len(items),reason='host_disabled_or_dispatcher_stopping')
            if decision['yield_count']:
                for jid,item in sorted(items,key=lambda x:x[1]['started'],reverse=True)[:decision['yield_count']]:
                    item['yield_reason']=decision['reason']
                    request=JOBS/jid/'YIELD_REQUEST.json'
                    if not request.exists():atomic_json(request,dict(timestamp=now(),reason=decision['reason']))
                cooldown[gpu]=time.monotonic()+(policy.poll_seconds if decision['reason'] in ['matched_job_epoch_slowdown','efficiency_cap'] else policy.yield_cooldown_seconds)
            if draining or len(failed)>=2 or time.monotonic()<cooldown[gpu] or not decision['launch']:continue
            if not available_cpu:
                decision.update(launch=False,reason='host_cpu_reserve');continue
            free_ram,total_ram=cpu_memory_available()
            if free_ram<max(8192,total_ram*.15)+3072:
                decision.update(launch=False,reason='host_ram_reserve');continue
            # At most one new process per card per scan; reassess its measured load first.
            for spec in manifest:
                if spec['task'] not in ['train','diagnose_reference']:continue
                jid=spec['id'];out=JOBS/jid
                if jid in running or jid in finished or jid in quarantined:continue
                if leased is not None and jid not in leased:continue
                if DEPLOYMENT is None and (out/'REMOTE_RESERVATION.json').exists():continue
                if (out/'SUCCESS.json').exists():finished.add(jid);continue
                if (out/'FAILURE.json').exists():quarantined.add(jid);continue
                out.mkdir(exist_ok=True);claim=(out/'worker.lock').open('a+')
                try:fcntl.flock(claim,fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError:claim.close();continue
                if (out/'SUCCESS.json').exists() or (out/'FAILURE.json').exists() or (DEPLOYMENT is None and (out/'REMOTE_RESERVATION.json').exists()):claim.close();continue
                # Only the holder of this job's lock can clear an old resource-yield request.
                (out/'YIELD_REQUEST.json').unlink(missing_ok=True)
                log=(out/'worker.log').open('a')
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu,PYTHONNOUSERSITE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
                child=subprocess.Popen([sys.executable,str(RUN/'operations/scheduled_worker.py'),'--job-id',jid],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,pass_fds=(claim.fileno(),))
                running[jid]=dict(process=child,lock=claim,log=log,gpu=gpu,started=time.monotonic());last_launch[gpu]=time.monotonic()
                print(now(),'launch',jid,'gpu',gpu,'concurrent',len(items)+1,flush=True);break
        active=[dict(id=jid,pid=x['process'].pid,gpu=x['gpu'],host=host) for jid,x in running.items()]
        atomic_json(folder/(tag+'.json'),dict(timestamp=now(),pid=os.getpid(),host=host,enabled=not draining,active=active,gpus=gpus,decisions=decisions,
            max_workers_per_gpu=policy.max_workers_per_gpu,resource_policy=policy.__dict__,waiting_for_resources=not active,failed_on_host=failed,circuit_breaker=len(failed)>=2,draining=draining))
        state=status_collect()
        if not running and (draining or len(failed)>=2 or state['completed']+state['failed']>=len(manifest)):break
        time.sleep(policy.poll_seconds)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--probe',action='store_true');ap.add_argument('--max-workers-per-gpu',type=int,default=12)
    ap.add_argument('--allow-shared-compute',action='store_true');ap.add_argument('--warmup-seconds',type=int,default=40);a=ap.parse_args()
    if a.probe:print(json.dumps(probe(),ensure_ascii=False))
    else:run(Policy(max_workers_per_gpu=a.max_workers_per_gpu,allow_shared_compute=a.allow_shared_compute,warmup_seconds=a.warmup_seconds))
