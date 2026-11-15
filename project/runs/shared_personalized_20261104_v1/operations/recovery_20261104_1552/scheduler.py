"""Adaptive multiple-process GPU admission; only our workers yield and resume."""
from pathlib import Path
import sys,os,json,time,socket,subprocess,fcntl,collections,argparse,signal
RUN=Path(__file__).resolve().parents[1];sys.path.insert(0,str(RUN/'code'))
from settings import JOBS,DATA,atomic_json,now,sha
from probe import probe
from gpu_policy import Policy,decide
from efficiency_v3 import EfficiencyGovernor
from resource_limits import effective_memory_available

def run(gpu_ids,remote=False):
    folder=RUN/'schedulers';folder.mkdir(exist_ok=True)
    lock=(folder/'scheduler.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert json.loads((RUN/'operations/CHECKS.json').read_text())['passed']
    assert json.loads((DATA/'READY.json').read_text())['passed']
    for rel,digest in json.loads((RUN/'SCIENCE_FREEZE.json').read_text())['files'].items():assert sha(RUN/rel)==digest,rel
    policy=Policy(max_workers_per_gpu=6,warmup_seconds=25,initial_worker_mib=3072,allow_shared_compute=False)
    governor=EfficiencyGovernor(JOBS,6);running={};samples=collections.defaultdict(list);last=collections.defaultdict(float);cooldown=collections.defaultdict(float)
    budget=collections.defaultdict(lambda:3072);failed=[];draining=False
    def stop(sig,frame):
        nonlocal draining
        draining=True
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    manifest=json.loads((RUN/'manifest.json').read_text());lastcpu=None
    while True:
        for jid,item in list(running.items()):
            rc=item['p'].poll()
            if rc is None:continue
            item['lock'].close();item['log'].close();del running[jid]
            if rc==75:
                cooldown[item['gpu']]=time.monotonic()+policy.yield_cooldown_seconds
                yp=JOBS/jid/'YIELDED.json'
                if yp.exists() and json.loads(yp.read_text()).get('reason')=='cuda_oom':budget[item['gpu']]=int(budget[item['gpu']]*1.5)
            elif rc:failed.append(jid)
        if (RUN/'operations/STOP.json').exists():draining=True
        try:gpus=probe(x['p'].pid for x in running.values())
        except Exception as exc:
            atomic_json(folder/'status.json',dict(timestamp=now(),pid=os.getpid(),error=str(exc)));time.sleep(10);continue
        lease=None
        if remote:
            lp=RUN/'operations/LEASE.json';lease=set()
            if lp.exists() and time.time()-lp.stat().st_mtime<180:lease=set(json.loads(lp.read_text())['jobs'])
        decisions={};ram,totalram=effective_memory_available()
        cpu=list(map(int,Path('/proc/stat').read_text().splitlines()[0].split()[1:9]));total=sum(cpu);idle=cpu[3]+cpu[4]
        cpu_ok=lastcpu is not None and (idle-lastcpu[1])/max(1,total-lastcpu[0])>.15;lastcpu=(total,idle)
        ordered_gpus=sorted(gpu_ids,key=lambda g:sum(x['gpu']==g for x in running.values())) if not remote else gpu_ids
        for gpu in ordered_gpus:
            g=gpus[gpu];samples[gpu].append(g);samples[gpu]=samples[gpu][-6:]
            items=[(j,x) for j,x in running.items() if x['gpu']==gpu]
            observed=[g['process_memory_mib'].get(str(x['p'].pid),0) for _,x in items]
            if any(observed):budget[gpu]=max(2048,int(max(observed)*1.25+256))
            starting=sum(x['p'].pid not in g['own_pids'] for _,x in items)
            decision=decide(samples[gpu],len(items),starting,time.monotonic()-last[gpu],budget[gpu],policy)
            # A second worker trial on an otherwise unoccupied GPU must pass
            # the measured throughput / slowdown governor before expansion.
            if len(items)==1 and not g['other_pids'] and not starting and decision['reason']=='compute_target_reached' and decision.get('additional_memory_slots',0)>0 and time.monotonic()-last[gpu]>25:
                decision.update(launch=True,reason='two_worker_throughput_trial')
            decision=governor.control(gpu,items,decision)
            if draining:decision.update(launch=False,yield_count=len(items),reason='cooperative_stop')
            if decision.get('yield_count'):
                for jid,item in sorted(items,key=lambda p:p[1]['started'],reverse=True)[:decision['yield_count']]:
                    atomic_json(JOBS/jid/'YIELD_REQUEST.json',dict(timestamp=now(),reason=decision['reason']))
            decisions[gpu]=decision
            ram_reserve=8192 if not remote else max(8192,totalram*.15)
            if not cpu_ok or ram<ram_reserve+3072:decision.update(launch=False,reason='cpu_or_ram_reserve')
            if not decision['launch'] or len(failed)>=2 or time.monotonic()<cooldown[gpu]:continue
            for spec in manifest:
                jid=spec['id'];out=JOBS/jid
                if not all((RUN/dep).exists() for dep in spec.get('dependencies',[])):continue
                cfg=spec.get('config')
                if cfg is None:cfg=json.loads((RUN/f'assets/selected/{spec["config_from"]}.json').read_text())['config']
                needed=max(budget[gpu],4096 if cfg['width']==128 else 2560)
                if g['free_mib']-max(4096,int(.2*g['total_mib']))-starting*budget[gpu]<needed:continue
                if jid in running or jid in failed:continue
                if lease is not None and jid not in lease:continue
                if not remote and (out/'REMOTE.json').exists():continue
                if (out/'SUCCESS.json').exists() or (out/'FAILURE.json').exists():continue
                out.mkdir(parents=True,exist_ok=True);claim=(out/'worker.lock').open('a+')
                try:fcntl.flock(claim,fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError:claim.close();continue
                if (out/'SUCCESS.json').exists() or (out/'REMOTE.json').exists():claim.close();continue
                (out/'YIELD_REQUEST.json').unlink(missing_ok=True)
                log=(out/'worker.log').open('a');env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONNOUSERSITE='1')
                p=subprocess.Popen([sys.executable,'-u',str(RUN/'code/train.py'),'--job-id',jid],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,pass_fds=(claim.fileno(),))
                running[jid]=dict(p=p,lock=claim,log=log,gpu=gpu,started=time.monotonic());last[gpu]=time.monotonic()
                ram-=3072
                print(now(),'launch',jid,'gpu',gpu,'concurrency',len(items)+1,flush=True);break
        active=[dict(id=j,pid=x['p'].pid,gpu=x['gpu'],host=socket.gethostname()) for j,x in running.items()]
        atomic_json(folder/'status.json',dict(timestamp=now(),pid=os.getpid(),host=socket.gethostname(),active=active,gpus={g:gpus[g] for g in gpu_ids},decisions=decisions,failed=failed,circuit_breaker=len(failed)>=2,draining=draining))
        if not running and (draining or len(failed)>=2):break
        if not remote and not running and all((JOBS/s['id']/'SUCCESS.json').exists() for s in manifest):break
        time.sleep(10)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--gpus',required=True);ap.add_argument('--remote',action='store_true');a=ap.parse_args();run(a.gpus.split(','),a.remote)
