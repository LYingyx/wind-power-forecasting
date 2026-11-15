"""Two workers per used GPU, with memory reserve, resumable OOM and bounded third-worker trials."""
from pathlib import Path
import sys,os,json,time,socket,subprocess,fcntl,argparse,collections,signal,math
RUN=Path(__file__).resolve().parents[1];sys.path.insert(0,str(RUN/'code'))
from settings import JOBS,atomic_json,now,sha
from probe import probe
from resource_limits import effective_memory_available
from storage_guard import storage_headroom
from efficiency_v3 import EfficiencyGovernor


def main(host):
    lock=(RUN/'schedulers/scheduler.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert json.loads((RUN/'operations/CHECKS.json').read_text())['passed']
    for path,value in json.loads((RUN/'SCIENCE_FREEZE.json').read_text())['files'].items():assert sha(RUN/path)==value,path
    policy=json.loads((RUN/'operations/RESOURCES.json').read_text());pool=list(map(str,policy[host]['gpus']))
    governor=EfficiencyGovernor(JOBS,3);running={};failed=set();last=collections.defaultdict(float);cooldown=collections.defaultdict(float)
    assignments=json.loads((RUN/'operations/ASSIGNMENTS.json').read_text());stopping=False
    def stop(*args):
        nonlocal stopping
        stopping=True
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    while True:
        for jid,item in list(running.items()):
            rc=item['p'].poll()
            if rc is None:continue
            item['lock'].close();item['log'].close();del running[jid]
            if rc==75:cooldown[item['gpu']]=time.monotonic()+30
            elif rc:failed.add(jid)
        stopping=stopping or (RUN/'operations/STOP.json').exists()
        specs=json.loads((RUN/'manifest.json').read_text())+json.loads((RUN/'diagnostic_manifest.json').read_text())
        eligible=[s for s in specs if assignments[s['id']]==host and all((RUN/d).exists() for d in s.get('dependencies',[])) and
                  not any((JOBS/s['id']/f).exists() for f in ['SUCCESS.json','FAILURE.json'])]
        # Concentrate short queues into pairs instead of one worker on many cards.
        used=[g for g in pool if any(x['gpu']==g for x in running.values())]
        limit=min(len(pool),max(1,len(eligible)//2))
        selected=used+[g for g in pool if g not in used][:max(0,limit-len(used))]
        try:
            gpus=probe(x['p'].pid for x in running.values());ram,totalram=effective_memory_available()
            storage=storage_headroom(RUN)
        except Exception as exc:
            atomic_json(RUN/'schedulers/status.json',dict(timestamp=now(),host=host,error=repr(exc)));time.sleep(10);continue
        decisions={}
        for gpu in sorted(selected,key=lambda g:sum(x['gpu']==g for x in running.values())):
            items=[(j,x) for j,x in running.items() if x['gpu']==gpu];g=gpus[gpu];count=len(items)
            reserve=max(policy['minimum_free_mib'],int(g['total_mib']*policy['memory_reserve_fraction']))
            observed=[g['process_memory_mib'].get(str(x['p'].pid),0) for _,x in items]
            budget=max(2304,int(max(observed,default=0)*1.3+384))
            ready=not any(x['p'].pid not in g['own_pids'] for _,x in items)
            decision=dict(launch=count<2,yield_count=0,reason='fill_pair',free_mib=g['free_mib'],reserve_mib=reserve,worker_budget_mib=budget)
            trial=governor.control(gpu,items,dict(decision,launch=count<3))
            if count>=2:
                decision=trial
                if decision.get('yield_count') and count<=2:decision.update(yield_count=0,launch=False)
            if g['free_mib']<reserve and items:
                decision.update(launch=False,yield_count=1,reason='memory_reserve')
            if stopping:decision.update(launch=False,yield_count=count,reason='cooperative_stop')
            if decision.get('yield_count'):
                for jid,item in sorted(items,key=lambda it:it[1]['started'],reverse=True)[:decision['yield_count']]:
                    if not item['diagnostic']:atomic_json(JOBS/jid/'YIELD_REQUEST.json',dict(timestamp=now(),reason=decision['reason']))
            if g['free_mib']<reserve+budget or not ready or time.monotonic()-last[gpu]<20 or time.monotonic()<cooldown[gpu]:
                decision.update(launch=False,reason='memory_or_warmup')
            if ram<max(8192,totalram*.1)+2048 or not storage['ok']:decision.update(launch=False,reason='host_memory_or_storage')
            if len(failed)>=2:decision.update(launch=False,reason='failure_circuit_breaker')
            decisions[gpu]=decision
            if not decision['launch']:continue
            for spec in eligible:
                jid=spec['id']
                if jid in running or jid in failed:continue
                out=JOBS/jid;out.mkdir(parents=True,exist_ok=True);claim=(out/'worker.lock').open('a+')
                try:fcntl.flock(claim,fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError:claim.close();continue
                (out/'YIELD_REQUEST.json').unlink(missing_ok=True)
                diagnostic=spec.get('task')=='reference_evaluation'
                entry='reference_evaluation.py' if diagnostic else 'train.py'
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONNOUSERSITE='1')
                log=(out/'worker.log').open('a')
                p=subprocess.Popen([sys.executable,'-u',str(RUN/'code'/entry),'--job-id',jid],env=env,
                    stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,pass_fds=(claim.fileno(),))
                running[jid]=dict(p=p,lock=claim,log=log,gpu=gpu,started=time.monotonic(),diagnostic=diagnostic)
                last[gpu]=time.monotonic();ram-=2048
                print(now(),'launch',jid,'gpu',gpu,'workers',count+1,flush=True);break
        active=[dict(id=j,pid=x['p'].pid,gpu=x['gpu'],host=host) for j,x in running.items()]
        atomic_json(RUN/'schedulers/status.json',dict(timestamp=now(),host=host,pid=os.getpid(),active=active,
                    decisions=decisions,failed=sorted(failed),eligible=len(eligible),stopping=stopping,storage=storage))
        if not running and (stopping or len(failed)>=2):break
        time.sleep(8)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--host',required=True);main(p.parse_args().host)
