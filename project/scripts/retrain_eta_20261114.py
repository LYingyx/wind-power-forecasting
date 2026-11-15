"""Read-only three-host snapshot and dependency-aware ETA scenarios (not promises)."""
from pathlib import Path
import json,subprocess,concurrent.futures,datetime,statistics,collections,heapq,argparse
PROJECT=Path(__file__).resolve().parents[1];ROOT=PROJECT/'runs/daily_protocol_restart_20261114_v1';EXTRA=PROJECT/'runs/daily_ablation_restart_20261114_v1'
CODE='''import json,pathlib,socket
r=pathlib.Path(ROOT); rows=[]
for p in (r/'jobs').iterdir():
 def read(n,default):
  try:return json.loads((p/n).read_text())
  except (OSError,ValueError):return default
 s=read('spec.json',{});h=read('history.json',[]);g=read('progress.json',{});m=read('metrics.json',{})
 if s or h or g:rows.append(dict(id=p.name,spec=s,history=h,progress=g,done=(p/'SUCCESS.json').exists(),metrics=m))
print(json.dumps(dict(host=socket.gethostname(),jobs=rows)))'''
def read(host):
    if host=='local':args=['python3','-c',CODE.replace('ROOT',repr(str(ROOT)))];inp=None
    else:args=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=8',host,'python3 -'];inp=CODE.replace('ROOT',repr('/home/liying/wind_physics_project/project/runs/'+ROOT.name))
    p=subprocess.run(args,input=inp,text=True,capture_output=True,check=True,timeout=60);return json.loads(p.stdout)
def atomic(p,x):
    t=p.with_suffix('.tmp');t.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n');t.replace(p)
def main():
    if (EXTRA/'LAUNCH_STATUS.json').exists() and not (ROOT/'COMPLETE.json').exists():
        print('Sequential ETA is superseded by concurrent queues. Use operations/resource_expansion_20261114/LIVE.json and per-run histories; do not overwrite the marked historical estimate.');return
    with concurrent.futures.ThreadPoolExecutor(3) as ex:snaps=list(ex.map(read,['local','v1006001','v1006002']))
    status=json.loads((ROOT/'STATUS.json').read_text());now=datetime.datetime.now().astimezone();active={x['id'] for x in status['active']};jobs={};measure=collections.defaultdict(list);host_stats=[]
    for snap in snaps:
        by=collections.defaultdict(list)
        for j in snap['jobs']:
            if j['id'] not in jobs or len(j['history'])>len(jobs[j['id']]['history']) or j['done']:jobs[j['id']]=j
            if j['id'] in active and j['history']:
                key=(j['spec']['family'],j['spec']['config']['width']);recent=[e['seconds'] for e in j['history'][-3:]];measure[key].extend(recent);by[key].extend(recent)
        host_stats.append(dict(host=snap['host'],active=sum(j['id'] in active for j in snap['jobs']),epoch_seconds={f'{f}_{w}':dict(median=statistics.median(v),observations=len(v)) for (f,w),v in by.items()}))
    done={jid for jid,j in jobs.items() if j['done']};manifest=json.loads((ROOT/'manifest.json').read_text());extra=json.loads((EXTRA/'manifest.json').read_text())
    rates={k:statistics.median(v) for k,v in measure.items()}
    # B0 historical timing comes only from this new run; all-fold validation sizes differ.
    for j in jobs.values():
        if j['spec'].get('family')=='B0' and j['history']:rates['B0',64]=max(110.,statistics.median([e['seconds'] for e in j['history'][-3:]]))
    def rate(s,scale=1.):
        fam=s['family'];width=s.get('config',{}).get('width',128)
        r=rates.get((fam,width),rates.get((fam,64),300.)*(1.25 if width==128 else 1.))
        if s.get('condition') in ['cbh','wrong_cbh']:r*=1.2
        if fam=='B0' and s.get('fold',-1)==-1:r*=1.5
        if 'station' in s:r=r/55*1.15
        return r*scale
    def simulate(specs,cap,tune_epochs,main_epochs,ref_epochs,scale=1.,observed=False):
        completed=set(done) if observed else set();todo=[s for s in specs if s['id'] not in completed];queue=[];t=0.;serial=0
        def deps(s):
            ds=[]
            if 'config_from' in s:
                group=s['config_from'];ds.extend(x['id'] for x in specs if x['stage']=='tune' and x['group']==group)
            if s.get('condition') in ['cb','cbh','anchor','wrong_cb','wrong_cbh']:
                ds.extend(x['id'] for x in specs if x['family']=='B0' and x['seed']==s['seed'])
            return ds
        def cost(s):
            target=min(s['epochs'],tune_epochs if s['stage']=='tune' else ref_epochs if s['family']=='B0' else main_epochs)
            j=jobs.get(s['id'],{}) if observed else {};h=j.get('history',[]);p=j.get('progress',{});ep=h[-1]['epoch'] if h else 0
            if ep:target=max(target,ep+1)
            target=min(target,s['epochs']);fraction=0.
            if p.get('epoch')==ep+1:fraction=min(1.,p['step']/max(1,p['total_steps']))*.7
            return max(20.,(max(0,target-ep-fraction)+.5)*rate(s,scale))+30.
        # Honor already active work even if the future admission cap is lower.
        if observed:
            for s in todo[:]:
                if s['id'] in active:serial+=1;heapq.heappush(queue,(cost(s),serial,s['id']));todo.remove(s)
        while todo or queue:
            while len(queue)<cap:
                s=next((s for s in todo if all(x in completed for x in deps(s))),None)
                if s is None:break
                todo.remove(s);serial+=1;heapq.heappush(queue,(t+cost(s),serial,s['id']))
            if not queue:raise RuntimeError('Unresolved ETA dependency')
            t,_,jid=heapq.heappop(queue);completed.add(jid)
        return t
    scenarios=[]
    # Lower/central are hypotheses about early stopping; upper is full epoch budgets
    # with fewer slots and a 15% slowdown, conditional on these machines remaining usable.
    current=max(1,len(active));options=[('较快早停',min(current,48),10,18,12,1.),('中性规划',min(current,36),20,35,25,1.),('全部跑满预算',min(current,30),30,80,80,1.15)]
    for label,cap,te,me,re,scale in options:
        shared=simulate(manifest,cap,te,me,re,scale,True)+900
        station_rate=statistics.median([rate(dict(family=f,station=0),scale) for f in ['W0','TF']])
        pe=min(30,12 if label=='较快早停' else 24 if label=='中性规划' else 30)
        # Includes epoch0/final evaluation, setup/checkpoints/Z1, cache prep and deploy.
        personal=1100*((pe+2)*station_rate+60)/cap+1800
        ablation=simulate(extra,cap,te,me,re,scale)+900
        total=shared+personal+ablation
        scenarios.append(dict(scenario=label,worker_slots=cap,assumed_epochs=dict(tune=te,shared_and_independent=me,B0=re,personalization=pe),shared_remaining_hours=shared/3600,personalization_hours=personal/3600,ablation_and_independent_hours=ablation/3600,total_remaining_hours=total/3600,shared_finish=(now+datetime.timedelta(seconds=shared)).isoformat(),all_training_finish=(now+datetime.timedelta(seconds=total)).isoformat()))
    snapshot=dict(timestamp=now.isoformat(),status={k:status[k] for k in ['timestamp','planned','completed','pending','failed','active','errors']},snapshots=snaps)
    atomic(ROOT/'operations/ETA_SNAPSHOT.json',snapshot)
    result=dict(timestamp=now.isoformat(),shared_status=dict(planned=180,completed=status['completed'],active=len(active),pending=status['pending'],failed=status['failed'],errors=status['errors']),hosts=host_stats,epoch_rate_seconds={f'{f}_{w}':r for (f,w),r in rates.items()},scenarios=scenarios,total_training_jobs=2571,scope=dict(shared=180,personalization=1100,new_ablation_and_independent=1291),limitations=['Only shared epochs measured. Per-site costs scaled from full55-farm epochs with15% allowance; personalization cache preparation/Z1/setup included approximately.','Early-stop epochs and future GPU availability unknown; this is not a guaranteed finish date or a measured confidence interval.','Full-budget scenario is conditional, not an absolute bound if GPUs become unavailable.','2024 model reevaluation and conditional LM model extensions excluded.','Forecast is for initial queue phase; after stage1 finishes, use measured stage2/3 histories to revise rather than reuse these assumptions.'])
    atomic(PROJECT/'RETRAIN_ETA_20261114.json',result)
    print(json.dumps({k:result[k] for k in ['timestamp','shared_status','hosts','scenarios']},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
