"""Read-only parallel-queue snapshot; conditional ETA, not a completion promise."""
from pathlib import Path
import json, subprocess, concurrent.futures, datetime, statistics, collections, heapq

P = Path(__file__).resolve().parents[1]
NAMES = ['daily_protocol_restart_20261114_v1', 'daily_ablation_restart_20261114_v1',
         'daily_site_finetune_20261114_v1']
OUT = P/'operations/eta_recovery_20261115'
CODE = '''from pathlib import Path
import json,time
base=Path(BASE);result={}
def read(p,default):
 try:return json.loads(p.read_text())
 except (OSError,ValueError):return default
for name in NAMES:
 r=base/name;rows=[]
 for p in (r/'jobs').glob('*'):
  if not p.is_dir():continue
  s=read(p/'spec.json',{})
  if s:rows.append(dict(id=p.name,spec=s,history=read(p/'history.json',[]),progress=read(p/'progress.json',{}),done=(p/'SUCCESS.json').exists()))
 st=read(r/'schedulers/status.json',{})
 result[name]=dict(jobs=rows,scheduler=st,age_seconds=time.time()-(r/'schedulers/status.json').stat().st_mtime if st else None)
print(json.dumps(result))'''

def read_host(host):
    base = str(P/'runs') if host == 'local' else '/home/liying/wind_physics_project/project/runs'
    code = CODE.replace('BASE', repr(base)).replace('NAMES', repr(NAMES))
    cmd = ['python3', '-'] if host == 'local' else ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8', host, 'python3 -']
    p = subprocess.run(cmd, input=code, text=True, capture_output=True, timeout=90, check=True)
    return host, json.loads(p.stdout)

def write(p, value):
    tmp = p.with_suffix('.tmp');tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n');tmp.replace(p)

def main():
    if all((P/'runs'/n/'COMPLETE.json').exists() for n in [NAMES[0], NAMES[2]]):
        from retrain_eta_tail_20261115 import main as tail_main
        return tail_main()
    with concurrent.futures.ThreadPoolExecutor(3) as pool:
        snaps = dict(pool.map(read_host, ['local', 'v1006001', 'v1006002']))
    now = datetime.datetime.now().astimezone(); jobs = {}; statuses = {}; cards = []; manifests = {}
    for name in NAMES:
        r = P/'runs'/name
        manifests[name] = json.loads((r/'manifest.json').read_text()) if (r/'manifest.json').exists() else []
        local = {j['id']:j for j in snaps['local'][name]['jobs']}
        merged = dict(local)
        for marker in (r/'jobs').glob('*/REMOTE.json'):
            owner = json.loads(marker.read_text())['host']
            remote = next((j for j in snaps[owner][name]['jobs'] if j['id'] == marker.parent.name), None)
            if remote:merged[remote['id']] = remote
        jobs[name] = merged
        done = sum((r/'jobs'/s['id']/'SUCCESS.json').exists() for s in manifests[name])
        active = []; failed = []
        for host, data in snaps.items():
            item = data[name]; st = item['scheduler']
            if item['age_seconds'] is None or item['age_seconds'] > 120:continue
            active.extend(st.get('active', []));failed.extend(st.get('failed', []))
            for gpu,d in st.get('decisions', {}).items():
                g = st.get('gpus', {}).get(gpu, {})
                cards.append(dict(run=name, host=host, gpu=gpu, workers=sum(a['gpu']==gpu for a in st.get('active', [])),
                                  utilization=g.get('utilization'), free_mib=g.get('free_mib'), reason=d.get('reason'), efficiency=d.get('efficiency')))
        status = json.loads((r/'STATUS.json').read_text()) if (r/'STATUS.json').exists() else {}
        statuses[name] = dict(completed=done, planned=len(manifests[name]) or 1100, active=active,
                              failed=sorted(set(failed+status.get('failed', []))), errors=status.get('errors', {}))
    # Use only the new protocol's completed main jobs for empirical stop epochs and rates.
    epochs = collections.defaultdict(list); rates = collections.defaultdict(list)
    for name in NAMES[:2]:
        for j in jobs[name].values():
            s=j['spec'];h=j['history']
            if not j['done'] or not h:continue
            key=s['family'];epochs[(s['stage'],key)].append(h[-1]['epoch'])
            if s['stage']!='tune' and key!='B0' and 'station' not in s:
                rates[key].extend(e['seconds'] for e in h[-3:])
    base_rates = {k:statistics.median(v) for k,v in rates.items()}
    epoch_summary = {f'{s}/{f}':dict(n=len(v),median=statistics.median(v),minimum=min(v),maximum=max(v)) for (s,f),v in epochs.items()}
    def spec_rate(s,scale,site_factor):
        rate=base_rates.get(s['family'],300.)
        cfg=s.get('config',{})
        if cfg.get('width')==128:rate*=1.15
        if 'station' in s:rate=rate/55*site_factor
        return rate*scale
    def simulate(name,slots,tune,main,independent,scale,site_factor,setup):
        todo=[];done=set();clock=0.;queue=[];work=collections.Counter()
        for s in manifests[name]:
            j=jobs[name].get(s['id'],{})
            if j.get('done'):done.add(s['id']);continue
            h=j.get('history',[]);ep=h[-1]['epoch'] if h else 0
            target=tune if s['stage']=='tune' else independent if 'station' in s else main
            target=min(s['epochs'],max(target,ep+1))
            # Final evaluation and per-job setup/return are explicitly budgeted.
            cost=(max(0,target-ep)+1)*spec_rate(s,scale,site_factor)+setup
            kind='independent' if 'station' in s else 'shared_size'
            work[kind]+=cost
            deps=[]
            group=s.get('config_from')
            if group and not (P/'runs'/name/'assets/selected'/f'{group}.json').exists():
                deps=[x['id'] for x in manifests[name] if x['stage']=='tune' and x['group']==group]
            todo.append((s['id'],cost,deps))
        while todo or queue:
            while len(queue)<slots:
                item=next((x for x in todo if all(d in done for d in x[2])),None)
                if item is None:break
                todo.remove(item);heapq.heappush(queue,(clock+item[1],item[0]))
            if not queue:raise RuntimeError('Unresolved ETA dependency')
            clock,jid=heapq.heappop(queue);done.add(jid)
        return clock,dict(work)
    scenarios=[]
    # No capacity from the currently unavailable 6002 is assumed. Slots are scenario
    # assumptions; actual workers are reported separately above. Single-site timings
    # still require calibration when the first formal jobs finish.
    options=[dict(label='较快情景',shared_slots=8,ablation_slots=20,personal_slots=24,tune=8,main=14,independent=20,personal=10,scale=1.,site_factor=1.15,setup=45),
             dict(label='中性规划',shared_slots=6,ablation_slots=15,personal_slots=16,tune=10,main=18,independent=35,personal=18,scale=1.1,site_factor=1.3,setup=60),
             dict(label='保守规划',shared_slots=4,ablation_slots=10,personal_slots=10,tune=15,main=24,independent=60,personal=30,scale=1.2,site_factor=1.5,setup=90),
             dict(label='全部跑满预算压力情景',shared_slots=4,ablation_slots=10,personal_slots=10,tune=30,main=80,independent=80,personal=30,scale=1.2,site_factor=1.5,setup=90)]
    for opt in options:
        common=[opt[k] for k in ['tune','main','independent','scale','site_factor','setup']]
        shared,sw=simulate(NAMES[0],opt['shared_slots'],*common)
        ablation,aw=simulate(NAMES[1],opt['ablation_slots'],*common)
        shared+=600;ablation+=900
        sr=statistics.median([base_rates.get(f,300.) for f in ['W0','TF']])/55*opt['site_factor']*opt['scale']
        personal=(1100-statuses[NAMES[2]]['completed'])*((opt['personal']+2)*sr+opt['setup']+30)/opt['personal_slots']+2700
        total=max(ablation,shared+personal)
        scenarios.append(dict(**opt,shared_remaining_hours=shared/3600,ablation_and_independent_hours=ablation/3600,
                              personalization_after_shared_hours=personal/3600,total_remaining_hours=total/3600,
                              shared_finish=(now+datetime.timedelta(seconds=shared)).isoformat(),
                              all_training_finish=(now+datetime.timedelta(seconds=total)).isoformat(),remaining_worker_seconds=dict(shared=sw,ablation=aw)))
    result=dict(timestamp=now.isoformat(),total_planned=2571,total_completed=sum(s['completed'] for s in statuses.values()),
                total_active=sum(len(s['active']) for s in statuses.values()),runs=statuses,cards=cards,
                measured_main_epoch_seconds=base_rates,observed_stopping_epochs=epoch_summary,scenarios=scenarios,
                assumptions=['6002当前不可用，不计入预计容量；6001保持当前7张可用卡，4090余量可用于后续逐站微调。',
                             '并行计算 max(补充消融及独立训练, 共享剩余+逐站微调)，含依赖/准备/回传余量。',
                             '单站耗时按55站共享训练缩放并加准备开销，尚无本轮正式单站实测，必须在首批完成后更新。',
                             '并发槽数及早停轮数为情景假设，范围不是置信区间；其他作业占用变化会推迟结束。',
                             '快照为只读跨主机采样，存在秒级采集时差。',
                             '仅统计2571项重训及其2023验证，不含2024复评和条件性LM模型扩展。'])
    OUT.mkdir(exist_ok=True);write(OUT/'ETA_SNAPSHOT.json',dict(timestamp=now.isoformat(),hosts=snaps))
    write(P/'RETRAIN_ETA_20261115.json',result)
    print(json.dumps({k:result[k] for k in ['timestamp','total_completed','total_active','measured_main_epoch_seconds','scenarios']},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
