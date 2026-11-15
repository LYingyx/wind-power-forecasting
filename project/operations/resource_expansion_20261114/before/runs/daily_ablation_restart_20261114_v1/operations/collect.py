"""Complete matched tables; no historic metrics, partial groups never pass."""
from pathlib import Path
import sys,json,csv
RUN=Path(__file__).resolve().parents[1];sys.path.insert(0,str(RUN/'code'))
from settings import *
import numpy as np

def collect():
    specs=json.loads((RUN/'manifest.json').read_text());done={}
    for s in specs:
        out=JOBS/s['id']
        if not (out/'SUCCESS.json').exists():continue
        d=json.loads((out/'metrics.json').read_text());assert d['requested_spec']==s and not d['smoke_only'];done[s['id']]=d
    tune=[s for s in specs if s['stage']=='tune'];selected=ASSETS/'selected/W0_BASE.json'
    if not selected.exists() and all(s['id'] in done for s in tune):
        table=[]
        for ci in range(8):
            pair=[s for s in tune if f'_c{ci}_' in s['id']];assert len(pair)==2
            table.append(dict(index=ci,config=pair[0]['config'],mae=float(np.mean([done[s['id']]['metrics']['mae'] for s in pair])),jobs=[s['id'] for s in pair]))
        best=min(table,key=lambda x:(x['mae'],x['index']));atomic_json(selected,dict(timestamp=now(),group='W0_BASE',config=best['config'],selected_index=best['index'],table=table,selected_by='2023 macroMAE averaged seeds40,41'))
    nf=json.loads((ASSETS/'COHORT.json').read_text())['count'];cohort=json.loads((ASSETS/'COHORT.json').read_text());records=[]
    shared=PROJECT/'runs/daily_protocol_restart_20261114_v1'
    source_specs=json.loads((shared/'manifest.json').read_text())
    for s in source_specs:
        q=shared/'jobs'/s['id']
        if s['stage'] not in ['baseline','mechanism','regional_memory'] or not (q/'SUCCESS.json').exists():continue
        d=json.loads((q/'metrics.json').read_text());assert d['requested_spec']==s and s['preprocess_fingerprint']==specs[0]['preprocess_fingerprint'];records.append(d)
    records.extend(d for d in done.values() if d['spec']['stage']!='tune')
    groups={};rows=[]
    for d in records:
        s=d['spec'];m=d['metrics'];g=groups.setdefault(s['group'],{})
        sites=[s['station']] if 'station' in s else range(nf)
        for i,station in enumerate(sites):
            key=(s['seed'],station);assert key not in g
            row=dict(group=s['group'],seed=s['seed'],station=station,station_id=cohort['station_ids'][station],mae=m['per_farm_mae'][i],rmse=m['per_farm_rmse'][i],high_power_rmse=m['high_power_per_farm_rmse'][i],high_power_count=m['high_power_counts'][i]);g[key]=row;rows.append(row)
    complete={g:len(v)==nf*len(SEEDS) for g,v in groups.items()};summary=[];per_farm=[]
    for g,rs in groups.items():
        v=list(rs.values());summary.append(dict(group=g,cells=len(v),expected=nf*len(SEEDS),complete=complete[g],mae=float(np.mean([r['mae'] for r in v])),rmse=float(np.mean([r['rmse'] for r in v]))))
        for station in range(nf):
            vv=[x for x in v if x['station']==station]
            if vv:per_farm.append(dict(group=g,station_id=cohort['station_ids'][station],name=cohort['names'][station],seeds=len(vv),mae=float(np.mean([x['mae'] for x in vv])),rmse=float(np.mean([x['rmse'] for x in vv]))))
    comparisons=[]
    def compare(a,b):
        if not complete.get(a) or not complete.get(b):return
        keys=sorted(groups[a]);assert keys==sorted(groups[b]);res=dict(candidate=a,reference=b)
        for metric in ['mae','rmse']:
            x=np.array([groups[a][k][metric] for k in keys]).reshape(5,nf);y=np.array([groups[b][k][metric] for k in keys]).reshape(5,nf)
            res[metric]=dict(change_percent=float(100*(x.mean()/y.mean()-1)),farms_improved=int((x.mean(0)<y.mean(0)).sum()),per_seed_change_percent=(100*(x.mean(1)/y.mean(1)-1)).tolist())
        comparisons.append(res)
    for g in groups:
        if g.startswith(('INPUT_','FACTOR_')):compare(g,('_'.join(g.split('_')[1:2]))+'_R3')
        if g.startswith('ENC_'):compare(g,'W0_R3' if '_neutral_' in g else 'M2_CBH')
        if g.startswith('INDEPENDENT_'):
            base=g.removeprefix('INDEPENDENT_');compare(g,'SHARED_W0_BASE' if base=='W0_BASE' else base)
    for cond in ['neutral','cbh']:
        for a,b in [('E2N','E2L'),('E2N','E0'),('E2','E2N'),('E1','E0')]:compare(f'ENC_{cond}_{a}',f'ENC_{cond}_{b}')
    for inp in ['BASE','R3']:compare(f'INDEPENDENT_W0_{inp}',f'INDEPENDENT_TF_{inp}')
    compare('SHARED_W0_BASE','TF_BASE');compare('W0_R3','SHARED_W0_BASE')
    interaction=[]
    for f in ['W0','TF']:
        gs=[f+'_R3',f'FACTOR_{f}_H0W1',f'FACTOR_{f}_H1W0',f'FACTOR_{f}_H0W0']
        if all(complete.get(g) for g in gs):
            for metric in ['mae','rmse']:
                keys=sorted(groups[gs[0]]);a,b,c,d=[np.array([groups[g][k][metric] for k in keys]).reshape(5,nf) for g in gs]
                v=(c-a)-(d-b);interaction.append(dict(family=f,metric=metric,weather_gain_with_real_history=float((c-a).mean()),weather_gain_with_constant_history=float((d-b).mean()),interaction=float(v.mean()),positive_farms=int((v.mean(0)>0).sum()),per_seed=v.mean(1).tolist()))
    for name,values in [('all_seed_metrics.csv',rows),('summary.csv',summary),('per_farm_metrics.csv',per_farm)]:
        if values:
            with (RUN/'analysis'/name).open('w') as out:w=csv.DictWriter(out,fieldnames=list(values[0]));w.writeheader();w.writerows(values)
    result=dict(timestamp=now(),stage='COMPLETE_REVIEW_ABLATIONS' if len(done)==len(specs) else 'pending',completed=len(done),planned=len(specs),groups=summary,comparisons=comparisons,history_weather_interaction=interaction,test2024_read=False)
    atomic_json(RUN/'analysis/GATES.json',result)
    text='# 新协议消融与独立训练\n\n'+f'完成 {len(done)}/{len(specs)}。仅完整 55 站 × 5 seeds 的组进行比较；旧协议成绩未导入。\n\n|组别|完成单元|MAE|RMSE|\n|---|---:|---:|---:|\n'
    for s in summary:text+=f'|{s["group"]}|{s["cells"]}/{s["expected"]}|{s["mae"]:.6f}|{s["rmse"]:.6f}|\n'
    (RUN/'analysis/RESULTS_CN.md').write_text(text)
    return result
if __name__=='__main__':print(json.dumps(collect(),ensure_ascii=False))
