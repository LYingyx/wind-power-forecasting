from pathlib import Path
import sys,json,csv,collections
RUN=Path(__file__).resolve().parents[1];sys.path.insert(0,str(RUN/'code'))
from settings import *
import numpy as np

def collect():
    manifest=json.loads((RUN/'manifest.json').read_text());cohort=json.loads((ASSETS/'COHORT.json').read_text());nf=cohort['count'];rows=[];done={};groups=collections.defaultdict(dict)
    for family in ['W0','TF']:
        for seed in SEEDS:
            d=json.loads((RUN/'pretrained'/f'{family}_r{seed}.metrics.json').read_text());m=d['metrics']
            for s in range(nf):
                row=dict(group=family+'_P0_SHARED',family=family,mode='P0_SHARED',seed=seed,station=s,station_id=cohort['station_ids'][s],name=cohort['names'][s],mae=m['per_farm_mae'][s],rmse=m['per_farm_rmse'][s],high_power_rmse=m['high_power_per_farm_rmse'][s],high_count=m['high_power_counts'][s],selected_epoch=0)
                rows.append(row);groups[row['group']][seed,s]=row
    for spec in manifest:
        out=JOBS/spec['id']
        if not (out/'SUCCESS.json').exists():continue
        d=json.loads((out/'metrics.json').read_text());assert d['requested_spec']==spec and not d['smoke_only'] and d['evaluation_year']==2023
        done[spec['id']]=d;m=d['metrics'];s=spec['station'];seed=spec['seed'];row=dict(group=spec['group'],family=spec['family'],mode=spec['mode'],seed=seed,station=s,station_id=spec['station_id'],name=cohort['names'][s],mae=m['mae'],rmse=m['rmse'],high_power_rmse=m['high_power_rmse'],high_count=m['high_power_counts'][0],selected_epoch=d['checkpoint_epoch'])
        rows.append(row);groups[row['group']][seed,s]=row
    summary=[]
    for group,rs in sorted(groups.items()):
        vals=list(rs.values());summary.append(dict(group=group,completed=len(vals),expected=nf*len(SEEDS),mae=float(np.mean([v['mae'] for v in vals])),rmse=float(np.mean([v['rmse'] for v in vals])),epoch0_selected=sum(v['selected_epoch']==0 for v in vals),complete=len(vals)==nf*len(SEEDS)))
    for name,data in [('per_station_seed.csv',rows),('summary.csv',summary)]:
        p=RUN/'analysis'/name;temp=p.with_suffix('.tmp')
        with temp.open('w') as f:w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
        os.replace(temp,p)
    def compare(a,b):
        if len(groups[a])!=nf*len(SEEDS) or len(groups[b])!=nf*len(SEEDS):return None
        out=dict(candidate=a,reference=b);passed=True
        for metric in ['mae','rmse']:
            x=np.array([[groups[a][seed,s][metric] for s in range(nf)] for seed in SEEDS]);y=np.array([[groups[b][seed,s][metric] for s in range(nf)] for seed in SEEDS]);xf=x.mean(0);yf=y.mean(0);delta=100*(x.mean()/y.mean()-1);wins=int((xf<yf).sum());ok=bool(delta<=-.1 and wins>nf/2);passed&=ok
            rng=np.random.default_rng(1104);ix=rng.integers(nf,size=(10000,nf));ci=np.quantile(100*(xf[ix].mean(1)/yf[ix].mean(1)-1),[.025,.975])
            out[metric]=dict(change_percent=float(delta),farms_improved=wins,total_farms=nf,passed=ok,per_seed_change_percent=(100*(x.mean(1)/y.mean(1)-1)).tolist(),farm_bootstrap95_percent=ci.tolist())
        x=np.mean([v['high_power_rmse'] for v in groups[a].values() if v['high_count']>0]);y=np.mean([v['high_power_rmse'] for v in groups[b].values() if v['high_count']>0]);delta=float(100*(x/y-1));out['high_power_rmse_change_percent']=delta;out['passed']=bool(passed and delta<=.1);return out
    pairs=[(f'{f}_{m}',f'{f}_P0_SHARED') for f in ['W0','TF'] for m in ['P1_FULL','P2_HEAD']]+[(f'W0_{m}',f'TF_{m}') for m in ['P0_SHARED','P1_FULL','P2_HEAD']]+[(f'{f}_P1_FULL',f'{f}_P2_HEAD') for f in ['W0','TF']]
    comparisons=[c for a,b in pairs if (c:=compare(a,b)) is not None];complete=len(done)==len(manifest)
    selected={}
    if complete:
        for family in ['W0','TF']:
            opts=[s for s in summary if s['group'].startswith(family+'_')];selected[family]=min(opts,key=lambda x:(x['mae'],x['group']))['group']
        comparisons.append(compare(selected['W0'],selected['TF']))
    gate=dict(timestamp=now(),stage='COMPLETE_REVIEW_PERSONALIZATION' if complete else 'pending',completed=len(done),planned=len(manifest),groups=summary,comparisons=comparisons,selected_by_2023_mae=selected,test2024_read=False,automatic_followup=False)
    atomic_json(RUN/'analysis/GATES.json',gate)
    text='# 共享预训练 + 逐站微调\n\n'+f'更新时间：{now()}。完成 {len(done)}/{len(manifest)} 次微调；55 站 × 5 seeds；2023 选择，未评价 2024。\n\n'
    text+='|组别|完成|MAE|RMSE|选第 0 轮|\n|---|---:|---:|---:|---:|\n'
    for s in summary:text+=f'|{s["group"]}|{s["completed"]}/275|{s["mae"]:.6f}|{s["rmse"]:.6f}|{s["epoch0_selected"]}|\n'
    text+='\n未满 275 项的组仅展示进度，不能与完整组比较。微调前第 0 轮也参加按 2023 MAE 的选模，选择集的改善不能当作独立确认。\n'
    for c in comparisons:text+=f'\n{c["candidate"]} 对 {c["reference"]}：MAE {c["mae"]["change_percent"]:+.4f}%（{c["mae"]["farms_improved"]}/55），RMSE {c["rmse"]["change_percent"]:+.4f}%（{c["rmse"]["farms_improved"]}/55），高功率 RMSE {c["high_power_rmse_change_percent"]:+.4f}%，验收 {c["passed"]}。\n'
    (RUN/'analysis/RESULTS_CN.md').write_text(text)
    return gate

if __name__=='__main__':print(json.dumps(collect(),ensure_ascii=False))
