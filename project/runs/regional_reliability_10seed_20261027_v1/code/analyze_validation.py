"""Predeclared selection and replication decisions; retain raw farm/seed values."""
from settings import *
from validation_design import *
import numpy as np,pandas as pd

def main():
    manifest=json.loads((RUN/'manifest.json').read_text());sources=[(s,JOBS/s['id']) for s in manifest]
    extension=RUN.name!=SCREEN_RUN.name
    if extension:
        reuse=json.loads((RUN/'REUSED_SCREEN.json').read_text())
        for item in reuse:
            path=Path(item['path']);assert sha(path/'metrics.json')==item['metrics_sha256'] and sha(path/'diagnostics.json')==item['diagnostics_sha256']
            sources.append((item['spec'],path))
    expected_seeds=10 if extension else 2
    overall=[];diagnostic=[];leads=[]
    for spec,out in sources:
        assert (out/'SUCCESS.json').exists(),out
        result=json.loads((out/'metrics.json').read_text());sel=result['selections']['mae'];meta={k:spec[k] for k in ['configuration','variant','site','seed']}
        for horizon,score in [('all',sel['metrics']),*sel['metrics']['horizons'].items()]:
            overall.append(dict(**meta,horizon=horizon,epoch=sel['epoch'],**{k:score[k] for k in ['n','mae','rmse','bias']}))
        for row in json.loads((out/'diagnostics.json').read_text())['rows']:diagnostic.append(dict(**meta,**row))
        for i,(mae,rmse) in enumerate(zip(sel['metrics']['lead_mae'],sel['metrics']['lead_rmse'])):
            leads.append(dict(**meta,lead_hours=(i+1)/4,mae=mae,rmse=rmse))
    a=pd.DataFrame(overall);d=pd.DataFrame(diagnostic);l=pd.DataFrame(leads)
    for name,frame in [('overall_per_farm_seed',a),('scenarios_per_farm_seed',d),('lead_per_farm_seed',l)]:frame.to_csv(RUN/'analysis'/f'{name}.csv',index=False)
    with np.load(ASSETS/'matched_subsets.npz') as z:common=np.flatnonzero(z['MATCH_INNER'][0].sum(1)>0)
    effects=[];cache={}
    def effect(config,variant,ref,horizon='all',group=None,label=None,support=None):
        key=(config,variant,ref,horizon,group,label,support)
        if key in cache:return cache[key]
        frame=a if group is None else d[(d.group==group)&(d.bin==label)]
        frame=frame[(frame.configuration==config)&(frame.horizon==horizon)]
        if support=='common':frame=frame[frame.site.isin(common)]
        x=frame[frame.variant==variant];y=frame[frame.variant==ref]
        if x.empty or y.empty:return None
        pair=x.merge(y,on=['site','seed'],suffixes=('_a','_b'),validate='one_to_one')
        assert (pair.n_a==pair.n_b).all()
        counts=pair.groupby('site').seed.nunique();assert (counts==expected_seeds).all(),key
        f=pair.groupby('site').mean(numeric_only=True)
        values=dict(configuration=config,variant=variant,reference=ref,horizon=horizon,group=group or 'overall',bin=label or 'all',support=support or 'all',farms=len(f))
        rng=np.random.default_rng(20261027);idx=rng.integers(0,len(f),(10000,len(f)))
        for m in ['mae','rmse']:
            xa=f[m+'_a'].to_numpy();xb=f[m+'_b'].to_numpy();v=100*(xa.mean()/xb.mean()-1)
            boots=100*(xa[idx].mean(1)/xb[idx].mean(1)-1)
            values[m]=dict(effect_percent=float(v),wins=int((xa<xb).sum()),ci_low=float(np.quantile(boots,.025)),ci_high=float(np.quantile(boots,.975)),upper95=float(np.quantile(boots,.95)))
        values['bias_change']=float((f.bias_a-f.bias_b).mean())
        for m in ['mae','rmse']:effects.append({k:v for k,v in values.items() if k not in ['mae','rmse']}|dict(metric=m,**values[m]))
        cache[key]=values;return values
    gates={};selected=set();geometry={};removal={}
    def dual(e,threshold=-.1,majority=True):
        return e is not None and all(e[m]['effect_percent']<=threshold and (not majority or e[m]['wins']>e['farms']/2) for m in ['mae','rmse'])
    for config in ['W0',TF]:
        variants=set(a[a.configuration==config].variant)
        for v in sorted(variants-set(REFERENCES)):
            for ref in ['N0','N3','R3_RAW']:
                for horizon in ['all','0_1h','1_3h','3_6h']:
                    effect(config,v,ref,horizon)
                    effect(config,v,ref,horizon,'current_power','high')
            if 'C2_'+v in variants:effect(config,v,'C2_'+v)
        for v in GATES+['R3_MEAN_DELTA']:
            if v not in variants or 'C2_'+v not in variants:continue
            comparisons={ref:effect(config,v,ref) for ref in ['N0','N3','R3_RAW','C2_'+v]}
            base=all(dual(comparisons[ref]) for ref in ['N0','N3'])
            preserve=dual(comparisons['R3_RAW'],.1,False)
            short=all(dual(effect(config,v,'R3_RAW',h),.1,False) for h in ['0_1h','1_3h'])
            high={ref+'/'+h:effect(config,v,ref,h,'current_power','high') for ref in ['N0','N3'] for h in ['all','3_6h']}
            protected=all(e['rmse']['effect_percent']<=.1 for e in high.values())
            control=dual(comparisons['C2_'+v],0,False) and all(comparisons['C2_'+v][m]['effect_percent']<0 for m in ['mae','rmse'])
            tail=effect(config,v,'R3_RAW','3_6h','current_power','high')
            benefit=dual(comparisons['R3_RAW'],-.1,False) or dual(tail,-.1,False)
            passed=base and preserve and short and protected and control and benefit
            gates[config+'/'+v]=dict(passes=bool(passed),base_dual=base,overall_preserved=preserve,short_preserved=short,
                high_protected=protected,beats_control=control,benefit=benefit,comparisons=comparisons,high=high,tail_vs_RAW=tail)
            if passed:
                family=GATES if v in GATES else [v]
                selected.update((config,x) for q in family for x in [q,'C2_'+q])
        if config=='W0':
            for v in REMOVE:
                if v not in variants:continue
                overall=effect(config,v,'R3_RAW');hs=[effect(config,v,'R3_RAW',h,'current_power','high') for h in ['all','3_6h']]
                passed=dual(overall,.1,False) and all(h['rmse']['effect_percent']<=.1 for h in hs)
                removal[v]=dict(replicate=passed,overall=overall,high=hs,
                    overall_noninferiority=all(overall[m]['upper95']<=.1 for m in ['mae','rmse']))
                if passed:selected.update((config,x) for x in REMOVE)
            for v in MATCH:
                for ref in MATCH:
                    if v==ref:continue
                    e=effect(config,v,ref,support='common')
                    if e is not None:geometry[v+'/'+ref]=dict(replicate=dual(e),effect=e)
            for full,match in zip(FULL,MATCH):
                e=effect(config,full,match,support='common')
                if e is not None:geometry[full+'/'+match]=dict(replicate=dual(e),effect=e)
            if any(x['replicate'] for x in geometry.values()):selected.update((config,x) for x in MATCH+FULL)
    pd.DataFrame(effects).to_csv(RUN/'analysis/equal_farm_effects.csv',index=False)
    # All detailed scenarios retained, including outcome diagnostics; no selective omission.
    scenario_rows=[]
    for config in a.configuration.unique():
        for v in a[a.configuration==config].variant.unique():
            for ref in ['N0','N3','R3_RAW']:
                if v==ref:continue
                aa=d[(d.configuration==config)&(d.variant==v)];bb=d[(d.configuration==config)&(d.variant==ref)]
                merged=aa.merge(bb,on=['site','seed','group','bin','horizon'],suffixes=('_a','_b'),validate='one_to_one')
                for key,frame in merged.groupby(['group','bin','horizon']):
                    f=frame.groupby('site').mean(numeric_only=True)
                    if not len(f):continue
                    row=dict(configuration=config,variant=v,reference=ref,group=key[0],bin=key[1],horizon=key[2],farms=len(f))
                    for m in ['mae','rmse']:
                        row[m+'_percent']=float(100*(f[m+'_a'].mean()/f[m+'_b'].mean()-1));row[m+'_wins']=int((f[m+'_a']<f[m+'_b']).sum())
                    scenario_rows.append(row)
    pd.DataFrame(scenario_rows).to_csv(RUN/'analysis/scenario_effects.csv',index=False)
    split_rows=[]
    high=d[(d.group=='current_power')&(d.bin=='high')]
    for label,seeds in [('screen42_43',[42,43]),('replicate44_51',list(range(44,52))),('combined42_51',list(range(42,52)))]:
        if not set(seeds).issubset(set(a.seed)):continue
        for scope,frame in [('overall',a),('high',high)]:
            frame=frame[frame.seed.isin(seeds)]
            for config in frame.configuration.unique():
                f=frame[frame.configuration==config]
                for v in f.variant.unique():
                    for ref in ['N0','N3','R3_RAW']:
                        if v==ref:continue
                        pair=f[f.variant==v].merge(f[f.variant==ref],on=['site','seed','horizon'],suffixes=('_a','_b'),validate='one_to_one')
                        for horizon,rows in pair.groupby('horizon'):
                            farms=rows.groupby('site').mean(numeric_only=True)
                            for metric in ['mae','rmse']:
                                split_rows.append(dict(seed_group=label,scope=scope,configuration=config,variant=v,reference=ref,horizon=horizon,metric=metric,
                                    effect_percent=float(100*(farms[metric+'_a'].mean()/farms[metric+'_b'].mean()-1)),wins=int((farms[metric+'_a']<farms[metric+'_b']).sum()),farms=len(farms)))
    pd.DataFrame(split_rows).to_csv(RUN/'analysis/replication_split_effects.csv',index=False)
    recommendation=dict(timestamp=now(),complete=True,extension=extension,open_2024=False,gates=gates,removal=removal,
        geometry=geometry,selected_extension_arms=sorted(selected),common_support_sites=common.tolist(),
        interpretation='selected-year evidence; subgroup outcomes unavailable at origin; geometry/deletion extensions are mechanism checks, not final model selection')
    atomic_json(RUN/'analysis/SELECTION_RECOMMENDATION.json',recommendation)
    lines=['# 区域可靠性与距离验证',f'完成 {len(sources)} 个结果；2023；每站 {expected_seeds} 个种子。',
        '指标先站内平均种子，再风场等权；负值改善。完整逐站逐种子值、失败工况和详细分组见同目录 CSV。',
        '距离匹配比较只用所有圈均有候选的共同支持站；FULL/MATCH 均为等权、同一摘要槽位，N3固定。',
        '历史/天气真实状态不在错配中替换；C2仅将全部邻站观测共同换到2022同月同钟点档案，仍有年际分布混杂。','',
        '|骨干/候选|整体对N0 MAE/RMSE|高功率3–6h对N0 RMSE|全部门槛通过|','|---|---:|---:|---|']
    for name,g in gates.items():
        e=g['comparisons']['N0'];h=g['high']['N0/3_6h'];lines.append(f"|{name}|{e['mae']['effect_percent']:+.4f}% / {e['rmse']['effect_percent']:+.4f}%|{h['rmse']['effect_percent']:+.4f}%|{g['passes']}|")
    lines+=['','自动扩种子清单：'+json.dumps(sorted(selected),ensure_ascii=False),'','不以点估计非劣判定等价，不开启2024。']
    (RUN/'analysis/REPORT_CN.md').write_text('\n'.join(lines)+'\n')
    return recommendation

if __name__=='__main__':main()
