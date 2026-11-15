"""Report selection seeds, new seeds and all ten separately; equal-farm paired effects."""
from settings import *
from extend_seeds import TF
import pandas as pd
import numpy as np

def main():
    activation=json.loads((RUN/'ACTIVATION.json').read_text())
    rows=[];diagnostics=[];leads=[]
    sources=json.loads((RUN/'REUSED_SOURCES.json').read_text())
    for s in json.loads((RUN/'manifest.json').read_text()):
        if (JOBS/s['id']/'SUCCESS.json').exists():sources.append(dict(spec=s,path=str(JOBS/s['id'])))
    for entry in sources:
        spec=entry['spec'];out=Path(entry['path'])
        if entry.get('metrics_sha256'):assert sha(out/'metrics.json')==entry['metrics_sha256']
        result=json.loads((out/'metrics.json').read_text());selected=result['selections']['mae']
        meta={k:spec[k] for k in ['configuration','variant','site','seed']}
        for horizon,score in [('all',selected['metrics']),*selected['metrics']['horizons'].items()]:
            rows.append(dict(**meta,horizon=horizon,**{k:score[k] for k in ['n','mae','rmse']}))
        for row in json.loads((out/'diagnostics.json').read_text())['rows']:
            diagnostics.append(dict(**meta,**row))
        for i,(mae,rmse) in enumerate(zip(selected['metrics']['lead_mae'],selected['metrics']['lead_rmse'])):
            leads.append(dict(**meta,lead_hours=(i+1)/4,n=selected['metrics']['n']//24,mae=mae,rmse=rmse))
    tables={'overall':pd.DataFrame(rows),'scenarios':pd.DataFrame(diagnostics),'leads':pd.DataFrame(leads)}
    effects=[]
    edges=[]
    for config in ['W0',TF]:
        arms=[v for c,v in activation['arms'] if c==config]
        edges.append((config,'N3',config,'N0'))
        for variant in ['R3_RAW','R3_ALLPOOL']:
            if variant in arms:
                edges.extend((config,variant,config,ref) for ref in ['N0','N3','C2_'+variant])
    common=set(v for c,v in activation['arms'] if c=='W0')&set(v for c,v in activation['arms'] if c==TF)
    edges.extend(('W0',v,TF,v) for v in common if not v.startswith('C2_'))
    keys={'overall':['horizon'],'scenarios':['group','bin','horizon'],'leads':['lead_hours']}
    for table,df in tables.items():
        df.to_csv(RUN/'analysis'/f'{table}_per_farm_seed.csv',index=False)
        for label,seeds in [('selection_2',[42,43]),('new_8',list(range(44,52))),('all_10',list(range(42,52)))]:
            f=df[df.seed.isin(seeds)]
            for cc,cv,rc,rv in edges:
                a=f[(f.configuration==cc)&(f.variant==cv)];b=f[(f.configuration==rc)&(f.variant==rv)]
                paired=a.merge(b,on=['site','seed',*keys[table]],suffixes=('_a','_b'),validate='one_to_one')
                if paired.empty:continue
                assert (paired.n_a==paired.n_b).all()
                for values,g in paired.groupby(keys[table],dropna=False):
                    if not isinstance(values,tuple):values=(values,)
                    complete=g.groupby('site').seed.agg(lambda s:set(s)==set(seeds))
                    g=g[g.site.isin(complete[complete].index)]
                    if g.empty:continue
                    for metric in ['mae','rmse']:
                        farm=g.groupby('site')[[metric+'_a',metric+'_b']].mean()
                        ca=farm[metric+'_a'];ref=farm[metric+'_b']
                        effects.append(dict(table=table,cohort=label,candidate_configuration=cc,candidate_variant=cv,
                            reference_configuration=rc,reference_variant=rv,**dict(zip(keys[table],values)),
                            metric=metric,farms=len(farm),pairs=len(g),candidate=float(ca.mean()),reference=float(ref.mean()),
                            effect_percent=float(100*(ca.mean()/ref.mean()-1)),improved_farms=int((ca<ref).sum())))
    effect=pd.DataFrame(effects);effect.to_csv(RUN/'analysis/paired_effects.csv',index=False)
    lines=['# 条件 10 种子扩展','',
        '2022 拟合、2023 复验。原两种子用于筛选；新增八种子衡量优化随机性的稳定性，仍不是独立年份复评。',
        '站内先平均种子，再对 56 站等权平均；两个指标都使用 MAE 选出的权重。', '',
        '|种子组|候选|对照|指标|变化|改善风场|完整风场|','|---|---|---|---|---:|---:|---:|']
    if not effect.empty:
        for r in effect[(effect.table=='overall')&(effect.horizon=='all')].itertuples():
            lines.append(f'|{r.cohort}|{r.candidate_configuration}/{r.candidate_variant}|{r.reference_configuration}/{r.reference_variant}|{r.metric}|{r.effect_percent:+.4f}%|{r.improved_farms}|{r.farms}|')
    (RUN/'analysis/REPORT_CN.md').write_text('\n'.join(lines)+'\n')
    return len(sources)

if __name__=='__main__':print(main())
