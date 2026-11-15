"""Full seed-level neighbor results and the user-specified promotion gate."""
from settings import *
from collect_physics import collect
import pandas as pd,numpy as np,itertools
TF='TF_z1_lr0.0002_dp0.05'

def main():
    status=collect();manifest=json.loads((RUN/'manifest.json').read_text());actual=[];diag=[];inter=[]
    for spec in manifest:
        out=JOBS/spec['id']
        if not (out/'SUCCESS.json').exists():continue
        result=json.loads((out/'metrics.json').read_text())
        score_result=json.loads((ATTRIBUTION_RUN/'jobs'/spec['source_job']/'metrics.json').read_text()) if spec['task']=='diagnose_reference' else result
        meta={k:spec[k] for k in ['id','configuration','variant','site','seed','backbone']}
        for criterion,s in score_result['selections'].items():
            for horizon,score in [('all',s['metrics']),*s['metrics']['horizons'].items()]:actual.append(dict(**meta,selection=criterion,horizon=horizon,epoch=s['epoch'],**{k:score[k] for k in ['n','mae','rmse','bias']}))
        for row in json.loads((out/'diagnostics.json').read_text())['rows']:diag.append(dict(**meta,**row))
        for kind,scores in result.get('interventions',{}).items():
            normal=result['selections']['mae']['metrics']
            for horizon,score in [('all',scores),*scores['horizons'].items()]:
                base=normal if horizon=='all' else normal['horizons'][horizon]
                inter.append(dict(**meta,intervention=kind,horizon=horizon,n=score['n'],mae=score['mae'],rmse=score['rmse'],bias=score['bias'],mae_normal=base['mae'],rmse_normal=base['rmse'],mae_difference=score['mae']-base['mae'],rmse_difference=score['rmse']-base['rmse']))
    folder=RUN/'analysis'
    if not actual:return status
    actual=pd.DataFrame(actual);assert not actual.duplicated(['configuration','variant','site','seed','selection','horizon']).any()
    actual.to_csv(folder/'all_actual_per_farm_seed.csv',index=False)
    pd.DataFrame(diag).to_csv(folder/'diagnostics_per_farm_seed.csv',index=False)
    if inter:pd.DataFrame(inter).to_csv(folder/'interventions_per_farm_seed.csv',index=False)
    edges=[('W0','N1','W0',v) for v in ['N0','N2','N2_SELF','N3']]
    edges += [('W0',v,'W0','N1') for v in ['N4_K4','N4_K16','N5_CORR','N7_D0','N7_D30']]
    edges += [(TF,'N1',TF,'N0'),('W0','N1',TF,'N1'),('W0','N0',TF,'N0')]
    pairs=[];summary=[]
    for cc,cv,rc,rv in edges:
        a=actual[(actual.configuration==cc)&(actual.variant==cv)];b=actual[(actual.configuration==rc)&(actual.variant==rv)]
        p=a.merge(b,on=['site','seed','selection','horizon'],suffixes=('_candidate','_reference'),validate='one_to_one')
        if p.empty:continue
        assert (p.n_candidate==p.n_reference).all()
        for m in ['mae','rmse']:p[m+'_difference']=p[m+'_candidate']-p[m+'_reference']
        pairs.append(p)
        for (selection,horizon),rows in p.groupby(['selection','horizon']):
            complete=rows.groupby('site').seed.nunique();rows=rows[rows.site.isin(complete[complete==2].index)]
            if rows.empty:continue
            for m in ['mae','rmse']:
                farm=rows.groupby('site')[[m+'_candidate',m+'_reference']].mean();ca=farm[m+'_candidate'].to_numpy();ref=farm[m+'_reference'].to_numpy()
                rng=np.random.default_rng(20261025);ix=rng.integers(0,len(ca),(2000,len(ca)));ci=100*(ca[ix].mean(1)/ref[ix].mean(1)-1)
                summary.append(dict(candidate_configuration=cc,candidate_variant=cv,reference_configuration=rc,reference_variant=rv,selection=selection,horizon=horizon,metric=m,farms=len(farm),pairs=len(rows),candidate=float(ca.mean()),reference=float(ref.mean()),effect_percent=float(100*(ca.mean()/ref.mean()-1)),improved_farms=int((ca<ref).sum()),improved_seed42=int((rows[rows.seed==42][m+'_difference']<0).sum()),improved_seed43=int((rows[rows.seed==43][m+'_difference']<0).sum()),ci_low=float(np.quantile(ci,.025)),ci_high=float(np.quantile(ci,.975))))
    if pairs:pd.concat(pairs,ignore_index=True).to_csv(folder/'paired_per_farm_seed.csv',index=False)
    effects=pd.DataFrame(summary)
    if not effects.empty:effects.to_csv(folder/'supplementary_effect_sizes.csv',index=False)
    done=status['completed']==status['planned'];gate={}
    if done:
        q=effects[(effects.candidate_configuration=='W0')&(effects.candidate_variant=='N1')&(effects.reference_configuration=='W0')&(effects.selection=='mae')&(effects.horizon=='all')&(effects.metric=='mae')].set_index('reference_variant')
        assert all(q.loc[v,'pairs']==112 for v in ['N0','N2','N3','N2_SELF'])
        gate=dict(relative_change_vs_N0_percent=float(q.loc['N0','effect_percent']),passes_0p1_percent=bool(q.loc['N0','effect_percent']<=-.1),beats_N2=bool(q.loc['N2','effect_percent']<0),beats_N3=bool(q.loc['N3','effect_percent']<0),beats_self_capacity_control=bool(q.loc['N2_SELF','effect_percent']<0))
        gate['passes_user_promotion_rule']=gate['passes_0p1_percent'] and gate['beats_N2'] and gate['beats_N3']
    atomic_json(folder/'NEIGHBOR_SELECTION_RECOMMENDATION.json',dict(timestamp=now(),complete=done,test_year_accessed=False,primary_variant='N1',gate=gate,next='Review seed-level effects, capacity/aggregate/surrogate controls and diagnostics; freeze revised joint 2024 protocol separately. This file never starts 2024.'))
    lines=['# SEL邻站筛选结果','',f"更新：{now()}；完成{status['completed']}/{status['planned']}。2024未访问。",'',
        '逐风场逐seed原值见 all_actual_per_farm_seed.csv，配对差见 paired_per_farm_seed.csv，场景见 diagnostics_per_farm_seed.csv，固定权重干预见 interventions_per_farm_seed.csv。结构和对照解释见 README_CN.md。','',
        '补充百分比为站内seed平均后再风场等权，负值改善；仅汇总两seed齐全的风场。bootstrap区间未校正空间相关或多重选择。0.1%仅用于用户指定的升阶门槛，更小改善仍保留。置零、身份错配、倒序及固定权重延迟是敏感性试验，不能单独证明价值。','']
    if not effects.empty:
        q=effects[(effects.selection=='mae')&(effects.horizon=='all')]
        lines+=['|候选|参考|指标|变化|改善站seed42/43|完整风场|','|---|---|---|---:|---|---:|']
        for row in q.itertuples():lines.append(f'|{row.candidate_configuration}/{row.candidate_variant}|{row.reference_configuration}/{row.reference_variant}|{row.metric}|{row.effect_percent:+.4f}%|{row.improved_seed42}/{row.improved_seed43}|{row.farms}|')
    lines+=['','升阶判断：'+json.dumps(gate,ensure_ascii=False),'','没有50km内近邻的风场仅1个，相关分组只作个案；N2验证用2022档案，年际分布差异仍是其局限。']
    (folder/'REPORT_CN.md').write_text('\n'.join(lines)+'\n')
    return dict(**status,actual_rows=len(actual),diagnostic_rows=len(diag),intervention_rows=len(inter))

if __name__=='__main__':print(json.dumps(main()))
