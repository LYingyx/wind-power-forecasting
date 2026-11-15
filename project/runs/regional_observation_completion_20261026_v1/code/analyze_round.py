"""Complete D contrasts, scenario effects and a prespecified simplification rule."""
from settings import *
from collect_physics import collect
from comparison_utils import effect_pairs
from regional_inputs import W0_VARIANTS, TF_VARIANTS
import pandas as pd, numpy as np
TF='TF_z1_lr0.0002_dp0.05'

def main():
    state=collect();actual=[];diagnostic=[];lead=[];interventions=[]
    for spec in json.loads((RUN/'manifest.json').read_text()):
        out=JOBS/spec['id']
        if not (out/'SUCCESS.json').exists():continue
        result=json.loads((out/'metrics.json').read_text())
        meta={k:spec[k] for k in ['id','configuration','variant','site','seed','backbone']}
        for criterion,item in result['selections'].items():
            for horizon,score in [('all',item['metrics']),*item['metrics']['horizons'].items()]:
                actual.append(dict(**meta,selection=criterion,horizon=horizon,epoch=item['epoch'],**{k:score[k] for k in ['n','mae','rmse','bias']}))
            if criterion=='mae':
                for i,(mae,rmse) in enumerate(zip(item['metrics']['lead_mae'],item['metrics']['lead_rmse'])):
                    lead.append(dict(**meta,lead_hours=(i+1)/4,n=item['metrics']['n']//24,mae=mae,rmse=rmse))
        for row in json.loads((out/'diagnostics.json').read_text())['rows']:diagnostic.append(dict(**meta,**row))
        normal=result['selections']['mae']['metrics']
        for kind,score in result.get('interventions',{}).items():
            for horizon,part in [('all',score),*score['horizons'].items()]:
                base=normal if horizon=='all' else normal['horizons'][horizon]
                interventions.append(dict(**meta,intervention=kind,horizon=horizon,n=part['n'],
                    mae=part['mae'],rmse=part['rmse'],mae_normal=base['mae'],rmse_normal=base['rmse']))
    folder=RUN/'analysis';frames={name:pd.DataFrame(rows) for name,rows in [
        ('all_actual_per_farm_seed',actual),('diagnostics_per_farm_seed',diagnostic),
        ('lead_per_farm_seed',lead),('interventions_per_farm_seed',interventions)]}
    for name,frame in frames.items():
        if not frame.empty:frame.to_csv(folder/(name+'.csv'),index=False)
    edges=[('W0',v,'W0',ref) for v in ['R3_RAW',*W0_VARIANTS] for ref in ['N0','N3','R3_RAW'] if v!=ref]
    edges += [('W0',v,'W0','C2_'+v) for v in ['R3_RAW','R3_ALLPOOL']]
    edges += [('W0','R3_RAW','W0','SELF_R3_RAW'),('W0','R3_ALLPOOL_MEAN','W0','R3_MEAN')]
    edges += [(TF,v,TF,ref) for v in TF_VARIANTS for ref in ['N0','N3','R3_RAW'] if v!=ref]
    edges += [(TF,v,TF,'C2_'+v) for v in ['R3_RAW','R3_ALLPOOL']]
    edges += [('W0',v,TF,v) for v in ['N0','N3','R3_RAW','R3_ALLPOOL']]
    edges=list(dict.fromkeys(edges));effects={}
    for name,keys in [('all_actual_per_farm_seed',['selection','horizon']),
                      ('diagnostics_per_farm_seed',['group','bin','horizon']),('lead_per_farm_seed',['lead_hours'])]:
        if frames[name].empty:continue
        pairs,eff=effect_pairs(frames[name],edges,keys);effects[name]=eff
        if not eff.empty:eff.to_csv(folder/(name+'_effects.csv'),index=False)
        if name=='all_actual_per_farm_seed' and not pairs.empty:pairs.to_csv(folder/'paired_per_farm_seed.csv',index=False)
    inter=[];df=frames['interventions_per_farm_seed']
    if not df.empty:
        for key,g in df.groupby(['configuration','variant','intervention','horizon']):
            sites=g.groupby('site').seed.nunique();g=g[g.site.isin(sites[sites==2].index)]
            if g.empty:continue
            f=g.groupby('site')[['mae','rmse','mae_normal','rmse_normal']].mean()
            for metric in ['mae','rmse']:
                inter.append(dict(zip(['configuration','variant','intervention','horizon'],key),metric=metric,
                    farms=len(f),effect_percent=100*(f[metric].mean()/f[metric+'_normal'].mean()-1),
                    worse_farms=int((f[metric]>f[metric+'_normal']).sum())))
        pd.DataFrame(inter).to_csv(folder/'intervention_equal_farm_effects.csv',index=False)
    complete=state['completed']==state['planned'];gates={};noninferiority={};selected=None
    if complete:
        overall=effects['all_actual_per_farm_seed'];high=effects['diagnostics_per_farm_seed']
        for config in ['W0',TF]:
            for variant in ['R3_RAW','R3_ALLPOOL']:
                q=overall[(overall.candidate_configuration==config)&(overall.reference_configuration==config)&
                    (overall.candidate_variant==variant)&(overall.selection=='mae')&(overall.horizon=='all')].set_index(['reference_variant','metric'])
                comparisons={ref:{metric:q.loc[(ref,metric)].to_dict() for metric in ['mae','rmse']} for ref in ['N0','N3','C2_'+variant]}
                assert all(comparisons[r][m]['farms']==56 for r in comparisons for m in ['mae','rmse'])
                h=high[(high.candidate_configuration==config)&(high.reference_configuration==config)&
                    (high.candidate_variant==variant)&(high.reference_variant=='N3')&(high.group=='current_power')&
                    (high.bin=='high')&(high.horizon=='all')&(high.metric=='rmse')].iloc[0]
                dual=all(comparisons[r][m]['effect_percent']<=-.1 and comparisons[r][m]['improved_farms']>=29 for r in ['N0','N3'] for m in ['mae','rmse'])
                controls=all(comparisons['C2_'+variant][m]['effect_percent']<0 for m in ['mae','rmse'])
                gates[config+'/'+variant]=dict(comparisons=comparisons,dual=dual,beats_control=controls,
                    high_power_rmse_percent=float(h.effect_percent),passes=bool(dual and controls and h.effect_percent<=.1))
        f=frames['all_actual_per_farm_seed'];f=f[(f.configuration=='W0')&(f.selection=='mae')&(f.horizon=='all')]
        a=f[f.variant=='R3_ALLPOOL'].groupby('site')[['mae','rmse']].mean().sort_index()
        b=f[f.variant=='R3_RAW'].groupby('site')[['mae','rmse']].mean().sort_index()
        assert a.index.equals(b.index) and len(a)==56
        indices=np.random.default_rng(20261026).integers(0,56,(10000,56))
        for metric in ['mae','rmse']:
            samples=100*(a[metric].to_numpy()[indices].mean(1)/b[metric].to_numpy()[indices].mean(1)-1)
            upper=float(np.quantile(samples,.95))
            noninferiority[metric]=dict(upper_one_sided_95_percent=upper,passes=upper<=.1)
        simple=gates['W0/R3_ALLPOOL']['passes'] and all(v['passes'] for v in noninferiority.values())
        selected='R3_ALLPOOL' if simple else 'R3_RAW' if gates['W0/R3_RAW']['passes'] else 'N3'
    recommendation=dict(timestamp=now(),complete=complete,primary='R3_RAW',gates=gates,
        simplification=noninferiority,recommended_variant=selected,open_2024=False,
        next='Review D controls and final E input-factor definitions; freeze E comparison manifest before evaluating 2024.',
        interpretation='2023 selection evidence, not causal proof; fixed-weight perturbations can be out of distribution')
    atomic_json(folder/'D_SELECTION_RECOMMENDATION.json',recommendation)
    lines=['# 区域实况补充 D 阶段',f"完成 {state['completed']}/{state['planned']}；2022 拟合、2023 选择，2024 未开。",'',
        '效应：两 seed 先在站内平均，再风场等权。主指标均取 MAE-selected 权重；负值为改善。', '',
        '分环拆解在保留 N3 时衡量额外环统计的条件增量；不等于隔离该距离带的全部信息。错配及缺失是敏感性分析，不能单独识别传播因果。','']
    if 'all_actual_per_farm_seed' in effects:
        e=effects['all_actual_per_farm_seed'];e=e[(e.selection=='mae')&(e.horizon=='all')]
        lines+=['|候选|参考|指标|变化|改善站数|完整风场|','|---|---|---|---:|---:|---:|']
        for r in e.itertuples():lines.append(f'|{r.candidate_configuration}/{r.candidate_variant}|{r.reference_configuration}/{r.reference_variant}|{r.metric}|{r.effect_percent:+.4f}%|{r.improved_farms}|{r.farms}|')
    lines+=['','预设判定：','```json',json.dumps(recommendation,ensure_ascii=False,indent=2),'```']
    (folder/'REPORT_CN.md').write_text('\n'.join(lines)+'\n')
    if complete:
        import matplotlib;matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        e=effects['lead_per_farm_seed'];fig,axes=plt.subplots(1,2,figsize=(10,3.6))
        for ax,metric in zip(axes,['mae','rmse']):
            for v in ['R3_RAW','R3_ALLPOOL','R3_INNER','R3_MIDDLE','R3_OUTER']:
                q=e[(e.candidate_configuration=='W0')&(e.reference_configuration=='W0')&
                    (e.candidate_variant==v)&(e.reference_variant=='N3')&(e.metric==metric)].sort_values('lead_hours')
                ax.plot(q.lead_hours,q.effect_percent,label=v)
            ax.axhline(0,color='black',lw=.7);ax.set(xlabel='Lead (h)',ylabel=metric.upper()+' change vs N3 (%)');ax.legend(fontsize=7)
        fig.tight_layout();fig.savefig(folder/'lead_effects.png',dpi=180);fig.savefig(folder/'lead_effects.pdf');plt.close(fig)
    return state

if __name__=='__main__':print(json.dumps(main()))
