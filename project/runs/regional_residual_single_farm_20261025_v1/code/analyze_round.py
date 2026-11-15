"""Paired, equal-farm effects and predeclared dual-metric gates; no 2024 launch."""
from settings import *
from collect_physics import collect
import pandas as pd,numpy as np
TF='TF_z1_lr0.0002_dp0.05'

def effect_pairs(frame,edges,keys):
    pairs=[];effects=[]
    for cc,cv,rc,rv in edges:
        a=frame[(frame.configuration==cc)&(frame.variant==cv)];b=frame[(frame.configuration==rc)&(frame.variant==rv)]
        if a.empty or b.empty:continue
        pair=a.merge(b,on=['site','seed',*keys],suffixes=('_candidate','_reference'),validate='one_to_one')
        if pair.empty:continue
        assert (pair.n_candidate==pair.n_reference).all()
        for metric in ['mae','rmse']:pair[metric+'_difference']=pair[metric+'_candidate']-pair[metric+'_reference']
        pairs.append(pair)
        for values,rows in pair.groupby(keys,dropna=False):
            if not isinstance(values,tuple):values=(values,)
            complete=rows.groupby('site').seed.nunique();rows=rows[rows.site.isin(complete[complete==(1 if cc=='ARX' else 2)].index)]
            if rows.empty:continue
            for metric in ['mae','rmse']:
                farm=rows.groupby('site')[[metric+'_candidate',metric+'_reference']].mean();ca=farm[metric+'_candidate'];ref=farm[metric+'_reference']
                effects.append(dict(candidate_configuration=cc,candidate_variant=cv,reference_configuration=rc,reference_variant=rv,**dict(zip(keys,values)),metric=metric,farms=len(farm),pairs=len(rows),candidate=float(ca.mean()),reference=float(ref.mean()),effect_percent=float(100*(ca.mean()/ref.mean()-1)),improved_farms=int((ca<ref).sum()),improved_seed42=int((rows[rows.seed==42][metric+'_difference']<0).sum()),improved_seed43=int((rows[rows.seed==43][metric+'_difference']<0).sum())))
    return pd.concat(pairs,ignore_index=True) if pairs else pd.DataFrame(),pd.DataFrame(effects)

def main():
    state=collect();actual=[];diag=[];inter=[];lead=[];arx=[]
    for spec in json.loads((RUN/'manifest.json').read_text()):
        out=JOBS/spec['id']
        if not (out/'SUCCESS.json').exists():continue
        result=json.loads((out/'metrics.json').read_text());meta={k:spec[k] for k in ['id','configuration','variant','site','seed','backbone']}
        for criterion,item in result['selections'].items():
            dest=arx if spec['task']=='arx' else actual
            for horizon,score in [('all',item['metrics']),*item['metrics']['horizons'].items()]:
                dest.append(dict(**dict(meta,variant=criterion) if spec['task']=='arx' else meta,selection='mae' if spec['task']=='arx' else criterion,horizon=horizon,epoch=item['epoch'],**{k:score[k] for k in ['n','mae','rmse','bias']}))
            if spec['task']!='arx' and criterion=='mae':
                for k,(mae,rmse) in enumerate(zip(item['metrics']['lead_mae'],item['metrics']['lead_rmse'])):lead.append(dict(**meta,lead_hours=(k+1)/4,n=item['metrics']['n']//24,mae=mae,rmse=rmse))
        if spec['task']!='arx':
            for row in json.loads((out/'diagnostics.json').read_text())['rows']:diag.append(dict(**meta,**row))
        for kind,scores in result.get('interventions',{}).items():
            normal=result['selections']['mae']['metrics']
            for horizon,score in [('all',scores),*scores['horizons'].items()]:
                base=normal if horizon=='all' else normal['horizons'][horizon]
                inter.append(dict(**meta,intervention=kind,horizon=horizon,mae=score['mae'],rmse=score['rmse'],mae_normal=base['mae'],rmse_normal=base['rmse']))
    folder=RUN/'analysis'
    frames={name:pd.DataFrame(rows) for name,rows in [('all_actual_per_farm_seed',actual),('diagnostics_per_farm_seed',diag),('interventions_per_farm_seed',inter),('lead_per_farm_seed',lead),('arx_per_farm',arx)]}
    for name,frame in frames.items():
        if not frame.empty:frame.to_csv(folder/(name+'.csv'),index=False)
    edges=[('W0',v,'W0',ref) for v in ['R1','R2','R3','R4'] for ref in ['N0','N1','N3','C1_'+v,'C2_'+v]]
    edges += [('W0','R1','W0','R1_RAWMAP'),('W0','R1','W0','R1_SELF'),('W0','R2','W0','R2_PAIR'),('W0','R3','W0','R3_RAW'),('W0','R3','W0','R1'),('W0','R4','W0','R3')]
    edges += [(TF,v,TF,ref) for v in ['R1','R2','R3','R4'] for ref in ['N0','N1','N3']]
    edges += [('W0',v,TF,v) for v in ['N0','N1','N3','R1','R2','R3','R4']]
    effects={}
    for name,keys in [('all_actual_per_farm_seed',['selection','horizon']),('diagnostics_per_farm_seed',['group','bin','horizon']),('lead_per_farm_seed',['lead_hours']),('arx_per_farm',['selection','horizon'])]:
        frame=frames[name]
        if frame.empty:continue
        ee=edges if name!='arx_per_farm' else [('ARX',v,'ARX',r) for v in ['REG_RAW','REG_RES','REG_PAIR'] for r in ['LOCAL']]+[('ARX','REG_RES','ARX','REG_RAW'),('ARX','REG_PAIR','ARX','REG_RES')]
        paired,ef=effect_pairs(frame,ee,keys);effects[name]=ef
        if not ef.empty:ef.to_csv(folder/(name+'_effects.csv'),index=False)
        if name=='all_actual_per_farm_seed' and not paired.empty:paired.to_csv(folder/'paired_per_farm_seed.csv',index=False)
    # Deterministic ARX is compared once per farm with the two-seed neural mean.
    arx_neural=[]
    if not frames['arx_per_farm'].empty and not frames['all_actual_per_farm_seed'].empty:
        nf=frames['all_actual_per_farm_seed'];nf=nf[(nf.selection=='mae')&(nf.horizon=='all')]
        af=frames['arx_per_farm'];af=af[af.horizon=='all']
        for (config,variant),rows in nf.groupby(['configuration','variant']):
            counts=rows.groupby('site').seed.nunique();rows=rows[rows.site.isin(counts[counts==2].index)]
            farm=rows.groupby('site')[['mae','rmse']].mean().reset_index()
            for name,rr in af.groupby('variant'):
                joined=farm.merge(rr[['site','mae','rmse']],on='site',suffixes=('_neural','_arx'),validate='one_to_one')
                if joined.empty:continue
                for metric in ['mae','rmse']:
                    ca=joined[metric+'_neural'];ref=joined[metric+'_arx']
                    arx_neural.append(dict(configuration=config,variant=variant,arx=name,metric=metric,farms=len(joined),neural=float(ca.mean()),arx_score=float(ref.mean()),neural_change_percent=float(100*(ca.mean()/ref.mean()-1)),neural_improved_farms=int((ca<ref).sum())))
        pd.DataFrame(arx_neural).to_csv(folder/'neural_vs_arx_equal_farm.csv',index=False)
    complete=state['completed']==state['planned'];gates={}
    if complete:
        overall=effects['all_actual_per_farm_seed'];overall=overall[(overall.candidate_configuration=='W0')&(overall.reference_configuration=='W0')&(overall.selection=='mae')&(overall.horizon=='all')]
        high=effects['diagnostics_per_farm_seed'];high=high[(high.candidate_configuration=='W0')&(high.reference_configuration=='W0')&(high.group=='current_power')&(high.bin=='high')&(high.horizon=='all')&(high.metric=='rmse')&(high.reference_variant=='N3')]
        for variant in ['R1','R2','R3','R4']:
            q=overall[overall.candidate_variant==variant].set_index(['reference_variant','metric']);h=high[high.candidate_variant==variant].iloc[0]
            assert all(q.loc[(ref,m),'farms']==56 for ref in ['N1','N3','C1_'+variant,'C2_'+variant] for m in ['mae','rmse']) and h.farms==56
            comparisons={ref:{m:dict(effect_percent=float(q.loc[(ref,m),'effect_percent']),improved_farms=int(q.loc[(ref,m),'improved_farms'])) for m in ['mae','rmse']} for ref in ['N1','N3','C1_'+variant,'C2_'+variant]}
            dual=all(comparisons[r][m]['effect_percent']<=-.1 and comparisons[r][m]['improved_farms']>=29 for r in ['N1','N3'] for m in ['mae','rmse'])
            controls=all(comparisons[r][m]['effect_percent']<0 for r in ['C1_'+variant,'C2_'+variant] for m in ['mae','rmse'])
            gates[variant]=dict(comparisons=comparisons,dual_metric_gate=dual,beats_matched_controls=controls,high_power_rmse_vs_N3_percent=float(h.effect_percent),high_power_gate=bool(h.effect_percent<=.1),passes=bool(dual and controls and h.effect_percent<=.1))
    atomic_json(folder/'RESIDUAL_SELECTION_RECOMMENDATION.json',dict(timestamp=now(),complete=complete,primary='R1',gates=gates,secondary=['R2','R3','R4'],open_2024=False,next='Review controls and seed-level results; separately freeze any 2024 protocol. If all R fail, retain N3 as regional observations baseline.'))
    lines=['# 区域残差筛选',f"完成 {state['completed']}/{state['planned']}；2022 拟合、2023 选择，未开 2024。",'','效应量：站内两 seed 平均，再对风场等权，报告候选均值/参考均值−1。所有原值保留；小于 0.1% 的变化也报告。分组只使用两 seed 齐全且可配对的风场。', '','残差包含功率映射和运行因素，相关性与固定权重干预不能单独证明气象误差传播；筛选年份上的优势仍有选择偏差。','']
    if 'all_actual_per_farm_seed' in effects and not effects['all_actual_per_farm_seed'].empty:
        e=effects['all_actual_per_farm_seed'];q=e[(e.selection=='mae')&(e.horizon=='all')]
        lines+=['|候选|参考|指标|变化|改善风场|完整风场|','|---|---|---|---:|---:|---:|']
        for x in q.itertuples():lines.append(f'|{x.candidate_configuration}/{x.candidate_variant}|{x.reference_configuration}/{x.reference_variant}|{x.metric}|{x.effect_percent:+.4f}%|{x.improved_farms}|{x.farms}|')
    lines+=['','门槛判断：',json.dumps(gates,ensure_ascii=False,indent=2),'','ARX 不复制随机 seed；LOCAL/REG_RAW/REG_RES/REG_PAIR 使用相同标准化、稀疏算法与选择网格。']
    (folder/'REPORT_CN.md').write_text('\n'.join(lines)+'\n')
    if complete:
        import matplotlib;matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        e=effects['lead_per_farm_seed'];fig,ax=plt.subplots(1,2,figsize=(10,3.5))
        for axis,metric in zip(ax,['mae','rmse']):
            for variant in ['R1','R2','R3','R4']:
                q=e[(e.candidate_configuration=='W0')&(e.reference_configuration=='W0')&(e.reference_variant=='N3')&(e.candidate_variant==variant)&(e.metric==metric)].sort_values('lead_hours')
                axis.plot(q.lead_hours,q.effect_percent,label=variant)
            axis.axhline(0,color='black',lw=.7);axis.set(xlabel='Forecast lead (h)',ylabel=metric.upper()+' change vs N3 (%)');axis.legend()
        fig.tight_layout();fig.savefig(folder/'lead_effects.png',dpi=180);fig.savefig(folder/'lead_effects.pdf');plt.close(fig)
    return state
if __name__=='__main__':print(json.dumps(main()))
