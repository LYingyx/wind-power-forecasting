"""Post-completion review; preserves frozen training code and all actual scores."""
from pathlib import Path
import json,hashlib,datetime
import pandas as pd
import numpy as np

R=Path(__file__).resolve().parents[1];A=R/'analysis'
DEFAULT='TF_z0_lr0.001_dp0';TFSTAR='TF_z1_lr0.0002_dp0.05'

def md(df):
    cols=list(df.columns);lines=['|'+'|'.join(cols)+'|','|'+'|'.join(['---']*len(cols))+'|']
    for row in df.itertuples(index=False,name=None):
        lines.append('|'+'|'.join(str(x) for x in row)+'|')
    return '\n'.join(lines)

def main():
    audit=json.loads((R/'COMPLETION_AUDIT.json').read_text());assert audit['passed'] and audit['completed']==6944
    df=pd.read_csv(A/'all_actual_per_farm_seed.csv');assert len(df)==55552
    assert not df.duplicated(['configuration','variant','site','seed','selection','horizon']).any()
    d=df[(df.selection=='mae')&(df.horizon=='all')].copy()
    assert d.groupby(['configuration','variant']).size().eq(112).all()
    diagnostics=pd.read_csv(A/'diagnostics_per_farm_seed.csv')
    assert diagnostics.groupby(['configuration','variant','site','seed']).ngroups==6944
    paired=[];summaries=[]
    def compare(label,left,right):
        x=d[(d.configuration==left[0])&(d.variant==left[1])]
        y=d[(d.configuration==right[0])&(d.variant==right[1])]
        p=x.merge(y,on=['site','seed'],suffixes=('_candidate','_reference'),validate='one_to_one')
        assert len(p)==112 and (p.n_candidate==p.n_reference).all()
        p['comparison']=label
        for m in ['mae','rmse']:
            p[m+'_difference']=p[m+'_candidate']-p[m+'_reference']
            farms=p.groupby('site')[[m+'_candidate',m+'_reference']].mean();ca=farms[m+'_candidate'];ref=farms[m+'_reference']
            summaries.append(dict(comparison=label,metric=m,candidate_configuration=left[0],candidate_variant=left[1],reference_configuration=right[0],reference_variant=right[1],candidate_equal_farm=float(ca.mean()),reference_equal_farm=float(ref.mean()),effect_percent=float((ca.mean()/ref.mean()-1)*100),improved_farms=int((ca<ref).sum()),improved_seed42=int((p.loc[p.seed==42,m+'_difference']<0).sum()),improved_seed43=int((p.loc[p.seed==43,m+'_difference']<0).sum())))
        paired.append(p[['comparison','site','seed','configuration_candidate','variant_candidate','configuration_reference','variant_reference','mae_candidate','mae_reference','mae_difference','rmse_candidate','rmse_reference','rmse_difference']])
    for v in ['SPD','SPD3','SPD_LV','SPD3_LV','RHO','DENS_NORHO','DENS_UNCOUPLED','DENS','SEL','ALL']:compare('W0_'+v+'_vs_BASE',('W0',v),('W0','BASE'))
    for v in ['DENS_NORHO','DENS_UNCOUPLED']:compare('W0_DENS_vs_'+v,('W0','DENS'),('W0',v))
    for v in ['SEL_NO_DENS','SEL_NO_DYN','SEL_NO_STAB','SEL_NO_STATE','SEL_REF','SEL_REF_DELTA','SEL_DENS_NORHO','SEL_DENS_UNCOUPLED','DENS','ALL']:compare('W0_SEL_vs_'+v,('W0','SEL'),('W0',v))
    for v in ['BASE','DENS']:
        compare('W0_vs_TFdefault_'+v,('W0',v),(DEFAULT,v))
        compare('W0_vs_TFstar_'+v,('W0',v),(TFSTAR,v))
        compare('TFstar_vs_TFdefault_'+v,(TFSTAR,v),(DEFAULT,v))
    compare('TFstar_DENS_vs_BASE',(TFSTAR,'DENS'),(TFSTAR,'BASE'))
    compare('TFdefault_SEL_vs_BASE',(DEFAULT,'SEL'),(DEFAULT,'BASE'))
    compare('W0_vs_TFdefault_SEL',('W0','SEL'),(DEFAULT,'SEL'))
    summary=pd.DataFrame(summaries);summary.to_csv(A/'key_comparison_effects.csv',index=False,float_format='%.12g')
    pd.concat(paired,ignore_index=True).to_csv(A/'key_comparisons_per_farm_seed.csv',index=False,float_format='%.12g')
    # Readable exact values for all 56 farms and both seeds; no means in this file.
    selected=[('W0','BASE'),('W0','SPD'),('W0','DENS_NORHO'),('W0','DENS'),('W0','SEL'),(DEFAULT,'SEL'),(TFSTAR,'BASE'),(TFSTAR,'DENS')]
    text=['# 关键模型逐风场逐seed实测值','', '全部为2023验证集，MAE训练、MAE选轮、全时效；每一行是一站、一个seed的真实结果，不跨站或跨seed平均。其余模型、RMSE选轮及时效结果见 all_actual_per_farm_seed.csv。','']
    for config,variant in selected:
        sub=d[(d.configuration==config)&(d.variant==variant)].sort_values(['site','seed'])
        view=sub[['site','seed','epoch','mae','rmse','bias']].copy()
        for k in ['mae','rmse','bias']:view[k]=view[k].map(lambda v:f'{v:.10f}')
        text += [f'## {config} / {variant}','',md(view),'']
    (A/'KEY_MODELS_PER_FARM_SEED_CN.md').write_text('\n'.join(text)+'\n')
    # Matching scenario masks must have identical counts for every site and seed.
    strata=[]
    for label,left,right in [('W0_DENS_vs_BASE',('W0','DENS'),('W0','BASE')),('W0_SPD_vs_BASE',('W0','SPD'),('W0','BASE')),('W0_DENS_vs_DENS_NORHO',('W0','DENS'),('W0','DENS_NORHO')),('W0_SEL_vs_BASE',('W0','SEL'),('W0','BASE')),('W0_SEL_vs_DENS',('W0','SEL'),('W0','DENS')),('W0_SEL_vs_SEL_DENS_NORHO',('W0','SEL'),('W0','SEL_DENS_NORHO')),('TFstar_DENS_vs_BASE',(TFSTAR,'DENS'),(TFSTAR,'BASE'))]:
        x=diagnostics[(diagnostics.configuration==left[0])&(diagnostics.variant==left[1])]
        y=diagnostics[(diagnostics.configuration==right[0])&(diagnostics.variant==right[1])]
        p=x.merge(y,on=['site','seed','group','bin','horizon'],suffixes=('_candidate','_reference'),validate='one_to_one')
        assert len(p)==len(x)==len(y) and (p.n_candidate==p.n_reference).all()
        for (group,bin_,horizon),rows in p.groupby(['group','bin','horizon']):
            for metric in ['mae','rmse']:
                per_farm=rows.groupby('site')[[metric+'_candidate',metric+'_reference']].mean()
                ca=per_farm[metric+'_candidate'];ref=per_farm[metric+'_reference']
                strata.append(dict(comparison=label,group=group,bin=bin_,horizon=horizon,metric=metric,farms=len(per_farm),pairs=len(rows),n_per_seed=int(rows[rows.seed==42].n_candidate.sum()),candidate_equal_farm=ca.mean(),reference_equal_farm=ref.mean(),effect_percent=100*(ca.mean()/ref.mean()-1),improved_farms=int((ca<ref).sum()),improved_seed_pairs=int((rows[metric+'_candidate']<rows[metric+'_reference']).sum())))
    strata=pd.DataFrame(strata);strata.to_csv(A/'scenario_effect_sizes.csv',index=False,float_format='%.12g')
    protocol=json.loads((R/'PROTOCOL.json').read_text());reco=json.loads((A/'SELECTION_RECOMMENDATION.json').read_text())
    assert reco['recommended_main']['variant']=='SEL' and reco['recommended_tf_configuration']['configuration']==TFSTAR
    catalog=[]
    for (config,variant),rows in d.groupby(['configuration','variant']):
        catalog.append(dict(configuration=config,variant=variant,backbone='W0/NEUTRAL_LAST' if config=='W0' else 'Transformer_FC',extra_groups=' + '.join(protocol['feature_groups'][variant]) or 'none',trainable_parameters=int(rows.trainable_parameters.iloc[0]),input_farms='target_only',training='2022',selection='2023',seeds='42,43',actual_rows=112))
        assert rows.trainable_parameters.nunique()==1
    pd.DataFrame(catalog).to_csv(A/'MODEL_CATALOG.csv',index=False)
    print('KEY EFFECTS')
    print(summary.to_string(index=False))
    print('SCENARIO MAE')
    print(strata[(strata.horizon=='all')&(strata.metric=='mae')&(strata.group.isin(['nwp_speed','current_power','forecast_wind_change_1h']))&(strata.comparison.isin(['W0_DENS_vs_BASE','W0_SEL_vs_BASE']))][['comparison','group','bin','effect_percent','improved_farms']].to_string(index=False))
    checks=dict(timestamp=datetime.datetime.now().astimezone().isoformat(),actual_rows=len(df),complete_model_configurations=d.groupby(['configuration','variant']).ngroups,scenario_rows=len(diagnostics),scenario_masks_match=True,key_pairs=sum(len(p) for p in paired),all_actual_cells_have_112_pairs=True,source_freeze_unchanged=all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in json.loads((R/'SOURCE_FREEZE.json').read_text())['files'].items()))
    assert checks['source_freeze_unchanged']
    (A/'POST_REVIEW_CHECKS.json').write_text(json.dumps(checks,indent=2)+'\n')

if __name__=='__main__':main()
