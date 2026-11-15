"""Compact supplementary summaries; existing farm/seed source tables stay authoritative."""
from pathlib import Path
import json, hashlib
import pandas as pd
import numpy as np

P=Path('/shared_data_ssd2/liying/wind_cluster_project_20261018/project')
N=P/'runs/neighbor_observation_sel_single_farm_20261025_v1'
D=P/'runs/upstream_advection_diagnostic_20261025'
OUT=Path(__file__).resolve().parent
TF='TF_z1_lr0.0002_dp0.05'

def summarize(p,metric):
    farms=p.groupby('site')[[metric+'_c',metric+'_r']].mean()
    assert len(p)==2*len(farms)
    c,r=farms[metric+'_c'].to_numpy(),farms[metric+'_r'].to_numpy()
    rng=np.random.default_rng(20261025);ix=rng.integers(0,len(c),(2000,len(c)))
    bootstrap=100*(c[ix].mean(1)/r[ix].mean(1)-1)
    result=dict(metric=metric,farms=len(farms),pairs=len(p),effect_percent=float(100*(c.mean()/r.mean()-1)),improved_farms=int((c<r).sum()),exploratory_ci_low=float(np.quantile(bootstrap,.025)),exploratory_ci_high=float(np.quantile(bootstrap,.975)))
    for seed,x in p.groupby('seed'):
        result[f'effect_seed{seed}']=float(100*(x[metric+'_c'].mean()/x[metric+'_r'].mean()-1))
        result[f'improved_seed{seed}']=int((x[metric+'_c']<x[metric+'_r']).sum())
    return result

def main():
    source=N/'analysis/all_actual_per_farm_seed.csv'
    a=pd.read_csv(source);a=a[a.selection=='mae']
    rows=[]
    for cfg,cv,rv in [('W0','N1','N0'),('W0','N3','N0'),('W0','N1','N3'),('W0','N1','N2'),('W0','N1','N2_SELF'),(TF,'N1','N0'),('W0','N7_D0','N1'),('W0','N7_D30','N1')]:
        p=a[(a.configuration==cfg)&(a.variant==cv)].merge(a[(a.configuration==cfg)&(a.variant==rv)],on=['site','seed','horizon'],suffixes=('_c','_r'),validate='one_to_one')
        assert (p.n_c==p.n_r).all() and len(p)==448
        for horizon,x in p.groupby('horizon'):
            for metric in ['mae','rmse']:rows.append(dict(configuration=cfg,candidate=cv,reference=rv,horizon=horizon,**summarize(x,metric)))
    effects=pd.DataFrame(rows);effects.to_csv(OUT/'model_comparisons.csv',index=False)
    q=effects[(effects.configuration=='W0')&(effects.candidate=='N1')&(effects.reference=='N0')&(effects.horizon=='all')].set_index('metric')
    gate=dict(mae_effect_percent=float(q.loc['mae','effect_percent']),rmse_effect_percent=float(q.loc['rmse','effect_percent']),both_below_minus_0p1=bool((q.effect_percent<=-.1).all()),selection='same MAE-selected checkpoint for both metrics',note='Retrospective application of the newly requested two-metric gate; original frozen gate used MAE. No change to old freeze or recommendation.')
    diag=pd.read_csv(N/'analysis/diagnostics_per_farm_seed.csv');rows=[]
    for cv,rv in [('N1','N0'),('N3','N0'),('N1','N3')]:
        p=diag[(diag.configuration=='W0')&(diag.variant==cv)].merge(diag[(diag.configuration=='W0')&(diag.variant==rv)],on=['site','seed','group','bin','horizon'],suffixes=('_c','_r'),validate='one_to_one')
        assert (p.n_c==p.n_r).all()
        for (group,label,horizon),x in p.groupby(['group','bin','horizon']):
            for metric in ['mae','rmse']:rows.append(dict(candidate=cv,reference=rv,group=group,bin=label,horizon=horizon,**summarize(x,metric)))
    pd.DataFrame(rows).to_csv(OUT/'scenario_comparisons.csv',index=False)
    intervention=pd.read_csv(N/'analysis/interventions_per_farm_seed.csv');rows=[]
    intervention=intervention[intervention.variant=='N1']
    for (cfg,kind,horizon),x in intervention.groupby(['configuration','intervention','horizon']):
        p=x.rename(columns={m:m+'_c' for m in ['mae','rmse']}|{m+'_normal':m+'_r' for m in ['mae','rmse']})
        for metric in ['mae','rmse']:rows.append(dict(configuration=cfg,intervention=kind,horizon=horizon,**summarize(p,metric)))
    pd.DataFrame(rows).to_csv(OUT/'intervention_comparisons.csv',index=False)
    old=pd.read_csv(D/'incremental_predictability.csv');old['baseline']='original';tables=[]
    corrected=OUT/'paired_predictability_corrected.csv'
    if corrected.exists():
        new=pd.read_csv(corrected);new['cls']=new.cls.map(dict(up='A_UP',down='B_UP',cross='CROSS'));new['b']=new.target
        old=pd.concat([old,new],ignore_index=True)
    for (baseline,cls,horizon),x in old.groupby(['baseline','cls','horizon']):
        for metric in ['mse','rmse','mae']:
            base=np.sqrt(x.mse_base) if metric=='rmse' else x[metric+'_base']
            full=np.sqrt(x.mse_full) if metric=='rmse' else x[metric+'_full']
            change=100*(full/base-1)
            pertarget=pd.DataFrame(dict(target=x.b,base=base,full=full)).groupby('target').mean()
            tables.append(dict(baseline=baseline,cls=cls,horizon=horizon,metric=metric,pairs=len(x),targets=x.b.nunique(),median_pair_change=float(change.median()),mean_pair_change=float(change.mean()),improved_pair_fraction=float((change<0).mean()),target_equal_metric_ratio_percent=float(100*(pertarget.full.mean()/pertarget.base.mean()-1))))
    pd.DataFrame(tables).to_csv(OUT/'linear_diagnostic_summaries.csv',index=False)
    lag=pd.read_csv(D/'lagged_crosscorrelation.csv')
    with np.load(N/'assets/neighbor_tables.npz') as topology:
        mask=topology['distance_mask'][:,:8];distance=topology['distance_geometry'][:,:8,2]
        topology_summary=dict(valid_directed_N1_edges=int(mask.sum()),edges_below_20km_excluded_from_old_diagnostic=int(((distance<20)&mask).sum()),median_N1_distance_km=float(np.median(distance[mask])))
    orig=old[old.baseline=='original'];counter=orig.groupby('b').size()
    diagnostic_audit=dict(unordered_pairs=int(len(lag[lag.cls=='A_UP'])),predicted_targets=int(len(counter)),missing_target_indices=sorted(set(range(56))-set(counter.index.astype(int))),A_UP_expected_lag_exceeds_search=int(((lag.cls=='A_UP')&(lag.expected_lag_h>4)).sum()))
    audit=dict(screen_completion=json.loads((N/'COMPLETION_AUDIT.json').read_text()),new_dual_metric_gate=gate,original_diagnostic=diagnostic_audit,topology=topology_summary,review_reads_2024_target_or_weather_values=False,aggregation='Supplementary seed means within farm then equal farm. Original raw farm/seed tables retained. Pair diagnostic summaries are not equivalent to model/farm metrics.',ci_limit='IID farm bootstrap exploratory only; unadjusted for spatial dependence and repeated selection.')
    (OUT/'REVIEW_AUDIT.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n')
    inputs=[source,N/'analysis/diagnostics_per_farm_seed.csv',N/'analysis/interventions_per_farm_seed.csv',D/'upstream_advection.py',D/'REPORT_CN.md',D/'lagged_crosscorrelation.csv',D/'incremental_predictability.csv']
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
    (OUT/'REVIEW_SOURCES.json').write_text(json.dumps(hashes,indent=2)+'\n')
    print(effects[effects.horizon=='all'].to_string(index=False));print(json.dumps(audit,ensure_ascii=False))

if __name__=='__main__':main()
