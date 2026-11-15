"""Paired, equal-farm exploratory summaries; outcomes never become inputs."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent
paths=sorted((HERE/'jobs').glob('*/RESULT.json'))
assert len(paths)==112, len(paths)
rows=[];paired=[];audits=[]
for path in paths:
    a=json.loads(path.read_text());rows+=a['rows'];paired+=a['paired'];audits+=a['audit']
r=pd.DataFrame(rows);p=pd.DataFrame(paired)
assert set(zip(r.site,r.seed))=={(i,j) for i in range(56) for j in [44,45]}
keys=['site','seed','group','bin','horizon']
base=r[r.variant=='N0']
den=base[(base.group=='current_power')&(base.bin=='high')][['site','seed','horizon','n']].rename(columns={'n':'total_high_n'})
r=r.merge(den,on=['site','seed','horizon'],validate='many_to_one')
p=p.merge(den,on=['site','seed','horizon'],validate='many_to_one')
for col in ['delta_sq_sum','cross_sum','update_sq_sum']:
    p[col.replace('_sum','')+'_contribution']=p[col]/p.total_high_n
p['fraction_of_high']=p.n/p.total_high_n
r['fraction_of_high']=r.n/r.total_high_n
r['positive_mse_contribution']=r.positive_sq_sum/r.total_high_n
r['negative_mse_contribution']=r.negative_sq_sum/r.total_high_n
g=r[r.variant!='N0'].merge(base,on=keys,suffixes=('','_base'),validate='many_to_one')
assert (g.n==g.n_base).all()
g=g.merge(p,on=keys+['variant'],suffixes=('','_pair'),validate='one_to_one')
g['delta_mae']=g.mae-g.mae_base;g['delta_rmse']=g.rmse-g.rmse_base
g.to_csv(HERE/'REPLAY_PAIRED_PER_FARM_SEED.csv',index=False)
result=[]
for key,frame in g.groupby(['variant','group','bin','horizon']):
    # Equal seed mean inside farm; equal farm mean after that.
    f=frame.groupby('site').mean(numeric_only=True)
    row=dict(zip(['variant','group','bin','horizon'],key))
    row.update(farms=len(f),cells_per_seed=frame.n.sum()/2,
        mae=float(f.mae.mean()),mae_base=float(f.mae_base.mean()),
        rmse=float(f.rmse.mean()),rmse_base=float(f.rmse_base.mean()),
        mae_percent=float(100*(f.mae.mean()/f.mae_base.mean()-1)),
        rmse_percent=float(100*(f.rmse.mean()/f.rmse_base.mean()-1)),
        bias=float(f.bias.mean()),bias_base=float(f.bias_base.mean()),
        update=float(f.mean_update.mean()),actual_change=float(f.mean_actual_change.mean()),
        improved_mae=int((f.delta_mae<0).sum()),improved_rmse=int((f.delta_rmse<0).sum()),
        max_metric_reproduction_difference=max(a['metric_max_difference'] for a in audits))
    # Absent groups contribute zero to the total 56-farm error, not missingness.
    for col in ['fraction_of_high','delta_sq_contribution','cross_contribution','update_sq_contribution']:
        row[col]=float(f[col].sum()/56)
    row['positive_mse_delta']=float(((frame.positive_sq_sum-frame.positive_sq_sum_base)/frame.total_high_n).sum()/112)
    row['negative_mse_delta']=float(((frame.negative_sq_sum-frame.negative_sq_sum_base)/frame.total_high_n).sum()/112)
    row['prediction_above1_fraction']=float(frame.prediction_above1.sum()/frame.n.sum())
    row['target_above1_fraction']=float(frame.target_above1.sum()/frame.n.sum())
    result.append(row)
summary=pd.DataFrame(result)
summary.to_csv(HERE/'REPLAY_GROUP_EFFECTS.csv',index=False)
# Additivity is a meaningful diagnostic of the accounting, not a fitted test.
for variant in p.variant.unique():
    for horizon in p.horizon.unique():
        total=summary[(summary.variant==variant)&(summary.group=='current_power')&(summary.horizon==horizon)].iloc[0]
        for family in ['outcome_from_origin','nwp_change_from_origin','region_minus_local','outcome_x_nwp','outcome_x_region','local_hour','origin_power_level','region_change_1h','history_change_1h']:
            parts=summary[(summary.variant==variant)&(summary.group==family)&(summary.horizon==horizon)]
            assert abs(parts.fraction_of_high.sum()-1)<1e-10,(variant,horizon,family)
            for col in ['delta_sq_contribution','cross_contribution','update_sq_contribution','positive_mse_delta','negative_mse_delta']:
                assert abs(parts[col].sum()-total[col])<1e-10,(variant,horizon,family,col)
audit=dict(jobs=112,farms=56,seeds=[44,45],arms=['N0','N3','R3_RAW','R3_ALLPOOL'],checkpoint_records=len(audits),
           selection_year=2023,training=False,predictions_saved=False,exploratory=True,
           max_metric_reproduction_difference=max(a['metric_max_difference'] for a in audits),
           decomposition_checks_passed=True,checkpoint_audits=audits)
(HERE/'REPLAY_AUDIT.json').write_text(json.dumps(audit,indent=2)+'\n')
show=summary[(summary.variant=='R3_RAW')&(summary.horizon.isin(['all','3_6h']))&(summary.group.isin(['current_power','outcome_from_origin','nwp_change_from_origin','region_minus_local','region_change_1h','origin_power_level']))]
print(show[['group','bin','horizon','farms','fraction_of_high','mae_percent','rmse_percent','bias_base','bias','update','delta_sq_contribution','cross_contribution','update_sq_contribution']].round(7).to_string(index=False))
