from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
HERE=Path(__file__).resolve().parent;P=HERE.parents[1]
E=P/'runs/regional_observation_10seed_20261026_v1/analysis'
D=P/'runs/regional_observation_completion_20261026_v1'
parts=[]
for chunk in pd.read_csv(E/'scenarios_per_farm_seed.csv',chunksize=100000):
    parts.append(chunk[(chunk.group=='current_power')&(chunk.bin=='high')])
a=pd.concat(parts,ignore_index=True)
a['mse']=a.rmse**2;a['bias_squared']=a.bias**2;a['centered_mse']=a.mse-a.bias_squared
topology=[]
with np.load(D/'assets/topology.npz') as z:
    for site in range(56):
        distances=z['geometry'][site,:,2];valid=z['mask'][site];caps=z['capacity'][site]
        row=dict(site=site,near8_within20=int(((distances[:8]<20)&valid[:8]).sum()))
        for index,(lo,hi) in enumerate([(0,20),(20,60),(60,150.0001)]):
            mask=valid&(distances>=lo)&(distances<hi);w=caps*mask
            row[f'ring{index}_n']=int(mask.sum());row[f'ring{index}_neff']=float(w.sum()**2/max(float((w*w).sum()),1e-12))
        topology.append(row)
top=pd.DataFrame(topology);top.to_csv(HERE/'TOPOLOGY.csv',index=False)
summary=a.groupby(['configuration','variant','horizon','site'])[['n','mae','rmse','bias','mse','bias_squared','centered_mse']].mean().groupby(['configuration','variant','horizon']).mean().reset_index()
summary.to_csv(HERE/'HIGH_POWER_10SEED.csv',index=False)
effects=[];farm_rows=[]
for config in a.configuration.unique():
    f=a[a.configuration==config]
    for variant,ref in [('N3','N0'),('R3_RAW','N0'),('R3_RAW','N3'),('R3_ALLPOOL','N0'),('R3_ALLPOOL','R3_RAW')]:
        g=f[f.variant==variant].merge(f[f.variant==ref],on=['site','seed','horizon'],suffixes=('_a','_b'),validate='one_to_one')
        assert (g.n_a==g.n_b).all()
        for horizon,v in g.groupby('horizon'):
            farms=v.groupby('site').mean(numeric_only=True)
            row=dict(configuration=config,variant=variant,reference=ref,horizon=horizon,
                mae_percent=100*(farms.mae_a.mean()/farms.mae_b.mean()-1),
                rmse_percent=100*(farms.rmse_a.mean()/farms.rmse_b.mean()-1),
                improved_mae=int((farms.mae_a<farms.mae_b).sum()),improved_rmse=int((farms.rmse_a<farms.rmse_b).sum()),
                bias_delta=farms.bias_a.mean()-farms.bias_b.mean(),
                delta_mse=farms.mse_a.mean()-farms.mse_b.mean(),
                delta_bias_squared=farms.bias_squared_a.mean()-farms.bias_squared_b.mean(),
                delta_centered_mse=farms.centered_mse_a.mean()-farms.centered_mse_b.mean())
            assert abs(row['delta_mse']-row['delta_bias_squared']-row['delta_centered_mse'])<1e-12
            effects.append(row)
            if variant=='R3_RAW' and ref=='N0' and horizon=='all':
                for site,x in farms.iterrows():farm_rows.append(dict(configuration=config,site=site,n=x.n_a,
                    mae_percent=100*(x.mae_a/x.mae_b-1),rmse_percent=100*(x.rmse_a/x.rmse_b-1),
                    bias_reference=x.bias_b,bias_candidate=x.bias_a,delta_mse=x.mse_a-x.mse_b))
pd.DataFrame(effects).to_csv(HERE/'HIGH_POWER_EFFECTS.csv',index=False)
farms=pd.DataFrame(farm_rows).merge(top,on='site');farms.to_csv(HERE/'HIGH_POWER_FARMS.csv',index=False)
numeric={}
for config,f in farms.groupby('configuration'):
    numeric[config]=dict(high_fraction_median=float(np.median(f.n/(32550*24))),high_origins_min=float(f.n.min()/24),high_origins_max=float(f.n.max()/24),
        high_rmse_median_effect=float(f.rmse_percent.median()),
        correlations={key:dict(spearman=float(spearmanr(f.rmse_percent,f[key]).statistic)) for key in ['n','ring0_n','ring1_n','ring2_n','ring0_neff','ring1_neff','ring2_neff']})
(HERE/'EXISTING_SUMMARY.json').write_text(json.dumps(numeric,indent=2)+'\n')
print(pd.DataFrame(effects).round(7).to_string(index=False))
print(json.dumps(numeric))

# D two-seed removal controls, using the same MAE-selected models.
overall=pd.read_csv(D/'analysis/all_actual_per_farm_seed.csv')
overall=overall[(overall.configuration=='W0')&(overall.selection=='mae')&(overall.horizon=='all')]
parts=[]
for chunk in pd.read_csv(D/'analysis/diagnostics_per_farm_seed.csv',chunksize=100000):
    parts.append(chunk[(chunk.configuration=='W0')&(chunk.group=='current_power')&(chunk.bin=='high')&(chunk.horizon=='all')])
high=pd.concat(parts,ignore_index=True)
ablation=[]
for scope,table in [('all',overall),('high_power',high)]:
    ref=table[table.variant=='R3_RAW']
    for variant in ['R3_NO_INNER','R3_NO_MIDDLE','R3_NO_OUTER','R3_NO_N3','R3_ALLPOOL','R3_MEAN','R3_MEAN_DELTA']:
        paired=table[table.variant==variant].merge(ref,on=['site','seed'],suffixes=('','_b'),validate='one_to_one')
        farms=paired.groupby('site').mean(numeric_only=True)
        ablation.append(dict(scope=scope,variant=variant,reference='R3_RAW',farms=len(farms),
            mae_percent=100*(farms.mae.mean()/farms.mae_b.mean()-1),rmse_percent=100*(farms.rmse.mean()/farms.rmse_b.mean()-1),
            mae_wins=int((farms.mae<farms.mae_b).sum()),rmse_wins=int((farms.rmse<farms.rmse_b).sum())))
pd.DataFrame(ablation).to_csv(HERE/'D_ABLATION_EFFECTS.csv',index=False)
