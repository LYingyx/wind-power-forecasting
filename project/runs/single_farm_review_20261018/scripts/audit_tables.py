"""Rebuild protocol-separated evidence and outcome-independent farm selection.

Run with project/.runtime/chronos/bin/python. No training; originals are read-only.
"""
from pathlib import Path
import hashlib
import json
import sqlite3
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parents[1]
ROOT = OUT.parents[1]
ART = ROOT / 'artifacts'
SOURCES = set()

def read_json(path):
    path = Path(path); SOURCES.add(path)
    return json.loads(path.read_text())

def read_csv(path, **kw):
    path = Path(path); SOURCES.add(path)
    return pd.read_csv(path, **kw)

def fmt(x):
    if isinstance(x, (float, np.floating)): return '' if pd.isna(x) else f'{x:.6f}'
    return str(x).replace('|', '/')

def save(df, name, preview=True):
    df.to_csv(OUT/'tables'/f'{name}.csv', index=False)
    if preview:
        s = '| ' + ' | '.join(df.columns) + ' |\n|' + '|'.join(['---']*len(df.columns)) + '|\n'
        s += '\n'.join('| ' + ' | '.join(fmt(x) for x in row) + ' |' for row in df.itertuples(index=False,name=None))
        (OUT/'tables'/f'{name}.md').write_text(s+'\n')
        df.to_latex(OUT/'tables'/f'{name}.tex', index=False, escape=True, float_format=lambda x: f'{x:.6f}')

def bootstrap(delta, seed=20261018):
    d = np.asarray(delta, dtype=float); d = d[np.isfinite(d)]
    rng = np.random.default_rng(seed)
    lo,hi = np.quantile(d[rng.integers(len(d),size=(20000,len(d)))].mean(1), [.025,.975])
    return dict(delta=float(d.mean()), ci_low=lo, ci_high=hi, wins=int((d<0).sum()), units=len(d))

def summary(df, keys):
    # Equal seeds/seasons inside farm, then equal farms. Keep seed SD separate.
    rows=[]
    for key,g in df.groupby(keys, dropna=False):
        key = key if isinstance(key,tuple) else (key,)
        farm = g.groupby('station')[['mae','rmse','r2']].mean()
        seed = g.groupby(['seed','station'])['mae'].mean().groupby('seed').mean()
        row=dict(zip(keys,key)); row.update(mae=farm.mae.mean(),
            rmse_macro=farm.rmse.mean(),r2_macro=farm.r2.mean(),seed_sd=seed.std(ddof=1),
            farm_sd=farm.mae.std(ddof=1),farms=len(farm),seeds=g.seed.nunique(),records=len(g))
        rows.append(row)
    return pd.DataFrame(rows)

def kmedoids(x,k):
    # Deterministic PAM: globally best single swap until convergence.
    distance=np.sqrt(((x[:,None]-x[None,:])**2).sum(-1))
    med=[int(distance.sum(0).argmin())]
    while len(med)<k:
        med.append(int(distance[:,med].min(1).argmax()))
    score=distance[:,med].min(1).sum()
    for _ in range(100):
        best=(score,None)
        for i in range(k):
            for j in range(len(x)):
                if j in med: continue
                candidate=med.copy();candidate[i]=j
                cost=distance[:,candidate].min(1).sum()
                if cost<best[0]-1e-10: best=(cost,candidate)
        if best[1] is None: break
        score,med=best
    med=sorted(med)
    return med, np.argmin(distance[:,med],axis=1), distance[:,med].min(1)

def descriptors(ids, target=False):
    cache=ROOT/'data/model_cache_pad5_f8'
    st=cache/'static'
    source_ids=np.load(st/'station_id.npy').astype(str)
    meta=read_csv(ART/'cross_farm_shared_specific_20260920_v1/data/farm_quality_audit.csv',dtype={'station_id':str}).set_index('station_id')
    if not target:
        hist=np.load(cache/'train/station_history.npy',mmap_mode='r')
        origins=np.load(cache/'train/origin_unix.npy')
    rows=[]
    for station in ids:
        if target:
            folder=ART/f'cross_farm_shared_specific_20260920_v1/data/transfer_station_cache_v2/{station}/train'
            h=np.load(folder/'station_history.npy',mmap_mode='r').reshape(-1,24)
            origins=np.load(folder/'origin_unix.npy')
        else: h=hist[:,list(source_ids).index(station),:]
        v=np.asarray(h[:,-1],dtype=float)
        ramp=np.abs(np.asarray(h[:,-1],dtype=float)-np.asarray(h[:,-5],dtype=float))
        m=meta.loc[station]
        rows.append(dict(station_id=station,station_name=m['name'],latitude=m.latitude,longitude=m.longitude,
            capacity_mw=m.capacity_mw,descriptor_year=2023 if target else 2022,origins=len(v),
            origin_min=int(origins.min()),origin_max=int(origins.max()),
            mean_cf=v.mean(),std_cf=v.std(ddof=0),low_cf_fraction=(v<.05).mean(),
            high_cf_fraction=(v>=.8).mean(),ramp_1h_q90=np.quantile(ramp,.9),
            coverage_train=m['coverage_2023' if target else 'coverage_2022']))
    return pd.DataFrame(rows).sort_values('station_id').reset_index(drop=True)

def select_farms(df,k,label):
    raw=df[['latitude','longitude','capacity_mw','mean_cf','std_cf','low_cf_fraction','high_cf_fraction','ramp_1h_q90']].to_numpy(float)
    raw[:,2]=np.log1p(raw[:,2])
    center=np.median(raw,axis=0); scale=np.quantile(raw,.75,axis=0)-np.quantile(raw,.25,axis=0)
    scale=np.where(scale>1e-9,scale,1)
    # Four equal feature groups: geography, size, CF distribution, dynamics.
    weights=np.sqrt(np.array([.5,.5,1,.25,.25,.25,.25,1]))
    z=(raw-center)/scale*weights
    med,groups,dist=kmedoids(z,k)
    df=df.copy();df['cluster']=groups+1;df['distance_to_medoid']=dist
    df['representative']=False;df.loc[med,'representative']=True
    save(df,label+'_descriptors',False)
    selected=df.loc[med].copy();selected['represented_farms']=[int((groups==i).sum()) for i in range(k)]
    save(selected,label+'_representatives')
    return dict(features=['latitude','longitude','log1p_capacity','mean_cf','std_cf','low_cf_fraction','high_cf_fraction','ramp_1h_q90'],
                centers=center.tolist(),iqr=scale.tolist(),weights=weights.tolist(),k=k,algorithm='deterministic PAM single-swap',
                selected=selected.station_id.tolist(),uses_prediction_errors=False,uses_test_descriptors=False)

def main():
    conn=sqlite3.connect(f'file:{ROOT}/results/CATALOG.sqlite?mode=ro',uri=True)
    obs=pd.read_sql_query('select * from observations',conn);conn.close()
    SOURCES.add(ROOT/'results/CATALOG.sqlite')
    SOURCES.add(ROOT/'data/model_cache_pad5_f8/manifest.json')
    obs['seed']=pd.to_numeric(obs.seed)
    dup=obs.duplicated(['experiment','run_id','split']).sum();assert dup==0,dup
    save(obs.groupby(['experiment','stage','cohort','split'],dropna=False).agg(records=('mae','size'),runs=('run_id','nunique'),mae_missing=('mae',lambda x:x.isna().sum()),checkpoints=('checkpoint_exists','sum')).reset_index(),'experiment_inventory',False)
    ids=np.load(ROOT/'data/model_cache_pad5_f8/static/station_id.npy').astype(str)
    selection=read_json(ART/'cross_farm_shared_specific_20260920_v1/data/selection_protocol.json')
    assert set(ids)==set(selection['source_station_ids'])
    assert not set(ids)&set(selection['final_heldout_station_ids'])
    assert not set(selection['transfer_dev_station_ids'])&set(selection['final_heldout_station_ids'])
    feature_selection={
        'source56':select_farms(descriptors(ids),6,'source56'),
        'target20':select_farms(descriptors(selection['final_heldout_station_ids'],True),4,'target20'),
        'limitation':'Retrospective algorithm fixed without using model errors; not preregistered before original experiments. Descriptors use available valid cache origins, not whole-calendar raw coverage.'}
    (OUT/'farm_selection.json').write_text(json.dumps(feature_selection,ensure_ascii=False,indent=2)+'\n')

    # Source baselines: expand primary metric JSON to retain the station axis.
    shared=obs[obs.experiment.eq('shared_all_station_baselines_20260922_v1')]
    station_rows=[];lead_rows=[]
    for _,r in shared.iterrows():
        d=read_json(ROOT/r.metric_file)[r.split]
        assert list(map(str,d['station_ids']))==list(ids)
        for j,station in enumerate(ids):
            station_rows.append(dict(model=r.model,split=r.split,seed=r.seed,station=station,
                mae=d['per_station_mae'][j],rmse=d['per_station_rmse'][j],r2=d['per_station_r2'][j]))
        for j,(mae,mse) in enumerate(zip(d['mae_by_lead'],d['mse_by_lead'])):
            lead_rows.append(dict(model=r.model,split=r.split,seed=r.seed,lead_hours=(j+1)/4,mae=mae,mse=mse))
    source=pd.DataFrame(station_rows)
    save(source,'shared_baselines_station_seed',False)
    save(pd.DataFrame(lead_rows),'shared_baselines_lead_seed',False)
    save(summary(source,['model','split']).sort_values(['split','mae']),'shared_baselines_full_year')
    pilot=obs[obs.experiment.eq('expanded_all_station_baselines_20260919_v1')]
    assert len(pilot)==8400 and pilot.station.nunique()==56 and pilot.seed.nunique()==10
    ps=summary(pilot,['model','split']).sort_values('mae');ps['protocol']='2022/2023 monthly-date pilot: 3156/3349 origins; not full-year'
    save(ps,'independent_baselines_pilot_only')
    save(pilot[['model','station','seed','split','mae','rmse','r2','metric_file']],'independent_baselines_pilot_station_seed',False)

    # Fully matched sharing/local-input 2x2, sourced outside scalar catalog.
    two=read_csv(ART/'cross_farm_shared_specific_20260920_v1/analysis/two_by_two_station_seed.csv')
    two['station']=ids[two.site_index.astype(int)];two=two.rename(columns={'cell':'model'})
    assert len(two)==4*10*2*56
    save(two,'sharing_2x2_station_seed',False)
    agg=two.groupby(['model','split','seed']).mae.mean().groupby(['model','split']).agg(['mean','std']).reset_index()
    save(agg,'sharing_2x2')
    contrasts=[]
    for split,g in two.groupby('split'):
        w=g.groupby(['station','model']).mae.mean().unstack()
        for left,right in [('SL','IL'),('SC','IC'),('SC','SL'),('IC','IL')]:
            contrasts.append(dict(split=split,comparison=f'{left}-{right}',unit='farm_after_seed_mean',**bootstrap(w[left]-w[right])))
    save(pd.DataFrame(contrasts),'sharing_2x2_contrasts')
    indep=read_csv(ART/'cbh_per_station_independent_20260918_v1/analysis/independent_per_station_seed_long.csv',dtype={'station_id':str})
    h=indep.groupby(['model','split','seed']).mae_cf.mean().groupby(['model','split']).agg(['mean','std']).reset_index()
    save(h,'independent_cbh_full_year')

    # Conditions retain FP16 protocol; expand per-farm metrics for stronger reporting.
    cond=[]
    for split in ['val','test']:
        for model in ['CBH','CB','CH','CNEUTRAL']:
            for seed in range(42,52):
                d=read_json(ART/f'p0i_condition_ablation_completion_20261014_v1/analysis/{split}/{model}_seed{seed}/metrics.json')
                for st,mae in zip(ids,d['per_station_mae_cf']):
                    cond.append(dict(model=model,split=split,seed=seed,station=st,mae=mae))
    cond=pd.DataFrame(cond);save(cond,'condition_ablation_station_seed',False)
    save(cond.groupby(['model','split','seed']).mae.mean().groupby(['model','split']).agg(['mean','std']).reset_index(),'condition_ablation_fp16')
    cc=[]
    for split,g in cond.groupby('split'):
        w=g.groupby(['station','model']).mae.mean().unstack()
        for left,right in [('CBH','CNEUTRAL'),('CB','CNEUTRAL'),('CH','CNEUTRAL'),('CBH','CB'),('CBH','CH')]:
            cc.append(dict(split=split,comparison=f'{left}-{right}',unit='farm_after_seed_mean',**bootstrap(w[left]-w[right])))
    save(pd.DataFrame(cc),'condition_farm_contrasts')
    weather=obs[obs.experiment.eq('column_weather_20260928_v1')]
    save(weather.groupby(['model','split']).mae.agg(['mean','std','count']).reset_index(),'weather_ablation_fp32')
    wd=read_json(ART/'final_e2c_extensions_20260930_v1/analysis/weather_encoder_factorial_ablation.json')
    wc=[]
    for d in wd['contrasts']:
        wc.append(dict(contrast=d['name'],coefficients=json.dumps(d['coefficients']),split=d['split'],delta=d['delta'],
             ci_low=d['bootstrap_95pct'][0],ci_high=d['bootstrap_95pct'][1],wins_seeds=d['negative_seeds'],wins_farms=d['negative_stations'],
             interval_unit='archived equal-week seed x farm x 7-day bootstrap; point delta is valid-origin-weighted'))
    save(pd.DataFrame(wc),'weather_factorial_contrasts')
    mo=read_json(ART/'pretrained_transformer_transfer_20260927_v1/analysis/matched_output_report.json')
    mr=[]
    for split in ['val','test']:
        for r in mo['records']:
            for model in ['direct','correction']:
                for st,value in zip(ids,r[model][f'per_station_{split}']):
                    mr.append(dict(model=model,split=split,seed=r['seed'],station=st,mae=value))
    mr=pd.DataFrame(mr);save(mr,'matched_output_station_seed',False)
    save(mr.groupby(['model','split','seed']).mae.mean().groupby(['model','split']).agg(['mean','std']).reset_index(),'matched_output_fp16')
    mc=[]
    for split,g in mr.groupby('split'):
        w=g.groupby(['station','model']).mae.mean().unstack()
        mc.append(dict(split=split,comparison='correction-direct',unit='farm_after_seed_mean',**bootstrap(w.correction-w.direct)))
    save(pd.DataFrame(mc),'matched_output_contrasts')

    # Target comparisons: common seeds 42--46; checkpoint selection labels are NOT in training-budget count.
    target=obs[obs.cohort.eq('final') & obs.stage.eq('target_main') & obs.split.eq('test') & obs.seed.between(42,46)].copy()
    target=target[target.experiment.isin(['selected30_fewshot_baselines_20260920_v1','cross_farm_shared_specific_20260920_v1','pretrained_transformer_transfer_20260927_v1','final_e2c_extensions_20260930_v1'])]
    target['model']=target.model.replace({'E1_pretrained_TRANSFORMER_FC_all_parameter_finetune':'Pretrained_Transformer','final_E2C_T5_full_correction_adaptation':'E2C_T5'})
    target['model']=np.where(target.experiment.eq('selected30_fewshot_baselines_20260920_v1'),'Scratch_'+target.model,target.model)
    assert not target.duplicated(['model','station','seed','budget','season']).any()
    for model in ['E2C_T5','Pretrained_Transformer','T5_r16'] + ['Scratch_'+m for m in pilot.model.unique()]:
        g=target[target.model.eq(model)]
        assert len(g)==1400 and g.station.nunique()==20 and g.seed.nunique()==5,(model,len(g))
    save(target[['model','station','seed','budget','season','mae','rmse','r2','precision','metric_file']],'target20_common_seed_metrics',False)
    save(summary(target,['model','budget']),'target20_all_baselines_common_seeds')
    frozen=read_json(ART/'selected30_fewshot_baselines_20260920_v1/analysis/dev_family_selection_pre_final.json')
    dev=obs[obs.experiment.eq('selected30_fewshot_baselines_20260920_v1') & obs.cohort.eq('dev') & obs.stage.eq('target_main') & obs.split.eq('val')]
    devmean=dev.groupby(['budget','model','station']).mae.mean().groupby(['budget','model']).mean()
    for b,d in frozen['budgets'].items(): assert devmean.loc[b].idxmin()==d['selected_family'],b
    frozen_rows=[]
    for b,d in frozen['budgets'].items():
        g=target[target.budget.eq(b)&target.model.eq('Scratch_'+d['selected_family'])].copy()
        g['model']='Scratch_dev_selected';g['selected_family']=d['selected_family'];frozen_rows.append(g)
    selected=pd.concat(frozen_rows)
    save(summary(selected,['selected_family','budget']),'target20_scratch_dev_selected')
    comp=pd.concat([target,selected])
    crows=[]
    for b in ['7d','30d','90d','180d','full']:
        q=comp[comp.budget.eq(b)]
        for left,right in [('E2C_T5','Pretrained_Transformer'),('E2C_T5','T5_r16'),('E2C_T5','Scratch_dev_selected'),('Pretrained_Transformer','Scratch_dev_selected'),('T5_r16','Scratch_dev_selected')]:
            keys=['station','seed','season']
            l=q[q.model.eq(left)].set_index(keys).mae
            r=q[q.model.eq(right)].set_index(keys).mae
            assert l.index.equals(r.index) or set(l.index)==set(r.index),(left,right,b)
            delta=(l-r).groupby('station').mean()
            crows.append(dict(budget=b,comparison=f'{left}-{right}',unit='20 farms; mean paired seeds and season blocks',**bootstrap(delta)))
    save(pd.DataFrame(crows),'target20_paired_contrasts')
    mains=comp[comp.model.isin(['E2C_T5','Pretrained_Transformer','T5_r16','Scratch_dev_selected'])]
    save(summary(mains,['model','budget']),'target20_main_comparison')
    farms=mains.groupby(['station','model','budget'])[['mae','rmse','r2']].mean().reset_index()
    save(farms,'target20_per_farm_budget',False)
    # All ten scratch seeds retained in a separate descriptive table.
    fullscratch=obs[obs.experiment.eq('selected30_fewshot_baselines_20260920_v1') & obs.cohort.eq('final') & obs.split.eq('test') & obs.stage.eq('target_main')]
    save(summary(fullscratch,['model','budget']),'target20_scratch_all10seeds')

    # Local-only viable reference against shared baselines, not the E2C architecture.
    local=two[two.model.eq('SL')].copy();local['model']='CBH_SL_local_only'
    source_for_pair=pd.concat([source,local],ignore_index=True)
    sr=[]
    for split,g in source_for_pair.groupby('split'):
        w=g.groupby(['station','model']).mae.mean().unstack()
        sr.append(dict(split=split,comparison='CBH_SL_local_only-TRANSFORMER_FC',unit='farm_after_seed_mean',**bootstrap(w.CBH_SL_local_only-w.TRANSFORMER_FC)))
    save(pd.DataFrame(sr),'local_only_cbh_vs_shared_transformer')
    audit=dict(observation_records=len(obs),unique_runs=obs[['experiment','run_id']].drop_duplicates().shape[0],
               duplicate_run_split_keys=int(dup),source_farms=len(ids),dev_farms=10,final_farms=20,
               target_common_seeds=[42,43,44,45,46],pilot_runs=len(pilot),
               limitations=['No completed new training in this review.', 'Source baseline full-year vs independent pilot are nonmatched.',
                 'Target budgets count adaptation labels only; full 2024 H1 labels select checkpoints.',
                 'Current E2C requires cluster-history context; pure local E2C has not been evaluated.'])
    (OUT/'audit_status.json').write_text(json.dumps(audit,indent=2,ensure_ascii=False)+'\n')
    sources=[]
    for f in sorted(SOURCES):
        sources.append(dict(path=str(f.relative_to(ROOT)),bytes=f.stat().st_size,sha256=hashlib.sha256(f.read_bytes()).hexdigest()))
    (OUT/'table_sources.json').write_text(json.dumps(sources,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(audit,ensure_ascii=False,indent=2),flush=True)
    print('FARMS',json.dumps(feature_selection,ensure_ascii=False),flush=True)

if __name__=='__main__': main()
