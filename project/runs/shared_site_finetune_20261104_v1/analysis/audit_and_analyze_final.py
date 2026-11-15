from pathlib import Path
import sys,json,csv,collections,datetime
import numpy as np
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'code'));from settings import sha,atomic_json,now
manifest=json.loads((R/'manifest.json').read_text());cohort=json.loads((R/'assets/COHORT.json').read_text());nf=cohort['count'];seeds=list(range(42,47));bygroup=collections.defaultdict(dict);raw={};audits=[]
# Frozen code, selected source checkpoints, data and normalization are unchanged.
for rel,digest in json.loads((R/'SCIENCE_FREEZE.json').read_text())['files'].items():assert sha(R/rel)==digest,rel
for rel,digest in json.loads((R/'SOURCE_DATA_FREEZE.json').read_text())['files'].items():assert sha(R/rel)==digest,rel
sources={}
for family in ['W0','TF']:
 for seed in seeds:
  d=json.loads((R/'pretrained'/f'{family}_r{seed}.metrics.json').read_text());assert sha(R/'pretrained'/f'{family}_r{seed}.pt')==d['checkpoint_sha256'];sources[family,seed]=d
  m=d['metrics']
  for site in range(nf):
   lm=np.asarray(m['lead_mae'][site]);lr=np.asarray(m['lead_rmse'][site]);horizons={label:dict(mae=float(lm[a:b].mean()),rmse=float(np.sqrt((lr[a:b]**2).mean()))) for label,a,b in [('0_1h',0,4),('1_3h',4,12),('3_6h',12,24)]}
   bygroup[family+'_P0_SHARED'][seed,site]=dict(mae=m['per_farm_mae'][site],rmse=m['per_farm_rmse'][site],high_power_rmse=m['high_power_per_farm_rmse'][site],lead_mae=lm.tolist(),lead_rmse=lr.tolist(),horizons=horizons,selected_epoch=0,high_count=m['high_power_counts'][site])
for spec in manifest:
 j=spec['id'];p=R/'jobs'/j;assert (p/'SUCCESS.json').exists() and not (p/'FAILURE.json').exists();d=json.loads((p/'metrics.json').read_text());raw[j]=d
 assert d['requested_spec']==spec and d['spec']==spec and not d['smoke_only'];assert d['evaluation_year']==2023 and d['station_id']==cohort['station_ids'][spec['station']]
 assert d['source_sha256']==sources[spec['family'],spec['seed']]['checkpoint_sha256'];assert sha(p/'best.pt')==d['checkpoint_sha256'];m=d['metrics'];b=d['baseline_metrics'];assert m['counts']==b['counts']==[29647]
 assert len(m['lead_mae'])==len(m['lead_rmse'])==1 and len(m['lead_mae'][0])==len(m['lead_rmse'][0])==24
 for key in ['mae','rmse','high_power_rmse']:assert np.isfinite(m[key]) and m[key]>=0
 assert np.isclose(np.mean(m['lead_mae']),m['mae'],atol=1e-12);assert np.isclose(np.sqrt(np.mean(np.asarray(m['lead_rmse'])**2)),m['rmse'],atol=1e-12)
 s=spec['station'];seed=spec['seed'];pre=sources[spec['family'],seed]['metrics']
 for key in ['mae','rmse']:assert np.isclose(b[key],pre['per_farm_'+key][s],rtol=2e-5,atol=2e-7)
 h=json.loads((p/'history.json').read_text());candidates=[(0,b['mae'])]+[(v['epoch'],v['validation_mae']) for v in h];selected=min(candidates,key=lambda z:z[1])[0]
 assert selected==d['checkpoint_epoch'];assert np.isclose(m['mae'],min(v for e,v in candidates),atol=1e-9)
 assert len(h)==d['trained_epochs']<=spec['epochs'];assert h[-1]['epoch']==len(h)
 bygroup[spec['group']][seed,s]=dict(mae=m['mae'],rmse=m['rmse'],high_power_rmse=m['high_power_rmse'],lead_mae=m['lead_mae'][0],lead_rmse=m['lead_rmse'][0],horizons=m['horizons'],selected_epoch=selected,high_count=m['high_power_counts'][0],trained_epochs=d['trained_epochs'])
 audits.append(dict(id=j,seed=seed,station=s,checkpoint_epoch=selected,trained_epochs=len(h),checkpoint_sha256=d['checkpoint_sha256'],timestamp=d['timestamp']))
assert len(raw)==1100
for g,values in bygroup.items():assert set(values)=={(seed,s) for seed in seeds for s in range(nf)},g

def array(g,key):return np.array([[bygroup[g][seed,s][key] for s in range(nf)] for seed in seeds])
def comparison(a,b):
 d=dict(candidate=a,reference=b)
 for key in ['mae','rmse','high_power_rmse']:
  x=array(a,key);y=array(b,key);xf=x.mean(0);yf=y.mean(0);delta=100*(x.mean()/y.mean()-1)
  d[key]=dict(change_percent=float(delta),wins=int((xf<yf).sum()),wins_absolute_difference_over_1e_7=int((yf-xf>1e-7).sum()),per_seed_change_percent=(100*(x.mean(1)/y.mean(1)-1)).tolist(),leave_one_seed_out_change_percent={str(seed):float(100*(np.delete(x,i,axis=0).mean()/np.delete(y,i,axis=0).mean()-1)) for i,seed in enumerate(seeds)})
  rng=np.random.default_rng(1106);draws=[]
  for _ in range(5000):
   si=rng.integers(5,size=5);fi=rng.integers(nf,size=nf);draws.append(100*(x[np.ix_(si,fi)].mean()/y[np.ix_(si,fi)].mean()-1))
  d[key]['paired_seed_and_farm_bootstrap95_percent']=np.quantile(draws,[.025,.975]).tolist()
 d['horizons']={}
 for hz in ['0_1h','1_3h','3_6h']:
  d['horizons'][hz]={key:float(100*(np.mean([v['horizons'][hz][key] for v in bygroup[a].values()])/np.mean([v['horizons'][hz][key] for v in bygroup[b].values()])-1)) for key in ['mae','rmse']}
 return d
pairs=[(f'{f}_{m}',f'{f}_P0_SHARED') for f in ['W0','TF'] for m in ['P1_FULL','P2_HEAD']]+[(f'W0_{m}',f'TF_{m}') for m in ['P0_SHARED','P1_FULL','P2_HEAD']]+[(f'{f}_P1_FULL',f'{f}_P2_HEAD') for f in ['W0','TF']]
comparisons=[comparison(a,b) for a,b in pairs]
gate=json.loads((R/'analysis/GATES.json').read_text())
for c in gate['comparisons']:
 d=next(v for v in comparisons if v['candidate']==c['candidate'] and v['reference']==c['reference'])
 for key in ['mae','rmse']:assert np.isclose(c[key]['change_percent'],d[key]['change_percent'],atol=1e-10);assert c[key]['farms_improved']==d[key]['wins']
summary=[];per_seed=[];farmrows=[];horizonrows=[];leadrows=[]
for g,values in bygroup.items():
 rec=dict(group=g,mae=float(array(g,'mae').mean()),rmse=float(array(g,'rmse').mean()),high_power_rmse=float(array(g,'high_power_rmse').mean()),epoch0_count=sum(v['selected_epoch']==0 for v in values.values()),epoch0_percent=100*np.mean([v['selected_epoch']==0 for v in values.values()]),all5seeds_epoch0_farms=sum(all(values[seed,s]['selected_epoch']==0 for seed in seeds) for s in range(nf)),selected_epoch_median=float(np.median([v['selected_epoch'] for v in values.values()])),selected_epoch30_count=sum(v['selected_epoch']==30 for v in values.values()))
 if '_P0_' not in g:
  ds=[d for d in raw.values() if d['spec']['group']==g];rec.update(trainable_parameters=ds[0]['trainable_parameter_count'],total_parameters=ds[0]['parameter_count'],median_trained_epochs=float(np.median([d['trained_epochs'] for d in ds])))
 summary.append(rec)
 for seed in seeds:per_seed.append(dict(group=g,seed=seed,**{k:float(np.mean([values[seed,s][k] for s in range(nf)])) for k in ['mae','rmse','high_power_rmse']},epoch0_count=sum(values[seed,s]['selected_epoch']==0 for s in range(nf))))
 for s in range(nf):
  farmrows.append(dict(group=g,station=s,station_id=cohort['station_ids'][s],name=cohort['names'][s],**{k:float(np.mean([values[seed,s][k] for seed in seeds])) for k in ['mae','rmse','high_power_rmse']},epoch0_count=sum(values[seed,s]['selected_epoch']==0 for seed in seeds)))
 for hz in ['0_1h','1_3h','3_6h']:horizonrows.append(dict(group=g,horizon=hz,**{k:float(np.mean([v['horizons'][hz][k] for v in values.values()])) for k in ['mae','rmse']}))
 for lead in range(24):leadrows.append(dict(group=g,lead_minutes=(lead+1)*15,**{k:float(np.mean([v['lead_'+k][lead] for v in values.values()])) for k in ['mae','rmse']}))
for name,rows in [('per_seed_final.csv',per_seed),('per_farm_final.csv',farmrows),('horizon_summary.csv',horizonrows),('lead_summary.csv',leadrows)]:
 with (R/'analysis'/name).open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
audit=dict(timestamp=now(),passed=True,planned=1100,verified=1100,farms=55,seeds=seeds,source_references=550,evaluation_year=2023,samples_per_station=29647,last_training_timestamp=max(d['timestamp'] for d in raw.values()),code_data_and_checkpoint_hashes_verified=True,selected_epoch_recomputed=True,rows=audits)
atomic_json(R/'analysis/FINAL_AUDIT.json',audit);analysis=dict(timestamp=now(),summary=summary,comparisons=comparisons,horizons=horizonrows,bootstrap_note='Exploratory paired resampling of 5 seeds and 55 farms; spatial dependence and validation-selection bias are not corrected. Not a confirmatory confidence interval.',scope='2023 selection only; no 2024 evaluation; no independent-from-scratch matched baseline');atomic_json(R/'analysis/FINAL_ANALYSIS.json',analysis)
print('AUDIT',json.dumps({k:v for k,v in audit.items() if k!='rows'},ensure_ascii=False));print('SUMMARY',json.dumps(summary,ensure_ascii=False))
for c in comparisons:print('COMPARE',json.dumps(c,ensure_ascii=False))
