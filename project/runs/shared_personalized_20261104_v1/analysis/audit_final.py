from pathlib import Path
import sys,json,collections,datetime
import numpy as np
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'code'));from settings import sha,atomic_json,now
manifest=json.loads((R/'manifest.json').read_text());cohort=json.loads((R/'assets/COHORT.json').read_text());nf=cohort['count'];records=[];groups=collections.defaultdict(list)
for spec in manifest:
 p=R/'jobs'/spec['id'];d=json.loads((p/'metrics.json').read_text());assert (p/'SUCCESS.json').exists();assert not (p/'FAILURE.json').exists();assert d['requested_spec']==spec;assert not d['smoke_only']
 expected_year=2022 if spec['stage']=='reference' and spec['fold']>=0 else 2023;assert d['evaluation_year']==expected_year
 assert sha(p/'best.pt')==d['checkpoint_sha256']
 if 'reference_sha256' in d:assert sha(p/'reference_prediction.npy')==d['reference_sha256']
 if 'selection_sha256' in d['spec']:assert sha(R/'assets/selected'/f"{spec['config_from']}.json")==d['spec']['selection_sha256']
 m=d['metrics'];mae=np.asarray(m['per_farm_mae']);rmse=np.asarray(m['per_farm_rmse']);lm=np.asarray(m['lead_mae']);lr=np.asarray(m['lead_rmse']);cnt=np.asarray(m['counts'])
 assert mae.shape==rmse.shape==(nf,) and lm.shape==lr.shape==(nf,24) and cnt.shape==(nf,)
 for v in [mae,rmse,lm,lr,np.asarray(m['high_power_per_farm_rmse'])]:assert np.isfinite(v).all() and (v>=0).all()
 assert (cnt>0).all() and (cnt==cnt[0]).all()
 assert np.isclose(mae.mean(),m['mae'],rtol=1e-12) and np.isclose(rmse.mean(),m['rmse'],rtol=1e-12)
 assert np.allclose(lm.mean(1),mae,rtol=1e-12) and np.allclose(np.sqrt((lr**2).mean(1)),rmse,rtol=1e-12)
 history=json.loads((p/'history.json').read_text());best=min(history,key=lambda z:z['validation_mae']);assert best['epoch']==d['checkpoint_epoch']
 if expected_year==2023:assert np.isclose(best['validation_mae'],m['mae'],rtol=1e-7)
 if spec['stage'] in ['baseline','mechanism']:groups[spec['group']].append(d)
 records.append(dict(id=spec['id'],stage=spec['stage'],evaluation_year=d['evaluation_year'],best_sha256=d['checkpoint_sha256'],best_epoch=d['checkpoint_epoch'],epochs=len(history),timestamp=d['timestamp'],count_per_farm=int(cnt[0])))
for group,ds in groups.items():assert sorted(d['spec']['seed'] for d in ds)==[42,43,44,45,46],group
counts={x['count_per_farm'] for x in records if x['stage'] in ['baseline','mechanism']};assert len(counts)==1
freeze=json.loads((R/'SCIENCE_FREEZE.json').read_text());before=json.loads((R/'operations/final_collection_20261104/SCIENCE_FREEZE.json').read_text());changed=[f for f,h in before['files'].items() if sha(R/f)!=h];assert changed==['operations/collect.py'],changed
for f,h in freeze['files'].items():assert sha(R/f)==h,f
# Immutable data hashes include both protocol caches, station topology and training-only normalization.
for f,h in json.loads((R/'DATA_FREEZE.json').read_text())['files'].items():assert sha(R/f)==h,f
for seed in range(42,47):
 folder=R/'data'/f'reference_r{seed}';ready=json.loads((folder/'READY.json').read_text());assert ready['passed'] and ready['seed']==seed
 for f,h in ready['files'].items():assert sha(folder/f)==h,(seed,f)
 for j,h in ready['sources'].items():assert sha(R/'jobs'/j/'best.pt')==h,(seed,j)
# Independently recalculate every reported pair and frozen gate without the collector.
gate=json.loads((R/'analysis/GATES.json').read_text());checked=set();pairs={}
for c in gate['comparisons']:
 a,b=c['candidate'],c['reference'];passed=True
 for metric in ['mae','rmse']:
  x=np.array([d['metrics']['per_farm_'+metric] for d in groups[a]]).mean(0);y=np.array([d['metrics']['per_farm_'+metric] for d in groups[b]]).mean(0)
  delta=100*(x.mean()/y.mean()-1);wins=int((x<y).sum());assert np.isclose(delta,c[metric]['change_percent'],atol=1e-10) and wins==c[metric]['farms_improved']
  passed&=delta<=-.1 and wins>nf/2
 x=np.mean([d['metrics']['high_power_rmse'] for d in groups[a]]);y=np.mean([d['metrics']['high_power_rmse'] for d in groups[b]]);passed&=100*(x/y-1)<=.1
 assert bool(passed)==c['passed'];pairs[a,b]=bool(passed);checked.add((a,b))
strongest=min(['TF_R3','TIDE_R3','TFT_R3'],key=lambda g:np.mean([d['metrics']['mae'] for d in groups[g]]));eligible=[a for a,b in [('M1_CB','M4_WRONG_CB'),('M2_CBH','M4H_WRONG_CBH')] if all(pairs[a,ref] for ref in ['W0_R3',b,strongest])];assert eligible==gate['eligible']
report=dict(timestamp=now(),passed=True,planned=len(manifest),verified=len(records),by_stage=dict(collections.Counter(s['stage'] for s in manifest)),formal_groups=len(groups),farms=nf,seeds=list(range(42,47)),val_count_per_farm=next(iter(counts)),last_training_result=max(x['timestamp'] for x in records),checked_pairs=len(checked),gate=gate['stage'],eligible=eligible,repairs=['Convert NumPy comparison bool to Python bool for JSON serialization; no metric/formula/training changes'],unchanged_science_and_data_hashes=True,records=records)
atomic_json(R/'analysis/FINAL_COMPLETENESS_AUDIT.json',report);print(json.dumps({k:v for k,v in report.items() if k!='records'},ensure_ascii=False,indent=2))
