"""All independent-run results; exact seed/farm pairing and residual diagnostics.

No prediction copies, no training or model selection, no averaging across fits.
"""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
import json,os,time,hashlib
import numpy as np
import pandas as pd

R=Path(__file__).resolve().parents[1];P=R.parents[1];O=P/'runs/independent_single_farm_20261021_v1'
A=R/'analysis';OUT=A/'rmse_review_20261024';OUT.mkdir(exist_ok=True)
OT=O/'seedwise_review_20261022/tables'
def read(path,**kw):return pd.read_csv(path,float_precision='round_trip',low_memory=False,**kw)
def save(df,name):
 df=pd.DataFrame(df);target=OUT/name;temp=target.with_name(target.name+'.tmp')
 df.to_csv(temp,index=False,float_format='%.17g',encoding='utf-8-sig',compression='gzip' if name.endswith('.gz') else None);os.replace(temp,target);return df
def mapped(f):return {'NEUTRAL_LAST':'W0','TRANSFORMER_FC':'TRANSFORMER_FC__W0','TIDE_FC':'TIDE_FC__W0'}.get(f,f)

if '--residuals-only' in __import__('sys').argv:
 data=read(OUT/'all_actual_results.csv')
 paired_count=int(read(OUT/'all_comparisons_by_seed.csv').n_farms.sum())
else:
 new=read(A/'results_per_farm_seed.csv');old=read(OT/'formal_and_ablation_results.csv');selected=read(OT/'selected_baseline_resolved.csv')
 old=pd.concat([old,selected],ignore_index=True).drop_duplicates(['family','site','seed','split'])
 old['arm']=old.family.map(mapped);old['run']='independent_20261021';new['run']='adaptive_20261022'
 columns=['arm','site','station_id','station_name','seed','split','year','source_job_id','mae','rmse','bias','n_origins','outside_0_1_fraction','stage','row_kind','run']
 data=pd.concat([new[new.stage!='tune'][columns],old[columns]],ignore_index=True).drop_duplicates(['arm','site','seed','year'],keep='first')
 assert not data.duplicated(['arm','site','seed','year']).any()
 save(data,'all_actual_results.csv')
 save(data.groupby(['arm','year','seed']).agg(farms=('site','nunique')).reset_index(),'coverage_by_seed.csv')
 BASE=['TRANSFORMER_FC__W0','TIDE_FC__W0','SELECTED_BASELINE','NBEATSX_FC','TIMEXER_FC','TFT_FC','TCN_FC','DLINEAR_FC','GRU_H','PATCHTST_H','CNN_W','B0','W0','TRANSFORMER_FC__S1']
 CONTROL=[('ST11','ST00'),('ST11','ST10'),('ST11','ST01'),('ST11','ST_NO_H'),('S3','S2'),('S4','S3'),('T2_7','T1_7'),('T2_8','T1_8'),('T1_8','T1_7'),('T2_8','T2_7'),('H1','W0'),('W0','H2'),('W0','CLIM_LAST')]
 for b in ['TRANSFORMER_FC','TIDE_FC']:CONTROL.extend([(b+'__ST11',b+'__ST00'),(b+'__ST11',b+'__S1')])
 COMPARISONS=sorted(set([(c,b) for c in data.arm.unique() for b in BASE if c!=b]+CONTROL))
 def pairs(frame,comparisons,extra=()):
  tables=[];keys=['site','seed','year',*extra]
  for candidate,reference in comparisons:
   x=frame[frame.arm==candidate].merge(frame[frame.arm==reference],on=keys,suffixes=('_candidate','_reference'),validate='one_to_one')
   for metric in ['mae','rmse']:
    x[metric+'_gain']=x[metric+'_reference']-x[metric+'_candidate']
   x['both_improve']=(x.mae_gain>0)&(x.rmse_gain>0)
   x['mae_improve_rmse_worse']=(x.mae_gain>0)&(x.rmse_gain<0)
   x['mae_worse_rmse_improve']=(x.mae_gain<0)&(x.rmse_gain>0)
   x['both_worse']=(x.mae_gain<0)&(x.rmse_gain<0)
   tables.append(x)
  return pd.concat(tables,ignore_index=True)
 def counts(frame,extra=()):
  out=[]
  for key,g in frame.groupby(['arm_candidate','arm_reference','year','seed',*extra]):
   row=dict(zip(['candidate','reference','year','seed',*extra],key));row['n_farms']=len(g)
   for metric in ['mae','rmse']:
    d=g[metric+'_gain'];row.update({metric+'_better':int((d>0).sum()),metric+'_worse':int((d<0).sum()),metric+'_tie':int((d==0).sum()),metric+'_missing':int(d.isna().sum())})
   for metric in ['both_improve','mae_improve_rmse_worse','mae_worse_rmse_improve','both_worse']:row[metric]=int(g[metric].sum())
   out.append(row)
  return pd.DataFrame(out)
 paired=save(pairs(data,COMPARISONS),'all_paired_actual_results.csv.gz');save(counts(paired),'all_comparisons_by_seed.csv')
 FOCUS=['E2C_CBH','W0','S1','S2','S3','S4','T1_7','T1_8','T2_7','T2_8','ST00','ST10','ST01','ST11','ST_NO_H','H1','H2','CLIM_LAST','H2_CLIM','TRANSFORMER_FC__S1','TRANSFORMER_FC__ST00','TRANSFORMER_FC__ST11','TIDE_FC__S1','TIDE_FC__ST00','TIDE_FC__ST11']
 SCOMP=sorted(set([(c,b) for c in FOCUS for b in ['TRANSFORMER_FC__W0','TIDE_FC__W0','W0'] if c!=b]+CONTROL))
 for fn,oldfn,key in [('horizon_per_farm_seed.csv',None,'horizon'),('scenario_per_farm_seed.csv.gz','scenario_results.csv.gz','scenario')]:
  f=read(A/fn)
  if oldfn:
   g=read(OT/oldfn);g=g[(g.stage!='tune')&(g.family=='E2C_CBH')];g['arm']=g.family.map(mapped)
  else:
   g=read(OT/'lead_results.csv.gz');g=g[(g.stage!='tune')&(g.family=='E2C_CBH')].copy()
   g['horizon']=np.select([g.lead_step<=4,g.lead_step<=12],['0_1h','1_3h'],default='3_6h');g['mse']=g.rmse**2
   g=g.groupby(['family','site','station_id','station_name','seed','year','horizon'],as_index=False).agg(mae=('mae','mean'),mse=('mse','mean'))
   g['rmse']=np.sqrt(g.mse);g['arm']=g.family.map(mapped)
  cols=['arm','site','station_id','station_name','seed','year',key,'mae','rmse']
  f=pd.concat([f[cols],g[cols]],ignore_index=True);ps=pairs(f,SCOMP,[key]);save(ps,f'{key}_paired_actual_results.csv.gz');save(counts(ps,[key]),f'{key}_comparisons_by_seed.csv')
 # Same-fit 2023/2024 direction transition, without averaging years.
 v=paired[paired.year==2023];t=paired[paired.year==2024]
 cross=v.merge(t,on=['arm_candidate','arm_reference','site','seed'],suffixes=('_2023','_2024'),validate='one_to_one')
 cr=[]
 for key,g in cross.groupby(['arm_candidate','arm_reference','seed']):
  for metric in ['mae','rmse']:
   a,b=g[metric+'_gain_2023'],g[metric+'_gain_2024'];cr.append(dict(candidate=key[0],reference=key[1],seed=key[2],metric=metric,n_farms=len(g),better_both=int(((a>0)&(b>0)).sum()),better_val_worse_test=int(((a>0)&(b<0)).sum()),worse_val_better_test=int(((a<0)&(b>0)).sum()),worse_both=int(((a<0)&(b<0)).sum())))
 save(cr,'cross_year_direction_counts.csv')
 print('ALL_METRIC_TABLES_COMPLETE',len(data),'fits_years',len(paired),'pairs',flush=True)

 paired_count=len(paired)

# Direct residual diagnostics cover all farms and all available seeds of the focal models.
DETAIL_ARMS=['E2C_CBH','W0','ST00','ST11','S2','TRANSFORMER_FC__W0','TIDE_FC__W0','TRANSFORMER_FC__S1']
DETAIL_PAIRS=[('ST11','TRANSFORMER_FC__W0'),('ST11','TIDE_FC__W0'),('ST11','W0'),('ST11','ST00'),('ST11','TRANSFORMER_FC__S1'),('W0','TRANSFORMER_FC__W0'),('ST00','TRANSFORMER_FC__W0'),('E2C_CBH','TRANSFORMER_FC__W0'),('S2','W0')]
test=data[(data.year==2024)&data.arm.isin(DETAIL_ARMS)]
def prediction_path(row):
 if row['run']=='independent_20261021' or row['row_kind']=='legacy_comparator':return O/'jobs'/row['source_job_id']/'test_prediction.npy'
 if row['row_kind']=='legacy_reuse':return R/'evaluation'/f"eval_reuse_W0_s{int(row['site']):02d}_r{int(row['seed'])}"/'matched_prediction.npy'
 return R/'evaluation'/('eval_'+row['source_job_id'])/'matched_prediction.npy'
def residual_task(site,seed,records):
 y=np.load(O/f'prepared/test/{site:02d}/target.npy').astype(np.float64);n=y.size;stats=[];lookup={};errors={};preds={};bins=[];pairrows=[]
 ident=dict(site=site,seed=seed,year=2024,station_id=records[0]['station_id'],station_name=records[0]['station_name'])
 for row in records:
  arm=row['arm'];pred=np.load(prediction_path(row)).astype(np.float64);assert pred.shape==y.shape
  e=pred-y;ae=np.abs(e);sq=e*e
  mae=float(ae.mean());mse=float(sq.mean());rmse=float(np.sqrt(mse));assert abs(mae-row['mae'])<1e-12 and abs(rmse-row['rmse'])<1e-12
  q=np.quantile(ae,[.5,.9,.95,.99,.999]);tail90=ae>=q[1];tail99=ae>=q[3];outside=(pred<0)|(pred>1)
  clip=np.clip(pred,0,1);ce=clip-y
  stat=dict(**ident,arm=arm,n_elements=n,mae=mae,rmse=rmse,bias=float(e.mean()),centered_error_variance=float(e.var()),
            q50_abs=float(q[0]),q90_abs=float(q[1]),q95_abs=float(q[2]),q99_abs=float(q[3]),q999_abs=float(q[4]),max_abs=float(ae.max()),
            top10pct_sse_fraction=float(sq[tail90].sum()/sq.sum()),top1pct_sse_fraction=float(sq[tail99].sum()/sq.sum()),
            prediction_min=float(pred.min()),prediction_max=float(pred.max()),outside_0_1_fraction=float(outside.mean()),outside_sse_fraction=float(sq[outside].sum()/sq.sum()),
            clipped_mae_diagnostic=float(np.abs(ce).mean()),clipped_rmse_diagnostic=float(np.sqrt((ce*ce).mean())),truth_min=float(y.min()),truth_max=float(y.max()))
  stats.append(stat);lookup[arm]=stat;errors[arm]=e;preds[arm]=pred
 for candidate,reference in DETAIL_PAIRS:
  if candidate not in errors or reference not in errors:continue
  c,b=errors[candidate],errors[reference];ca,ba=np.abs(c),np.abs(b);delta=c*c-b*b;lc,lb=lookup[candidate],lookup[reference]
  row=dict(**ident,candidate=candidate,reference=reference,mae_gain=lb['mae']-lc['mae'],rmse_gain=lb['rmse']-lc['rmse'],
       mse_increase=float(delta.mean()),bias_squared_increase=lc['bias']**2-lb['bias']**2,
       centered_error_variance_increase=lc['centered_error_variance']-lb['centered_error_variance'],
       sample_abs_error_better_fraction=float((ca<ba).mean()),clipped_mae_gain=lb['clipped_mae_diagnostic']-lc['clipped_mae_diagnostic'],
       clipped_rmse_gain=lb['clipped_rmse_diagnostic']-lc['clipped_rmse_diagnostic'])
  for field in ['q50_abs','q90_abs','q95_abs','q99_abs','q999_abs','max_abs','top10pct_sse_fraction','outside_0_1_fraction']:
   row[field+'_candidate']=lc[field];row[field+'_reference']=lb[field]
  assert abs(row['mse_increase']-row['bias_squared_increase']-row['centered_error_variance_increase'])<1e-12
  sharedmax=np.maximum(ca,ba);boundaries=[(0,.1),(.1,.2),(.2,.4),(.4,np.inf)];parts=[]
  for lower,upper in boundaries:
   mask=(sharedmax>=lower)&(sharedmax<upper);count=int(mask.sum());contrib=float(delta[mask].sum()/n);parts.append(contrib)
   bins.append(dict(**ident,candidate=candidate,reference=reference,partition='symmetric_max_abs_error',lower_cf=lower,upper_cf=upper if np.isfinite(upper) else None,
                    n_elements=count,fraction=count/n,mse_increase_contribution=contrib,mae_increase_contribution=float((ca[mask]-ba[mask]).sum()/n)))
  assert abs(sum(parts)-row['mse_increase'])<1e-12
  for horizon,a,z in [('0_1h',0,4),('1_3h',4,12),('3_6h',12,24)]:
   row['mse_increase_contribution_'+horizon]=float(delta[:,a:z].sum()/n)
  outside=((preds[candidate]<0)|(preds[candidate]>1)|(preds[reference]<0)|(preds[reference]>1))
  row['mse_increase_contribution_either_outside']=float(delta[outside].sum()/n)
  row['mse_increase_contribution_both_inside']=float(delta[~outside].sum()/n)
  pairrows.append(row)
 return stats,pairrows,bins
tasks=[(int(site),int(seed),g.to_dict('records')) for (site,seed),g in test.groupby(['site','seed'])]
stats=[];detail=[];bins=[];start=last=time.monotonic()
with ThreadPoolExecutor(max_workers=6) as pool:
 for i,future in enumerate(as_completed([pool.submit(residual_task,*args) for args in tasks]),1):
  s,d,b=future.result();stats.extend(s);detail.extend(d);bins.extend(b)
  if time.monotonic()-last>20:print('RESIDUALS',i,'/',len(tasks),'seconds',round(time.monotonic()-start),flush=True);last=time.monotonic()
save(stats,'residual_statistics_per_farm_seed.csv');save(detail,'residual_pair_diagnostics.csv');save(bins,'squared_error_partition_per_farm_seed.csv')
summary=[]
dd=pd.DataFrame(detail)
for key,g in dd.groupby(['candidate','reference','seed']):
 worse=g.rmse_gain<0;trade=(g.mae_gain>0)&worse
 for name,mask in [('all',np.ones(len(g),bool)),('rmse_worse',worse),('mae_better_rmse_worse',trade)]:
  z=g.loc[mask];summary.append(dict(candidate=key[0],reference=key[1],seed=key[2],subset=name,n_farms=len(z),
   q50_better=int((z.q50_abs_candidate<z.q50_abs_reference).sum()),q90_worse=int((z.q90_abs_candidate>z.q90_abs_reference).sum()),q99_worse=int((z.q99_abs_candidate>z.q99_abs_reference).sum()),
   variance_increased=int((z.centered_error_variance_increase>0).sum()),variance_more_than_bias=int((z.centered_error_variance_increase>z.bias_squared_increase).sum()),
   both_inside_contribution_positive=int((z.mse_increase_contribution_both_inside>0).sum()),clipped_rmse_still_worse=int((z.clipped_rmse_gain<0).sum()),
   long_horizon_contribution_positive=int((z.mse_increase_contribution_3_6h>0).sum())))
save(summary,'residual_diagnosis_counts_by_seed.csv')
check=dict(passed=True,source_training_jobs=dict(previous=9500,new=8016),source_new_evaluation_jobs=7840,
           formal_results_rows=len(data),paired_rows=paired_count,residual_models_checked=len(stats),residual_pairs_checked=len(detail),
           all_focal_predictions_recompute_original_metrics=True,mse_decomposition_verified=True,no_cross_farm_seed_averages=True,
           baseline_training_loss='MAE',checkpoint_selection='validation_MAE',raw_predictions_copied=False,
           analysis_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
(OUT/'CHECK.json').write_text(json.dumps(check,indent=2)+'\n');print('FINISHED',check,flush=True)
