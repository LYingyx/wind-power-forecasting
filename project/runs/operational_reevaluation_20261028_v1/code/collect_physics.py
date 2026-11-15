from settings import *
import csv,collections
import numpy as np
_CACHE={};_LAST=-1
META={n:np.load(CACHE/'static'/f'{n}.npy') for n in ['station_id','station_name','station_capacity_mw']}

def rows_for(spec,year,m):
 site=spec['site'];cap=float(META['station_capacity_mw'][site]);out=[]
 for horizon,score in [('all',m),*m['horizons'].items()]:
  out.append(dict(configuration=spec['configuration'],site=site,station_id=str(META['station_id'][site]),station_name=str(META['station_name'][site]),capacity_mw=cap,seed=spec.get('seed',-1),year=year,horizon=horizon,n=score['n'],mae_cf=score['mae'],rmse_cf=score['rmse'],bias_cf=score['bias'],mae_mw=score['mae']*cap,rmse_mw=score['rmse']*cap,bias_mw=score['bias']*cap))
 return out

def csv_write(path,rows):
 if not rows:return
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+f'.{os.getpid()}.tmp')
 with tmp.open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 os.replace(tmp,path)

def collect(export=True):
 global _LAST
 specs=json.loads((RUN/'manifest.json').read_text());byid={s['id']:s for s in specs}
 completed={p.parent.name for p in JOBS.glob('*/SUCCESS.json')}&byid.keys();failed=({p.parent.name for p in JOBS.glob('*/FAILURE.json')}&byid.keys())-completed
 if not export:
  state=dict(timestamp=now(),planned=len(specs),completed=len(completed),failed=len(failed),failed_jobs=sorted(failed),active=[],pending=len(specs)-len(completed)-len(failed),training_completed=sum(byid[j]['task']=='train' for j in completed),replay_completed=sum(byid[j]['task']=='evaluate' for j in completed),test_year_evaluated=bool(completed))
  atomic_json(RUN/'STATUS.json',state);return state
 # Export parent validation results as they arrive, including during dependency wait.
 for folder,kind in [(PARENT/'jobs','parent'),(JOBS,'next')]:
  for receipt in folder.glob('*/SUCCESS.json'):
   key=(kind,receipt.parent.name)
   if key in _CACHE:continue
   path=receipt.parent
   if kind=='next' and path.name not in completed:continue
   s=json.loads((path/'spec.json').read_text());m=json.loads((path/'metrics.json').read_text());rows=[];scenario=[];leads=[]
   if kind=='parent' or s['task']=='train':rows+=rows_for(s,2023,m['selections']['mae']['metrics'])
   if kind=='next':
    test=json.loads((path/'test.json').read_text());rows+=rows_for(s,2024,test['metrics'])
    for x in test['diagnostics']:scenario.append(dict(configuration=s['configuration'],station_id=str(META['station_id'][s['site']]),site=s['site'],seed=s['seed'],year=2024,**x))
    for i,(a,b) in enumerate(zip(test['metrics']['lead_mae'],test['metrics']['lead_rmse'])):leads.append(dict(configuration=s['configuration'],station_id=str(META['station_id'][s['site']]),site=s['site'],seed=s['seed'],year=2024,lead_minutes=(i+1)*15,mae_cf=a,rmse_cf=b))
   _CACHE[key]=(rows,scenario,leads)
 p=RUN/'analysis/PERSISTENCE.json'
 if p.exists() and ('persistence','all') not in _CACHE:
  d=json.loads(p.read_text());_CACHE['persistence','all']=([r for x in d['rows'] for r in rows_for(dict(configuration='PERSISTENCE',site=x['site'],seed=-1),x['year'],x['metrics'])],[],[])
 if len(_CACHE)!=_LAST:
  rows=[r for v in _CACHE.values() for r in v[0]];scenario=[r for v in _CACHE.values() for r in v[1]];leads=[r for v in _CACHE.values() for r in v[2]]
  csv_write(RUN/'analysis/per_farm_seed_year.csv',rows);csv_write(RUN/'analysis/scenarios_2024.csv',scenario);csv_write(RUN/'analysis/leads_2024.csv',leads)
  stations=collections.defaultdict(list);groups=collections.defaultdict(list)
  for x in rows:stations[x['station_id']].append(x);groups[x['configuration'],x['station_id'],x['year'],x['horizon']].append(x)
  for site,rs in stations.items():csv_write(RUN/'analysis/stations'/f'{site}.csv',rs)
  summary=[]
  for (config,station,year,horizon),rs in sorted(groups.items()):
   summary.append(dict(configuration=config,station_id=station,station_name=rs[0]['station_name'],year=year,horizon=horizon,seeds=len(rs),**{k+'_mean':float(np.mean([r[k] for r in rs])) for k in ['mae_cf','rmse_cf','bias_cf','mae_mw','rmse_mw']},**{k+'_std':float(np.std([r[k] for r in rs],ddof=1)) if len(rs)>1 else 0. for k in ['mae_cf','rmse_cf']}))
  csv_write(RUN/'analysis/per_farm_summary.csv',summary)
  bymodel=collections.defaultdict(list)
  for x in summary:bymodel[x['configuration'],x['year'],x['horizon']].append(x)
  macro=[dict(configuration=c,year=y,horizon=h,farms=len(rs),min_seeds=min(x['seeds'] for x in rs),max_seeds=max(x['seeds'] for x in rs),mae_cf=float(np.mean([x['mae_cf_mean'] for x in rs])),rmse_cf=float(np.mean([x['rmse_cf_mean'] for x in rs]))) for (c,y,h),rs in sorted(bymodel.items())]
  csv_write(RUN/'analysis/model_summary.csv',macro)
  text=['# 业务协议：逐场站基线与2024复评',f'更新：{now()}。先站内平均种子，再风场等权平均；2024剔除11月。','|模型|年份|风场数|种子范围|MAE(CF)|RMSE(CF)|','|---|---:|---:|---:|---:|---:|']
  text += [f"|{x['configuration']}|{x['year']}|{x['farms']}|{x['min_seeds']}–{x['max_seeds']}|{x['mae_cf']:.6f}|{x['rmse_cf']:.6f}|" for x in macro if x['horizon']=='all']
  text += ['','基线超参数继承此前2023选择，不称为新业务协议下重新调优的最优配置。缺失行表示任务未完成，不能用部分站点主张整体优势。','完整逐站逐种子表：per_farm_seed_year.csv；逐站文件：stations/<station_id>.csv；分时效：leads_2024.csv；场景：scenarios_2024.csv。']
  (RUN/'analysis/REPORT_CN.md').write_text('\n'.join(text)+'\n');_LAST=len(_CACHE)
 state=dict(timestamp=now(),planned=len(specs),completed=len(completed),failed=len(failed),failed_jobs=sorted(failed),active=[],pending=len(specs)-len(completed)-len(failed),training_completed=sum(byid[j]['task']=='train' for j in completed),replay_completed=sum(byid[j]['task']=='evaluate' for j in completed),test_year_evaluated=bool(completed))
 atomic_json(RUN/'STATUS.json',state);return state
if __name__=='__main__':print(json.dumps(collect()))
