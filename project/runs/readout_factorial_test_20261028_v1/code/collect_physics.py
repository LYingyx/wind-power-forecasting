from settings import *
import csv,collections
import numpy as np
_CACHE={}
META={n:np.load(CACHE/'static'/f'{n}.npy') for n in ['station_id','station_name','station_capacity_mw']}

def csv_write(path,rows):
    if not rows:return
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    with tmp.open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    os.replace(tmp,path)

def collect(export=True):
    specs=json.loads((RUN/'manifest.json').read_text());byid={s['id']:s for s in specs}
    done={p.parent.name for p in JOBS.glob('*/SUCCESS.json')}&byid.keys()
    failed=({p.parent.name for p in JOBS.glob('*/FAILURE.json')}&byid.keys())-done
    state=dict(timestamp=now(),planned=len(specs),completed=len(done),failed=len(failed),failed_jobs=sorted(failed),active=[],pending=len(specs)-len(done)-len(failed),training_completed=0,replay_completed=len(done),test_year_evaluated=bool(done))
    if export:
        for jid in sorted(done):
            if jid in _CACHE:continue
            p=JOBS/jid;s=byid[jid];m=json.loads((p/'metrics.json').read_text())['selections']['mae']['metrics']
            diag=json.loads((p/'diagnostics.json').read_text())['rows']
            cap=float(META['station_capacity_mw'][s['site']])
            common=dict(configuration=s['configuration'],site=s['site'],station_id=str(META['station_id'][s['site']]),station_name=str(META['station_name'][s['site']]),capacity_mw=cap,seed=s['seed'],year=2024)
            rows=[dict(**common,horizon=hz,n=x['n'],mae_cf=x['mae'],rmse_cf=x['rmse'],bias_cf=x['bias'],mae_mw=x['mae']*cap,rmse_mw=x['rmse']*cap) for hz,x in [('all',m),*m['horizons'].items()]]
            leads=[dict(**common,lead_minutes=(i+1)*15,mae_cf=a,rmse_cf=b) for i,(a,b) in enumerate(zip(m['lead_mae'],m['lead_rmse']))]
            _CACHE[jid]=(rows,[dict(**common,**d) for d in diag],leads)
        rows=[x for v in _CACHE.values() for x in v[0]]
        csv_write(RUN/'analysis/per_farm_seed_year.csv',rows)
        csv_write(RUN/'analysis/scenarios_2024.csv',[x for v in _CACHE.values() for x in v[1]])
        csv_write(RUN/'analysis/leads_2024.csv',[x for v in _CACHE.values() for x in v[2]])
        farms=collections.defaultdict(list);groups=collections.defaultdict(list)
        for x in rows:farms[x['station_id']].append(x);groups[x['configuration'],x['station_id'],x['horizon']].append(x)
        for site,rs in farms.items():csv_write(RUN/'analysis/stations'/f'{site}.csv',rs)
        summary=[dict(configuration=c,station_id=f,horizon=h,seeds=len(rs),**{k:float(np.mean([x[k] for x in rs])) for k in ['mae_cf','rmse_cf','mae_mw','rmse_mw']}) for (c,f,h),rs in sorted(groups.items())]
        csv_write(RUN/'analysis/per_farm_summary.csv',summary)
        macro=collections.defaultdict(list)
        for x in summary:macro[x['configuration'],x['horizon']].append(x)
        csv_write(RUN/'analysis/model_summary.csv',[dict(configuration=c,horizon=h,farms=len(rs),min_seeds=min(x['seeds'] for x in rs),max_seeds=max(x['seeds'] for x in rs),mae_cf=float(np.mean([x['mae_cf'] for x in rs])),rmse_cf=float(np.mean([x['rmse_cf'] for x in rs]))) for (c,h),rs in sorted(macro.items())])
    atomic_json(RUN/'STATUS.json',state)
    return state

if __name__=='__main__':print(json.dumps(collect()))
