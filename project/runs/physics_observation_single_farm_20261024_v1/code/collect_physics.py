"""Small reproducible result tables, never aggregate farm/seed scores."""
from settings import *
import csv,collections

def collect():
    manifest=json.loads((RUN/'manifest.json').read_text());rows=[];complete=[];fail=[];active=[]
    for spec in manifest:
        folder=JOBS/spec['id']
        if (folder/'SUCCESS.json').exists():
            complete.append(spec['id']);result=json.loads((folder/'metrics.json').read_text())
            for criterion,s in result['selections'].items():
                for horizon,metrics in [('all',s['metrics']),*s['metrics']['horizons'].items()]:
                    rows.append({**spec,'selection':criterion,'horizon':horizon,'epoch':s['epoch'],
                        **{k:metrics[k] for k in ['n','mae','rmse','bias']},'checkpoint':str(folder/s['checkpoint'])})
        elif (folder/'FAILURE.json').exists():fail.append(spec['id'])
    for p in (RUN/'schedulers').glob('*.json') if (RUN/'schedulers').exists() else []:
        try:
            s=json.loads(p.read_text());active.extend(s.get('active',[]))
        except (OSError,ValueError):pass
    out=RUN/'analysis';out.mkdir(exist_ok=True)
    def table(name,values):
        if not values:return
        path=out/name;tmp=path.with_name(path.name+f'.{os.getpid()}.tmp')
        with tmp.open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(values[0]));writer.writeheader();writer.writerows(values)
        os.replace(tmp,path)
    table('new_actual_per_farm_seed.csv',rows)
    index={(r['site'],r['seed'],r['backbone'],r['loss'],r['variant'],r['capacity_control'],r['delay_minutes'],r['selection'],r['horizon']):r for r in rows}
    pairs=[]
    for row in rows:
        references=['BASE']
        mapping={'DELTA':['REF'],'SHIFT_FIXED':['SHIFT_ZERO','HIST'],'SHIFT_SAMPLE':['SHIFT_FIXED','HIST'],
            'CURVE_FREE':['PC_INPUT'],'CURVE_BIAS':['CURVE_FREE','PC_INPUT'],'NEIGH_ADVECT':['NEIGH_GENERIC']}
        references+=mapping.get(row['variant'],[])
        reference_keys=[(row['site'],row['seed'],row['backbone'],row['loss'],ref,False,row['delay_minutes'],row['selection'],row['horizon']) for ref in references]
        if row['loss']!='MAE':reference_keys.append((row['site'],row['seed'],row['backbone'],'MAE',row['variant'],row['capacity_control'],row['delay_minutes'],row['selection'],row['horizon']))
        if row['backbone']=='W0':reference_keys.append((row['site'],row['seed'],'TF',row['loss'],row['variant'],row['capacity_control'],row['delay_minutes'],row['selection'],row['horizon']))
        if not row['capacity_control']:reference_keys.append((row['site'],row['seed'],row['backbone'],row['loss'],row['variant'],True,row['delay_minutes'],row['selection'],row['horizon']))
        for key in dict.fromkeys(reference_keys):
            other=index.get(key)
            if other is None or other['id']==row['id']:continue
            pairs.append(dict(site=row['site'],seed=row['seed'],backbone=row['backbone'],loss=row['loss'],selection=row['selection'],horizon=row['horizon'],candidate=row['id'],reference=other['id'],candidate_mae=row['mae'],reference_mae=other['mae'],candidate_rmse=row['rmse'],reference_rmse=other['rmse'],mae_difference=row['mae']-other['mae'],rmse_difference=row['rmse']-other['rmse'],mae_better=row['mae']<other['mae'],rmse_better=row['rmse']<other['rmse']))
    table('new_paired_per_farm_seed.csv',pairs)
    status=dict(timestamp=now(),planned=len(manifest),completed=len(complete),failed=len(fail),active=active,pending=len(manifest)-len(complete)-len(fail)-len({s['id'] for s in active}),failed_jobs=fail,
        source_freeze=(RUN/'SOURCE_FREEZE.json').exists(),diagnostics_complete=(out/'DIAGNOSTICS_COMPLETE.json').exists(),test_year_accessed=False)
    atomic_json(RUN/'STATUS.json',status)
    if len(complete)==len(manifest):atomic_json(RUN/'SCREENING_COMPLETE.json',status)
    return status
if __name__=='__main__':print(json.dumps(collect(),ensure_ascii=False))
