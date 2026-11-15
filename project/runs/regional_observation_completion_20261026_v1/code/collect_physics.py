"""Per-farm, per-seed ledger with dynamic stage/task counts."""
from settings import *
import csv,collections
_cache={};_written=-1
def collect():
    global _written
    specs=json.loads((RUN/'manifest.json').read_text());byid={s['id']:s for s in specs}
    complete={p.parent.name for p in JOBS.glob('*/SUCCESS.json')}&byid.keys()
    failed=({p.parent.name for p in JOBS.glob('*/FAILURE.json')}&byid.keys())-complete
    for jid in complete-_cache.keys():_cache[jid]=json.loads((JOBS/jid/'metrics.json').read_text())
    if len(complete)!=_written:
        rows=[]
        for jid in sorted(complete):
            m=_cache[jid];s=byid[jid]
            for criterion,selected in m['selections'].items():
                for horizon,score in [('all',selected['metrics']),*selected['metrics']['horizons'].items()]:
                    rows.append(dict(id=jid,stage=s['stage'],task=s['task'],configuration=s['configuration'],backbone=s['backbone'],variant=s['variant'],site=s['site'],seed=s['seed'],selection=criterion,horizon=horizon,epoch=selected['epoch'],**{k:score[k] for k in ['n','mae','rmse','bias']}))
        if rows:
            p=RUN/'analysis/actual_per_farm_seed.csv';tmp=p.with_name(p.name+f'.{os.getpid()}.tmp')
            with tmp.open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
            os.replace(tmp,p)
        _written=len(complete)
    state=dict(timestamp=now(),planned=len(specs),completed=len(complete),failed=len(failed),failed_jobs=sorted(failed),active=[],pending=len(specs)-len(complete)-len(failed),planned_training=sum(s['task']=='train' for s in specs),training_completed=sum(byid[j]['task']=='train' for j in complete),planned_reference_diagnostics=sum(s['task']=='diagnose_reference' for s in specs),reference_diagnostics_completed=sum(byid[j]['task']=='diagnose_reference' for j in complete),planned_arx=sum(s['task']=='arx' for s in specs),arx_completed=sum(byid[j]['task']=='arx' for j in complete),test_year_accessed=False,stages={stage:dict(planned=sum(s['stage']==stage for s in specs),completed=sum(byid[j]['stage']==stage for j in complete)) for stage in sorted({s['stage'] for s in specs})})
    atomic_json(RUN/'STATUS.json',state);return state
if __name__=='__main__':print(json.dumps(collect()))
