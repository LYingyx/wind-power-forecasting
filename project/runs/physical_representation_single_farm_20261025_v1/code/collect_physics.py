"""Small per-task/per-seed score table and separate training/replay completion counts."""
from settings import *
import csv,collections
_cache={};_written=-1

def collect():
    global _written
    specs=json.loads((RUN/'manifest.json').read_text());byid={s['id']:s for s in specs}
    complete={p.parent.name for p in JOBS.glob('*/SUCCESS.json')} & byid.keys()
    failed=({p.parent.name for p in JOBS.glob('*/FAILURE.json')} & byid.keys())-complete
    for jid in complete-_cache.keys():_cache[jid]=json.loads((JOBS/jid/'metrics.json').read_text())
    if len(complete)!=_written:
        rows=[]
        for jid in sorted(complete):
            m=_cache[jid];spec=byid[jid]
            for criterion,selection in m['selections'].items():
                for horizon,score in [('all',selection['metrics']),*selection['metrics']['horizons'].items()]:
                    rows.append(dict(id=jid,stage=spec['stage'],task=spec['task'],site=spec['site'],seed=spec['seed'],configuration=spec['configuration'],backbone=spec['backbone'],variant=spec['variant'],selection=criterion,horizon=horizon,epoch=selection['epoch'],**{k:score[k] for k in ['n','mae','rmse','bias']}))
        if rows:
            out=RUN/'analysis/actual_per_farm_seed.csv';tmp=out.with_name(out.name+f'.{os.getpid()}.tmp')
            with tmp.open('w') as f:
                writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
            os.replace(tmp,out)
        _written=len(complete)
    stages={stage:dict(planned=sum(s['stage']==stage for s in specs),completed=sum(byid[j]['stage']==stage for j in complete)) for stage in ['A1','A2','A3','A4_REFERENCE']}
    state=dict(timestamp=now(),planned=len(specs),completed=len(complete),failed=len(failed),failed_jobs=sorted(failed),active=[],pending=len(specs)-len(complete)-len(failed),
        planned_training=6272,training_completed=sum(byid[j]['task']=='train' for j in complete),planned_reference_diagnostics=672,reference_diagnostics_completed=sum(byid[j]['task']=='diagnose_reference' for j in complete),
        stages=stages,source_freeze=(RUN/'SOURCE_FREEZE.json').exists(),diagnostics_complete=len(complete)==len(specs),test_year_accessed=False)
    atomic_json(RUN/'STATUS.json',state)
    return state

if __name__=='__main__':print(json.dumps(collect()))
