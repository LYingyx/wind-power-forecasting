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
   m=_cache[jid];s=byid[jid];selected=m['selections']['mae']
   for horizon,score in [('all',selected['metrics']),*selected['metrics']['horizons'].items()]:
    rows.append(dict(id=jid,configuration=s['configuration'],backbone=s['backbone'],variant=s['variant'],site=s['site'],seed=s['seed'],horizon=horizon,epoch=selected['epoch'],**{k:score[k] for k in ['n','mae','rmse','bias']}))
  if rows:
   p=RUN/'analysis/per_farm_seed.csv';tmp=p.with_name(p.name+f'.{os.getpid()}.tmp')
   with tmp.open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
   os.replace(tmp,p)
  _written=len(complete)
 state=dict(timestamp=now(),planned=len(specs),completed=len(complete),failed=len(failed),failed_jobs=sorted(failed),active=[],pending=len(specs)-len(complete)-len(failed),test_year_evaluated=False)
 atomic_json(RUN/'STATUS.json',state);return state
if __name__=='__main__':print(json.dumps(collect()))
