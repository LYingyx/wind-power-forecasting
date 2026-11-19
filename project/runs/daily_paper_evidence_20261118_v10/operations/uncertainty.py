"""Paired moving-block bootstrap over daily NWP issues, conditional on trained seeds."""
from pathlib import Path
import sys,json,hashlib
import numpy as np
RUN=Path(__file__).resolve().parents[1];sys.path.insert(0,str(RUN/'code'))
from settings import atomic_json,now

def report():
 specs=json.loads((RUN/'manifest.json').read_text());paths={}
 for s in specs:
  for split in ['val','test']:
   p=RUN/'jobs'/s['id']/f'issue_stats_{split}.npz'
   if p.exists() and (p.parent/'DONE.json').exists():paths[s['group'],s['seed'],split]=p
 out=RUN/'analysis/ISSUE_BLOCK_UNCERTAINTY.json';prior=json.loads(out.read_text()) if out.exists() else {};results=prior.get('results',{})
 changed=False
 for split in ['val','test']:
  for ref in ['R0','TF_E2L','TF_R3']:
   seeds=[s for s in range(42,52) if ('H-FULL',s,split) in paths and (ref,s,split) in paths]
   if len(seeds)<5:continue
   seeds=list(range(42,52)) if seeds==list(range(42,52)) else list(range(42,47))
   if any(('H-FULL',s,split) not in paths or (ref,s,split) not in paths for s in seeds):continue
   key=split+':H-FULL/'+ref
   if key in results and results[key]['seeds']==seeds:continue
   aa=[];ss=[];bb=[];tt=[];counts=None;issues=None
   for seed in seeds:
    with np.load(paths['H-FULL',seed,split]) as a,np.load(paths[ref,seed,split]) as b:
     assert np.array_equal(a['issue'],b['issue']) and np.array_equal(a['counts'],b['counts'])
     if counts is None:counts=a['counts'].copy();issues=a['issue'].copy()
     assert np.array_equal(counts,a['counts']) and np.array_equal(issues,a['issue'])
     aa.append(a['abs_sum'].sum(-1));ss.append(a['sq_sum'].sum(-1));bb.append(b['abs_sum'].sum(-1));tt.append(b['sq_sum'].sum(-1))
   # Seven calendar-day blocks never bridge excluded-month gaps. Resample
   # blocks until the number of observed issue days is recovered, then truncate.
   rng=np.random.default_rng(20261118);n=len(issues);weights=np.zeros((1000,n))
   blocks=[np.where((issues>=d)&(issues<d+7*86400))[0] for d in issues]
   for i in range(1000):
    draw=[]
    while len(draw)<n:draw.extend(blocks[rng.integers(n)].tolist())
    weights[i]=np.bincount(draw[:n],minlength=n)
   den=weights@counts*24
   def metric(arr,square=False):
    values=np.stack(arr).transpose(1,0,2).reshape(n,-1)
    sums=(weights@values).reshape(1000,len(seeds),55)
    v=sums/np.maximum(den[:,None,:],1)
    if square:v=np.sqrt(v)
    return v.mean((1,2))
   dmae=100*(metric(aa)/metric(bb)-1);drmse=100*(metric(ss,True)/metric(tt,True)-1)
   results[key]=dict(seeds=seeds,issues=n,replicates=1000,block_calendar_days=7,
     mae_percent_interval_95=np.percentile(dmae,[2.5,97.5]).tolist(),rmse_percent_interval_95=np.percentile(drmse,[2.5,97.5]).tolist(),
     scope='conditional on these seeds and fixed farms; not a test of unseen years or independent farm samples')
   changed=True
 if changed:atomic_json(out,dict(timestamp=now(),results=results))
 return results
if __name__=='__main__':report()
