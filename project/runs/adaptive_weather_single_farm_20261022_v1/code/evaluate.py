"""Frozen-model 2024 evaluation and predeclared weather/weight interventions."""
from config import *
from models import Model
from data import Data,score
from train import save_array
import argparse,traceback

def donors(split,seed,limit=None):
 origin=np.load(CACHE/split/'origin_unix.npy');phase=np.load(CACHE/split/'origin_phase.npy');age=np.load(CACHE/split/'nwp_age_minutes.npy');cycle=np.load(CACHE/split/'weather_cycle_index.npy')
 origin,phase,age,cycle=(x[:limit] for x in [origin,phase,age,cycle])
 month=(origin+28800).astype('datetime64[s]').astype('datetime64[M]').astype(int);day=(origin+28800)//86400
 keys=np.stack((month,phase,age),-1);result=np.arange(len(origin));rng=np.random.default_rng(seed)
 for key in np.unique(keys,axis=0):
  ix=np.flatnonzero((keys==key).all(1));assert len(np.unique(day[ix]))>=2,'Insufficient dates for within-month intervention'
  ok=False
  for shift in rng.permutation(np.arange(1,len(ix))):
   donor=np.roll(ix,int(shift))
   if np.all(day[ix]!=day[donor]) and np.all(cycle[ix]!=cycle[donor]):result[ix]=donor;ok=True;break
  assert ok,'No valid within-stratum derangement'
 assert np.array_equal(np.sort(result),np.arange(len(result)))
 assert np.all(day!=day[result]) and np.array_equal(phase,phase[result])
 return result

@torch.no_grad()
def run(s,smoke=False):
 configure();device='cuda:0' if torch.cuda.is_available() else 'cpu'
 if not smoke:assert (RUN/'ALL_SELECTIONS_FROZEN.json').exists()
 split='val' if smoke else 'test'
 out=RUN/('smoke_evaluation' if smoke else 'evaluation')/s['id'];out.mkdir(parents=True,exist_ok=True)
 if (out/'SUCCESS.json').exists():return
 atomic_json(out/'spec.json',s)
 source=(OLD if s.get('legacy_source') else RUN)/('smoke_jobs' if smoke else 'jobs')/s['source_job_id'];checkpoint=source/'best.pt'
 m=Model(s).to(device).eval();state=torch.load(checkpoint,map_location=device,weights_only=False)['model']
 if s.get('legacy_source'):m.backbone.load_state_dict(state)
 else:m.load_state_dict(state)
 data=Data(split,s['site'],s['k'],device,limit=2048 if smoke else None)
 if s['front']=='H1':data.load_base(s['seed'],device)
 truth=data.y.cpu().numpy();hist=data.h.cpu().numpy()
 variants=[('matched',None)]
 diagnostic=s['seed']<=46 and (s['front'] in FRONTS or s['front']=='W0')
 if diagnostic:
  variants += [(f'weather_shuffle_{seed}',seed) for seed in PROTOCOL['intervention_seeds']]
  if m.frontend.has_space or m.frontend.has_time:variants += [(f'weight_shuffle_{seed}',seed) for seed in PROTOCOL['intervention_seeds']]
 result=dict(**s,checkpoint_sha256=sha(checkpoint),selection_freeze_sha256=sha(RUN/'ALL_SELECTIONS_FROZEN.json') if not smoke else None,source_checkpoint=str(checkpoint.relative_to(ROOT)),interventions={},smoke_only=smoke,evaluation_split=split)
 weights_sample=[];indices_sample=np.linspace(0,data.n-1,32,dtype=int);summary=[]
 for label,permutation_seed in variants:
  path=out/f'{label}_metrics.json'
  attention_missing=label=='matched' and (m.frontend.has_space or m.frontend.has_time) and not ((out/'attention_cases.npz').exists() and (out/'attention_diagnostics.npy').exists())
  if path.exists() and (out/f'{label}_prediction.npy').exists() and not attention_missing:result['interventions'][label]=json.loads(path.read_text());continue
  donor=None
  if permutation_seed is not None:
   donor_np=donors(split,permutation_seed,data.n);save_array(out/f'donors_{permutation_seed}.npy',donor_np);donor=torch.as_tensor(donor_np,device=device)
   assert torch.equal(data.times,data.times[donor]),'Donor relative time mismatch'
  predictions=[]
  for st in range(0,data.n,256):
   ix=slice(st,min(st+256,data.n));args=list(data.batch(ix));override=None
   if label.startswith('weather_shuffle'):args[1]=data.weather[data.cycle[donor[ix]]]
   elif label.startswith('weight_shuffle'):
    dh,dw,dp,dt,_=data.batch(donor[ix]);override=m.frontend.weights(dh,dw,dp,dt)
   prediction,weights=m(*args,override=override,return_weights=True);assert torch.isfinite(prediction).all();predictions.append(prediction.cpu().numpy())
   if label=='matched' and weights:
    chosen=indices_sample[(indices_sample>=st)&(indices_sample<st+len(prediction))]-st
    if len(chosen):weights_sample.append({k:v[chosen].cpu().numpy() for k,v in weights.items()})
    # Per-origin/lead centroids and temporal locations; no seed/farm aggregation.
    b=len(prediction);centroid=torch.zeros(b,24,2,device=device);time_expected=torch.zeros(b,24,device=device);time_peak=torch.zeros(b,24,device=device)
    if 'space' in weights:
     sw=weights['space']
     if sw.ndim==4:sw=torch.einsum('bln,blnk->blk',weights['time'],sw)
     centroid=torch.einsum('blk,kc->blc',sw,m.frontend.geometry)*(111.32*.25)
    if 'time' in weights:
     tw=weights['time'];ts=args[3][:,:tw.shape[-1]];time_expected=torch.einsum('bln,bn->bl',tw,ts);time_peak=ts.gather(1,tw.argmax(-1))
    # Issued point wind at each lead, for physical association rather than causal claims.
    from models import interpolate_grid
    pw=interpolate_grid(m.frontend.point_nodes(args[1])[:,:,None],args[3])[:,:,0]*m.frontend.nwp_std+m.frontend.nwp_mean
    summary.append(torch.cat((centroid,time_expected[...,None],time_peak[...,None],pw[...,1:3]),-1).cpu().numpy())
  pred=np.concatenate(predictions);save_array(out/f'{label}_prediction.npy',pred)
  metrics=score(pred,truth,hist,s['site'],split);metrics['prediction_sha256']=sha(out/f'{label}_prediction.npy');metrics['permutation_seed']=permutation_seed
  atomic_json(path,metrics);result['interventions'][label]=metrics
  atomic_json(out/'progress.json',dict(completed_variant=label,variants_total=len(variants),timestamp=now()))
 if weights_sample:
  np.savez_compressed(out/'attention_cases.npz',origin_indices=indices_sample,**{k:np.concatenate([x[k] for x in weights_sample]) for k in weights_sample[0]})
 if summary:save_array(out/'attention_diagnostics.npy',np.concatenate(summary))
 result[split]=result['interventions']['matched'];result['completed_at']=now()
 atomic_json(out/'metrics.json',result);atomic_json(out/'SUCCESS.json',dict(timestamp=now(),variants=len(variants),checkpoint_sha256=result['checkpoint_sha256']))

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--spec',required=True);ap.add_argument('--smoke',action='store_true');a=ap.parse_args();s=json.loads(Path(a.spec).read_text())
 try:run(s,a.smoke)
 except BaseException:atomic_json(RUN/('smoke_evaluation' if a.smoke else 'evaluation')/s['id']/'FAILURE.json',dict(timestamp=now(),traceback=traceback.format_exc()));raise
