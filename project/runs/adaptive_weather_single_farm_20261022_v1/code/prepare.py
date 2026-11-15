"""Cycle-indexed true 7x7 patches, train-only priors; no new labels or weather."""
from config import *
import argparse

def metadata():
 s={p.stem:np.load(p,allow_pickle=False) for p in (CACHE/'static').glob('*.npy')}
 lat,lon=s['weather_latitude'],s['weather_longitude'];ids=[];xy=[];weights=[]
 for a,b in zip(s['station_latitude'],s['station_longitude']):
  r=int(np.abs(lat-a).argmin());c=int(np.abs(lon-b).argmin())
  rows=np.arange(r-3,r+4);cols=np.arange(c-3,c+4)
  assert rows.min()>=0 and rows.max()<len(lat) and cols.min()>=0 and cols.max()<len(lon)
  rr,cc=np.meshgrid(rows,cols,indexing='ij');ids.append((rr*len(lon)+cc).ravel())
  dx=(lon[cc]-b)*111.32*np.cos(np.deg2rad(a));dy=(lat[rr]-a)*111.32;xy.append(np.stack([dx,dy],-1).reshape(49,2)/(111.32*.25))
  north=int(np.searchsorted(-lat,-a,side='right')-1);west=int(np.searchsorted(lon,b,side='right')-1)
  fx=(b-lon[west])/(lon[west+1]-lon[west]);fy=(a-lat[north+1])/(lat[north]-lat[north+1])
  wt=np.zeros((7,7),np.float32)
  for dr,dc,w in [(0,0,(1-fx)*fy),(0,1,fx*fy),(1,0,(1-fx)*(1-fy)),(1,1,fx*(1-fy))]:
   wt[north+dr-rows[0],west+dc-cols[0]]=w
  assert abs(wt.sum()-1)<1e-6;weights.append(wt.ravel())
 return s,np.asarray(ids),np.asarray(xy,np.float32),np.asarray(weights,np.float32)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--splits',nargs='+',default=['train','val']);a=ap.parse_args()
 s,ix,xy,wt=metadata();dest=RUN/'prepared';dest.mkdir(exist_ok=True)
 for name,x in [('patch_indices',ix),('geometry',xy),('bilinear_weights',wt)]:np.save(dest/f'{name}.npy',x)
 audits={}
 for split in a.splits:
  out=dest/split;out.mkdir(exist_ok=True)
  if (out/'SUCCESS.json').exists():audits[split]=json.loads((out/'SUCCESS.json').read_text());continue
  grid=np.load(CACHE/split/'weather.npy',mmap_mode='r');n=len(grid)
  time=np.load(CACHE/split/'weather_time_offset_hours.npy');cyc=np.load(CACHE/split/'weather_cycle_index.npy');origin=np.load(CACHE/split/'origin_unix.npy');valid=np.load(CACHE/split/'weather_valid_unix.npy')
  assert np.array_equal((valid[cyc]-origin[:,None])/3600,time) and np.all(np.diff(time,axis=1)>0)
  sums=np.zeros((56,8,49,14),np.float64);max_error=0.
  local=np.load(ROOT/f'artifacts/shared_all_station_baselines_20260922_v1/data/local_weather/{split}.npy',mmap_mode='r')
  # One contiguous write per station avoids thousands of network mmap page faults.
  z=np.array(grid).reshape(n,8,14,600)
  for site in range(56):
   block=np.ascontiguousarray(z[...,ix[site]].transpose(0,1,3,2))
   assert np.isfinite(block).all();np.save(out/f'site{site:02d}.npy',block)
   sums[site]=block.sum(0,dtype=np.float64)
   reproduced=(block*wt[site][None,None,:,None]).sum(2)
   max_error=max(max_error,float(np.max(np.abs(reproduced-local[:,site]))))
   if site%8==0:print(split,'site',site,'cycles',n,flush=True)
  del z,block
  assert max_error<1e-5,max_error
  if split=='train':
   np.save(dest/'weather_prior.npy',(sums/n).astype(np.float32))
   hp=np.load(CACHE/'train/station_history.npy',mmap_mode='r');np.save(dest/'history_prior.npy',hp.mean(0,dtype=np.float64).astype(np.float32))
  info=dict(split=split,cycles=n,samples=len(cyc),shape=[n,8,49,14],bilinear_max_error=max_error,prior_fit_split='train',time_offsets_verified=True,completed_at=now())
  atomic_json(out/'SUCCESS.json',info);audits[split]=info
 atomic_json(dest/'AUDIT.json',dict(splits={sp:json.loads((dest/sp/'SUCCESS.json').read_text()) for sp in ['train','val','test'] if (dest/sp/'SUCCESS.json').exists()},true_patch_sizes=[3,5,7],train_only_priors=True,source_manifest_sha256=sha(CACHE/'manifest.json')))

if __name__=='__main__':main()
