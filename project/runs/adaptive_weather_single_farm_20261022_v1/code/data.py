from config import *
from models import patch_select

class Data:
 def __init__(self,split,site,k,device,base=False,limit=None):
  if split=='test':assert (RUN/'ALL_SELECTIONS_FROZEN.json').exists(),'2024 evaluation before selection freeze'
  src=CACHE/split
  def tensor(a,dtype=torch.float32):return torch.as_tensor(np.array(a),dtype=dtype,device=device)
  self.h=tensor(np.load(OLD/f'prepared/{split}/{site:02d}/history.npy',mmap_mode='r')[:limit])
  self.y=tensor(np.load(OLD/f'prepared/{split}/{site:02d}/target.npy',mmap_mode='r')[:limit])
  self.phase=tensor(np.load(src/'origin_phase.npy')[:limit],torch.long)
  self.times=tensor(np.load(src/'weather_time_offset_hours.npy')[:limit])
  ci=np.load(src/'weather_cycle_index.npy')[:limit];cycles=np.unique(ci)
  raw=np.load(RUN/f'prepared/{split}/site{site:02d}.npy',mmap_mode='r')
  self.weather=tensor(np.asarray(raw[cycles])[:,:,patch_select(k)])
  self.cycle=tensor(np.searchsorted(cycles,ci),torch.long)
  self.n=len(self.h);self.split=split;self.site=site;self.base=None
 def load_base(self,seed,device):
  p=OLD/'jobs'/f'reference_B0_s{self.site:02d}_r{seed}'/f'{self.split}_prediction.npy'
  self.base=torch.tensor(np.array(np.load(p,mmap_mode='r')[:self.n]),device=device)
 def batch(self,ix):return (self.h[ix],self.weather[self.cycle[ix]],self.phase[ix],self.times[ix],self.base[ix] if self.base is not None else None)

def score(pred,truth,history,site,split):
 p=np.asarray(pred,dtype=np.float64);y=np.asarray(truth,dtype=np.float64);e=p-y;n=len(e)
 origin=np.load(CACHE/split/'origin_unix.npy')[:n];dates=(origin+8*3600).astype('datetime64[s]');months=dates.astype('datetime64[M]').astype(int)%12+1;hours=((origin+8*3600)//3600)%24
 desc=json.loads((OLD/f'prepared/train/{site:02d}/train_descriptors.json').read_text());cur=history[:,-1]
 masks={}
 for label,ms in [('DJF',[12,1,2]),('MAM',[3,4,5]),('JJA',[6,7,8]),('SON',[9,10,11])]:masks['season_'+label]=np.isin(months,ms)
 for k in range(4):masks[f'local_hour_{k*6:02d}_{(k+1)*6:02d}']=(hours>=k*6)&(hours<(k+1)*6)
 for name,lo,hi in [('low',-np.inf,.1),('medium_low',.1,.3),('medium_high',.3,.7),('high',.7,np.inf)]:masks['current_cf_'+name]=(cur>=lo)&(cur<hi)
 masks['observed_recent_ramp']=np.abs(history[:,-1]-history[:,-5])>desc['ramp_q90'];masks['observed_nonramp']=~masks['observed_recent_ramp']
 masks['future_ramp_diagnostic_only']=(np.abs(y-np.concatenate((history[:,-4:],y[:,:-4]),1))>=.2).any(1)
 def calc(a):return dict(n_origins=len(a),mae=float(np.abs(a).mean()),rmse=float(np.sqrt((a*a).mean())),bias=float(a.mean())) if len(a) else dict(n_origins=0,mae=None,rmse=None,bias=None)
 result=calc(e);result['lead_mae']=np.abs(e).mean(0).tolist();result['lead_rmse']=np.sqrt((e*e).mean(0)).tolist()
 result['horizons']={name:calc(e[:,a:b]) for name,a,b in [('0_1h',0,4),('1_3h',4,12),('3_6h',12,24)]}
 result['scenarios']={k:calc(e[v]) for k,v in masks.items()};result['outside_0_1_fraction']=float(((p<0)|(p>1)).mean())
 result['daily_utc']={str(d):dict(n=int((origin//86400==d).sum()),abs_sum=float(np.abs(e[origin//86400==d]).sum()),sq_sum=float((e[origin//86400==d]**2).sum())) for d in np.unique(origin//86400)}
 return result
