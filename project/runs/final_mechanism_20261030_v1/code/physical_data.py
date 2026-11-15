from settings import *
import numpy as np,torch
from operational_farm import OperationalFarmData
from operational import interpolate_nodes
from neighbor_inputs import topology,window

def interpolate(nodes,times,leads):
    # Compatibility interface for checks; no extrapolation permitted.
    q=torch.as_tensor(leads,device=nodes.device,dtype=nodes.dtype)
    if q.ndim==1:q=q[None].expand(len(nodes),-1)
    assert torch.all(q>=times[:,:1]) and torch.all(q<=times[:,-1:])
    right=torch.searchsorted(times.contiguous(),q.contiguous()).clamp(1,times.shape[1]-1);left=right-1
    a=((q-times.gather(1,left))/(times.gather(1,right)-times.gather(1,left)))
    lo=nodes.gather(1,left[...,None].expand(-1,-1,nodes.shape[-1]));hi=nodes.gather(1,right[...,None].expand(-1,-1,nodes.shape[-1]))
    return lo+(hi-lo)*a[...,None]

class FarmData(OperationalFarmData):
 def __init__(self,split,site,variant,device='cpu',stats=None,limit=None,delay_minutes=15,capacity_control=False,seed=42):
    assert split in ['train','val','test']
    if split=='test':raise RuntimeError('2024 access disabled until a separate frozen follow-up')
    feature='BASE' if variant in ['B0','PERSISTENCE','BASE'] else 'SEL'
    super().__init__(CACHE,split,site,feature,stats,device,limit)
    self.variant=variant;self.seed=seed;self.anchor=self.h[:,-1:].clone();self.actual_history=self.h.clone();self.extra_keys=['anchor']
    tensor=lambda a:torch.as_tensor(np.array(a),dtype=torch.float32,device=device)
    if stats is None:
        self.stats.update(history_mean=self.h.mean(0).cpu().tolist(),weather_mean=self.weather.mean(0).cpu().tolist())
    if variant in ['H0W1','H0W0']:
        self.h=tensor(self.stats['history_mean'])[None].expand_as(self.h).clone()
        # STATE is recomputed from the fixed history, never the actual history.
        import physics
        state=physics.operating_state(self.h.cpu().numpy(),self.stats['flat_threshold'])
        mu=np.asarray(self.stats['mean'][-7:]);sd=np.asarray(self.stats['std'][-7:])
        self.x[...,-7:]=tensor((state-mu)/sd)[:,None]
    if variant in ['H1W0','H0W0']:
        self.weather=tensor(self.stats['weather_mean'])[None].expand_as(self.weather).clone()
        # Weather-derived SEL channels replaced by training expectations.
        self.x[...,:15]=0
    regional=variant in ['N3','R3_RAW','C2_R3_RAW']
    if regional:
        mismatch=variant=='C2_R3_RAW'
        source='train' if mismatch else split
        rows=np.load(ASSETS/f'mismatch_{split}_r{seed}.npy')[:self.n] if mismatch else np.arange(self.n)
        ids,geo,mask,cap=topology(site,'N3')
        hist=np.load(CACHE/source/'station_history.npy',mmap_mode='r')
        self.neighbor_raw=tensor(hist[rows[:,None],ids[None]])
        self.extra_keys+=['neighbor_raw']
        if variant!='N3':
            values=np.load(ASSETS/f'rings_{source}.npy',mmap_mode='r')[rows,site]
            if stats is None:self.stats['ring_mean']=values.mean(0).tolist();self.stats['ring_std']=values.std(0).clip(1e-4).tolist()
            self.region_features=tensor((values-np.asarray(self.stats['ring_mean']))/np.asarray(self.stats['ring_std']))
            self.extra_keys+=['region_features']
    if variant in ['AGE','LOCAL_ERR','C2_LOCAL_ERR','REGION_ERR','C2_REGION_ERR']:
        self.age=tensor((self.origin-self.issue_unix)/3600/24)[:,None]
        self.extra_keys+=['age']
    if variant in ['LOCAL_ERR','C2_LOCAL_ERR','REGION_ERR','C2_REGION_ERR']:
        mismatch=variant.startswith('C2_');source='train' if mismatch else split
        rows=np.load(ASSETS/f'mismatch_{split}_r{seed}.npy')[:self.n] if mismatch else np.arange(self.n)
        errors=np.load(ASSETS/f'error_history_{source}.npy',mmap_mode='r')[rows,site,-12:]
        if variant in ['REGION_ERR','C2_REGION_ERR']:
            errors=np.concatenate([errors,np.load(ASSETS/f'error_rings_{source}.npy',mmap_mode='r')[rows,site]],-1)
        if stats is None:self.stats['error_mean']=errors.mean(0).tolist();self.stats['error_std']=errors.std(0).clip(1e-4).tolist()
        self.error_features=tensor((errors-np.asarray(self.stats['error_mean']))/np.asarray(self.stats['error_std']))
        self.extra_keys+=['error_features']
 def batch(self,indices):
    out=super().batch(indices)
    out.update({k:getattr(self,k)[indices] for k in self.extra_keys})
    return out

UnfilteredFarmData=FarmData
class FarmData(UnfilteredFarmData):
 def __init__(self,split,*args,**kwargs):
    super().__init__(split,*args,**kwargs)
    if split=='val':
        n=self.n
        keep=np.load(ASSETS/'val_keep.npy')[:n]
        assert keep.dtype==np.bool_ and keep.shape==(n,)
        indices=np.flatnonzero(keep)
        def subset(v):
            if isinstance(v,torch.Tensor) and v.ndim and len(v)==n:
                return v[torch.as_tensor(indices,device=v.device)]
            if isinstance(v,np.ndarray) and v.ndim and len(v)==n:return v[indices]
            return v
        for key,value in list(vars(self).items()):
            if key=='diagnostic':setattr(self,key,{k:subset(v) for k,v in value.items()})
            elif key!='stats':setattr(self,key,subset(value))
        self.n=len(indices)
        assert self.n>0 and len(self.origin)==self.n
        from operational import november_exclusions
        assert not np.logical_or.reduce(list(november_exclusions(self.origin,2023).values())).any()

ReadoutFarmData=FarmData
class FarmData(ReadoutFarmData):
 def __init__(self,split,site,variant,*args,**kwargs):
    cell=variant.removeprefix('R3_') if variant.startswith('R3_H') else 'H1W1'
    assert cell in ['H1W1','H1W0','H0W1','H0W0']
    super().__init__(split,site,'R3_RAW',*args,**kwargs)
    self.variant=variant
    if cell[1]=='0':
        prior=torch.as_tensor(self.stats['history_mean'],device=self.h.device,dtype=self.h.dtype)
        self.h=prior[None].expand_as(self.h).clone()
        import physics
        state=physics.operating_state(self.h.cpu().numpy(),self.stats['flat_threshold'])
        z=(state-np.asarray(self.stats['mean'][-7:]))/np.asarray(self.stats['std'][-7:])
        self.x[...,-7:]=torch.as_tensor(z,device=self.x.device,dtype=self.x.dtype)[:,None]
    if cell[3]=='0':
        prior=torch.as_tensor(self.stats['weather_mean'],device=self.weather.device,dtype=self.weather.dtype)
        self.weather=prior[None].expand_as(self.weather).clone()
        self.x[...,:15]=0
        # Regional-attention query also reads physical u/v: close this otherwise hidden weather route.
        mu=torch.as_tensor(np.load(CACHE/'static/nwp_mean.npy'),device=self.weather.device,dtype=self.weather.dtype)
        sd=torch.as_tensor(np.load(CACHE/'static/nwp_std.npy'),device=self.weather.device,dtype=self.weather.dtype)
        self.physical=self.weather*sd+mu
