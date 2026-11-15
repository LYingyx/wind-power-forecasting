from settings import *
import numpy as np,torch
from operational import interpolate_nodes
from neighbor_inputs import topology
import physics

class FarmData:
    def __init__(self,split,spec,device='cpu',stats=None,limit=None,load_labels=True):
        assert split in ['train','val']
        assert (DATA/'READY.json').exists()
        if stats is None:assert split=='train'
        site=spec['site'];age=spec['age_shift'];c=DATA/'common'/split;w=DATA/f'age{age}'/split
        self.origin=np.load(c/'origin_unix.npy')[:limit];self.n=len(self.origin)
        self.h_numpy=np.array(np.load(c/'station_history.npy',mmap_mode='r')[:limit,site])
        self.y_numpy=np.array(np.load(c/'station_future.npy',mmap_mode='r')[:limit,site])
        cyc=np.load(c/'cycle.npy')[:limit];offset=np.load(c/'weather_time_offset_hours.npy')[:limit]
        future=np.load(w/'future.npy',mmap_mode='r')[cyc,site];past=np.load(w/'past.npy',mmap_mode='r')[cyc,site]
        dyn_nodes=np.load(w/'dynamics.npy',mmap_mode='r')[cyc,site]
        weather=interpolate_nodes(future,offset,np.arange(1,25)/4);dyn=interpolate_nodes(dyn_nodes,offset,np.arange(1,25)/4)
        mu=np.load(CACHE/'static/nwp_mean.npy');sd=np.load(CACHE/'static/nwp_std.npy')
        phys=weather*sd+mu;elev=float(np.load(DATA/'elevation.npy')[site])
        den,stab=physics.thermodynamics(phys,elev)
        flat=float(np.quantile(abs(np.diff(self.h_numpy,axis=1)),.1)) if stats is None else stats['flat_threshold']
        state=physics.operating_state(self.h_numpy,flat)
        raw=np.concatenate([den,dyn,stab,np.broadcast_to(state[:,None],(self.n,24,7))],-1)
        ring=np.array(np.load(c/'rings.npy',mmap_mode='r')[:limit,site])
        reg=np.array(np.load(c/'region_history.npy',mmap_mode='r')[:limit,site]).reshape(self.n,-1)
        obs=np.concatenate([self.h_numpy,reg],-1)
        if stats is None:
            stats=dict(site=site,age_shift=age,fitted_split='train',flat_threshold=flat,
                mean=raw.mean((0,1),dtype='f8').tolist(),std=raw.std((0,1),dtype='f8').clip(1e-4).tolist(),
                ring_mean=ring.mean(0).tolist(),ring_std=ring.std(0).clip(1e-4).tolist(),
                obs_min=obs.min(0).tolist(),obs_max=obs.max(0).tolist(),obs_mean=obs.mean(0).tolist(),obs_std=obs.std(0).clip(1e-4).tolist())
        assert stats['fitted_split']=='train' and stats['site']==site and stats['age_shift']==age
        self.stats=stats
        if spec['variant']=='E5':
            donors=np.load(c/f'donor_r{spec["seed"]}.npy')[:limit];tc=DATA/'common/train'
            hh=np.load(tc/'station_history.npy',mmap_mode='r')[donors,site]
            rr=np.load(tc/'region_history.npy',mmap_mode='r')[donors,site].reshape(self.n,-1)
            obs=np.concatenate([hh,rr],-1)
        obs=(np.clip(obs,stats['obs_min'],stats['obs_max'])-np.array(stats['obs_mean']))/np.array(stats['obs_std'])
        if spec['variant']=='E3':obs[:]=0
        if spec['variant']=='E4':obs[:,24:]=0
        ids,_,_,_=topology(site,'N3')
        neighbor=np.load(c/'station_history.npy',mmap_mode='r')[:limit][:,ids]
        times=np.concatenate([offset[:,:1]+np.arange(-6,0)[None],offset],-1)
        arrays=dict(h=self.h_numpy,anchor=self.h_numpy[:,-1:],phase=np.load(c/'origin_phase.npy')[:limit],
            weather=weather,physical=phys,x=(raw-np.array(stats['mean']))/np.array(stats['std']),
            neighbor_raw=neighbor,region_features=(ring-np.array(stats['ring_mean']))/np.array(stats['ring_std']),
            nodes=future,past=past,offset=offset,all_times=times,
            age=((self.origin-np.load(c/'issue0.npy')[:limit])/3600+age)[:,None]/24,
            corr_obs=obs,dynamics=dyn)
        self.inputs={k:torch.as_tensor(np.array(v),device=device,dtype=torch.long if k=='phase' else torch.float32) for k,v in arrays.items()}
        self.y=torch.as_tensor(self.y_numpy,device=device)
        # Targets are intentionally outside the forward-input dictionary.
        self.era=None
        if load_labels:self.era=torch.as_tensor(np.array(np.load(DATA/'labels'/split/'future_uv.npy',mmap_mode='r')[cyc,site]),device=device)
    def batch(self,idx):return {k:v[idx] for k,v in self.inputs.items()}
