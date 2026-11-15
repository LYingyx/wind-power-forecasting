"""Per-farm inputs, train-only scaling, causal neighbor observations."""
from settings import *
from physics import thermodynamics, calendar, operating_state
import numpy as np
import torch

FEATURES={
 'BASE':[], 'CAP':['capacity'], 'CAL':['calendar'], 'REF':['reference'],
 'DELTA':['reference','delta'], 'DENS':['density'], 'STAB':['stability'],
 'DYN':['dynamics'], 'STATE':['state'], 'CYCLE':['disagreement'],
 'ALL':['calendar','reference','delta','density','stability','dynamics','state','disagreement'],
 'HIST':['historical'], 'SHIFT_ZERO':['historical'], 'SHIFT_FIXED':['historical'], 'SHIFT_SAMPLE':['historical'],
 'PC_INPUT':['reference','density'], 'CURVE_FREE':['reference','density'], 'CURVE_BIAS':['reference','density'],
 'NEIGH_GENERIC':['calendar'], 'NEIGH_ADVECT':['calendar'],
}

# Keep all A1 variants at three nominal projection inputs; inactive slots are zero.
A1=['SPD','SPD3','SPD_LV','SPD3_LV','RHO','DENS_NORHO','DENS_UNCOUPLED']
for _v in A1:FEATURES[_v]=[_v.lower()]
FEATURES['SEL']=['density','dynamics','stability','state']
for _label,_group in [('DENS','density'),('DYN','dynamics'),('STAB','stability'),('STATE','state')]:
    FEATURES['SEL_NO_'+_label]=['zero_'+g if g==_group else g for g in FEATURES['SEL']]
FEATURES['SEL_REF']=FEATURES['SEL']+['reference']
FEATURES['SEL_REF_DELTA']=FEATURES['SEL']+['reference','delta']
for _v in A1:FEATURES['SEL_'+_v]=[_v.lower(),'dynamics','stability','state']

def representation_features(physical,density):
    v=np.hypot(physical[...,1],physical[...,2]);v1000=np.hypot(physical[...,10],physical[...,12]);v925=np.hypot(physical[...,11],physical[...,13])
    zero=np.zeros_like(v);rho=density[...,0]
    stack=lambda *xs:np.stack(xs,-1).astype('f4')
    return dict(spd=stack(v/10,zero,zero),spd3=stack(v**3/1000,zero,zero),
        spd_lv=stack(v/10,v1000/10,v925/10),spd3_lv=stack(v**3/1000,v1000**3/1000,v925**3/1000),
        rho=stack(rho,zero,zero),dens_norho=stack(np.ones_like(v),v/10,1.225*v**3/1000),
        dens_uncoupled=stack(rho,v/10,1.225*v**3/1000))

def interpolate(nodes,times,queries):
    """Piecewise linear, clamped at endpoints, differentiable in query times."""
    if queries.ndim==1:queries=queries[None].expand(len(nodes),-1)
    # Nodes are hourly and contiguous, possibly from different known issuance cycles.
    x=(queries-times[:,:1]).clamp(0,nodes.shape[1]-1)
    lo=x.floor().long().clamp(max=nodes.shape[1]-2);frac=x-lo
    left=nodes.gather(1,lo[...,None].expand(-1,-1,nodes.shape[-1]))
    right=nodes.gather(1,(lo+1)[...,None].expand(-1,-1,nodes.shape[-1]))
    return left+(right-left)*frac[...,None]

class FarmData:
    def __init__(self,split,site,variant,device='cpu',stats=None,limit=None,delay_minutes=15,capacity_control=False):
        assert split in ['train','val'], '2024 excluded from fitting/selection'
        assert (PREPARED/split/'READY.json').exists()
        self.site=site;self.split=split;self.variant=variant;self.device=device
        def tensor(a,dtype=torch.float32):return torch.as_tensor(np.array(a),dtype=dtype)
        self.h=tensor(np.load(OLD/f'prepared/{split}/{site:02d}/history.npy',mmap_mode='r')[:limit])
        self.y=tensor(np.load(OLD/f'prepared/{split}/{site:02d}/target.npy',mmap_mode='r')[:limit]);self.n=len(self.h)
        src=CACHE/split;self.origin=np.load(src/'origin_unix.npy')[:self.n]
        ci=np.load(src/'weather_cycle_index.npy')[:self.n];cycles=np.unique(ci);self.cycle=tensor(np.searchsorted(cycles,ci),torch.long)
        self.phase=tensor(np.load(src/'origin_phase.npy')[:self.n],torch.long)
        self.times=tensor(np.load(src/'weather_time_offset_hours.npy')[:self.n])
        self.nodes=tensor(np.load(LOCAL_WEATHER/(split+'.npy'),mmap_mode='r')[cycles,site])
        self.past=tensor(np.load(PREPARED/split/'past_weather.npy',mmap_mode='r')[cycles,site])
        self.past_times=tensor((np.load(PREPARED/split/'past_valid_unix.npy')[ci]-self.origin[:,None])/3600.)
        self.all_times=torch.cat((self.past_times,self.times),1)
        leads=torch.arange(1,25)/4
        nodes=self.nodes[self.cycle];allnodes=torch.cat((self.past[self.cycle],nodes),1)
        self.weather=interpolate(nodes,self.times,leads)
        self.reference=interpolate(allnodes,self.all_times,torch.zeros(1))[:,0]
        mean=tensor(np.load(CACHE/'static/nwp_mean.npy'));std=tensor(np.load(CACHE/'static/nwp_std.npy'))
        self.physical=self.weather*std+mean;self.reference_physical=self.reference*std+mean
        elevation=float(np.load(PREPARED/'elevation.npy')[site])
        density,stability=thermodynamics(self.physical.numpy(),elevation)
        self.density=tensor(density)
        self.ref_density=tensor(thermodynamics(self.reference_physical.numpy(),elevation)[0])
        train_h=np.load(OLD/f'prepared/train/{site:02d}/history.npy',mmap_mode='r')
        # Train-only, explicitly descriptive plateau threshold.
        flat=float(np.quantile(np.abs(np.diff(train_h,axis=1)),.1))
        state=operating_state(self.h.numpy(),flat)
        release=np.stack(((self.origin-np.load(PREPARED/split/'past_issue_unix.npy')[ci,-1])/86400.,
                          (self.origin-np.load(PREPARED/split/'future_issue_unix.npy')[ci,0])/86400.),-1).astype('f4')
        repeated=lambda x:torch.as_tensor(x,dtype=torch.float32)[:,None].expand(-1,24,-1)
        features={'capacity':torch.cat((self.weather,self.h[:,-1:,None].expand(-1,24,1),leads[None,:,None].expand(self.n,-1,1)/6),-1),
            'calendar':tensor(calendar(self.origin,leads.numpy())),
            'reference':repeated(np.concatenate((self.reference.numpy(),release),-1)),
            'delta':self.weather-self.reference[:,None], 'density':tensor(density),'stability':tensor(stability),
            'state':repeated(state),
            'historical':repeated(torch.cat((self.past[self.cycle].flatten(1),self.past_times/6,tensor(release)),1))}
        dyn=tensor(np.load(PREPARED/split/'dynamics.npy',mmap_mode='r')[cycles,site])[self.cycle]
        features['dynamics']=interpolate(dyn,self.times,leads)
        mask=tensor(np.load(PREPARED/split/'overlap_mask.npy')[ci])
        dpath=PREPARED/split/'disagreement.npy'
        diff=tensor(np.load(dpath,mmap_mode='r')[cycles,site])[self.cycle] if dpath.exists() else torch.zeros_like(nodes)
        # Mask is explicit; unavailable alternate cycles are not presented as agreement.
        features['disagreement']=interpolate(torch.cat((diff,mask[...,None]),-1),self.times,leads)
        features.update({k:tensor(v) for k,v in representation_features(self.physical.numpy(),density).items()})
        for group in ['density','dynamics','stability','state']:features['zero_'+group]=torch.zeros_like(features[group])
        self.diagnostic={k:features[k] for k in ['calendar','density','stability','dynamics','state']}
        groups=FEATURES[variant]
        raw=torch.cat([features[k] for k in groups],-1) if groups else torch.zeros(self.n,24,0)
        if capacity_control:
            assert not variant.startswith(('NEIGH_','SHIFT_','CURVE_'))
            # Exact same projection shape, but deterministic repeated baseline inputs only.
            old=features['capacity']
            raw=old.repeat(1,1,(raw.shape[-1]+old.shape[-1]-1)//old.shape[-1])[...,:raw.shape[-1]]
        if stats is None:
            assert split=='train'
            mu=raw.double().mean((0,1)).float() if raw.shape[-1] else torch.zeros(0)
            sd=raw.double().std((0,1),unbiased=False).float().clamp_min(1e-4) if raw.shape[-1] else torch.ones(0)
            stats={'mean':mu.tolist(),'std':sd.tolist(),'groups':groups,'flat_threshold':flat,'capacity_control':capacity_control}
        assert groups==stats['groups'] and capacity_control==stats['capacity_control']
        self.stats=stats;self.x=(raw-tensor(stats['mean']))/tensor(stats['std'])
        assert torch.isfinite(self.x).all()
        self.neighbors=None
        if variant.startswith('NEIGH_'):
            assert delay_minutes in [0,15,30]
            ids=np.load(PREPARED/'neighbor_ids.npy')[site]
            # Cached last observation is origin. Positive delay drops recent entries;
            # shorten the same history window rather than fabricate unavailable history.
            end=24-delay_minutes//15
            nh=np.load(src/'station_history.npy',mmap_mode='r')[:self.n,ids,:end]
            self.neighbors=tensor(nh)
            self.neighbor_ages=(torch.arange(end)-(end-1))/4-delay_minutes/60
            assert float(self.neighbor_ages.max())<=-delay_minutes/60
        self.historical_input=torch.cat((self.h,self.past[self.cycle].flatten(1),self.past_times/6),-1)
        for name in ['h','y','cycle','phase','times','nodes','past','past_times','all_times','weather','reference','physical','reference_physical','density','ref_density','x','historical_input','neighbors','neighbor_ages']:
            value=getattr(self,name,None)
            if torch.is_tensor(value):setattr(self,name,value.to(device))
    def batch(self,ix):
        d={k:getattr(self,k)[ix] for k in ['h','phase','times','all_times','weather','reference','physical','reference_physical','density','ref_density','x','historical_input']}
        d['allnodes']=torch.cat((self.past[self.cycle[ix]],self.nodes[self.cycle[ix]]),1)
        if self.neighbors is not None:d.update(neighbors=self.neighbors[ix],neighbor_ages=self.neighbor_ages)
        return d
