"""Weather-only supervision and explicit three-source response interactions."""
from settings import *
import numpy as np
import torch
from torch import nn
from operational import interpolate_nodes
from physical_models import PhysicalModel, frozen
from data import FarmData

class CorrectionData:
    def __init__(self, split, spec, device='cpu', stats=None, limit=None):
        assert split in ['train', 'val']
        c=DATA/'common'/split; w=DATA/'age0'/split; site=spec['site']
        self.origin=np.load(c/'origin_unix.npy')[:limit]; self.n=len(self.origin)
        cycle=np.load(c/'cycle.npy')[:limit]
        self.nodes=np.array(np.load(w/'future.npy',mmap_mode='r')[cycle,site])
        self.past=np.array(np.load(w/'past.npy',mmap_mode='r')[cycle,site])
        self.offset=np.load(c/'weather_time_offset_hours.npy')[:limit]
        h=np.array(np.load(c/'station_history.npy',mmap_mode='r')[:limit,site])
        reg=np.array(np.load(c/'region_history.npy',mmap_mode='r')[:limit,site]).reshape(self.n,-1)
        obs=np.concatenate([h,reg],-1)
        if stats is None:
            assert split=='train'
            stats={name:fn(obs,axis=0).tolist() for name,fn in [('obs_min',np.min),('obs_max',np.max),('obs_mean',np.mean)]}
            stats['obs_std']=obs.std(0).clip(1e-4).tolist()
        self.stats=stats
        obs=(np.clip(obs,stats['obs_min'],stats['obs_max'])-np.array(stats['obs_mean']))/np.array(stats['obs_std'])
        times=np.concatenate([self.offset[:,:1]+np.arange(-6,0)[None],self.offset],-1)
        age=((self.origin-np.load(c/'issue0.npy')[:limit])/3600)[:,None]/24
        self.context=np.concatenate([np.concatenate([self.past,self.nodes],1).reshape(self.n,-1),obs,times,age],-1).astype('f4')
        assert self.context.shape==(self.n,301)
        self.truth=np.array(np.load(DATA/'labels'/split/'future_uv.npy',mmap_mode='r')[cycle,site])
        self.mu=np.load(CACHE/'static/nwp_mean.npy'); self.sd=np.load(CACHE/'static/nwp_std.npy')
        self.x=torch.as_tensor(self.context,device=device)
        # Match the original forward's float32, on-device time division exactly.
        self.x[:,286:300]=self.x[:,286:300]/6
        self.context=self.x.detach().cpu().numpy().copy()
        self.raw=torch.as_tensor(self.nodes[...,1:3],device=device)
        self.y=torch.as_tensor(self.truth,device=device)

    def weather_context(self):
        return np.concatenate([self.context[:,:196],self.context[:,286:]],-1)

    def mos_features(self):
        f=self.nodes*self.sd+self.mu; p=self.past*self.sd+self.mu
        local=self.origin+28800; hour=(local%86400)/3600
        day=local.astype('datetime64[s]').astype('datetime64[D]')
        doy=(day-day.astype('datetime64[Y]')).astype(float)/365.25
        clock=np.stack([self.context[:,-1],np.sin(hour*2*np.pi/24),np.cos(hour*2*np.pi/24),np.sin(doy*2*np.pi),np.cos(doy*2*np.pi)],-1)
        return np.concatenate([f[...,:4].reshape(self.n,-1),p[...,:4].reshape(self.n,-1),np.linalg.norm(f[...,1:3],axis=-1),np.linalg.norm(p[...,1:3],axis=-1),clock],-1)

    def hourly_uv(self, uv):
        return interpolate_nodes(np.asarray(uv,dtype='f8')*self.sd[1:3]+self.mu[1:3],self.offset,np.arange(1,7))

    def metrics(self,uv):
        uv=np.asarray(uv,dtype='f8'); err=uv-self.truth
        pred=self.hourly_uv(uv); true=self.hourly_uv(self.truth)
        euv=pred-true; es=np.linalg.norm(pred,axis=-1)-np.linalg.norm(true,axis=-1)
        return dict(node_normalized_uv_mse=float(np.mean(err**2)),
            u_mse=float(np.mean(euv[...,0]**2)),v_mse=float(np.mean(euv[...,1]**2)),uv_mse=float(np.mean(euv**2)),
            speed_mse=float(np.mean(es**2)),speed_rmse=float(np.sqrt(np.mean(es**2))),
            uv_mse_1_3=float(np.mean(euv[:,:3]**2)),uv_mse_4_6=float(np.mean(euv[:,3:]**2)),
            speed_mse_1_3=float(np.mean(es[:,:3]**2)),speed_mse_4_6=float(np.mean(es[:,3:]**2)),
            lead_uv_mse=np.mean(euv**2,axis=(0,2)).tolist(),lead_speed_mse=np.mean(es**2,axis=0).tolist())

class PureCorrector(nn.Module):
    def __init__(self,seed):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed+61101)
            self.corrector=nn.Sequential(nn.Linear(301,128),nn.GELU(),nn.Linear(128,64),nn.GELU(),nn.Linear(64,16))
            nn.init.zeros_(self.corrector[-1].weight); nn.init.zeros_(self.corrector[-1].bias)
    def forward(self,x,raw):return raw+self.corrector(x).reshape(-1,8,2)

class InteractionData(FarmData):
    def __init__(self,split,spec,device='cpu',stats=None,limit=None):
        super().__init__(split,dict(spec,variant='E0'),device,stats,limit,load_labels=False)
        region=self.inputs['region_features']
        if spec['variant']=='I_WRONG':
            donor=np.load(DATA/'common'/split/f'donor_r{spec["seed"]}.npy')[:limit]
            raw=np.load(DATA/'common/train/rings.npy',mmap_mode='r')[donor,spec['site']]
            region=torch.as_tensor((raw-np.array(self.stats['ring_mean']))/np.array(self.stats['ring_std']),device=device,dtype=torch.float32)
        elif spec['variant']=='I_LOCAL':region=torch.zeros_like(region)
        self.inputs['interaction_region']=region

class ResponseInteraction(nn.Module):
    """Same 4512 trainable parameters in every new branch; no changed input set."""
    def __init__(self,seed,variant):
        super().__init__(); self.variant=variant
        # manual_seed also touches CUDA: preserve the E0 backbone dropout stream.
        devices=list(range(torch.cuda.device_count())) if torch.cuda.is_initialized() else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(seed+621102)
            self.h=nn.Linear(27,32); self.r=nn.Linear(15,32); self.w=nn.Linear(32,32)
            self.out=nn.Linear(32,64,bias=False); nn.init.zeros_(self.out.weight)
    def forward(self,d):
        b=len(d['h']); lead=torch.arange(1,25,device=d['h'].device,dtype=d['h'].dtype)/24
        t=torch.stack([lead,torch.sin(lead*2*torch.pi),torch.cos(lead*2*torch.pi)],-1)[None].expand(b,-1,-1)
        h=torch.tanh(self.h(torch.cat([d['corr_obs'][:,:24,None].transpose(1,2).expand(-1,24,-1),t],-1)))
        r=torch.tanh(self.r(torch.cat([d['interaction_region'][:,None].expand(-1,24,-1),t],-1)))
        w=self.w(torch.cat([d['weather'],d['x'][...,:15],t],-1))
        if self.variant=='I_ADD':z=(h+r+w)/(3**.5)
        elif self.variant=='I_XATTN':
            memory=torch.stack([h,r],-2)
            weights=(w.unsqueeze(-2)*memory).sum(-1)/(32**.5)
            z=(weights.softmax(-1).unsqueeze(-1)*memory).sum(-2)
        else:z=h*r*w
        return self.out(z)

class InteractionModel(nn.Module):
    def __init__(self,spec):
        super().__init__();self.spec=spec
        base=dict(spec,variant='R3_RAW',backbone='W0',head_width=32,head_current_power=False)
        self.base=PhysicalModel(base,22)
        self.interaction=ResponseInteraction(spec['seed'],spec['variant'])
    def forward(self,d):
        m=self.base
        extra=m.projection(d['x'])+m.neighbor(d)[0]+m.regional(d['region_features'])[:,None]+self.interaction(d)
        return m.backbone(d['h'],d['weather'],d['phase'],extra)+d['anchor']-d['h'][:,-1:]
