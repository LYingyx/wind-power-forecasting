"""Baseline-replayable query additions and explicit, falsifiable mechanisms."""
from settings import *
import sys,math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
sys.path.insert(0,str(PREVIOUS/'code'))
import models as frozen
from physical_data import interpolate
from neighbor_inputs import NEIGHBORS,topology,window,ages

ACTIVE=['history_encoder','history_projection','native_projection','align_out','align_v',
        'condition_projection','context_projection','transfer_adapter','decoder','readout']

class Backbone(nn.Module):
    def __init__(self,kind,site,seed):
        super().__init__();self.kind=kind;frozen.seed_all(seed)
        if kind=='W0':
            state=torch.load(OLD/f'jobs/reference_B0_s{site:02d}_r{seed}/best.pt',map_location='cpu',weights_only=False)['model']
            original=frozen.old.correction(site,seed,64,'NEUTRAL_LAST',state)
            for name in ACTIVE:setattr(self,name,getattr(original,name))
            self.register_buffer('site',original.site.detach().clone())
            self.condition_bias=nn.Parameter(original.condition_bias.detach().clone())
        else:self.original=frozen.old.named('TRANSFORMER_FC',64)
    def forward(self,h,w,phase,extra=None):
        if self.kind=='W0':
            b=len(h);lead=torch.arange(1,25,device=h.device,dtype=h.dtype)[None].expand(b,-1)/4
            time=torch.stack((lead/6,torch.sin(2*torch.pi*lead/24),torch.cos(2*torch.pi*lead/24)),-1)
            context=torch.cat((time,self.site[0][None,None].expand(b,24,-1),phase[:,None,None].to(h).expand(-1,24,-1)/3),-1)
            ht=torch.arange(-23,1,device=h.device,dtype=h.dtype)/24
            memory=self.history_encoder(self.history_projection(torch.stack((h,ht.expand_as(h)),-1)))
            wv=self.align_out(self.align_v(self.native_projection(w[:,None])[:,0]))
            raw=torch.cat((torch.zeros(b,24,66,device=h.device),lead[...,None]/6),-1)
            cond=self.condition_projection(raw)+self.condition_bias[:,0]
            query=self.context_projection(context)+wv+self.transfer_adapter(torch.cat((wv,cond),-1))
            if extra is not None:query=query+extra
            return h[:,-1:]+self.readout(self.decoder(query,memory)).squeeze(-1)
        m=self.original;b=len(h)
        clock=torch.linspace(-1,0,24,device=h.device,dtype=h.dtype)[None,:,None].expand(b,-1,-1)
        memory=m.h_enc(m.pe(m.h_in(torch.cat((h[...,None],clock),-1))))
        lead=torch.arange(1,25,device=h.device,dtype=h.dtype)[None].expand(b,-1)/24
        features=torch.stack((lead,torch.sin(2*math.pi*lead),torch.cos(2*math.pi*lead),phase[:,None].float()/3+lead*0),-1)
        query=m.pe(m.w_in(torch.cat((w,features),-1)))
        if extra is not None:query=query+extra
        out=m.dec(query,memory)
        return h[:,-1:]+m.head(torch.cat((out,h[:,-1:,None].expand(-1,24,-1)),-1)).squeeze(-1)

class NeighborReadout(nn.Module):
    def __init__(self,site,advection):
        super().__init__();self.advection=advection
        self.query=nn.Sequential(nn.Linear(30,32),nn.GELU(),nn.Linear(32,32))
        self.key=nn.Sequential(nn.Linear(6,32),nn.GELU(),nn.Linear(32,32))
        self.value=nn.Linear(2,32,bias=False);self.out=nn.Linear(32,64,bias=False)
        nn.init.zeros_(self.out.weight)
        self.register_buffer('geometry',torch.from_numpy(np.load(PREPARED/'neighbor_geometry.npy')[site]))
        self.register_buffer('mask',torch.from_numpy(np.load(PREPARED/'neighbor_mask.npy')[site]))
    def forward(self,d,intervention=None):
        h=d['h'];p=d['neighbors'];b,k,t=p.shape;geo=self.geometry
        if intervention=='neighbor_reverse':p=p.flip(-1)
        if intervention=='neighbor_identity':p=p.roll(1,1)
        if intervention=='neighbor_zero':p=torch.zeros_like(p)
        delta=torch.diff(p,dim=-1,prepend=p[:,:,:1])
        age=d['neighbor_ages'][None,None].expand(b,k,-1)
        geometry=geo[None,:,None].expand(b,-1,t,-1)/100
        key=self.key(torch.cat((p[...,None],delta[...,None],age[...,None]/6,geometry),-1).flatten(1,2))
        value=self.value(torch.stack((p,delta),-1).flatten(1,2))
        lead=torch.arange(1,25,device=h.device,dtype=h.dtype)/4
        time=torch.stack((lead/6,torch.sin(2*torch.pi*lead/24),torch.cos(2*torch.pi*lead/24)),-1)
        uv=d['physical'][...,[1,2]]/10
        q=self.query(torch.cat((h[:,None].expand(-1,24,-1),time[None].expand(b,-1,-1),d['phase'][:,None,None].to(h).expand(-1,24,1)/3,uv),-1))
        # Null token permits no neighbor contribution, including stations with no close peers.
        key=torch.cat((key,torch.zeros(b,1,32,device=h.device)),1)
        value=torch.cat((value,torch.zeros(b,1,32,device=h.device)),1)
        logits=torch.einsum('bld,bnd->bln',q,key)/math.sqrt(32)
        mask=torch.cat((self.mask[:,None].expand(-1,t).flatten(),torch.ones(1,dtype=torch.bool,device=h.device)))
        logits=logits.masked_fill(~mask[None,None],-1e4)
        if self.advection:
            wind=uv*10
            if intervention=='direction_rotate':wind=-wind
            speed=wind.norm(dim=-1).clamp_min(2.)
            direction=wind/speed[...,None]
            along=torch.einsum('bld,kd->blk',direction,geo[:,:2])
            cosine=along/geo[:,2].clamp_min(.1)
            travel=(along.clamp_min(0)/(speed[...,None]*3.6)).clamp(max=12)
            arrival=lead[None,:,None,None]-age[:,None]
            prior=2*(cosine-1)[...,None]-.5*((arrival-travel[...,None])/1.)**2
            # Fixed finite prior, not proof of atmospheric signal transport at 10m speed.
            prior=prior.clamp(min=-12).flatten(2)
            logits=logits+torch.cat((prior,torch.zeros(b,24,1,device=h.device)),2)
        weights=logits.softmax(-1)
        return self.out(torch.einsum('bln,bnd->bld',weights,value)),weights

class ResponseCurve(nn.Module):
    def __init__(self,monotone):
        super().__init__();self.monotone=monotone
        self.increments=nn.Parameter(torch.full((16,),-2. if monotone else float(F.softplus(torch.tensor(-2.)))))
        self.amplitude=nn.Parameter(torch.tensor(1.));self.decay=nn.Parameter(torch.tensor(2.));self.gate=nn.Parameter(torch.zeros(()))
        self.register_buffer('knots',torch.linspace(0,24,16))
    def response(self,v):
        weights=F.softplus(self.increments) if self.monotone else self.increments
        basis=((v[...,None]-self.knots)/1.6).clamp(0,1)
        return F.softplus(self.amplitude)*(basis*weights).sum(-1)/weights.abs().sum().clamp_min(1e-6)
    def forward(self,d):
        # Density-adjusted wind proxy, in m/s. No claim that this is observed hub wind.
        m=self.response(d['density'][...,1]*10);m0=self.response(d['ref_density'][:,1]*10)
        lead=torch.arange(1,25,device=m.device,dtype=m.dtype)/4
        alpha=torch.exp(-lead/(.25+F.softplus(self.decay)))
        structured=m+alpha[None]*(d['h'][:,-1:]-m0[:,None])
        return self.gate.tanh()*(structured-d['h'][:,-1:])

class ObservationReadout(nn.Module):
    def __init__(self,site,variant,delay=15):
        super().__init__();self.variant=variant;self.delay=delay
        ids,geo,mask,capacity=topology(site,variant)
        self.query=nn.Sequential(nn.Linear(30,32),nn.GELU(),nn.Linear(32,32))
        self.key=nn.Sequential(nn.Linear(6,32),nn.GELU(),nn.Linear(32,32))
        self.value=nn.Linear(2,32,bias=False);self.out=nn.Linear(32,64,bias=False);nn.init.zeros_(self.out.weight)
        self.register_buffer('geometry',torch.from_numpy(geo));self.register_buffer('mask',torch.from_numpy(mask))
        self.register_buffer('capacity',torch.as_tensor(capacity,dtype=torch.float32))
    def values(self,d,intervention=None):
        delay=int(intervention[5:]) if intervention and intervention.startswith('delay') else self.delay
        p=d['neighbor_raw'][...,window(delay)]
        if self.variant=='N2_SELF':p=d['h'][:,None,window(delay)].expand(-1,p.shape[1],-1)
        if self.variant=='N3':
            weight=self.capacity*self.mask
            pooled=(p*weight[None,:,None]).sum(1)/weight.sum().clamp_min(1e-9)
            p=pooled[:,None].expand_as(p)
        if intervention=='neighbor_identity':
            valid=self.mask.nonzero().flatten()
            if len(valid)>1:
                p=p.clone();p[:,valid]=p[:,valid.roll(1)]
        if intervention=='neighbor_reverse':p=p.flip(-1)
        if intervention=='neighbor_zero':p=torch.zeros_like(p)
        return p,torch.as_tensor(ages(delay),device=p.device,dtype=p.dtype)
    def forward(self,d,intervention=None):
        h=d['h'];p,agevector=self.values(d,intervention);b,k,t=p.shape
        delta=torch.diff(p,dim=-1,prepend=p[:,:,:1]);age=agevector[None,None].expand(b,k,-1)
        geometry=self.geometry[None,:,None].expand(b,-1,t,-1)/100
        key=self.key(torch.cat((p[...,None],delta[...,None],age[...,None]/6,geometry),-1).flatten(1,2))
        value=self.value(torch.stack((p,delta),-1).flatten(1,2))
        lead=torch.arange(1,25,device=h.device,dtype=h.dtype)/4
        time=torch.stack((lead/6,torch.sin(2*torch.pi*lead/24),torch.cos(2*torch.pi*lead/24)),-1)
        query=self.query(torch.cat((h[:,None].expand(-1,24,-1),time[None].expand(b,-1,-1),d['phase'][:,None,None].to(h).expand(-1,24,1)/3,d['physical'][...,[1,2]]/10),-1))
        key=torch.cat((key,torch.zeros(b,1,32,device=h.device)),1);value=torch.cat((value,torch.zeros(b,1,32,device=h.device)),1)
        logits=torch.einsum('bld,bnd->bln',query,key)/math.sqrt(32)
        mask=torch.cat((self.mask[:,None].expand(-1,t).flatten(),torch.ones(1,dtype=torch.bool,device=h.device)))
        weights=logits.masked_fill(~mask[None,None],-1e4).softmax(-1)
        return self.out(torch.einsum('bln,bnd->bld',weights,value)),weights

class LegacyPhysicalModel(nn.Module):
    def __init__(self,spec,feature_dim,historical_prior=None):
        super().__init__();self.spec=spec;self.variant=spec['variant']
        self.backbone=Backbone(spec['backbone'],spec['site'],spec['seed'])
        if spec['backbone']=='TF':
            if spec.get('tf_zero_init',False):
                head=[m for m in self.backbone.original.head.modules() if isinstance(m,nn.Linear)][-1]
                nn.init.zeros_(head.weight);nn.init.zeros_(head.bias)
            if 'tf_dropout' in spec:
                for module in self.backbone.original.modules():
                    if isinstance(module,nn.Dropout):module.p=spec['tf_dropout']
                    if isinstance(module,nn.MultiheadAttention):module.dropout=spec['tf_dropout']
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(spec['seed']+241024)
            self.projection=nn.Linear(feature_dim,64,bias=False) if feature_dim else None
            if self.projection is not None:nn.init.zeros_(self.projection.weight)
            if self.variant in NEIGHBORS:self.neighbor=ObservationReadout(spec['site'],self.variant,spec.get('delay_minutes',15))
            if self.variant.startswith('NEIGH_'):self.neighbor=NeighborReadout(spec['site'],self.variant=='NEIGH_ADVECT')
            if self.variant.startswith('CURVE_'):self.curve=ResponseCurve(self.variant=='CURVE_BIAS')
            if self.variant.startswith('SHIFT_'):
                self.shift=nn.Sequential(nn.Linear(114,32),nn.GELU(),nn.Linear(32,1))
                nn.init.zeros_(self.shift[-1].weight);nn.init.zeros_(self.shift[-1].bias)
                self.register_buffer('historical_prior',torch.as_tensor(historical_prior,dtype=torch.float32))
    def forward(self,d,intervention=None,return_details=False):
        extra=self.projection(d['x']) if self.projection is not None else None;details={};w=d['weather']
        if hasattr(self,'shift'):
            state=d['historical_input']
            if self.variant=='SHIFT_FIXED':state=self.historical_prior[None].expand_as(state)
            shift=self.shift(state).tanh()[:,0] # bounded +/- 1h, zero initialized
            if self.variant=='SHIFT_ZERO':shift=shift*0
            queries=torch.arange(1,25,device=w.device,dtype=w.dtype)[None]/4+shift[:,None]
            w=interpolate(d['allnodes'],d['all_times'],queries)
            # Shift zero must preserve the existing front, including t0 endpoint clamp.
            zero=interpolate(d['allnodes'],d['all_times'],torch.arange(1,25,device=w.device,dtype=w.dtype)/4)
            w=d['weather']+(w-zero);details['shift_hours']=shift
        if hasattr(self,'neighbor'):
            n,weights=self.neighbor(d,intervention);extra=n if extra is None else extra+n;details['neighbor_weights']=weights
        out=self.backbone(d['h'],w,d['phase'],extra)
        if hasattr(self,'curve'):out=out+self.curve(d)
        return (out,details) if return_details else out

    def groups(self):
        base=list(self.backbone.parameters());ids={id(p) for p in base}
        extra=[p for p in self.parameters() if id(p) not in ids]
        groups=[dict(params=base,lr=2e-4 if self.spec['backbone']=='W0' else self.spec.get('tf_lr',1e-3))]
        if extra:groups.append(dict(params=extra,lr=2e-4))
        return groups


from residual_inputs import RESIDUAL_VARIANTS, family
from residual_features import fields, summary as residual_summary

class ResidualReadout(nn.Module):
    def __init__(self,variant):
        super().__init__();self.variant=variant
        self.query=nn.Sequential(nn.Linear(30,32),nn.GELU(),nn.Linear(32,32))
        self.key=nn.Sequential(nn.Linear(10,32),nn.GELU(),nn.Linear(32,32))
        self.value=nn.Linear(6,32,bias=False);self.out=nn.Linear(32,64,bias=False);nn.init.zeros_(self.out.weight)
    def forward(self,d,intervention=None):
        p,m,r,delay=fields(d,self.variant,intervention);b,k,t=p.shape;h=d['h']
        values=torch.stack((p,r,m,torch.diff(p,prepend=p[:,:,:1],dim=-1),torch.diff(r,prepend=r[:,:,:1],dim=-1),torch.diff(m,prepend=m[:,:,:1],dim=-1)),-1)
        age=torch.as_tensor(ages(delay),device=h.device,dtype=h.dtype)[None,None,:,None].expand(b,k,t,1)/6
        geo=d['res_geometry'][None,:,None,:].expand(b,k,t,3)/100
        key=self.key(torch.cat((values,age,geo),-1).flatten(1,2));value=self.value(values.flatten(1,2))
        lead=torch.arange(1,25,device=h.device,dtype=h.dtype)/4
        time=torch.stack((lead/6,torch.sin(2*torch.pi*lead/24),torch.cos(2*torch.pi*lead/24)),-1)
        query=self.query(torch.cat((h[:,None].expand(-1,24,-1),time[None].expand(b,-1,-1),d['phase'][:,None,None].to(h).expand(-1,24,1)/3,d['physical'][...,[1,2]]/10),-1))
        key=torch.cat((key,torch.zeros(b,1,32,device=h.device)),1);value=torch.cat((value,torch.zeros(b,1,32,device=h.device)),1)
        mask=torch.cat((d['res_mask'][:,None].expand(-1,t).flatten(),torch.ones(1,dtype=torch.bool,device=h.device)))
        weights=(torch.einsum('bld,bnd->bln',query,key)/math.sqrt(32)).masked_fill(~mask[None,None],-1e4).softmax(-1)
        return self.out(torch.einsum('bln,bnd->bld',weights,value)),weights

class PhysicalModel(LegacyPhysicalModel):
    def __init__(self,spec,feature_dim,historical_prior=None):
        variant=spec['variant'];base=('N1' if family(variant)=='R2' else 'N3') if variant in RESIDUAL_VARIANTS else variant
        super().__init__(dict(spec,variant=base),feature_dim,historical_prior)
        self.spec=spec;self.variant=variant
        if variant not in RESIDUAL_VARIANTS:return
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(spec['seed']+251025)
            if family(variant)=='R2':self.neighbor=ResidualReadout(variant)
            else:
                self.residual_projection=nn.Linear(6 if family(variant)=='R1' else 36,64,bias=False)
                nn.init.zeros_(self.residual_projection.weight)
    def forward(self,d,intervention=None,return_details=False):
        if self.variant not in RESIDUAL_VARIANTS:return super().forward(d,intervention,return_details)
        extra=self.projection(d['x']);details={}
        if family(self.variant)=='R2':n,w=self.neighbor(d,intervention)
        else:
            dd=d
            if intervention=='power_identity':
                dd=dict(d,neighbor_raw=d['neighbor_raw'].clone())
                valid=self.neighbor.mask.nonzero().flatten()
                dd['neighbor_raw'][:,valid]=dd['neighbor_raw'][:,valid.roll(1)]
            n,w=self.neighbor(dd,intervention if intervention and intervention.startswith('delay') else None)
            f=d['res_features'] if intervention is None else (residual_summary(d,self.variant,intervention)-d['res_mean'])/d['res_std']
            extra=extra+self.residual_projection(f)[:,None]
        extra=extra+n;details['neighbor_weights']=w
        out=self.backbone(d['h'],d['weather'],d['phase'],extra)
        return (out,details) if return_details else out


ResidualPhysicalModel = PhysicalModel
from regional_inputs import REGIONAL_VARIANTS, latency, summary as regional_summary, neighbor_readout

class PhysicalModel(ResidualPhysicalModel):
    def __init__(self, spec, feature_dim, historical_prior=None):
        variant = spec['variant']
        base = dict(spec, variant='R3_RAW', delay_minutes=latency(variant, spec.get('delay_minutes', 15))) if variant in REGIONAL_VARIANTS else spec
        super().__init__(base, feature_dim, historical_prior)
        self.spec = spec; self.variant = variant
        self.region_delay = latency(variant, spec.get('delay_minutes', 15))

    def forward(self, d, intervention=None, return_details=False):
        if self.variant not in REGIONAL_VARIANTS:
            return super().forward(d, intervention, return_details)
        delay = int(intervention[5:]) if intervention and intervention.startswith('delay') else self.region_delay
        f = d['res_features'] if intervention is None else (regional_summary(d, self.variant, delay, intervention)-d['res_mean'])/d['res_std']
        n, weights = neighbor_readout(self.neighbor, d, delay, intervention)
        if self.variant == 'R3_NO_N3': n = n*0
        # Same addition order and parameter names as the original R3_RAW.
        extra = self.projection(d['x']) + self.residual_projection(f)[:, None]
        extra = extra + n
        out = self.backbone(d['h'], d['weather'], d['phase'], extra)
        return (out, dict(neighbor_weights=weights)) if return_details else out
