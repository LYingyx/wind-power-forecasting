from settings import *
import sys,math,copy,random
import numpy as np
import torch
from torch import nn
sys.path.insert(0,str(RUN/'code/legacy'))
import models as old
import common
common.CACHE=CACHE
old.CACHE=CACHE
old.static=lambda: {p.stem:np.load(p,allow_pickle=False) for p in (CACHE/'static').glob('*.npy')}
class Frozen:
    old=old
    @staticmethod
    def seed_all(seed):
        random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
        if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    @staticmethod
    def configure():
        torch.set_num_threads(1)
        try:torch.set_num_interop_threads(1)
        except RuntimeError:pass
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.backends.cudnn.benchmark=False
frozen=Frozen()
from neighbor_inputs import topology,window,ages
ACTIVE=['history_encoder','history_projection','native_projection','align_out','align_v','condition_projection','context_projection','transfer_adapter','decoder','readout']
class Backbone(nn.Module):
    def __init__(self,kind,site,seed):
        super().__init__();self.kind=kind;frozen.seed_all(seed)
        if kind=='W0':
            import copy
            cfg=copy.deepcopy(frozen.old.ref.CBH_CONFIG);cfg['condition_mode']='neutral'
            full=frozen.old.static()
            original=frozen.old.CachedCorrection(cfg,frozen.old.ref.single_site_static(full,site),24)
            _,sv=frozen.old.ref.canonical_static_vectors(full)
            original.site.copy_(sv[site:site+1])
            original.native_projection=frozen.old.WeatherProjection('E2C',full,str(ASSETS/'dem.json'),str(ASSETS/'column_stats.json'),64)
            original.native_projection.elevation=original.native_projection.elevation[:,site:site+1].clone()
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
            out=self.decoder(query,memory)
            if getattr(self,'readout_current_power',False):
                out=torch.cat((out,h[:,-1:,None].expand(-1,24,-1)),-1)
            return h[:,-1:]+self.readout(out).squeeze(-1)
        m=self.original;b=len(h)
        clock=torch.linspace(-1,0,24,device=h.device,dtype=h.dtype)[None,:,None].expand(b,-1,-1)
        memory=m.h_enc(m.pe(m.h_in(torch.cat((h[...,None],clock),-1))))
        lead=torch.arange(1,25,device=h.device,dtype=h.dtype)[None].expand(b,-1)/24
        features=torch.stack((lead,torch.sin(2*math.pi*lead),torch.cos(2*math.pi*lead),phase[:,None].float()/3+lead*0),-1)
        query=m.pe(m.w_in(torch.cat((w,features),-1)))
        if extra is not None:query=query+extra
        out=m.dec(query,memory)
        return h[:,-1:]+m.head(torch.cat((out,h[:,-1:,None].expand(-1,24,-1)),-1)).squeeze(-1)

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

class PhysicalModel(nn.Module):
 def __init__(self,spec,feature_dim,historical_prior=None):
    super().__init__();self.spec=spec;self.variant=spec['variant'];frozen.seed_all(spec['seed'])
    if self.variant=='B0':
        self.baseline=old.baseline(spec['site'],spec['seed']);return
    self.backbone=Backbone(spec['backbone'],spec['site'],spec['seed'])
    if spec['backbone']=='TF':
        head=[m for m in self.backbone.original.head.modules() if isinstance(m,nn.Linear)][-1]
        nn.init.zeros_(head.weight);nn.init.zeros_(head.bias)
        for m in self.backbone.original.modules():
            if isinstance(m,nn.Dropout):m.p=.05
            if isinstance(m,nn.MultiheadAttention):m.dropout=.05
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(spec['seed']+241024)
        self.projection=nn.Linear(feature_dim,64,bias=False);nn.init.zeros_(self.projection.weight)
        if self.variant in ['N3','R3_RAW','C2_R3_RAW']:
            self.neighbor=ObservationReadout(spec['site'],'N3',15)
        if self.variant in ['R3_RAW','C2_R3_RAW']:
            self.regional=nn.Linear(12,64,bias=False);nn.init.zeros_(self.regional.weight)
        if self.variant in ['AGE','LOCAL_ERR','C2_LOCAL_ERR','REGION_ERR','C2_REGION_ERR']:
            self.age_projection=nn.Linear(1,64,bias=False);nn.init.zeros_(self.age_projection.weight)
        if self.variant in ['LOCAL_ERR','C2_LOCAL_ERR','REGION_ERR','C2_REGION_ERR']:
            self.error_projection=nn.Linear(24 if 'REGION' in self.variant else 12,64,bias=False);nn.init.zeros_(self.error_projection.weight)
 def forward(self,d,intervention=None):
    if self.variant=='B0':
        return self.baseline(station_history=d['h'][:,None],cluster_history=d['h'],origin_phase=d['phase'])['station_factor'][:,0]
    extra=self.projection(d['x'])
    if hasattr(self,'neighbor'):extra=extra+self.neighbor(d)[0]
    if hasattr(self,'regional'):extra=extra+self.regional(d['region_features'])[:,None]
    if hasattr(self,'age_projection'):extra=extra+self.age_projection(d['age'])[:,None]
    if hasattr(self,'error_projection'):extra=extra+self.error_projection(d['error_features'])[:,None]
    return self.backbone(d['h'],d['weather'],d['phase'],extra)+d['anchor']-d['h'][:,-1:]
 def groups(self):return [dict(params=[p for p in self.parameters() if p.requires_grad],lr=2e-4)]

OriginalPhysicalModel=PhysicalModel
class PhysicalModel(OriginalPhysicalModel):
 def __init__(self,spec,feature_dim,historical_prior=None):
    if spec['backbone']!='NAMED':
        super().__init__(spec,feature_dim,historical_prior);return
    nn.Module.__init__(self);self.spec=spec;self.variant=spec['variant'];frozen.seed_all(spec['seed'])
    self.named_model=old.named(spec['family'],spec['model_config']['width'])
 def forward(self,d,intervention=None):
    if hasattr(self,'named_model'):return self.named_model(d['h'],d['weather'],d['phase'])
    return super().forward(d,intervention)
 def groups(self):
    cfg=self.spec.get('model_config',dict(lr=2e-4,weight_decay=1e-4))
    return [dict(params=[p for p in self.parameters() if p.requires_grad],lr=cfg['lr'],weight_decay=cfg['weight_decay'])]

ReferencePhysicalModel=PhysicalModel
class PhysicalModel(ReferencePhysicalModel):
 def __init__(self,spec,feature_dim,historical_prior=None):
    super().__init__(spec,feature_dim,historical_prior)
    if spec['backbone']=='W0':
        width=spec['head_width'];direct=spec['head_current_power']
        assert width in [32,64] and isinstance(direct,bool)
        self.backbone.readout_current_power=direct
        if width!=32 or direct:
            # Do not perturb shared module initialization or dropout RNG stream.
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(spec['seed']+281028)
                original=self.backbone.readout
                master=nn.Linear(64,64)
                current_column=nn.Linear(1,64,bias=False).weight.detach().clone()/8
                with torch.no_grad():
                    master.weight[:32].copy_(original[0].weight)
                    master.bias[:32].copy_(original[0].bias)
                self.backbone.readout=nn.Sequential(nn.Linear(64+int(direct),width),nn.GELU(),nn.Linear(width,1))
                with torch.no_grad():
                    self.backbone.readout[0].weight[:,:64].copy_(master.weight[:width])
                    self.backbone.readout[0].bias.copy_(master.bias[:width])
                    if direct:self.backbone.readout[0].weight[:,64:].copy_(current_column[:width])
                nn.init.zeros_(self.backbone.readout[-1].weight)
                nn.init.zeros_(self.backbone.readout[-1].bias)
