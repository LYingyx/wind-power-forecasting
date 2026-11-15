"""Matched, state/content-conditioned weather retrieval and fixed controls."""
from config import *
import sys,importlib.util,math
from torch import nn

LEGACY=Path(__file__).resolve().parent/'legacy'
sys.path.insert(0,str(LEGACY))
loader=importlib.util.spec_from_file_location('legacy_models',LEGACY/'models.py')
old=importlib.util.module_from_spec(loader);loader.loader.exec_module(old)
from wind_forcing.models.p0i_legacy import interpolate as legacy_interpolate

def patch_select(k):
 radius=k//2
 return np.array([r*7+c for r in range(3-radius,4+radius) for c in range(3-radius,4+radius)])

def interpolate_grid(w,times):
 # B,N,K,C -> legacy B,K,N,C -> B,L,K,C; same clamped interpolation.
 leads=torch.arange(1,25,device=w.device,dtype=w.dtype)[None].expand(len(w),-1)/4
 return legacy_interpolate(w.transpose(1,2),times,leads).transpose(1,2)

class Attention(nn.Module):
 def __init__(self,key_features):
  super().__init__();self.query=nn.Sequential(nn.Linear(28,32),nn.GELU(),nn.Linear(32,32))
  self.key=nn.Sequential(nn.Linear(key_features,32),nn.GELU(),nn.Linear(32,32,bias=False))
 def q(self,h,phase):
  l=torch.arange(1,25,device=h.device,dtype=h.dtype)/4
  time=torch.stack((l/6,torch.sin(2*torch.pi*l/24),torch.cos(2*torch.pi*l/24)),-1)
  return self.query(torch.cat((h[:,None].expand(-1,24,-1),time[None].expand(len(h),-1,-1),phase[:,None,None].to(h).expand(-1,24,1)/3),-1))

class Frontend(nn.Module):
 def __init__(self,site,kind,k):
  super().__init__();self.kind=kind;self.k=k;ix=patch_select(k)
  self.register_buffer('geometry',torch.from_numpy(np.load(RUN/'prepared/geometry.npy')[site,ix]))
  self.register_buffer('bilinear',torch.from_numpy(np.load(RUN/'prepared/bilinear_weights.npy')[site,ix]))
  self.register_buffer('prior',torch.from_numpy(np.load(RUN/'prepared/weather_prior.npy')[site,:,ix,:].transpose(1,0,2).copy()))
  # NumPy advanced indexing above puts patch axis first; canonical shape is N,K,C.
  assert self.prior.shape==(8,k*k,14),self.prior.shape
  self.register_buffer('hprior',torch.from_numpy(np.load(RUN/'prepared/history_prior.npy')[site]))
  self.register_buffer('nwp_mean',torch.from_numpy(np.load(CACHE/'static/nwp_mean.npy')))
  self.register_buffer('nwp_std',torch.from_numpy(np.load(CACHE/'static/nwp_std.npy')))
  self.has_space=kind in ['S2','S3','S4'] or kind.startswith('ST')
  self.has_time=kind.startswith('T') or kind.startswith('ST')
  if kind=='S1':self.flat=nn.Sequential(nn.Linear(14*k*k,32),nn.GELU(),nn.Linear(32,14))
  if self.has_space:self.spatial=Attention(17)
  if self.has_time:self.temporal=Attention(15)
  self.enabled=kind in FRONTS
  if self.enabled:self.gate=nn.Parameter(torch.zeros(()))
 def point_nodes(self,w):return (w*self.bilinear[None,None,:,None]).sum(2)
 def weights(self,h,w,phase,times):
  kind=self.kind;b=len(h);n=7 if kind.endswith('_7') else 8
  state=h if kind not in ['S2','T1_7','T1_8','ST00','ST_NO_H'] else self.hprior[None].expand_as(h)
  prior=self.prior[None].expand(b,-1,-1,-1)
  result={}
  if self.has_space:
   dynamic=kind in ['S3','S4','ST10','ST11','ST_NO_H']
   hs=h if dynamic and kind!='ST_NO_H' else self.hprior[None].expand_as(h)
   values=w if dynamic else prior
   if kind.startswith('ST'):
    geo=self.geometry[None,None].expand(b,8,-1,-1)
    kt=times[:,:,None,None].expand(-1,-1,self.k*self.k,1)/6
    key=self.spatial.key(torch.cat((values,geo,kt),-1))
    logits=torch.einsum('bld,bnkd->blnk',self.spatial.q(hs,phase),key)/math.sqrt(32)
   else:
    values=interpolate_grid(values,times)
    geo=self.geometry[None,None].expand(b,24,-1,-1)
    if kind=='S4':
     physical=values*self.nwp_std+self.nwp_mean
     u,v=physical[...,1],physical[...,2];speed=torch.sqrt(u*u+v*v).clamp_min(1e-3)
     dx,dy=geo[...,0],geo[...,1]
     geo=torch.stack((-(dx*u+dy*v)/speed,(-dx*v+dy*u)/speed),-1)
    kt=(torch.arange(1,25,device=h.device,dtype=h.dtype)/24)[None,:,None,None].expand(b,-1,self.k*self.k,1)
    key=self.spatial.key(torch.cat((values,geo,kt),-1))
    logits=torch.einsum('bld,blkd->blk',self.spatial.q(hs,phase),key)/math.sqrt(32)
   result['space']=logits.softmax(-1)
  if self.has_time:
   dynamic=kind in ['T2_7','T2_8','ST01','ST11','ST_NO_H']
   ht=h if dynamic and kind!='ST_NO_H' else self.hprior[None].expand_as(h)
   content=self.point_nodes(w if dynamic else prior)[:,:n]
   key=self.temporal.key(torch.cat((content,times[:,:n,None]/6),-1))
   result['time']=(torch.einsum('bld,bnd->bln',self.temporal.q(ht,phase),key)/math.sqrt(32)).softmax(-1)
  return result
 def forward(self,h,w,phase,times,override=None,return_weights=False):
  point=interpolate_grid(self.point_nodes(w)[:,:,None],times)[:,:,0]
  if not self.enabled:return (point,{}) if return_weights else point
  weights=self.weights(h,w,phase,times) if override is None else override
  kind=self.kind
  if kind=='S1':candidate=self.flat(interpolate_grid(w,times).flatten(2))
  elif kind.startswith('ST'):
   spatial=torch.einsum('blnk,bnkc->blnc',weights['space'],w)
   candidate=torch.einsum('bln,blnc->blc',weights['time'],spatial)
  elif kind.startswith('S'):candidate=torch.einsum('blk,blkc->blc',weights['space'],interpolate_grid(w,times))
  else:
   n=7 if kind.endswith('_7') else 8
   candidate=torch.einsum('bln,bnc->blc',weights['time'],self.point_nodes(w)[:,:n])
  out=point+self.gate.tanh()*(candidate-point)
  return (out,weights) if return_weights else out

class Model(nn.Module):
 def __init__(self,spec):
  super().__init__();self.spec=spec;site,seed,arm=spec['site'],spec['seed'],spec['front'];b=spec['backbone']
  seed_all(seed)
  if b=='NEUTRAL_LAST':
   refpath=OLD/'jobs'/f'reference_B0_s{site:02d}_r{seed}'/'best.pt'
   state=torch.load(refpath,map_location='cpu',weights_only=False)['model']
   self.backbone=old.correction(site,seed,64,'CB' if arm=='H1' else 'NEUTRAL_LAST',state)
   self.backbone.output_mode='last'
  else:self.backbone=old.named(b,spec['backbone_config']['width'])
  # Independent RNG stream: adding a front never changes backbone initialization.
  with torch.random.fork_rng(devices=[]):
   torch.manual_seed(seed+920221)
   self.frontend=Frontend(site,arm,spec['k'])
  self.register_buffer('hprior',torch.from_numpy(np.load(RUN/'prepared/history_prior.npy')[site]))
  self.register_buffer('climatology',torch.from_numpy(np.load(OLD/f'prepared/train/{site:02d}/weather_climatology.npy')))
 def forward(self,h,w,phase,times,base=None,override=None,return_weights=False):
  weather,weights=self.frontend(h,w,phase,times,override=override,return_weights=True)
  arm=self.spec['front'];m=self.backbone
  if arm in ['CLIM_LAST','H2_CLIM']:weather=self.climatology[None].expand(len(h),-1,-1)
  if self.spec['backbone']!='NEUTRAL_LAST':out=m(h,weather,phase)
  elif arm not in ['H2','H2_CLIM']:
   out=m(h,weather,phase,base if base is not None else torch.zeros_like(h))
  else:
   hp=self.hprior[None].expand_as(h);b=len(h)
   lead=torch.arange(1,25,device=h.device,dtype=h.dtype)[None].expand(b,-1)/4
   time=torch.stack((lead/6,torch.sin(2*torch.pi*lead/24),torch.cos(2*torch.pi*lead/24)),-1)
   context=torch.cat((time,m.site[0][None,None].expand(b,24,-1),phase[:,None,None].to(h).expand(-1,24,-1)/3),-1)
   ht=torch.arange(-23,1,device=h.device,dtype=h.dtype)/24
   memory=m.history_encoder(m.history_projection(torch.stack((hp,ht.expand_as(hp)),-1)))
   wv=m.align_out(m.align_v(m.native_projection(weather[:,None])[:,0]))
   raw=torch.cat((torch.zeros(b,24,64,device=h.device),torch.zeros(b,24,2,device=h.device),lead[...,None]/6),-1)
   cond=m.condition_projection(raw)+m.condition_bias[:,0]
   query=m.context_projection(context)+wv+m.transfer_adapter(torch.cat((wv,cond),-1))
   out=h[:,-1:]+m.readout(m.decoder(query,memory)).squeeze(-1)
  return (out,weights) if return_weights else out
 def groups(self):
  groups=[dict(params=[p for p in self.backbone.parameters() if p.requires_grad],lr=self.spec['backbone_config']['lr'])]
  pars=list(self.frontend.parameters())
  if pars:groups.append(dict(params=pars,lr=self.spec['front_lr']))
  return groups
