from settings import *
import numpy as np,torch
from torch import nn
from physical_models import PhysicalModel,frozen

def interp(nodes,offset):
    q=torch.arange(1,25,device=nodes.device,dtype=nodes.dtype)[None]/4-offset[:,:1]
    lo=q.floor().long().clamp(0,6);a=(q-lo)[...,None]
    return nodes.gather(1,lo[...,None].expand(-1,-1,nodes.shape[-1]))*(1-a)+nodes.gather(1,(lo+1)[...,None].expand(-1,-1,nodes.shape[-1]))*a

def thermo(w,z):
    t=w[...,3];ct=(t+.0065*z/2).clamp_min(180);ps=w[...,0]*torch.exp(-9.80665*z/(287.05*ct))
    th=(t-.0065*100).clamp_min(180);ph=ps*torch.exp(-9.80665*100/(287.05*((t+th)/2).clamp_min(180)));rho=ph/(287.05*th)
    ts=t*(1e5/ps.clamp_min(1)).pow(.2854);t925=w[...,9]*(1000/925)**.2854
    height925=w[...,5]/9.80665-z;valid=height925>100;height=(height925-10).clamp_min(1)
    du=w[...,11]-w[...,1];dv=w[...,13]-w[...,2]
    zero=torch.zeros_like(t)
    grad=torch.where(valid,100*(t925-ts)/height,zero)
    shear=torch.where(valid,100*torch.sqrt(du.square()+dv.square()+1e-12)/height,zero)
    rb=torch.where(valid,9.80665/ts.clamp_min(180)*(t925-ts)*height/(du.square()+dv.square()+1),zero).clamp(-5,5)
    alpha=torch.where(valid,np.log(10.)/torch.log(height925.clamp_min(101)/10),zero)
    u=w[...,1];v=w[...,2];speed=torch.sqrt(u.square()+v.square()+1e-12)
    hub=torch.sqrt((u+alpha*du).square()+(v+alpha*dv).square()+1e-12)
    den=torch.stack([rho/1.225,(rho/1.225).pow(1/3)*speed/10,rho*speed.pow(3)/1000],-1)
    stab=torch.stack([grad,shear,rb,valid.to(w.dtype),hub/10],-1)
    return den,stab

class CorrectedModel(nn.Module):
    def __init__(self,spec,stats):
        super().__init__();self.spec=spec;self.variant=spec['variant']
        base=dict(spec,variant='R3_RAW',backbone='W0',head_width=32,head_current_power=False)
        self.base=PhysicalModel(base,22)
        self.register_buffer('nwp_mu',torch.from_numpy(np.load(CACHE/'static/nwp_mean.npy')))
        self.register_buffer('nwp_sd',torch.from_numpy(np.load(CACHE/'static/nwp_std.npy')))
        self.register_buffer('sel_sd',torch.tensor(stats['std'],dtype=torch.float32))
        self.register_buffer('elevation',torch.tensor(float(np.load(DATA/'elevation.npy')[spec['site']])))
        if self.variant!='E0':
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(spec['seed']+61101)
                # All correction arms have identical capacity and initialization.
                self.corrector=nn.Sequential(nn.Linear(301,128),nn.GELU(),nn.Linear(128,64),nn.GELU(),nn.Linear(64,16))
                nn.init.zeros_(self.corrector[-1].weight);nn.init.zeros_(self.corrector[-1].bias)
    def forward(self,d):
        nodes=d['nodes']
        if self.variant=='E0':return self.base(d),nodes[...,1:3]
        context=torch.cat([torch.cat([d['past'],nodes],1).flatten(1),d['corr_obs'],d['all_times']/6,d['age']],-1)
        delta=self.corrector(context).reshape(-1,8,2)
        corrected=nodes.clone();corrected[...,1:3]=nodes[...,1:3]+delta
        new=dict(d);dw=interp(delta,d['offset']);weather=d['weather'].clone();weather[...,1:3]+=dw
        new['weather']=weather;physical=weather*self.nwp_sd+self.nwp_mu;new['physical']=physical
        oldden,oldstab=thermo(d['physical'],self.elevation);den,stab=thermo(physical,self.elevation)
        dyn_delta=torch.zeros_like(d['dynamics']);dyn_delta[...,2:4]=dw*self.nwp_sd[1:3]/10
        oldp=nodes*self.nwp_sd+self.nwp_mu;newp=corrected*self.nwp_sd+self.nwp_mu
        oldspeed=torch.sqrt(oldp[...,1].square()+oldp[...,2].square()+1e-12)
        newspeed=torch.sqrt(newp[...,1].square()+newp[...,2].square()+1e-12)
        temporal=torch.gradient(newspeed-oldspeed,dim=1)[0]
        dyn_delta[...,6]=interp(temporal[...,None],d['offset']).squeeze(-1)
        shift=torch.cat([den-oldden,dyn_delta,stab-oldstab,torch.zeros_like(d['x'][...,-7:])],-1)
        new['x']=d['x']+shift/self.sel_sd
        return self.base(new),corrected[...,1:3]
