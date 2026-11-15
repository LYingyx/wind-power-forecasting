"""Shared tensor implementation for training, interventions and audit checks."""
import torch
from residual_inputs import family
from neighbor_inputs import window

def fields(d,variant,intervention=None):
    delay=int(intervention[5:]) if intervention and intervention.startswith('delay') else 15
    p=d['res_power'][...,window(delay)];m=d['res_mapped'][...,window(delay)]
    mask=d['res_mask'];valid=mask.nonzero().flatten()
    if variant.startswith('C1_'):p=m
    if intervention=='power_identity' and len(valid)>1:
        p=p.clone();p[:,valid]=p[:,valid.roll(1)]
    r=p-m
    if intervention=='residual_identity' and len(valid)>1:
        r=r.clone();r[:,valid]=r[:,valid.roll(1)]
    if intervention=='residual_zero':r=torch.zeros_like(r)
    if variant=='R2_PAIR':r=torch.zeros_like(r)
    if variant=='R1_SELF':
        p=d['h'][:,None,window(delay)].expand_as(p)
        m=d['self_mapped'][:,None,window(delay)].expand_as(m);r=p-m
    return p,m,r,delay

def moment(x,weight):
    weight=weight/weight.sum().clamp_min(1e-12)
    mean=(x*weight[None,:,None]).sum(1)
    var=((x[:,:,-1]-mean[:,-1:])**2*weight[None]).sum(1)
    return torch.stack((mean[:,-1],var.clamp_min(0).sqrt(),mean[:,-1]-mean[:,-5]),-1)

def summary(d,variant,intervention=None):
    p,m,r,delay=fields(d,variant,intervention);weights=d['res_capacity']*d['res_mask']
    if family(variant)=='R1':
        return torch.cat((moment(p if variant=='R1_RAWMAP' else r,weights),moment(m,weights)),-1)
    assert family(variant) in ['R3','R4']
    distance=d['res_geometry'][:,2];out=[];direction=[]
    wind=d['reference_physical'][:,[1,2]]
    if intervention=='direction_rotate':wind=-wind
    unit=d['res_geometry'][:,:2]/distance[:,None].clamp_min(.1)
    # Smooth attenuation for uncertain wind direction at low speed, no hard gate.
    speed=torch.sqrt(wind.square().sum(-1)+4)
    cosine=(wind[:,None,:]*unit[None]).sum(-1)/speed[:,None]
    sine=(wind[:,None,0]*unit[None,:,1]-wind[:,None,1]*unit[None,:,0])/speed[:,None]
    for lo,hi in [(0,20),(20,60),(60,150.0001)]:
        w=weights*((distance>=lo)&(distance<hi));present=(w.sum()>0).to(p)
        rp=moment(r,w);mp=moment(m,w)
        if variant=='R3_RAW':rp=rp*0;mp=mp*0
        out += [moment(p,w),rp,mp,present.expand(len(p),1)]
        z=w/w.sum().clamp_min(1e-12)
        directional=torch.stack(((r[:,:,-1]*cosine*z[None]).sum(1),(r[:,:,-1]*sine*z[None]).sum(1)),-1)
        direction.append(directional if family(variant)=='R4' else directional*0)
    return torch.cat(out+direction,-1)
