"""Geometry-only masks and capacity-matched subsets, independent of outcomes."""
from settings import ASSETS
from validation_design import clean,MATCH,FULL
from functools import lru_cache
import numpy as np
import torch

@lru_cache(None)
def layout(site,variant,seed=42):
    v=clean(variant)
    with np.load(ASSETS/'topology.npz') as z:
        ids,geo,valid,cap=[z[n][site].copy() for n in ['ids','geometry','mask','capacity']]
    if v=='EXCLUDE20_REFILL':
        allowed=valid&(geo[:,2]>=20)
        order=np.r_[np.flatnonzero(allowed),np.flatnonzero(~allowed)]
        ids,geo,valid,cap=[a[order] for a in [ids,geo,valid,cap]]
    if v.startswith('EXCLUDE20_'):valid &= geo[:,2]>=20
    return ids,geo,valid,cap

@lru_cache(None)
def matched(site,variant,seed):
    with np.load(ASSETS/'matched_subsets.npz') as z:
        return z[clean(variant)][seed-42,site].copy()

def regional_moment_weights(d,variant):
    # Matched arms all use one pool, equal weights and the same number of farms.
    # Identity subsets are chosen using geometry/nameplate capacity only.
    v=clean(variant)
    if v in MATCH+FULL:return d['region_selection'].to(d['region_power'])
    return d['region_capacity']*d['region_mask']

def gate_inputs(d):
    h=d['h'];p=d['region_power'][...,-2];pold=d['region_power'][...,-6]
    w=d['region_capacity']*d['region_mask'];w=w/w.sum().clamp_min(1e-12)
    r=(p*w[None]).sum(1);change=((p-pold)*w[None]).sum(1)
    spread=((p-r[:,None]).square()*w[None]).sum(1).clamp_min(0).sqrt()
    speed=torch.linalg.vector_norm(d['physical'][...,[1,2]],dim=-1)
    ref=torch.linalg.vector_norm(d['reference_physical'][:,[1,2]],dim=-1)
    static=torch.stack((h[:,-1],h[:,-1]-h[:,-5],r,change,spread,r-h[:,-1]),-1)
    return torch.cat((static[:,None].expand(-1,24,-1),(speed/10)[...,None],((speed-ref[:,None])/10)[...,None]),-1)
