"""Raw regional observations with matched projections and fixed latency windows."""
import numpy as np
import torch
from settings import CACHE, ASSETS
from neighbor_inputs import window
from residual_inputs import topology

W0_VARIANTS = ['C2_R3_RAW', 'C2_R3_ALLPOOL', 'SELF_R3_RAW', 'R3_INNER', 'R3_MIDDLE', 'R3_OUTER',
               'R3_NO_INNER', 'R3_NO_MIDDLE', 'R3_NO_OUTER', 'R3_ALLPOOL',
               'R3_ALLPOOL_MEAN', 'R3_NO_N3', 'R3_MEAN', 'R3_MEAN_DELTA', 'R3_D0', 'R3_D30']
TF_VARIANTS = ['R3_RAW', 'C2_R3_RAW', 'R3_ALLPOOL', 'C2_R3_ALLPOOL']
REGIONAL_VARIANTS = set(W0_VARIANTS + TF_VARIANTS)
RINGS = [(0, 20), (20, 60), (60, 150.0001)]
INTERVENTIONS = [f'ring{i}_{kind}' for i in range(3) for kind in ['mismatch', 'zero']]
INTERVENTIONS += ['drop30', 'all_mismatch', 'delay0', 'delay15', 'delay30']


def latency(variant, default=15):
    return 0 if variant == 'R3_D0' else 30 if variant == 'R3_D30' else default


def observations(split, site, variant, seed, n):
    ids, geo, valid, cap = topology(site, True)
    mismatch = variant.startswith('C2_')
    source = np.load(CACHE/('train' if mismatch else split)/'station_history.npy', mmap_mode='r')
    rows = np.load(ASSETS/f'mismatch_{split}_r{seed}.npy', mmap_mode='r')[:n] if mismatch else np.arange(n)
    power = np.asarray(source[rows[:, None], ids[None], :], dtype='f4')
    return power, ids, geo, valid, cap


def moment(power, weights):
    if weights.ndim == 1:
        w = weights / weights.sum().clamp_min(1e-12)
        mean = (power*w[None, :, None]).sum(1)
        var = ((power[:, :, -1]-mean[:, -1:])**2*w[None]).sum(1)
    else:
        w = weights / weights.sum(1, keepdim=True).clamp_min(1e-12)
        mean = (power*w[:, :, None]).sum(1)
        var = ((power[:, :, -1]-mean[:, -1:])**2*w).sum(1)
    return torch.stack((mean[:, -1], var.clamp_min(0).sqrt(), mean[:, -1]-mean[:, -5]), -1)


def intervened(d, intervention):
    power = d['region_power']; valid = d['region_mask']
    if intervention == 'all_mismatch':
        power = d['region_donor']
    elif intervention == 'drop30':
        valid = valid[None] & d['region_keep']
    elif intervention and intervention.startswith('ring'):
        index, kind = intervention[4:].split('_', 1); lo, hi = RINGS[int(index)]
        ring = (d['region_geometry'][:, 2] >= lo) & (d['region_geometry'][:, 2] < hi)
        power = power.clone()
        power[:, ring] = d['region_donor'][:, ring] if kind == 'mismatch' else 0
    return power, valid


def summary(d, variant, delay=15, intervention=None):
    if variant.startswith('C2_'): variant = variant[3:]
    if intervention and intervention.startswith('delay'):
        delay = int(intervention[5:])
    power, valid = intervened(d, intervention)
    # SELF controls the extra summary capacity; the existing N3 input remains real.
    if variant == 'SELF_R3_RAW':
        power = d['h'][:, None].expand_as(power)
    power = power[..., window(delay)]
    weights = d['region_capacity']*valid
    distance = d['region_geometry'][:, 2]
    out = []
    for index, (lo, hi) in enumerate(RINGS):
        mask = (distance >= lo) & (distance < hi)
        if variant in ['R3_ALLPOOL', 'R3_ALLPOOL_MEAN']:
            mask = torch.ones_like(mask) if index == 0 else torch.zeros_like(mask)
        selected = {'R3_INNER': 0, 'R3_MIDDLE': 1, 'R3_OUTER': 2}.get(variant)
        removed = {'R3_NO_INNER': 0, 'R3_NO_MIDDLE': 1, 'R3_NO_OUTER': 2}.get(variant)
        if (selected is not None and index != selected) or index == removed:
            mask = torch.zeros_like(mask)
        weight = weights*mask
        values = moment(power, weight)
        if variant in ['R3_MEAN', 'R3_ALLPOOL_MEAN']:
            values = values*values.new_tensor([1, 0, 0])
        elif variant == 'R3_MEAN_DELTA':
            values = values*values.new_tensor([1, 0, 1])
        present = (weight.sum(-1) > 0).to(power)
        present = present.expand(len(power)) if present.ndim == 0 else present
        out += [values, torch.zeros_like(values), torch.zeros_like(values), present[:, None]]
    return torch.cat(out + [power.new_zeros(len(power), 6)], -1)


def neighbor_readout(module, d, delay, intervention):
    # Ring-only interventions hold the N3 base fixed. Global missingness/mismatch
    # affects both branches and renormalizes over the same retained neighbors.
    if intervention == 'drop30':
        p = d['region_power'][:, :8, window(delay)]
        weight = module.capacity[None]*module.mask[None]*d['region_keep'][:, :8]
        pooled = (p*weight[:, :, None]).sum(1)/weight.sum(1, keepdim=True).clamp_min(1e-9)
        absent = weight.sum(1) == 0
        dd = dict(d, neighbor_raw=d['neighbor_raw'].clone())
        dd['neighbor_raw'][:, :, window(delay)] = pooled[:, None].expand_as(p)
        value, attention = module(dd, 'delay'+str(delay))
        return value*(~absent)[:, None, None], attention
    dd = dict(d, neighbor_raw=d['region_donor'][:, :8]) if intervention == 'all_mismatch' else d
    return module(dd, 'delay'+str(delay))
