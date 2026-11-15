"""Fixed topology, fixed 22-sample history, split-safe seasonal surrogate control."""
from settings import *
import numpy as np

VARIANTS=['N0','N1','N2','N2_SELF','N3','N4_K4','N4_K16','N5_CORR','N7_D0','N7_D30']
NEIGHBORS=set(VARIANTS)-{'N0'}
INTERVENTION_VARIANTS={'N1','N4_K4','N4_K16','N5_CORR','N7_D0','N7_D30'}

def topology(site,variant):
    with np.load(ASSETS/'neighbor_tables.npz') as tab:
        label='corr' if variant=='N5_CORR' else 'distance'
        k=4 if variant=='N4_K4' else 16 if variant=='N4_K16' else 8
        ids=tab[label+'_ids'][site,:k].copy();geo=tab[label+'_geometry'][site,:k].copy();mask=tab[label+'_mask'][site,:k].copy()
        capacity=tab['capacity_mw'][ids].copy()
    assert site not in ids
    return ids,geo,mask,capacity

def window(delay):
    assert delay in [0,15,30]
    end=24-delay//15
    return slice(end-22,end)

def ages(delay):return (np.arange(22,dtype='f4')-21)/4-delay/60

def mismatch_map(split,seed):
    assert split in ['train','val'] and seed in SEEDS
    return np.load(ASSETS/f'mismatch_{split}_r{seed}.npy',mmap_mode='r')
