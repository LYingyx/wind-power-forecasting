"""Delayed regional history from the frozen 55-site cohort, no weather access."""
from settings import *
import numpy as np

def regional_history(h,top):
    # h [time, source station, history]; output [time,target,history,10].
    weights=top['ring_weights'];near=top['near_weights'];flags=top['ring_valid'].T
    x=np.zeros((len(h),h.shape[1],24,10),'f4')
    past=h[:,:,:-1]
    mean=np.einsum('rsj,njt->nstr',weights,past,optimize=True)
    second=np.einsum('rsj,njt->nstr',weights,past**2,optimize=True)
    x[:,:,1:,:3]=mean;x[:,:,1:,3:6]=np.sqrt(np.maximum(0,second-mean**2))
    x[:,:,1:,6]=np.einsum('sj,njt->nst',near,past,optimize=True)
    x[:,:,1:,7:]=flags[None,:,None,:]
    # At step0 only regional channels are missing; local history stays intact.
    return x

def main():
    top=np.load(ASSETS/'topology.npz');root=RUN/'rm_cache';root.mkdir(exist_ok=True)
    for split in ['train','val']:
        h=np.load(DATA/split/'h.npy',mmap_mode='r')
        p=root/f'{split}.npy';a=np.lib.format.open_memmap(p,mode='w+',dtype='f4',shape=(*h.shape[:2],24,10))
        for start in range(0,len(h),256):a[start:start+256]=regional_history(np.asarray(h[start:start+256]),top)
        a.flush();assert np.isfinite(a).all();print(split,a.shape,flush=True)
    atomic_json(root/'READY.json',dict(timestamp=now(),passed=True,source_freeze_sha256=sha(RUN/'SOURCE_DATA_FREEZE.json'),
        channels=['ring_mean_0','ring_mean_1','ring_mean_2','ring_std_0','ring_std_1','ring_std_2','near_mean','ring_exists_0','ring_exists_1','ring_exists_2'],
        neighbor_delay_minutes=15,first_regional_step_zero=True,scaling='capacity factor, population weighted std, binary flags',
        files={f'{s}.npy':sha(root/f'{s}.npy') for s in ['train','val']}))

if __name__=='__main__':main()
