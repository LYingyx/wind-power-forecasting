from settings import *
import numpy as np
from prepare import month,moments

def run():
 setup();split='test';t=np.load(CACHE/split/'origin_unix.npy');tr=np.load(CACHE/'train/origin_unix.npy');m=month(t);slot=(t+28800)%86400;trm=month(tr);trs=(tr+28800)%86400
 # Metadata guard, not a model evaluation: whole test power/NWP windows avoid November.
 from operational import november_exclusions
 assert not any(v.any() for v in november_exclusions(t).values())
 for seed in SEEDS:
  rng=np.random.default_rng(seed+20261028);donor=np.empty(len(t),'i8')
  for mm in np.unique(m):
   for ss in np.unique(slot):
    ix=np.flatnonzero((m==mm)&(slot==ss));pool=np.flatnonzero((trm==mm)&(trs==ss));assert len(pool)>1;donor[ix]=rng.choice(pool,len(ix))
  assert np.all(abs(tr[donor]-t)>=172800);np.save(ASSETS/f'mismatch_test_r{seed}.npy',donor)
 raw=np.load(CACHE/'test/station_history.npy',mmap_mode='r');topo=np.load(ASSETS/'topology.npz')
 out=np.lib.format.open_memmap(ASSETS/'rings_test.npy',mode='w+',dtype='f4',shape=(len(t),56,12))
 for site in range(56):
  ids=topo['ids'][site];dist=topo['geometry'][site,:,2];cap=topo['capacity'][site];valid=topo['mask'][site];p=np.asarray(raw[:,ids,1:23])
  out[:,site]=np.concatenate([moments(p,cap*valid*(dist>=lo)*(dist<hi)) for lo,hi in [(0,20),(20,60),(60,150.0001)]],-1)
 out.flush();atomic_json(ASSETS/'TEST_INPUT_AUDIT.json',dict(timestamp=now(),n_origins=len(t),month11_excluded=True,same_train_topology=True,neighbor_delay_minutes=15,test_labels_used_for_fit=False,model_evaluation_performed=False))
if __name__=='__main__':run()
