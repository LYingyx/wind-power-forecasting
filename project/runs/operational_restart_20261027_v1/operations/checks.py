from pathlib import Path
import sys,shutil,json,copy
RUN=Path(__file__).resolve().parents[1];sys.path.insert(0,str(RUN/'code'))
from settings import *
from physical_data import FarmData
from physical_models import PhysicalModel,frozen
from train_physics import run,predict
import torch,numpy as np

def main():
 frozen.configure();cases=[];fixtures=[]
 for b,v in [('W0','B0'),('W0','SEL'),('TF','SEL'),('W0','N3'),('W0','R3_RAW'),('TF','R3_RAW'),('W0','C2_R3_RAW'),('W0','H0W1'),('W0','H1W0'),('W0','H0W0'),('W0','AGE'),('W0','LOCAL_ERR'),('W0','C2_LOCAL_ERR'),('W0','REGION_ERR'),('W0','C2_REGION_ERR')]:
  d=FarmData('train',6,v,limit=64);val=FarmData('val',6,v,stats=d.stats,limit=16)
  assert val.stats==d.stats and val.data_fingerprint==FINGERPRINT
  assert np.all(d.available_unix<=d.origin) and np.all(d.origin<d.available_unix+43200)
  if v.startswith('H0'):
   assert torch.all(d.h==d.h[:1]);assert torch.all(d.x[...,-7:]==d.x[:1,...,-7:])
   assert torch.equal(d.anchor,d.actual_history[:,-1:])
  if v.endswith('W0'):
   assert torch.all(d.weather==d.weather[:1]) and torch.all(d.x[...,:15]==0)
  spec=dict(id=f'check_{b}_{v}',backbone=b,variant=v,site=6,seed=42,loss='MAE',task='train',configuration=b+'_'+v,stage='check',data_fingerprint=FINGERPRINT)
  model=PhysicalModel(spec,d.x.shape[-1]);batch=d.batch(slice(0,16));opt=torch.optim.AdamW(model.groups(),weight_decay=1e-4)
  for _ in range(2):
   model.train();opt.zero_grad();p=model(batch);assert torch.isfinite(p).all();loss=(p-d.y[:16]).abs().mean();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step()
  model.eval()
  with torch.no_grad():p=model(batch);expected=model({**batch,'physical':batch['physical'].clone()})
  assert torch.equal(p,expected)
  fixtures.append(dict(spec=spec,feature_dim=d.x.shape[-1],batch=batch,target=d.y[:16],weights=model.state_dict(),expected=p))
  cases.append(dict(backbone=b,variant=v,finite=True,train_only_stats=True));print('check',b,v,flush=True)
 # Exclusion, causal windows, and mismatched donors.
 topo=np.load(ASSETS/'topology.npz')
 for i in range(56):assert i not in topo['ids'][i]
 from neighbor_inputs import window
 assert np.arange(24)[window(15)][-1]==22
 tr=np.load(CACHE/'train/origin_unix.npy');va=np.load(CACHE/'val/origin_unix.npy')
 for split,t in [('train',tr),('val',va)]:
  ids=np.load(ASSETS/f'mismatch_{split}_r42.npy');donor=tr[ids]
  assert np.all((donor+28800)%86400==(t+28800)%86400)
  assert np.all(abs(donor-t)>=172800)
 # Real training loop, interruption then exact resume equivalence.
 base=dict(backbone='W0',variant='SEL',site=6,seed=42,loss='MAE',task='train',configuration='W0_SEL',stage='check',data_fingerprint=FINGERPRINT)
 for name in ['check_full','check_resume']:
  out=RUN/'smoke_jobs'/name
  if out.exists():shutil.rmtree(out)
 run(dict(base,id='check_full'),'cpu',True)
 run(dict(base,id='check_resume'),'cpu',True,stop_after_epoch=1)
 run(dict(base,id='check_resume'),'cpu',True)
 a=torch.load(RUN/'smoke_jobs/check_full/epoch_002.pt',weights_only=False)['model'];b=torch.load(RUN/'smoke_jobs/check_resume/epoch_002.pt',weights_only=False)['model']
 assert all(torch.equal(a[k],b[k]) for k in a),'Resume trajectory changed'
 torch.save(fixtures,RUN/'operations/compatibility_fixture.pt')
 atomic_json(RUN/'operations/CHECKS.json',dict(timestamp=now(),passed=True,cases=cases,causal_neighbor_delay=True,no_self_neighbor=True,month_clock_mismatch=True,exact_cpu_resume=True,year2024_evaluated=False))
 print('ALL SCIENTIFIC CHECKS PASSED',flush=True)
if __name__=='__main__':main()
