from pathlib import Path
import sys,json,copy,argparse,time
RUN=Path(__file__).resolve().parents[1];sys.path.insert(0,str(RUN/'code'))
import torch,numpy as np
from model import SharedModel,configure
from history_model import SharedModel as HistoryModel
from backbone import SharedModel as Backbone
from data import SharedData
from train import resolve
from settings import atomic_json,now,sha,ASSETS

def main(device):
 configure();specs=json.loads((RUN/'manifest.json').read_text());groups={}
 for s in specs:
  if s.get('task')!='evaluate' and s.get('family')!='LINEAR' and s['group'] not in groups:groups[s['group']]=resolve(s)
 base=groups['H-FULL'];base=dict(base,seed=42);data=SharedData('train',base);b,y,_,_=data.batch(np.arange(16)*55+np.arange(16),device)
 report={}
 for group,s in groups.items():
  m=SharedModel(s).to(device);m.train();loss=m.training_loss(b,y);loss.backward()
  assert torch.isfinite(loss);assert all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters()),group
  # Randomize trained pathways, otherwise zero readout hides leakage.
  with torch.no_grad():
   for p in m.parameters():
    if p.requires_grad and p.numel() and bool((p==0).all()):p.normal_(std=.01)
  m.eval()
  if s.get('refinement'):
   h=b['refinement_h'];c=b['state_c'];w=m.window_features(b);o=b['offset'];f=m.refiner
   with torch.no_grad():
    z=f(h,c,w,o);dc=(z-f(h,c+1,w,o)).abs().max().item()
    dh=(z-f(h.flip(-1),c,w,o)).abs().max().item()
    wx=w.clone();wx[:,:30]+=1;dw=(z-f(h,c,wx,o)).abs().max().item()
   if s.get('state_only_off') or s.get('history_mode')=='no_state':assert dc==0,(group,dc)
   elif not s.get('input_ablation'):assert dc>0,(group,dc)
   if s.get('state_only_off'):assert dh>0,group
   if s.get('window_mode')=='none':assert dw==0,(group,dw)
  if s.get('decoder_state_off'):
   altered=dict(b,x=b['x'].clone(),region=b['region']+2,near=b['near']+2);altered['x'][...,15:]+=3
   weather=m.weather_projection(b['weather'])
   with torch.no_grad():torch.testing.assert_close(m.decode(b,weather),m.decode(altered,weather),rtol=0,atol=0)
  if s.get('input_ablation')=='region':
   altered=dict(b,region=b['region']+2,near=b['near']+2,state_c=b['state_c'].clone());altered['state_c'][...,11:]+=1
   with torch.no_grad():torch.testing.assert_close(m(b),m(altered),rtol=0,atol=0)
  if s.get('full_window_baseline'):
   m.train();m.zero_grad();m.training_loss(b,y).backward();assert m.forecast_window_projection.weight.grad.abs().sum()>0,group
  report[group]=dict(parameters=sum(p.numel() for p in m.parameters()),finite_gradients=True)
 # Exact old H-FULL forward with nonzero old trained weights, plus strict checkpoints.
 for job in [s for s in specs if s.get('task')=='evaluate' and s['seed']==42]:
  ck=torch.load(RUN/job['checkpoint'],map_location='cpu',weights_only=False);s=ck['spec']
  
  if job['backbone']:m=Backbone(s)
  elif 'wind_mean' not in ck['model']:
   from parent_model import SharedModel as Parent
   m=Parent(s)
  else:m=SharedModel(s)
  m.load_state_dict(ck['model'],strict=True);m.to(device).eval()
  with torch.no_grad():assert torch.isfinite(m(b)).all()
  report[job['id']]=dict(strict_load=True)
 # Zero refiner must recover baseline and preserve training dropout RNG.
 s=base;ref=HistoryModel(dict(s,refinement=False,representation='R0')).to(device)
 m=SharedModel(s).to(device)
 with torch.no_grad():ref.head[-1].weight.normal_(std=.02);m.head.load_state_dict(ref.head.state_dict())
 for training in [False,True]:
  ref.train(training);m.train(training);rng=torch.get_rng_state();cuda=torch.cuda.get_rng_state() if device.startswith('cuda') else None
  expected=ref(b);torch.set_rng_state(rng)
  if cuda is not None:torch.cuda.set_rng_state(cuda)
  actual=m(b);torch.testing.assert_close(actual,expected,rtol=2e-5,atol=2e-6)
 # Wrong state only changes c, never history or labels; donors are training-only.
 ws=next(s for s in specs if s['group']=='H-wrongStateOnly');wr=SharedData('train',ws);wb,wy,_,_=wr.batch(np.arange(16)*55+np.arange(16),device)
 for k in b:
  if k!='state_c':torch.testing.assert_close(b[k],wb[k],rtol=0,atol=0)
 assert not torch.allclose(b['state_c'],wb['state_c']);torch.testing.assert_close(y,wy,rtol=0,atol=0)
 for split in ['train','val','test']:
  d=SharedData(split,ws);orig=np.asarray(d.a['origin'])+28800;don=np.load(RUN/'data/train/origin.npy')[d.state_donor]+28800
  assert np.all(orig%86400==don%86400)
  assert np.all(orig.astype('datetime64[s]').astype('datetime64[M]').astype(int)%12==don.astype('datetime64[s]').astype('datetime64[M]').astype(int)%12)
  assert np.all(abs(orig//86400-don//86400)>=2)
 # Full-information simple baseline uses exactly declared features.
 ls=next(s for s in specs if s.get('stage')=='tune');linear=SharedModel(ls).to(device)
 assert linear.features(b).shape==(16,1101);linear.training_loss(b,y).backward();assert linear.readout.weight.grad.abs().sum()>0
 # No training-loop/protocol drift.
 assert sha(RUN/'code/train.py')==json.loads((RUN/'PROTOCOL.json').read_text())['source_training_code_sha256']
 atomic_json(RUN/'operations/CHECKS.json',dict(timestamp=now(),device=device,passed=True,groups=report,
   controls_remove_only_intended_paths=True,donors_training_only=True,unchanged_training_loop=True,baseline_dropout_equivalence=True))
 print('PASS',device,len(report),'architectures/checkpoints',flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--device',default='cuda:0');main(p.parse_args().device)
