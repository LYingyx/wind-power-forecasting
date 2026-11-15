"""Behavioral tests for input isolation, matched graphs, and legacy compatibility."""
from config import *
from models import Model,Frontend,old,interpolate_grid
from data import Data,score
from train import save_torch
import tempfile

def main():
 configure();device='cuda:0' if torch.cuda.is_available() else 'cpu';checks={};site=6;seed=42
 d=Data('val',site,3,device,limit=16);h,w,p,t,_=d.batch(slice(None));s=spec('W0',site,seed,'check')
 model=Model(s).to(device).eval()
 ck=OLD/f'jobs/ablation_NEUTRAL_LAST_s{site:02d}_r{seed}/best.pt'
 model.backbone.load_state_dict(torch.load(ck,map_location=device,weights_only=False)['model'])
 with torch.no_grad():
  pred=model(h,w,p,t);expected=np.load(OLD/f'jobs/ablation_NEUTRAL_LAST_s{site:02d}_r{seed}/val_prediction.npy')[:len(h)]
  err=float(np.abs(pred.cpu().numpy()-expected).max());assert err<2e-6,err;checks['legacy_prediction_max_error']=err
  point=model.frontend(h,w,p,t);cached=np.load(OLD/f'prepared/val/{site:02d}/weather.npy')[:len(h)]
  err=float(np.abs(point.cpu().numpy()-cached).max());assert err<2e-6,err;checks['legacy_weather_max_error']=err
 for a,b in [('S2','S3'),('T1_7','T2_7'),('T1_8','T2_8'),('ST00','ST11'),('ST10','ST01'),('ST11','ST_NO_H')]:
  ma=Model(spec(a,site,seed,'check')).to(device).eval();mb=Model(spec(b,site,seed,'check')).to(device).eval()
  pa=dict(ma.named_parameters());pb=dict(mb.named_parameters());assert pa.keys()==pb.keys()
  assert all(torch.equal(pa[k],pb[k]) for k in pa),'initial parameters differ'
  with torch.no_grad():assert torch.equal(ma(h,w,p,t),mb(h,w,p,t))
  checks[a+'_'+b+'_matched']=True
  del ma,mb
 # Same times/phase, different history/content: static weights must be invariant.
 for arm in ['S2','T1_7','T1_8','ST00']:
  m=Model(spec(arm,site,seed,'check')).to(device).eval();f=m.frontend
  wa=f.weights(h,w,p,t);wb=f.weights(h.flip(0),w.flip(0),p,t)
  assert all(torch.equal(wa[k],wb[k]) for k in wa),arm
  checks[arm+'_static_invariance']=True
 # State-free adaptive extraction retains weather dependence.
 m=Model(spec('ST_NO_H',site,seed,'check')).to(device).eval();f=m.frontend
 wa=f.weights(h,w,p,t);wb=f.weights(h.flip(0),w,p,t);wc=f.weights(h,w.flip(0),p,t)
 assert all(torch.equal(wa[k],wb[k]) for k in wa)
 assert any(not torch.equal(wa[k],wc[k]) for k in wa);checks['state_free_weather_adaptive']=True
 for arm in ['H2','H2_CLIM']:
  m=Model(spec(arm,site,seed,'check')).to(device).eval()
  # A nonzero readout makes the information-isolation check non-vacuous.
  with torch.no_grad():m.backbone.readout[-1].weight.normal_(0,.1)
  hp=h.flip(0).clone();hp[:,-1]=h[:,-1]
  with torch.no_grad():assert torch.equal(m(h,w,p,t),m(hp,w,p,t))
  checks[arm+'_history_isolation']=True
 # Every arm has finite useful forward/backward paths after opening the common gate.
 params={}
 for arm in FORMAL+['W0']:
  m=Model(spec(arm,site,seed,'check')).to(device).train()
  if m.frontend.enabled:
   with torch.no_grad():m.frontend.gate.fill_(.2)
  if m.spec['backbone']=='NEUTRAL_LAST':
   with torch.no_grad():
    m.backbone.readout[-1].weight.normal_(0,.1)
    m.backbone.align_out.weight.normal_(0,.1)
  base=torch.zeros_like(h)
  loss=m(h,w,p,t,base).square().mean();loss.backward()
  fp=list(m.frontend.parameters());assert all(x.grad is not None and torch.isfinite(x.grad).all() for x in fp)
  assert all(float(x.grad.norm())>0 for x in fp),arm
  params[arm]=dict(total=sum(x.numel() for x in m.parameters()),front=sum(x.numel() for x in fp),active=sum(x.numel() for x in m.parameters() if x.grad is not None))
  del m
 checks['all_24_arms_forward_backward']=True
 # Whole/segment RMSE must equal pooled squared errors, not arithmetic lead RMSE.
 y=d.y.cpu().numpy();pred=y+np.arange(24,dtype=np.float64)[None]/100
 metrics=score(pred,y,h.cpu().numpy(),site,'val');assert abs(metrics['horizons']['3_6h']['rmse']-np.sqrt(np.mean((pred[:,12:]-y[:,12:])**2)))<1e-14
 checks['segment_rmse']=True
 m=Model(s).eval();x={k:v.clone() for k,v in m.state_dict().items()}
 with tempfile.TemporaryDirectory() as td:
  save_torch(Path(td)/'checkpoint.pt',dict(model=x,spec=s));m.load_state_dict(torch.load(Path(td)/'checkpoint.pt',weights_only=False)['model'])
  assert all(torch.equal(x[k],v) for k,v in m.state_dict().items())
 checks['checkpoint_roundtrip']=True
 atomic_json(RUN/'PREFLIGHT.json',dict(passed=True,checks=checks,parameters=params,device=device,completed_at=now()))
 print(json.dumps(checks,indent=2),flush=True)

if __name__=='__main__':main()
