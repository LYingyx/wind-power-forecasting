"""Numerical physics, zero initialization, label isolation and resume smoke checks."""
from pathlib import Path
import sys,json,argparse
RUN=Path(__file__).resolve().parents[1];sys.path.insert(0,str(RUN/'code'))
from settings import DATA,CACHE,atomic_json,now
from data import FarmData
from model import CorrectedModel,thermo,interp
from physical_models import frozen
import torch,numpy as np,physics
from operational import interpolate_nodes

def main(device):
    frozen.configure();manifest=json.loads((RUN/'manifest.json').read_text())
    spec=next(s for s in manifest if s['id']=='age12_E1_s00_r42')
    train=FarmData('train',spec,device,limit=96);val=FarmData('val',spec,device,stats=train.stats,limit=96)
    batch=train.batch(slice(0,16));model=CorrectedModel(spec,train.stats).to(device).eval()
    with torch.no_grad():
        p,uv=model(batch);base=model.base(batch)
        assert torch.equal(p,base),'Zero correction must preserve base prediction exactly'
        assert torch.equal(uv,batch['nodes'][...,1:3])
        # Arbitrary changes in ERA5 targets cannot change a deployed prediction.
        train.era.fill_(12345);p2,_=model(train.batch(slice(0,16)));assert torch.equal(p,p2)
        expected=interpolate_nodes(batch['nodes'].cpu().numpy(),batch['offset'].cpu().numpy(),np.arange(1,25)/4)
        assert np.max(abs(interp(batch['nodes'],batch['offset']).cpu().numpy()-expected))<2e-6
        den,stab=thermo(batch['physical'],model.elevation)
        nd,ns=physics.thermodynamics(batch['physical'].cpu().numpy(),float(model.elevation))
        assert np.max(abs(den.cpu().numpy()-nd))<1e-4
        assert np.max(abs(stab.cpu().numpy()-ns))<2e-3
    # ERA5 can influence gradients only through the explicitly supervised loss.
    model.train();model.zero_grad();_,uv=model(batch);target=val.era[:16]
    loss=(uv-target).square().mean();loss.backward();assert model.corrector[-1].weight.grad.abs().sum()>0
    counts={}
    for arm in ['E1','E2','E3','E4','E5']:
        sp=dict(spec,variant=arm);d=FarmData('train',sp,'cpu',limit=96);m=CorrectedModel(sp,d.stats)
        counts[arm]=sum(p.numel() for p in m.parameters())
        if arm=='E3':assert not torch.count_nonzero(d.inputs['corr_obs'])
        if arm=='E4':assert not torch.count_nonzero(d.inputs['corr_obs'][:,24:])
        assert 'era' not in d.batch(slice(0,2))
    assert len(set(counts.values()))==1
    # Both ages and all variants use identical power labels and origins.
    d0=FarmData('train',dict(spec,age_shift=0),'cpu',limit=96)
    assert np.array_equal(d0.origin,train.origin) and np.array_equal(d0.y_numpy,train.y_numpy)
    # Cross-device numerical fixture: checkpoint and a fixed deployment batch.
    model.eval()
    fixture=RUN/'operations/fixture.pt'
    if not fixture.exists():
        with torch.no_grad():
            model.corrector[-1].bias.copy_(torch.linspace(-.1,.1,16,device=device))
            # Nonzero base head also checks decoder and E2C kernels.
            model.base.backbone.readout[-1].weight.fill_(.01)
            prediction,wind=model(batch)
        torch.save(dict(spec=spec,stats=train.stats,model={k:v.cpu() for k,v in model.state_dict().items()},batch={k:v.cpu() for k,v in batch.items()},prediction=prediction.cpu(),wind=wind.cpu()),fixture)
    f=torch.load(fixture,map_location=device,weights_only=False);model.load_state_dict(f['model'])
    with torch.no_grad():pred,wind=model(f['batch'])
    pdiff=float((pred-f['prediction']).abs().max());wdiff=float((wind-f['wind']).abs().max())
    assert pdiff<3e-4 and wdiff<3e-5,(pdiff,wdiff)
    result=dict(timestamp=now(),passed=True,zero_correction_exact=True,labels_isolated=True,physics_matches_numpy=True,
        all_correction_arms_parameter_counts=counts,paired_sample_identity=True,fixture_prediction_max_abs_difference=pdiff,fixture_wind_max_abs_difference=wdiff,
        torch_version=torch.__version__,device=device,gpu_name=torch.cuda.get_device_name() if device.startswith('cuda') else None)
    atomic_json(RUN/'operations/CHECKS.json',result);print(json.dumps(result),flush=True)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--device',default='cpu');a=ap.parse_args();main(a.device)
