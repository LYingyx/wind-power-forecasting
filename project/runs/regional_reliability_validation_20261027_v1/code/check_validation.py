"""Scientific checks: complete exclusion, matching, causal delay, gate parity, resume."""
from settings import *
from validation_design import *
from validation_inputs import layout,matched
from physical_data import FarmData
from physical_models import PhysicalModel,frozen
from regional_inputs import summary
from diagnostics import stratified
from train_physics import run
import torch,numpy as np,gc

def main():
    frozen.configure();torch.set_num_threads(1);checks=[]
    for site in range(56):
        for v in REMOVE:
            ids,geo,valid,cap=layout(site,v)
            assert site not in ids and np.all(geo[valid,2]>=20)
            if v=='EXCLUDE20_REFILL':assert valid[:8].sum()==min(8,valid.sum())
        for seed in [42,43,44,51]:
            masks=[matched(site,v,seed) for v in MATCH];assert len({x.sum() for x in masks})==1
            assert all(x.sum()<=4 for x in masks)
    checks.append('both-route exclusion and refilled nearest eight; same-k geometry controls with no outcome use')
    specs=[]
    for v in sorted(NEW):
        train=FarmData('train',6,v,limit=32)
        val=FarmData('val',6,v,stats=train.stats,limit=32)
        s=spec('W0',v,6,42);model=PhysicalModel(s,train.x.shape[-1],train.historical_input.mean(0));model.eval()
        with torch.no_grad():
            model.residual_projection.weight.normal_(0,.01);model.neighbor.out.weight.normal_(0,.01)
            d=train.batch(slice(None));p=model(d);assert torch.isfinite(p).all()
            dd=dict(d,region_power=d['region_power'].clone(),neighbor_raw=d['neighbor_raw'].clone())
            dd['region_power'][...,-1]+=100;dd['neighbor_raw'][...,-1]+=100
            dd['res_features']=(summary(dd,v)-train.res_mean)/train.res_std
            assert torch.equal(p,model(dd)),v+' future/zero-latency leakage'
            if v in REMOVE:
                close=d['region_geometry'][:,2]<20
                dd=dict(d,region_power=d['region_power'].clone(),neighbor_raw=d['neighbor_raw'].clone())
                dd['region_power'][:,close]+=100;dd['neighbor_raw'][:,close[:8]]+=100
                dd['res_features']=(summary(dd,v)-train.res_mean)/train.res_std
                assert torch.equal(p,model(dd)),v+' incomplete exclusion'
            if clean(v) in GATES:
                y,details=model(d,return_details=True);g=details['gate'];assert ((g>=0)&(g<=1)).all()
                if clean(v)=='G_LEAD':assert (g[:,1:]<=g[:,:-1]).all()
                assert torch.allclose(y,details['local_prediction']+g*details['regional_update'])
        loss=(model(train.batch(slice(None)))-train.y).abs().mean();loss.backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        rows=stratified(p.detach().numpy(),train);assert any(x['group']=='high_outcome_from_origin' for x in rows)
        del train,val,model;gc.collect()
    checks.append('all new arms finite and differentiable; unavailable latest observations irrelevant; exclusion perturbation invariance; diagnostic partitions')
    # A trained RAW checkpoint must be exactly usable by G_FREE with no content changes.
    src=D_RUN/'jobs/TF_z1_lr0.0002_dp0.05_R3_RAW_s06_r42'
    old=json.loads((src/'metrics.json').read_text());stats=json.loads((src/'normalization.json').read_text())
    d=FarmData('val',6,'R3_RAW',stats=stats,limit=32);base=PhysicalModel(old,d.x.shape[-1]).eval()
    weights=torch.load(src/old['selections']['mae']['checkpoint'],weights_only=False)['model'];base.load_state_dict(weights)
    gate=PhysicalModel(spec(TF,'G_FREE',6,42),d.x.shape[-1]).eval();missing=gate.load_state_dict(weights,strict=False)
    assert all(x.startswith('reliability.') or x=='decay_raw' for x in missing.missing_keys) and not missing.unexpected_keys
    with torch.no_grad():assert torch.allclose(base(d.batch(slice(None))),gate(d.batch(slice(None))),atol=2e-7,rtol=0)
    gate.train()
    with torch.no_grad():
        gate.residual_projection.weight.zero_();gate.neighbor.out.weight.zero_()
        _,details=gate(d.batch(slice(None)),return_details=True)
        assert torch.count_nonzero(details['regional_update'])==0,'paired dropout must not invent regional updates'
    checks.append('learned RAW/G_FREE parity and shared dropout yield zero update when regional paths zero')
    for config,v in [('W0','EXCLUDE20_REFILL'),(TF,'G_STATE_LEAD')]:
        s=spec(config,v,6,42);s['id']='check_'+s['id'];run(s,'cpu',True)
    s=spec(TF,'G_STATE_LEAD',6,42);s['id']='check_resume';run(s,'cpu',True,stop_after_epoch=1);run(s,'cpu',True)
    a=RUN/'smoke_jobs'/('check_'+spec(TF,'G_STATE_LEAD',6,42)['id']);b=RUN/'smoke_jobs/check_resume'
    x=json.loads((a/'metrics.json').read_text());y=json.loads((b/'metrics.json').read_text())
    for criterion in ['mae','rmse']:
        assert x['selections'][criterion]['metrics']==y['selections'][criterion]['metrics']
        wx=torch.load(a/x['selections'][criterion]['checkpoint'],weights_only=False)['model'];wy=torch.load(b/y['selections'][criterion]['checkpoint'],weights_only=False)['model']
        assert all(torch.equal(wx[k],wy[k]) for k in wx)
    checks.append('real W0 and TF two-epoch fits; gated model optimizer/dropout resume exact')
    atomic_json(RUN/'operations/CHECKS.json',dict(timestamp=now(),passed=True,checks=checks,open_2024=False))
    print('PASSED',checks,flush=True)

if __name__=='__main__':main()
