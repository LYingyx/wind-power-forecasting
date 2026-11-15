from pathlib import Path
import sys,json,copy,shutil
RUN=Path(__file__).resolve().parents[1];sys.path.insert(0,str(RUN/'code'))
from settings import *
import numpy as np,torch
from physical_models import PhysicalModel,frozen
from reference_models import PhysicalModel as Reference
from physical_data import FarmData,UnfilteredFarmData
from operational import november_exclusions,plan_times
import fit_model

def run():
    frozen.configure();specs=json.loads((RUN/'manifest.json').read_text())[:5];checked=[]
    train=FarmData('train',0,'R3_RAW',limit=64)
    val=FarmData('val',0,'R3_RAW',stats=train.stats)
    raw=UnfilteredFarmData('val',0,'R3_RAW',stats=train.stats)
    keep=np.load(ASSETS/'val_keep.npy');n=int(keep.sum())
    assert val.n==n and val.stats==train.stats
    for key,v in vars(raw).items():
        filtered=getattr(val,key)
        if isinstance(v,torch.Tensor) and v.ndim and len(v)==raw.n:assert torch.equal(filtered,v[keep]),key
        if isinstance(v,np.ndarray) and v.ndim and len(v)==raw.n:assert np.array_equal(filtered,v[keep]),key
    for k in raw.diagnostic:assert torch.equal(val.diagnostic[k],raw.diagnostic[k][keep]),k
    assert not np.logical_or.reduce(list(november_exclusions(val.origin,2023).values())).any()
    # Boundary probes include both timezones and all feature/target windows.
    from operational import unix
    t=np.arange(unix('2023-10-30'),unix('2023-12-03'),900)
    masks=november_exclusions(t,2023);valid=~np.logical_or.reduce(list(masks.values()))
    assert not valid[(t>=unix('2023-11-01'))&(t<unix('2023-12-01'))].any()
    assert valid[0] and valid[-1]
    checked.append('validation mask excludes November/boundaries; every sample field aligns;2022 normalization untouched')
    batch=train.batch(slice(0,4));fixtures=[];models=[];rngs=[]
    for spec in specs:
        m=PhysicalModel(spec,train.x.shape[-1]);rngs.append(torch.get_rng_state().clone());m.eval()
        with torch.no_grad():assert torch.equal(m(batch),batch['anchor'].expand(-1,24))
        models.append(m)
        if spec['configuration'] in ['W0_H32','TF_R3_RAW']:
            old=Reference(spec,train.x.shape[-1]);assert set(m.state_dict())==set(old.state_dict())
            for k,v in m.state_dict().items():assert torch.equal(v,old.state_dict()[k]),k
            # Check parity after nonzero readout as well as at persistence initialization.
            head=m.backbone.readout if spec['backbone']=='W0' else m.backbone.original.head
            last=[x for x in head.modules() if isinstance(x,torch.nn.Linear)][-1]
            with torch.no_grad():last.weight.fill_(.01);last.bias.fill_(.02)
            old.load_state_dict(m.state_dict());old.eval()
            with torch.no_grad():assert torch.equal(m(batch),old(batch))
            with torch.no_grad():last.weight.zero_();last.bias.zero_()
        # Nonzero weights exercise full CPU/GPU numerical path, followed by a backward step remotely.
        fixture_model=copy.deepcopy(m)
        with torch.no_grad():
            head=fixture_model.backbone.readout if spec['backbone']=='W0' else fixture_model.backbone.original.head
            last=[x for x in head.modules() if isinstance(x,torch.nn.Linear)][-1]
            last.weight.fill_(.01);last.bias.fill_(.02)
            expected=fixture_model(batch)
        fixtures.append(dict(spec=spec,feature_dim=train.x.shape[-1],weights=fixture_model.state_dict(),batch=batch,expected=expected,target=train.y[:4]))
        m.train();loss=(m(batch)-train.y[:4]).abs().mean();loss.backward()
        assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
    for i in range(1,4):
        assert torch.equal(rngs[0],rngs[i])
        for k,v in models[0].state_dict().items():
            if not k.startswith('backbone.readout.'):assert torch.equal(v,models[i].state_dict()[k]),k
        assert torch.equal(models[0].backbone.readout[0].weight,models[i].backbone.readout[0].weight[:32,:64])
        assert torch.equal(models[0].backbone.readout[0].bias,models[i].backbone.readout[0].bias[:32])
    checked.append('five arms finite backward; reference architecture exact; shared weights/RNG/head coordinates equal;zero output initialization')
    torch.save(fixtures,RUN/'operations/compatibility_fixture.pt')
    del val,raw,models,fixtures
    # Real fit interruption and resume reproduce uninterrupted CPU weights exactly.
    full=dict(specs[3],id='SMOKE_FULL');resume=dict(specs[3],id='SMOKE_RESUME')
    for s in [full,resume]:
        p=RUN/'smoke_jobs'/s['id']
        if p.exists():shutil.rmtree(p)
    fit_model.run(full,'cpu',True)
    fit_model.run(resume,'cpu',True,stop_after_epoch=1)
    assert (RUN/'smoke_jobs'/resume['id']/'latest.pt').exists()
    fit_model.run(resume,'cpu',True)
    for criterion in ['mae','rmse']:
        weights=[];metrics=[]
        for s in [full,resume]:
            p=RUN/'smoke_jobs'/s['id'];selected=json.loads((p/'metrics.json').read_text())['selections'][criterion]
            weights.append(torch.load(p/selected['checkpoint'],weights_only=False)['model']);metrics.append(selected['metrics'])
        assert metrics[0]==metrics[1]
        for k in weights[0]:assert torch.equal(weights[0][k],weights[1][k]),k
    checked.append('interrupted/resumed fit exactly matches uninterrupted selected weights and metrics')
    try:FarmData('test',0,'R3_RAW',stats=train.stats,limit=1)
    except RuntimeError:pass
    else:raise AssertionError('Test access should be blocked')
    checked.append('test2024 access blocked in this run')
    atomic_json(RUN/'operations/CHECKS.json',dict(timestamp=now(),passed=True,checks=checked,validation_rows=n,fixture_cases=5))
    print(json.dumps(checked),flush=True)

if __name__=='__main__':run()
