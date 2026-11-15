"""Actual-data checks for equivalence, label isolation, gradients and donor validity."""
from pathlib import Path
import argparse, copy, json, sys, time
RUN=Path(__file__).resolve().parents[1];sys.path.insert(0,str(RUN/'code'))
import numpy as np
import torch
from settings import atomic_json,now,sha,DATA,ASSETS
from data import SharedData
from backbone import SharedModel as Original
from model import SharedModel,configure
from train import resolve
from representation import lerp,stateless_noise


def main(device):
    configure();started=time.monotonic()
    specs={s['group']:resolve(s) for s in json.loads((RUN/'manifest.json').read_text()) if s['seed']==42 and s['representation']!='SELECTED'}
    data=SharedData('train',specs['R3']);ids=np.arange(16)*55+np.arange(16)
    b,y,_,_=data.batch(ids,device)
    assert b['hourly'].shape==(16,8,14) and b['state_c'].shape==(16,17)
    # Every quarter-hour phase, including the edge 6h prediction, must match.
    assert set(b['phase'].cpu().tolist())=={0,1,2,3}
    torch.testing.assert_close(lerp(b['hourly'],b['offset']),b['weather'],rtol=2e-6,atol=2e-6)
    base=Original(dict(specs['R1'],representation='R0')).to(device).eval()
    cpu_rng=torch.get_rng_state().clone()
    cuda_rng=torch.cuda.get_rng_state().clone() if device.startswith('cuda') else None
    for group in ['R0+H','R1','R2','R3','TF_E2L','R4']:
        original=Original(specs[group])
        expected_cpu=torch.get_rng_state().clone()
        expected_cuda=torch.cuda.get_rng_state().clone() if device.startswith('cuda') else None
        checked=SharedModel(specs[group])
        assert torch.equal(expected_cpu,torch.get_rng_state()),('CPU RNG changed',group)
        if expected_cuda is not None:assert torch.equal(expected_cuda,torch.cuda.get_rng_state()),('CUDA RNG changed',group)
    torch.set_rng_state(cpu_rng)
    if cuda_rng is not None:torch.cuda.set_rng_state(cuda_rng)
    torch.manual_seed(123)
    with torch.no_grad():
        # A zero final forecast head would make any equivalence test vacuous.
        base.head[-1].weight.normal_(std=.1);base.head[-1].bias.fill_(.03)
        expected=base(b)
    diffs={};gradients={}
    for group in ['R0+H','R1','R2','R3']:
        model=SharedModel(specs[group]).to(device).eval()
        model.head[-1].load_state_dict(base.head[-1].state_dict())
        out=model(b)
        torch.testing.assert_close(out,expected,rtol=2e-5,atol=2e-6)
        diffs[group]=float((out-expected).abs().max().detach())
        if group in ['R2','R3']:
            (out-y).square().mean().backward()
            gradient=float(model.window.output.weight.grad.abs().sum())
            assert gradient>0;gradients[group]=gradient
        if group=='R0+H':
            (out-y).square().mean().backward();assert model.head[0].weight.grad[:,-2:].abs().sum()>0
    # State isolation: changing c changes only R3's generator after a nonzero update.
    for group,uses_state in [('R2',False),('R3',True)]:
        model=SharedModel(specs[group]).to(device).eval()
        with torch.no_grad():model.window.output.weight.normal_(std=.1)
        z,_=model.latent(b);a=model.window(z,b['offset'],b['state_c']);other=model.window(z,b['offset'],b['state_c']+1)
        assert bool((a-other).abs().max()>1e-7)==uses_state
    for group in ['R4','R4-noS','R4-wrongS','TF_R4','TF_E2L']:
        model=SharedModel(specs[group]).to(device)
        head=model.head if model.family=='W0' else model.original.head
        last=[m for m in head.modules() if isinstance(m,torch.nn.Linear)][-1]
        with torch.no_grad():last.weight.normal_(std=.1)
        small={k:v[:4] for k,v in b.items()};target=y[:4]
        model.train();loss=model.training_loss(small,target);assert torch.isfinite(loss)
        loss.backward();assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        if model.generative:
            assert model.residual_projection.weight.grad.abs().sum()>0
            assert model.flow.output.weight.grad.abs().sum()>0
            # Verify power/flow gradients beyond the zero-initialized projection.
            optimizer=torch.optim.AdamW(model.parameters(),lr=.0002)
            optimizer.step();optimizer.zero_grad();model.training_loss(small,target).backward()
            assert model.teacher.output.weight.grad.abs().sum()>0
            assert model.flow.output.weight.grad.abs().sum()>0
            model.eval()
            def forbidden(*args,**kwargs):raise AssertionError('Teacher used at inference')
            handle=model.teacher.register_forward_pre_hook(forbidden)
            rng=torch.get_rng_state().clone()
            with torch.no_grad():
                draws=model.prediction_samples(small);mean=model(small)
                assert draws.shape==(4,8,24) and torch.isfinite(draws).all()
                assert float(draws.std(1).mean())>0
                torch.testing.assert_close(mean,draws.mean(1))
                parts=torch.cat([model({k:v[:2] for k,v in small.items()}),model({k:v[2:] for k,v in small.items()})])
                torch.testing.assert_close(parts,mean,rtol=2e-5,atol=2e-6)
                assert torch.equal(rng,torch.get_rng_state())
            handle.remove()
        gradients[group]=float(loss.detach())
    # A wrong-state donor comes from training only, same month/clock, >=2 days away.
    for split in ['train','val']:
        wrong=SharedData(split,specs['R4-wrongS'])
        origin=np.asarray(wrong.a['origin'])+28800
        train_origin=np.load(DATA/'train/origin.npy')[wrong.state_donor]+28800
        assert np.all(origin%86400==train_origin%86400)
        assert np.all(origin.astype('datetime64[s]').astype('datetime64[M]').astype(int)%12==train_origin.astype('datetime64[s]').astype('datetime64[M]').astype(int)%12)
        assert np.all(abs(origin//86400-train_origin//86400)>=2)
        wb,_,_,_=wrong.batch(ids[:4],device)
        assert not torch.allclose(wb['state_c'],b['state_c'][:4])
    frozen=json.loads((RUN/'DATA_FREEZE.json').read_text())
    for path,value in frozen['files'].items():
        assert sha(RUN/path)==value,('data changed',path)
    report=dict(timestamp=now(),passed=True,device=device,seconds=time.monotonic()-started,
                equivalence_max_errors=diffs,gradient_checks=gradients,all_four_phases=True,
                inference_has_no_teacher=True,batch_invariant_sampling=True,wrong_state_donors_valid=True,
                backbone_cpu_cuda_initialization_rng_preserved=True,
                data_files_hashed=len(frozen['files']))
    atomic_json(RUN/'operations/CHECKS.json',report);print(json.dumps(report),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--device',default='cuda:0');main(p.parse_args().device)
