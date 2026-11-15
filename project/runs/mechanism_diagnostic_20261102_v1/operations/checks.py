from pathlib import Path
import sys,json,argparse
RUN=Path(__file__).resolve().parents[1];sys.path.insert(0,str(RUN/'code'))
from experiment import *

def main(device):
    frozen.configure();spec=dict(site=0,seed=42,age_shift=0,variant='E1')
    c=CorrectionData('train',spec,device,limit=96)
    old=FarmData('train',spec,device,limit=96)
    context=torch.cat([torch.cat([old.inputs['past'],old.inputs['nodes']],1).flatten(1),old.inputs['corr_obs'],old.inputs['all_times']/6,old.inputs['age']],-1)
    torch.testing.assert_close(c.x,context,rtol=0,atol=0)
    torch.testing.assert_close(c.y,old.era,rtol=0,atol=0)
    model=PureCorrector(42).to(device)
    torch.testing.assert_close(model(c.x,c.raw),c.raw,rtol=0,atol=0)
    fixture=torch.load(RUN/'operations/fixture.pt',map_location='cpu',weights_only=False)
    model.load_state_dict(fixture['model']);out=model(c.x,c.raw).detach().cpu()
    torch.testing.assert_close(out,fixture['prediction'],rtol=1e-5,atol=2e-6)
    counts={};diffs={}
    before=torch.cuda.get_rng_state().clone() if device.startswith('cuda') else torch.get_rng_state().clone()
    branch=ResponseInteraction(42,'I_MUL')
    after=torch.cuda.get_rng_state() if device.startswith('cuda') else torch.get_rng_state()
    assert torch.equal(before,after),'New interaction perturbed backbone dropout RNG'
    for v in ['I_MUL','I_ADD','I_XATTN','I_LOCAL','I_WRONG']:
        sp=dict(spec,variant=v);data=InteractionData('train',sp,device,limit=96)
        m=InteractionModel(sp).to(device).eval()
        with torch.no_grad():
            y=m(data.inputs);ref=m.base(data.inputs)
        torch.testing.assert_close(y,ref,rtol=0,atol=0)
        counts[v]=sum(p.numel() for p in m.interaction.parameters());diffs[v]=float((y-ref).abs().max())
        assert data.era is None
    assert len(set(counts.values()))==1
    atomic_json(RUN/'operations'/('CHECKS.json' if device=='cpu' else 'GPU_CHECKS.json'),dict(timestamp=now(),passed=True,device=device,identical_corrector_context_and_labels=True,backbone_rng_preserved=True,zero_initialization_exact=True,equal_branch_parameters=counts,zero_initialization_max_difference=diffs,numerical_max_difference=float((out-fixture['prediction']).abs().max())))
    print('CHECKS PASSED',device,counts,flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--device',default='cpu');a=ap.parse_args();main(a.device)
