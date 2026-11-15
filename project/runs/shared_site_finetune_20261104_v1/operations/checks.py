from pathlib import Path
import sys,json,copy,argparse
RUN=Path(__file__).resolve().parents[1];sys.path.insert(0,str(RUN/'code'))
from settings import *
from data import SharedData
from site_data import StationData
from model import SharedModel,configure
from train import load_model,run
import numpy as np,torch

def checks():
    configure();specs=json.loads((RUN/'manifest.json').read_text());results=[];fixtures=[]
    for family in ['W0','TF']:
        sp=next(s for s in specs if s['family']==family and s['mode']=='P1_FULL' and s['station']==0 and s['seed']==42)
        src=torch.load(RUN/sp['pretrained'],map_location='cpu',weights_only=False);shared=SharedData('val',src['spec']);site=StationData('val',src['spec'],0)
        ix=np.array([0,7,123,site.n-1]);a,y,_,_=shared.batch(ix*55);b,z,si,_=site.batch(ix)
        assert (si==0).all() and torch.equal(y,z)
        for k in a:assert torch.equal(a[k],b[k]),k
        m=SharedModel(src['spec']).eval();m.load_state_dict(src['model'])
        with torch.no_grad():out=m(b)
        fixtures.append(dict(spec=src['spec'],weights=src['model'],b=b,output=out))
        for mode in ['P1_FULL','P2_HEAD']:
            sp=next(s for s in specs if s['family']==family and s['mode']==mode and s['station']==0 and s['seed']==42)
            model,opt,_=load_model(sp,'cpu');before={k:p.detach().clone() for k,p in model.named_parameters()};model.train();loss=(model(b)-z).abs().mean();loss.backward();opt.step()
            changed=[k for k,p in model.named_parameters() if not torch.equal(before[k],p)]
            assert changed and torch.equal(model.embedding.weight[1:],before['embedding.weight'][1:])
            if mode=='P2_HEAD':
                prefix='head.' if family=='W0' else 'original.head.'
                assert all(k=='embedding.weight' or k.startswith(prefix) for k in changed),changed
            results.append(dict(family=family,mode=mode,changed_parameters=changed))
            s=dict(sp,id=f'CHECK_{family}_{mode}')
            assert run(s,smoke=True,device='cpu')==0
    sp=next(s for s in specs if s['family']=='W0' and s['mode']=='P1_FULL' and s['station']==0 and s['seed']==42)
    resumed=dict(sp,id='CHECK_RESUME');straight=dict(sp,id='CHECK_STRAIGHT')
    assert run(resumed,smoke=True,device='cpu',max_batches=1)==75
    assert run(resumed,smoke=True,device='cpu')==0
    assert run(straight,smoke=True,device='cpu')==0
    a=torch.load(RUN/'smoke_jobs/CHECK_RESUME/best.pt',map_location='cpu',weights_only=False);b=torch.load(RUN/'smoke_jobs/CHECK_STRAIGHT/best.pt',map_location='cpu',weights_only=False)
    assert all(torch.equal(a['model'][k],b['model'][k]) for k in a['model'])
    torch.save(fixtures,RUN/'operations/fixture.pt')
    atomic_json(RUN/'operations/CHECKS.json',dict(timestamp=now(),passed=True,checks=['exact_cached_input_equivalence','target_station_only','frozen_parameter_integrity','other_embedding_rows_unchanged','four_mode_smoke_training','checkpoint_resume_equivalence','epoch0_selection'],parameter_checks=results))
    print('CPU checks passed',flush=True)

def fixture():
    configure();cases=torch.load(RUN/'operations/fixture.pt',map_location='cpu',weights_only=False);diffs=[]
    for f in cases:
        m=SharedModel(f['spec']).cuda().eval();m.load_state_dict(f['weights'])
        with torch.no_grad():v=m({k:x.cuda() for k,x in f['b'].items()}).cpu()
        assert torch.allclose(v,f['output'],atol=3e-5,rtol=3e-5),(v-f['output']).abs().max().item()
        diffs.append(float((v-f['output']).abs().max()))
    atomic_json(RUN/'operations/GPU_CHECKS.json',dict(timestamp=now(),passed=True,gpu=torch.cuda.get_device_name(),max_errors=diffs));print('GPU checks passed',diffs,flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--fixture',action='store_true');a=p.parse_args();fixture() if a.fixture else checks()
