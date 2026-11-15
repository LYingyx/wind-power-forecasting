from pathlib import Path
import sys,json,argparse
RUN=Path(__file__).resolve().parents[1];sys.path.insert(0,str(RUN/'code'))
from settings import *
from data import SharedData
from site_data import StationData
from model import SharedModel,configure
from train import run
from z1 import self_test
from prepare_rm import regional_history
import numpy as np,torch

def checks():
    configure();assert self_test();specs=json.loads((RUN/'manifest.json').read_text());fixtures=[]
    top=np.load(ASSETS/'topology.npz');h=np.random.default_rng(3).normal(size=(2,55,24)).astype('f4')
    x=regional_history(h,top);assert np.all(x[:,:,0]==0)
    hh=h.copy();hh[:,:,-1]+=100
    assert np.array_equal(regional_history(hh,top),x),'contemporaneous neighbor leak'
    for s in [0,30,54]:
        hh=h.copy();hh[:,s]+=100
        assert np.allclose(regional_history(hh,top)[:,s],x[:,s]),'target in regional data'
        for r in range(3):
            w=top['ring_weights'][r,s]
            assert np.allclose(x[:,s,1:,r],np.einsum('j,njt->nt',w,h[:,:,:-1]),atol=1e-6)
    for family in ['W0','TF']:
        sp=next(s for s in specs if s['family']==family and s['stage']=='Z1' and s['station']==0 and s['seed']==42)
        src=torch.load(RUN/sp['pretrained'],map_location='cpu',weights_only=False);shared=SharedData('val',src['spec']);site=StationData('val',src['spec'],0)
        ix=np.array([0,7,123,site.n-1]);a,y,_,_=shared.batch(ix*55);b,z,_,_=site.batch(ix)
        assert torch.equal(y,z)
        for k in a:assert torch.equal(a[k],b[k]),k
        m=SharedModel(src['spec']).eval();m.load_state_dict(src['model'])
        with torch.no_grad():out=m(b)
        fixtures.append(dict(spec=src['spec'],weights=m.state_dict(),b=b,output=out))
        rm=next(s for s in specs if s['family']==family and s['stage']=='RM')
        rb,ry,_,_=SharedData('val',rm).batch(ix*55)
        mm=SharedModel(rm);assert not hasattr(mm,'regional') and not hasattr(mm,'neighbor')
        projection=mm.history_projection if family=='W0' else mm.original.h_in
        assert projection.in_features==12
        opt=torch.optim.AdamW(mm.parameters(),lr=.001);loss=(mm(rb)-ry).abs().mean();loss.backward();opt.step();mm.eval()
        with torch.no_grad():
            out=mm(rb);alter={**rb,'near':rb['near']+99,'region':rb['region']-99}
            assert torch.equal(out,mm(alter))
        assert out.shape==(len(ix),24) and torch.isfinite(out).all()
        fixtures.append(dict(spec=rm,weights=mm.state_dict(),b=rb,output=out))
    sp=next(s for s in specs if s['stage']=='Z1')
    for jid,maxb in [('CHECK_RESUME',1),('CHECK_RESUME',None),('CHECK_STRAIGHT',None)]:
        rc=run(dict(sp,id=jid),smoke=True,device='cpu',max_batches=maxb);assert rc==(75 if maxb else 0)
    a=torch.load(RUN/'smoke_jobs/CHECK_RESUME/best.pt',map_location='cpu',weights_only=False);b=torch.load(RUN/'smoke_jobs/CHECK_STRAIGHT/best.pt',map_location='cpu',weights_only=False)
    assert all(torch.equal(a['model'][k],b['model'][k]) for k in a['model'])
    torch.save(fixtures,RUN/'operations/fixture.pt')
    atomic_json(RUN/'operations/CHECKS.json',dict(timestamp=now(),passed=True,checks=['causal_rolling_vs_bruteforce','future_mutation_invariance','bin_and_fallback_boundaries','regional_lag_and_self_exclusion','unchanged_cached_P0_inputs','RM_query_removal_and_backward','E1_resume_equivalence']))
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
