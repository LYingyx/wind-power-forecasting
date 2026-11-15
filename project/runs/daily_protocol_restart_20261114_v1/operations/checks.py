from pathlib import Path
import sys,os,json,copy
RUN=Path(__file__).resolve().parents[1];sys.path.insert(0,str(RUN/'code'))
from settings import *
from data import SharedData
from model import SharedModel,configure
from prepare import column
from train import run
import numpy as np,torch,argparse

def checks(device='cpu'):
    configure();manifest=json.loads((RUN/'manifest.json').read_text());sp=manifest[0];d=SharedData('train',sp);b,y,si,ti=d.batch(np.arange(32),device)
    assert 'y' not in b and b['h'].shape==(32,24)
    cohort=json.loads((ASSETS/'COHORT.json').read_text());keep=np.asarray(cohort['original_indices']);assert len(keep)==d.farms
    topo=np.load(ASSETS/'topology.npz');assert np.all(np.isinf(np.diag(topo['distance']))) and np.all(np.diagonal(topo['ring_weights'],axis1=1,axis2=2)==0)
    assert topo['ids'].max()<d.farms
    h=np.load(DATA/'train/h.npy',mmap_mode='r')
    expected=np.einsum('ij,njt->nit',topo['near_weights'],h[:3,:,2:24]);assert np.allclose(expected,np.load(DATA/'train/near.npy',mmap_mode='r')[:3],atol=2e-6)
    from prepare_rm import regional_history
    from operational import select_issue
    fingerprint=json.loads((DATA/'READY.json').read_text())['preprocess_fingerprint']
    for split in ['train','val']:
        root=DATA/split;origin=np.load(root/'origin.npy')
        np.testing.assert_array_equal(origin,np.load(RUN/f'selection_inputs/{split}_origin.npy'))
        issue=np.load(root/'issue.npy');np.testing.assert_array_equal(issue,select_issue(origin))
        assert (issue%86400==0).all() and ((origin-issue>=54000)&(origin-issue<140400)).all()
        rows=np.unique(np.linspace(0,len(origin)-1,21,dtype=int))
        hh=np.load(root/'h.npy',mmap_mode='r')[rows]
        near=np.load(root/'near.npy',mmap_mode='r')[rows]
        np.testing.assert_allclose(near,np.einsum('ij,njt->nit',topo['near_weights'],hh[:,:,2:24]),atol=2e-6)
        rings=np.load(root/'rings.npy',mmap_mode='r')[rows]
        for ring in range(3):
            w=topo['ring_weights'][ring];cur=np.einsum('ij,nj->ni',w,hh[:,:,23])
            old=np.einsum('ij,nj->ni',w,hh[:,:,19]);second=np.einsum('ij,nj->ni',w,hh[:,:,23]**2)
            np.testing.assert_allclose(rings[:,:,4*ring],cur,atol=2e-6)
            np.testing.assert_allclose(rings[:,:,4*ring+1],np.sqrt(np.maximum(0,second-cur**2)),atol=2e-5)
            np.testing.assert_allclose(rings[:,:,4*ring+2],cur-old,atol=2e-6)
        rm=np.load(DATA/f'regional_history/{split}.npy',mmap_mode='r')[rows]
        np.testing.assert_allclose(rm,regional_history(hh,topo),atol=2e-6)
        # Verify the oldest and newest RM steps both use their exact local-history timestamp.
        for step in [0,23]:
            np.testing.assert_allclose(rm[:,:,step,6],np.einsum('ij,nj->ni',topo['near_weights'],hh[:,:,step]),atol=2e-6)
    assert sp['preprocess_fingerprint']==fingerprint
    from column_original import column_coordinates
    p=b['physical'];e=torch.as_tensor(np.load(ASSETS/'elevation.npy')[si,None],device=device).expand(-1,24)
    cc=column_coordinates(p,e).cpu().numpy();cn=np.stack([column(p[i].cpu().numpy(),float(e[i,0])) for i in range(len(si))]);assert np.allclose(cc,cn,atol=.002,rtol=2e-6)
    results=[]
    for family in ['W0','TF','TIDE','TFT','B0']:
        conditions=['neutral','cb','cbh','anchor','wrong_cb','wrong_cbh'] if family=='W0' else ['neutral','neutral_adapter','cb','wrong_cb'] if family=='TF' else ['neutral']
        for cond in conditions:
            spec=dict(sp,family=family,input='R3',condition=cond,config=dict(width=64,lr=2e-4,dropout=.05,weight_decay=1e-4))
            x={k:v.clone() for k,v in b.items()};x.update(base=x['h'][:,-1:].expand(-1,24).clone(),base_delta=x['h'].new_zeros(32,24),condition_h=x['h'].clone())
            model=SharedModel(spec).to(device);opt=torch.optim.AdamW(model.parameters(),lr=.001)
            for _ in range(2):
                model.train();pred=model(x);assert pred.shape==y.shape and torch.isfinite(pred).all();loss=(pred-y).abs().mean();loss.backward();assert all(torch.isfinite(z.grad).all() for z in model.parameters() if z.grad is not None);opt.step();opt.zero_grad(set_to_none=True)
            model.eval()
            with torch.no_grad():
                pred=model(x);perm=torch.randperm(len(y),device=device);shuffled=model({k:v[perm] for k,v in x.items()});assert torch.allclose(pred[perm],shuffled,atol=2e-5,rtol=2e-5)
            results.append(dict(family=family,condition=cond,loss=float(loss.detach()),parameters=sum(p.numel() for p in model.parameters())))
    for family in ['W0','TF']:
        spec=dict(sp,family=family,input='R3',rm=True,condition='neutral')
        data=SharedData('train',spec);inputs,target,_,_=data.batch(np.arange(32),device)
        model=SharedModel(spec).to(device);prediction=model(inputs)
        loss=(prediction-target).abs().mean();loss.backward()
        assert prediction.shape==target.shape and torch.isfinite(prediction).all()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        results.append(dict(family=family,condition='regional_history',loss=float(loss.detach())))
    t=np.load(DATA/'train/origin.npy');fold=np.load(ASSETS/'oof_fold.npy')
    for f in range(4):
        z=np.load(ASSETS/f'fold{f}.npz');tr=t[z['train']];va=t[z['valid']];he=t[z['heldout']]
        assert ((tr<he.min()-43200)|(tr>he.max()+43200)).all();assert ((va<he.min()-43200)|(va>he.max()+43200)).all()
        assert ((tr<va.min()-43200)|(tr>va.max()+43200)).all()
    for seed in SEEDS:
        donor=np.load(DATA/f'train/donor_r{seed}.npy');assert np.array_equal(fold,fold[donor]);assert (np.abs((t[donor]+28800)//86400-(t+28800)//86400)>=2).all()
    # Persist CPU fixture so remote GPU arithmetic is compared on exactly the same weights and inputs.
    spec=dict(sp,family='W0',input='R3',condition='neutral');model=SharedModel(spec).to(device).eval()
    with torch.no_grad():out=model(b)
    if device=='cpu':torch.save(dict(spec=spec,b=b,weights=model.state_dict(),output=out),RUN/'operations/fixture.pt')
    atomic_json(RUN/'operations'/('CHECKS.json' if device=='cpu' else 'GPU_CHECKS.json'),dict(timestamp=now(),passed=True,device=device,retained_farms=d.farms,tests=results,checks=['excluded_neighbor_absence','regional_available_through_origin','column_units','finite_gradients','permutation_equivariance','oof_target_exclusion12h','matched_month_clock_donors','no_future_label_in_forward']))
    print('checks passed',device,len(results),flush=True)
def fixture():
    configure();f=torch.load(RUN/'operations/fixture.pt',map_location='cpu',weights_only=False);m=SharedModel(f['spec']).cuda().eval();m.load_state_dict(f['weights']);b={k:v.cuda() for k,v in f['b'].items()}
    with torch.no_grad():pred=m(b).cpu()
    assert torch.allclose(pred,f['output'],atol=2e-5,rtol=2e-5)
    # Use nonzero head too; zero-initialized heads alone would be a vacuous fixture.
    with torch.no_grad():m.head[-1].weight.fill_(.01);m.head[-1].bias.fill_(.02)
    cm=SharedModel(f['spec']).eval();cm.load_state_dict({k:v.cpu() for k,v in m.state_dict().items()})
    with torch.no_grad():a=m(b).cpu();c=cm(f['b'])
    assert torch.allclose(a,c,atol=3e-5,rtol=3e-5),(a-c).abs().max().item()
    atomic_json(RUN/'operations/GPU_FIXTURE.json',dict(timestamp=now(),passed=True,max_absolute_error=float((a-c).abs().max()),gpu=torch.cuda.get_device_name()))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--device',default='cpu');p.add_argument('--fixture',action='store_true');a=p.parse_args();fixture() if a.fixture else checks(a.device)
