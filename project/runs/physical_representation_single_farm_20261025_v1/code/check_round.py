"""Scientific checks for new feature attribution and independent TF control factors."""
from settings import *
from physical_data import FarmData,FEATURES,A1,representation_features
from physical_models import PhysicalModel,frozen
from train_physics import run,predict,metrics
from diagnostics import stratified
import torch,numpy as np,importlib.util,shutil

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

def main():
    setup();frozen.configure();checks={}
    olddata=load('parent_data',PARENT_RUN/'code/physical_data.py');oldmodels=load('parent_model',PARENT_RUN/'code/physical_models.py')
    for variant in ['BASE','DENS','ALL']:
        a=FarmData('train',6,variant,limit=64);b=olddata.FarmData('train',6,variant,limit=64)
        assert torch.equal(a.x,b.x) and a.stats==b.stats
        for backbone in ['W0','TF']:
            spec=dict(site=6,seed=42,backbone=backbone,variant=variant)
            m=PhysicalModel(spec,a.x.shape[-1]);old=oldmodels.PhysicalModel(spec,b.x.shape[-1]);m.eval();old.eval()
            assert all(torch.equal(v,old.state_dict()[k]) for k,v in m.state_dict().items())
            with torch.no_grad():assert torch.equal(m(a.batch(slice(None))),old(b.batch(slice(None))))
    checks['existing_defaults_exact_replay']=True
    data=FarmData('train',6,'DENS',limit=64);physical=data.physical.numpy();density=data.density.numpy()
    values=representation_features(physical,density);v=np.hypot(physical[...,1],physical[...,2])
    np.testing.assert_allclose(values['spd'][...,0],v/10,rtol=1e-6)
    np.testing.assert_allclose(values['spd3'][...,0],v**3/1000,rtol=1e-6)
    assert np.all(values['spd'][...,1:]==0)
    fixed=density.copy();fixed[...,0]=1.;fixed[...,1]=v/10;fixed[...,2]=1.225*v**3/1000
    np.testing.assert_allclose(values['dens_norho'],fixed,rtol=1e-6)
    assert np.array_equal(values['dens_uncoupled'][...,0],density[...,0])
    checks['density_and_speed_decomposition_units']=True
    for variant in A1+[v for v in FEATURES if v.startswith('SEL')]:
        tr=FarmData('train',6,variant,limit=64);va=FarmData('val',6,variant,stats=tr.stats,limit=32)
        assert torch.isfinite(tr.x).all() and torch.isfinite(va.x).all()
        if variant in A1:assert tr.x.shape[-1]==3
        if variant.startswith('SEL_NO_'):assert tr.x.shape[-1]==22
        for backbone in ['W0','TF']:
            spec=dict(site=6,seed=42,backbone=backbone,variant=variant)
            m=PhysicalModel(spec,tr.x.shape[-1]);loss=(m(tr.batch(slice(None)))-tr.y).abs().mean();loss.backward()
            assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
    checks['all_new_variants_finite_gradients_and_matched_slots']=True
    tf0=dict(site=6,seed=42,backbone='TF',variant='DENS',tf_zero_init=True,tf_lr=2e-4,tf_dropout=.05)
    m=PhysicalModel(tf0,3);m.eval()
    with torch.no_grad():assert torch.equal(m(data.batch(slice(None))),data.h[:,-1:].expand(-1,24))
    assert m.groups()[0]['lr']==2e-4
    assert all(mod.p==.05 for mod in m.backbone.original.modules() if isinstance(mod,torch.nn.Dropout))
    assert all(mod.dropout==.05 for mod in m.backbone.original.modules() if isinstance(mod,torch.nn.MultiheadAttention))
    w=PhysicalModel(dict(tf0,backbone='W0'),3)
    checks['original_dropout_values']=sorted({float(mod.p) for mod in w.backbone.modules() if isinstance(mod,torch.nn.Dropout)})
    checks['tf_zero_head_persistence_and_independent_lr_dropout']=True
    full=FarmData('train',6,'SEL',limit=64)
    for variant,positions in [('SEL_NO_DENS',slice(0,3)),('SEL_NO_DYN',slice(3,10)),('SEL_NO_STAB',slice(10,15)),('SEL_NO_STATE',slice(15,22))]:
        ab=FarmData('train',6,variant,limit=64);want=full.x.clone();want[...,positions]=0
        assert torch.equal(ab.x,want)
    checks['leave_group_out_changes_only_named_slots']=True
    cases=[]
    for backbone,variant,zi,drop in [('W0','SPD3',False,0),('TF','DENS_UNCOUPLED',True,.05),('W0','SEL_NO_DYN',False,0),('TF','SEL_SPD_LV',False,0)]:
        spec=dict(id='smoke_'+backbone+'_'+variant,site=6,seed=42,backbone=backbone,variant=variant,loss='MAE',delay_minutes=15,capacity_control=False,tf_zero_init=zi,tf_dropout=drop,tf_lr=2e-4)
        run(spec,'cpu',smoke=True);folder=RUN/'smoke_jobs'/spec['id'];meta=json.loads((folder/'metrics.json').read_text());stats=json.loads((folder/'normalization.json').read_text())
        va=FarmData('val',6,variant,stats=stats,limit=64)
        for selection in meta['selections'].values():
            model=PhysicalModel(spec,va.x.shape[-1]);model.load_state_dict(torch.load(folder/selection['checkpoint'],weights_only=False)['model'])
            score=metrics(predict(model,va),va.y.numpy());assert score==selection['metrics']
        assert (folder/'diagnostics.json').exists();cases.append(spec['id']);shutil.rmtree(folder)
    # Optimizer/RNG-state continuation, including dropout and zero-initialized head.
    spec=dict(tf0,id='resume_full',loss='MAE',delay_minutes=15,capacity_control=False)
    resumed=dict(spec,id='resume_split');run(spec,'cpu',smoke=True);run(resumed,'cpu',smoke=True,stop_after_epoch=1);run(resumed,'cpu',smoke=True)
    a=json.loads((RUN/'smoke_jobs'/spec['id']/'metrics.json').read_text());b=json.loads((RUN/'smoke_jobs'/resumed['id']/'metrics.json').read_text())
    for k in a['selections']:
        left,right=a['selections'][k],b['selections'][k]
        assert left['epoch']==right['epoch'] and left['metrics']==right['metrics']
        la=torch.load(RUN/'smoke_jobs'/spec['id']/left['checkpoint'],weights_only=False)['model']
        rb=torch.load(RUN/'smoke_jobs'/resumed['id']/right['checkpoint'],weights_only=False)['model']
        assert all(torch.equal(v,rb[name]) for name,v in la.items())
    for s in [spec,resumed]:shutil.rmtree(RUN/'smoke_jobs'/s['id'])
    checks['checkpoint_resume_with_dropout_exact']=True
    oldfreeze=json.loads((PARENT_RUN/'SOURCE_FREEZE.json').read_text())['files'];assert all(sha(p)==h for p,h in oldfreeze.items())
    checks['parent_science_freeze_unchanged']=len(oldfreeze)
    result=dict(timestamp=now(),passed=True,checks=checks,smoke_cases=cases,not_formal_results=True)
    atomic_json(RUN/'CHECKS.json',result);atomic_json(RUN/'SMOKE_CHECK.json',result);atomic_json(RUN/'operations/CHECKS.json',result)
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
