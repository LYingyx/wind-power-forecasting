"""Meaningful leakage, topology, intervention, replay and resume checks."""
from settings import *
from physical_data import FarmData,FEATURES
from physical_models import PhysicalModel,ObservationReadout,frozen
from neighbor_inputs import VARIANTS,NEIGHBORS,topology,window,ages,mismatch_map
from train_physics import run,predict,metrics
import torch,numpy as np,importlib.util,shutil

def module(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def main():
    setup();frozen.configure();checks={}
    for site in range(56):
        for variant in ['N1','N4_K4','N4_K16','N5_CORR']:
            ids,geo,mask,cap=topology(site,variant)
            assert site not in ids and np.isfinite(geo).all() and (geo[mask,2]<=150).all()
            assert (cap>0).all()
    for delay in [0,15,30]:
        ix=np.arange(24)[window(delay)]
        assert len(ix)==22 and ix[-1]==23-delay//15
        np.testing.assert_array_equal(ages(delay),(ix-23)/4)
    checks['target_exclusion_radius_and_fixed_window_22']=True
    ready=json.loads((ASSETS/'READY.json').read_text())
    assert all(v['validation_donors_always_past'] for k,v in ready['mismatch_checks'].items() if k.startswith('val'))
    checks['surrogate_split_month_clock_and_day_separation']=True
    olddata=module('parent_data',ATTRIBUTION_RUN/'code/physical_data.py');oldmodel=module('parent_model',ATTRIBUTION_RUN/'code/physical_models.py')
    for bb in ['W0','TF']:
        oldspec=dict(site=6,seed=42,backbone=bb,variant='SEL',tf_zero_init=bb=='TF',tf_lr=2e-4,tf_dropout=.05)
        new=FarmData('train',6,'N0',limit=64);old=olddata.FarmData('train',6,'SEL',limit=64)
        assert torch.equal(new.x,old.x) and new.stats==old.stats
        a=PhysicalModel(dict(oldspec,variant='N0'),22);b=oldmodel.PhysicalModel(oldspec,22)
        assert all(torch.equal(v,b.state_dict()[k]) for k,v in a.state_dict().items())
        # A trained checkpoint prevents zero-head equality from hiding differences.
        source=ATTRIBUTION_RUN/'jobs'/('A2_W0_SEL_s06_r42' if bb=='W0' else 'A2_TF_z0_lr0.001_dp0_SEL_s06_r42')
        met=json.loads((source/'metrics.json').read_text());cp=source/met['selections']['mae']['checkpoint']
        weights=torch.load(cp,weights_only=False)['model'];a.load_state_dict(weights);b.load_state_dict(weights);a.eval();b.eval()
        with torch.no_grad():assert torch.equal(a(new.batch(slice(None))),b(old.batch(slice(None))))
    checks['N0_SEL_original_inputs_weights_and_predictions_exact']=True
    data=FarmData('train',6,'N1',limit=64);base=FarmData('train',6,'N0',limit=64)
    assert data.x.shape[-1]==22 and torch.equal(data.x,base.x) and 'calendar' not in FEATURES['N1']
    dimensions=[];initial={}
    for variant in NEIGHBORS:
        d=FarmData('train',6,variant,limit=64,seed=43,delay_minutes=0 if variant=='N7_D0' else 30 if variant=='N7_D30' else 15)
        spec=dict(site=6,seed=43,variant=variant,backbone='W0',delay_minutes=0 if variant=='N7_D0' else 30 if variant=='N7_D30' else 15)
        m=PhysicalModel(spec,22);dimensions.append(sum(p.numel() for p in m.parameters()));initial[variant]={n:p.detach().clone() for n,p in m.neighbor.named_parameters()}
        loss=(m(d.batch(slice(None)))-d.y).abs().mean();loss.backward();assert torch.isfinite(loss)
        assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
    assert len(set(dimensions))==1
    for v in ['N2','N2_SELF','N3']:
        assert all(torch.equal(t,initial[v][name]) for name,t in initial['N1'].items())
    checks['all_neighbor_arms_identical_parameter_count_and_control_initialization']=True
    batch=data.batch(slice(None));n=ObservationReadout(6,'N1');n.out.weight.data.normal_()
    original,_=n(batch)
    changed={**batch,'neighbor_raw':batch['neighbor_raw'].clone()};changed['neighbor_raw'][:,~n.mask]=1e5
    assert torch.equal(original,n(changed)[0])
    p,_=n.values(batch,'neighbor_identity');valid=n.mask.nonzero().flatten();raw,_=n.values(batch)
    assert torch.equal(p[:,valid],raw[:,valid.roll(1)]) and torch.equal(p[:,~n.mask],raw[:,~n.mask])
    assert torch.equal(n(batch,'delay15')[0],original)
    n.mask.zero_();assert torch.equal(n(batch)[0],torch.zeros_like(original))
    checks['invalid_neighbors_and_null_token_exact_zero_identity_valid_only']=True
    n=ObservationReadout(6,'N3');values,_=n.values(batch);weights=n.capacity*n.mask
    expected=(batch['neighbor_raw'][...,window(15)]*weights[None,:,None]).sum(1)/weights.sum()
    assert torch.equal(values,expected[:,None].expand_as(values))
    n=ObservationReadout(6,'N2_SELF');values,_=n.values(batch)
    assert torch.equal(values,batch['h'][:,None,window(15)].expand_as(values))
    checks['regional_capacity_weighting_and_self_only_control']=True
    cases=[]
    for bb,v in [('W0','N1'),('W0','N2'),('TF','N3'),('TF','N4_K16')]:
        spec=dict(id='smoke_'+bb+'_'+v,site=6,seed=42,backbone=bb,variant=v,loss='MAE',delay_minutes=15,capacity_control=False,tf_zero_init=True,tf_lr=2e-4,tf_dropout=.05)
        run(spec,'cpu',smoke=True);out=RUN/'smoke_jobs'/spec['id'];result=json.loads((out/'metrics.json').read_text())
        va=FarmData('val',6,v,stats=json.loads((out/'normalization.json').read_text()),limit=64,seed=42)
        for s in result['selections'].values():
            m=PhysicalModel(spec,22);m.load_state_dict(torch.load(out/s['checkpoint'],weights_only=False)['model'])
            assert metrics(predict(m,va),va.y.numpy())==s['metrics']
        if v in ['N1','N4_K16']:assert result['interventions']['delay15']==result['selections']['mae']['metrics']
        cases.append(spec['id']);shutil.rmtree(out)
    spec=dict(id='resume_full',site=6,seed=42,backbone='TF',variant='N1',loss='MAE',delay_minutes=15,tf_zero_init=True,tf_lr=2e-4,tf_dropout=.05)
    split=dict(spec,id='resume_split');run(spec,'cpu',smoke=True);run(split,'cpu',smoke=True,stop_after_epoch=1);run(split,'cpu',smoke=True)
    a=RUN/'smoke_jobs'/spec['id'];b=RUN/'smoke_jobs'/split['id'];am=json.loads((a/'metrics.json').read_text());bm=json.loads((b/'metrics.json').read_text())
    for criterion,x in am['selections'].items():
        y=bm['selections'][criterion];assert x['metrics']==y['metrics'] and x['epoch']==y['epoch']
        aw=torch.load(a/x['checkpoint'],weights_only=False)['model'];bw=torch.load(b/y['checkpoint'],weights_only=False)['model'];assert all(torch.equal(v,bw[k]) for k,v in aw.items())
    shutil.rmtree(a);shutil.rmtree(b);checks['resume_with_dropout_and_interventions_exact']=True
    oldfreeze=json.loads((ATTRIBUTION_RUN/'SOURCE_FREEZE.json').read_text())['files'];assert all(sha(p)==h for p,h in oldfreeze.items())
    checks['all_85_parent_scientific_sources_unchanged']=True
    report=dict(timestamp=now(),passed=True,checks=checks,smoke_cases=cases,smoke_is_not_formal=True)
    atomic_json(RUN/'CHECKS.json',report);atomic_json(RUN/'operations/CHECKS.json',report)
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
