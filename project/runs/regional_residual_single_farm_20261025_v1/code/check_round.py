"""Causality, control isolation, legacy parity, real-data training and resume checks."""
from settings import *
from residual_inputs import *
from physical_data import FarmData,LegacyFarmData
from physical_models import PhysicalModel,LegacyPhysicalModel,frozen
from residual_features import fields,summary,moment
from train_physics import run,predict
import numpy as np,torch,shutil

def main():
    frozen.configure();torch.set_num_threads(1);checks=[]
    for path,h in json.loads((NEIGHBOR_RUN/'SOURCE_FREEZE.json').read_text())['files'].items():assert sha(path)==h
    checks.append('all prior frozen files unchanged')
    for site,table in curves()['stations'].items():
        for c in [table['final'],*table['oof'].values()]:
            assert np.all(np.diff(c['x'])>0) and np.all(np.diff(c['y'])>=0) and min(c['y'])>=0 and max(c['y'])<=1.200001
    time=np.load(CACHE/'train/origin_unix.npy')-900
    for month,f in curves()['folds'].items():
        fit=(time<f['excluded_start_unix'])|(time>=f['excluded_end_unix'])
        assert not np.any(fit&(month_index(time)==int(month))) and int(fit.sum())==f['n_fit']
    for site in range(56):
        ids,g,mask,cap=topology(site,True);assert site not in ids and np.all(g[mask,2]<=150)
        from neighbor_inputs import topology as oldtop
        old=oldtop(site,'N1');assert np.array_equal(ids[:8],old[0]) and np.array_equal(mask[:8],old[2])
    checks.append('monotone curves, OOF month buffers, target exclusion, 150km mask and legacy neighbor parity')
    for split in ['train','val']:
        w=np.load(ASSETS/f'boundary_weather_{split}.npy');assert np.abs(w).max()<30,('normalization',split,np.abs(w).max())
        for delay in [0,15,30]:
            from neighbor_inputs import ages
            assert max(ages(delay))<=-delay/60
    train=FarmData('train',6,'R1',limit=64);val=FarmData('val',6,'R1',stats=train.stats,limit=64)
    p,m,r,_=fields(val.batch(slice(None)),'R1');assert torch.equal(r,p-m)
    c=summary(val.batch(slice(None)),'R1');assert torch.allclose(c,(val.res_features*val.res_std+val.res_mean),atol=1e-6)
    assert torch.equal(moment(torch.ones(2,3,6),torch.zeros(3)),torch.zeros(2,3))
    checks.append('latency windows, residual identity, summary normalization and empty neighborhoods')
    for variant in W0_VARIANTS:
        data=FarmData('train',6,variant,limit=32)
        spec=dict(id='check_'+variant,site=6,seed=42,backbone='W0',variant=variant,loss='MAE')
        model=PhysicalModel(spec,data.x.shape[-1],data.historical_input.mean(0));model.eval();batch=data.batch(slice(None))
        # Activate all output paths before isolation checks, avoiding a zero-init tautology.
        with torch.no_grad():
            model.neighbor.out.weight.normal_(0,.01)
            if hasattr(model,'residual_projection'):model.residual_projection.weight.normal_(0,.01)
            output=model(batch);assert torch.isfinite(output).all()
            assert torch.allclose(output,model(batch,intervention='delay15'),atol=1e-6)
            if variant.startswith('C1_'):
                changed=dict(batch,res_power=torch.randn_like(batch['res_power']))
                if 'res_features' in batch:changed['res_features']=(summary(changed,variant)-data.res_mean)/data.res_std
                assert torch.equal(output,model(changed))
            if variant in ['R1','R2','R3','R4']:
                for intervention in ['power_identity','residual_identity','residual_zero','direction_rotate','delay0','delay30']:assert torch.isfinite(model(batch,intervention=intervention)).all()
        if variant.startswith('C2_'):
            donor=np.load(NEIGHBOR_RUN/'assets/mismatch_train_r42.npy')[:32];ids=topology(6,family(variant) in ['R3','R4'])[0]
            expected=np.load(CACHE/'train/station_history.npy',mmap_mode='r')[donor[:,None],ids[None],:]
            assert np.array_equal(expected,data.res_power.numpy())
            assert np.array_equal(mapping_history('train',int(ids[0]),donor)[0],data.res_mapped[:,0].numpy())
        del data,model
    checks.append('all 16 model paths, active C1 observation isolation, joint C2 donors, exact delay15 and finite interventions')
    for variant in ['N1','N3']:
        source=NEIGHBOR_RUN/'jobs'/f'W0_{variant}_s06_r42';old=json.loads((source/'metrics.json').read_text());stats=json.loads((source/'normalization.json').read_text())
        data=FarmData('val',6,variant,stats=stats,limit=64);a=PhysicalModel(old,data.x.shape[-1]);b=LegacyPhysicalModel(old,data.x.shape[-1])
        state=torch.load(source/old['selections']['mae']['checkpoint'],map_location='cpu',weights_only=False)['model'];a.load_state_dict(state);b.load_state_dict(state)
        a.eval();b.eval()
        with torch.no_grad():assert torch.equal(a(data.batch(slice(None))),b(data.batch(slice(None))))
    checks.append('learned N1 and N3 checkpoint predictions exactly unchanged')
    for backbone,variant in [('W0','R1'),('W0','R2'),('W0','R3'),('W0','R4'),('TF','R2')]:
        spec=dict(id=f'check_fit_{backbone}_{variant}',site=6,seed=42,backbone=backbone,variant=variant,loss='MAE',tf_zero_init=True,tf_lr=.0002,tf_dropout=.05)
        run(spec,'cpu',True)
    spec=dict(id='check_resume_R2',site=6,seed=42,backbone='W0',variant='R2',loss='MAE',tf_zero_init=True,tf_lr=.0002,tf_dropout=.05)
    run(spec,'cpu',True,stop_after_epoch=1);run(spec,'cpu',True)
    aa=RUN/'smoke_jobs/check_fit_W0_R2';bb=RUN/'smoke_jobs/check_resume_R2'
    am=json.loads((aa/'metrics.json').read_text());bm=json.loads((bb/'metrics.json').read_text())
    for key in ['mae','rmse']:
        assert am['selections'][key]['metrics']==bm['selections'][key]['metrics']
        a=torch.load(aa/am['selections'][key]['checkpoint'],weights_only=False)['model'];b=torch.load(bb/bm['selections'][key]['checkpoint'],weights_only=False)['model']
        assert all(torch.equal(a[k],b[k]) for k in a)
    checks.append('five real-data two-epoch fits and bit-identical dropout/optimizer resume')
    from sparse_arx import run as run_arx
    run_arx(dict(id='check_ARX',site=6,seed=0,task='arx',variant='C3',stage='C3_ARX',backbone='ARX',configuration='ARX'),'cpu',True)
    checks.append('CPU sparse ARX all four families, alpha selection, finite predictions')
    atomic_json(RUN/'operations/CHECKS.json',dict(timestamp=now(),passed=True,checks=checks,smoke_only=True,test_year_accessed=False))
    print('CHECKS PASSED',checks,flush=True)
if __name__=='__main__':main()
