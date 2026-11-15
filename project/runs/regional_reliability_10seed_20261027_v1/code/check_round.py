"""Check learned-weight parity, as-of windows, ablations and resumable training."""
from settings import *
import numpy as np, torch
from physical_data import FarmData, ResidualFarmData
from physical_models import PhysicalModel, ResidualPhysicalModel, frozen
from regional_inputs import *
from train_physics import run

def main():
    frozen.configure(); torch.set_num_threads(1); checks=[]
    for path,h in json.loads((RESIDUAL_RUN/'SOURCE_FREEZE.json').read_text())['files'].items():assert sha(path)==h,path
    for site in range(56):
        ids,geo,valid,capacity=topology(site,True)
        assert site not in ids and np.all(geo[valid,2]<=150)
    checks.append('old frozen files, target exclusion and distance limits')
    source=RESIDUAL_RUN/'jobs/W0_R3_RAW_s06_r42'
    old=json.loads((source/'metrics.json').read_text());stats=json.loads((source/'normalization.json').read_text())
    a=FarmData('val',6,'R3_RAW',stats=stats,limit=64)
    b=ResidualFarmData('val',6,'R3_RAW',stats=stats,limit=64)
    assert torch.equal(a.x,b.x) and torch.equal(a.res_features,b.res_features)
    ma=PhysicalModel(old,a.x.shape[-1]);mb=ResidualPhysicalModel(old,b.x.shape[-1])
    weights=torch.load(source/old['selections']['mae']['checkpoint'],weights_only=False,map_location='cpu')['model']
    ma.load_state_dict(weights);mb.load_state_dict(weights);ma.eval();mb.eval()
    with torch.no_grad():
        assert torch.equal(ma(a.batch(slice(None))),mb(b.batch(slice(None))))
    checks.append('old learned R3_RAW summary and predictions bit-identical with raw-only loading')
    for variant in sorted(REGIONAL_VARIANTS):
        data=FarmData('train',6,variant,limit=32)
        spec=dict(site=6,seed=42,backbone='W0',variant=variant,delay_minutes=latency(variant))
        model=PhysicalModel(spec,data.x.shape[-1]);model.eval()
        with torch.no_grad():
            model.residual_projection.weight.normal_(0,.01);model.neighbor.out.weight.normal_(0,.01)
            d=data.batch(slice(None));normal=model(d)
            assert torch.isfinite(normal).all()
            assert torch.allclose(normal,model(d,'delay'+str(latency(variant))),atol=1e-6)
            if latency(variant)>0:
                dd=dict(d,region_power=d['region_power'].clone(),neighbor_raw=d['neighbor_raw'].clone())
                drop=latency(variant)//15
                dd['region_power'][...,-drop:]+=100;dd['neighbor_raw'][...,-drop:]+=100
                dd['res_features']=(summary(dd,variant,latency(variant))-data.res_mean)/data.res_std
                assert torch.equal(normal,model(dd))
            if variant=='R3_NO_N3':
                dd=dict(d,neighbor_raw=torch.randn_like(d['neighbor_raw']))
                assert torch.equal(normal,model(dd))
        if variant.startswith('C2_'):
            donor=np.load(ASSETS/'mismatch_train_r42.npy')[:32]
            ids=topology(6,True)[0]
            expected=np.load(CACHE/'train/station_history.npy',mmap_mode='r')[donor[:,None],ids[None],:]
            assert np.array_equal(expected,data.region_power.numpy())
        del data,model
    checks.append('all variant paths, learned nonzero routes, fixed latency and C2 joint donor isolation')
    d=a.batch(slice(None))
    with torch.no_grad():
        for intervention in INTERVENTIONS:assert torch.isfinite(ma(d,intervention)).all()
        assert torch.equal(ma(d,'drop30'),torch.cat([ma(a.batch(slice(i,i+16)),'drop30') for i in range(0,64,16)]))
    raw=summary(dict(d,region_mask=torch.zeros_like(d['region_mask'])),'R3_RAW')
    assert torch.equal(raw,torch.zeros_like(raw))
    for variant,active_columns in [('R3_INNER',[0,1,2,9]),('R3_MIDDLE',[10,11,12,19]),
                                   ('R3_OUTER',[20,21,22,29]),('R3_MEAN',[0,9,10,19,20,29])]:
        value=summary(d,variant);inactive=[i for i in range(36) if i not in active_columns]
        assert torch.count_nonzero(value[:,inactive])==0
    checks.append('empty neighborhoods, ring/statistic masks, finite interventions and batch-stable missingness')
    for backbone,variant in [('W0','R3_D0'),('TF','R3_RAW')]:
        spec=dict(id=f'check_{backbone}_{variant}',site=6,seed=42,backbone=backbone,variant=variant,
                  loss='MAE',delay_minutes=latency(variant),tf_zero_init=True,tf_dropout=.05,tf_lr=.0002)
        run(spec,'cpu',True)
    spec=dict(id='check_resume_D0',site=6,seed=42,backbone='W0',variant='R3_D0',loss='MAE',delay_minutes=0,
              tf_zero_init=True,tf_dropout=.05,tf_lr=.0002)
    run(spec,'cpu',True,stop_after_epoch=1);run(spec,'cpu',True)
    x=RUN/'smoke_jobs/check_W0_R3_D0';y=RUN/'smoke_jobs/check_resume_D0'
    xm=json.loads((x/'metrics.json').read_text());ym=json.loads((y/'metrics.json').read_text())
    for k in ['mae','rmse']:
        assert xm['selections'][k]['metrics']==ym['selections'][k]['metrics']
        xx=torch.load(x/xm['selections'][k]['checkpoint'],weights_only=False)['model']
        yy=torch.load(y/ym['selections'][k]['checkpoint'],weights_only=False)['model']
        assert all(torch.equal(xx[n],yy[n]) for n in xx)
    checks.append('real W0/TF two-epoch fits and exact optimizer/dropout resume')
    atomic_json(RUN/'operations/CHECKS.json',dict(timestamp=now(),passed=True,checks=checks,smoke_only=True,test_year_accessed=False))
    print('CHECKS PASSED',checks,flush=True)

if __name__=='__main__':main()
