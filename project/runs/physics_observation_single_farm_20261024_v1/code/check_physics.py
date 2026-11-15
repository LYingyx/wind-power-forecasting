"""Scientific invariants and real-data model replay, rather than implementation mirrors."""
from settings import *
from physics import *
from physical_data import FarmData,FEATURES,interpolate
from physical_models import *
from train_physics import loss_value
import gc

def check():
    frozen.configure();report={}
    # Exact geostrophic solution on a plane, both latitude orders.
    for latstep in [.25,-.25]:
        lat=45.;f=2*OMEGA*np.sin(np.deg2rad(lat));dx=np.deg2rad(.25)*6371000*np.cos(np.deg2rad(lat));dy=np.deg2rad(latstep)*6371000
        x=(np.arange(7)-3)*dx;y=(np.arange(7)-3)*dy
        patch=np.zeros((2,8,7,7,14));patch[...,5]=f*(4*x[None,:]-8*y[:,None]);patch[...,3]=280;patch[...,9]=280
        patch[...,0]=1e5+np.arange(8)[None,:,None,None]*100
        weights=np.zeros((7,7));weights[3,3]=1
        dynamic=dynamic_fields(patch,lat,latstep,.25,weights)
        np.testing.assert_allclose(dynamic[...,0],.8,atol=1e-6);np.testing.assert_allclose(dynamic[...,1],.4,atol=1e-6)
        np.testing.assert_allclose(dynamic[...,5],1,atol=1e-6)
    report['signed_geopotential_gradients_and_pressure_units']='pass'
    w=np.zeros((1,14));w[:,0]=101325;w[:,3]=288.15;w[:,9]=280;w[:,5]=800*G;w[:,1]=10
    d0,s0=thermodynamics(w,0);d1,s1=thermodynamics(w,1000)
    assert d1[0,0]<d0[0,0] and s1[0,3]==0 and np.isfinite(s1).all()
    report['surface_pressure_elevation_and_below_ground_mask']='pass'
    stamp=np.array(['2024-02-28T16:00:00'],dtype='datetime64[s]').astype('i8')
    c=calendar(stamp,np.array([0,6]));np.testing.assert_allclose(c[0,0,:2],[0,1],atol=1e-6);np.testing.assert_allclose(c[0,1,:2],[1,0],atol=1e-6)
    report['local_time_and_leap_year']='pass'
    ids,geo,mask=neighbor_geometry([45,45],[120,120.5],maximum_k=1)
    assert ids[0,0]==1 and geo[0,0,0]<0 and geo[1,0,0]>0
    report['neighbor_to_target_direction']='pass'
    nodes=torch.tensor([[[0.],[2.],[4.]]]);times=torch.tensor([[0.,1.,2.]])
    q=torch.tensor([[.25,1.25]],requires_grad=True);value=interpolate(nodes,times,q)
    torch.testing.assert_close(value[...,0],torch.tensor([[.5,2.5]]))
    value.sum().backward();torch.testing.assert_close(q.grad,torch.tensor([[2.,2.]]))
    report['temporal_shift_interpolation_gradient']='pass'
    # Actual data, both backbones, identical initial predictions across every addition.
    base=FarmData('train',6,'BASE',limit=8);db=base.batch(slice(None))
    ci=np.load(CACHE/'train/weather_cycle_index.npy')[:8]
    patch=torch.from_numpy(np.array(np.load(PREVIOUS/'prepared/train/site06.npy',mmap_mode='r')[ci]))
    weights=torch.from_numpy(np.load(PREVIOUS/'prepared/bilinear_weights.npy')[6])
    old_nodes=(patch*weights[None,None,:,None]).sum(2)
    old_weather=frozen.interpolate_grid(old_nodes[:,:,None],db['times'])[:,:,0]
    difference=float((old_weather-db['weather']).abs().max());assert difference<2e-6,difference
    report['frozen_weather_front_replay_max_abs']=difference
    for kind in ['W0','TF']:
        spec=dict(site=6,seed=42,backbone=kind,variant='BASE')
        m=PhysicalModel(spec,0);m.eval();pred=m(db).detach()
        if kind=='W0':
            st=torch.load(OLD/'jobs/reference_B0_s06_r42/best.pt',map_location='cpu',weights_only=False)['model']
            original=frozen.old.correction(6,42,64,'NEUTRAL_LAST',st);original.eval()
            expected=original(db['h'],db['weather'],db['phase'],torch.zeros_like(db['h'])).detach()
        else:
            frozen.seed_all(42);original=frozen.old.named('TRANSFORMER_FC',64);original.eval();expected=original(db['h'],db['weather'],db['phase']).detach()
        delta=float((pred-expected).abs().max());assert delta<2e-6,delta
        report[kind+'_original_replay_max_abs']=delta
        for variant in FEATURES:
            d=FarmData('train',6,variant,limit=8)
            model=PhysicalModel(dict(spec,variant=variant),d.x.shape[-1],d.historical_input.mean(0));model.eval()
            prediction,details=model(d.batch(slice(None)),return_details=True)
            delta=float((prediction.detach()-pred).abs().max());assert delta<2e-6,(variant,delta)
            loss=loss_value(prediction,d.y,'MIX');loss.backward()
            assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
            if variant=='NEIGH_ADVECT':assert torch.allclose(details['neighbor_weights'].sum(-1),torch.ones_like(pred))
            if variant.startswith('CURVE_'):
                curve=model.curve;v=torch.tensor([0.,5.,10.,20.,30.]);response=curve.response(v)
                if variant=='CURVE_BIAS':assert torch.all(response.diff()>=0)
                # Formula is anchored exactly at lead zero, with no extra duplicated bias.
                o=d.h[:,-1];m0=curve.response(d.ref_density[:,1]*10)
                torch.testing.assert_close(m0+torch.ones_like(m0)*(o-m0),o)
            report[kind+'_'+variant]=dict(replay_max_abs=delta,finite_gradients=True,parameters=sum(p.numel() for p in model.parameters()))
            del model,d;gc.collect()
    # Exact neighbor endpoint and no target farm in neighborhood.
    for delay in [0,15,30]:
        d=FarmData('train',6,'NEIGH_GENERIC',limit=8,delay_minutes=delay)
        assert float(d.neighbor_ages[-1])==-delay/60 and 6 not in np.load(PREPARED/'neighbor_ids.npy')[6]
    report['neighbor_availability_delay_checks']='pass'
    for variant in ['CAL','REF','DELTA','DENS','STAB','DYN','STATE','CYCLE','ALL']:
        real=FarmData('train',6,variant,limit=8)
        control=FarmData('train',6,variant,limit=8,capacity_control=True)
        assert real.x.shape==control.x.shape
        assert control.stats['capacity_control'] and not real.stats['capacity_control']
    report['matched_capacity_controls']='pass'
    atomic_json(RUN/'CHECKS.json',dict(timestamp=now(),checks=report,passed=True))
    print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':check()
