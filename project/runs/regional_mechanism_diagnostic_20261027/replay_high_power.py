"""Exploratory 2023 high-origin-power audit, frozen weights, no training/prediction files."""
from pathlib import Path
import os,sys,json,time,gc,argparse
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
HERE=Path(__file__).resolve().parent;PROJECT=HERE.parents[1]
SOURCE=PROJECT/'runs/regional_observation_10seed_20261026_v1'
sys.path.insert(0,str(SOURCE/'code'))
from settings import atomic_json,sha,now
from physical_data import FarmData
from physical_models import PhysicalModel,frozen
import numpy as np
import torch


def score(e,y,p,mask):
    v=e[mask];n=len(v)
    if not n:return None
    return dict(n=n,mae=float(abs(v).mean()),rmse=float(np.sqrt(np.mean(v*v))),
        bias=float(v.mean()),sq_sum=float(np.sum(v*v)),abs_sum=float(np.sum(abs(v))),
        mean_y=float(y[mask].mean()),mean_pred=float(p[mask].mean()),
        positive_sq_sum=float(np.square(v[v>0]).sum()),negative_sq_sum=float(np.square(v[v<0]).sum()),
        large_over_count=int((v>.2).sum()),large_under_count=int((v<-.2).sum()))


def main(site,seed):
    os.nice(15);frozen.configure();began=time.monotonic()
    out=HERE/'jobs'/f's{site:02d}_r{seed}'
    if (out/'SUCCESS.json').exists():return
    out.mkdir(parents=True,exist_ok=True)
    predictions={};audit=[];groups=None
    for variant in ['N0','N3','R3_RAW','R3_ALLPOOL']:
        jid=f'W0_{variant}_s{site:02d}_r{seed}';src=SOURCE/'jobs'/jid
        m=json.loads((src/'metrics.json').read_text());selection=m['selections']['mae'];cp=src/selection['checkpoint']
        assert sha(cp)==selection['sha256']
        stats=json.loads((src/'normalization.json').read_text())
        data=FarmData('val',site,variant,'cpu',stats=stats,seed=seed)
        ix=np.flatnonzero(data.h.numpy()[:,-1]>=.7)
        assert len(ix)>0
        if groups is None:
            y=data.y.numpy()[ix].astype('f8');h=data.h.numpy()[ix].astype('f8');origin=data.origin[ix]
            o=h[:,-1:];delta=y-o
            physical=data.physical.numpy()[ix];reference=data.reference_physical.numpy()[ix]
            wind_delta=np.hypot(physical[...,1],physical[...,2])-np.hypot(reference[:,1],reference[:,2])[:,None]
            hist_delta=h[:,-1]-h[:,-5]
            groups={'outcome_from_origin':{'drop_gt20':delta<-.2,'drop_5to20':(delta>=-.2)&(delta<-.05),'steady_5':abs(delta)<=.05,'rise_gt5':delta>.05},
                'nwp_change_from_origin':{'down':wind_delta<-1,'flat':abs(wind_delta)<=1,'up':wind_delta>1},
                'history_change_1h':{'down':hist_delta<-.1,'flat':abs(hist_delta)<=.1,'up':hist_delta>.1},
                'current_power':{'high':np.ones_like(y,dtype=bool)},
                # Already restricted by the original float32 CF>=0.7 mask.
                # Do not reapply its lower boundary after casting to float64.
                'origin_power_level':{'70_85':o[:,0]<.85,'85_100':(o[:,0]>=.85)&(o[:,0]<1),'above100':o[:,0]>=1}}
            localhour=((origin[:,None]+np.arange(1,25)[None]*900+8*3600)//3600)%24
            groups['local_hour']={str(k)+'_'+str(k+6):(localhour>=k)&(localhour<k+6) for k in [0,6,12,18]}
        model=PhysicalModel(m,data.x.shape[-1],torch.zeros(114)).eval()
        model.load_state_dict(torch.load(cp,map_location='cpu',weights_only=False)['model'])
        with torch.inference_mode():
            pred=torch.cat([model(data.batch(torch.as_tensor(ix[b:b+256]))) for b in range(0,len(ix),256)]).numpy().astype('f8')
        old=json.loads((src/'diagnostics.json').read_text())['rows']
        old=next(r for r in old if r['group']=='current_power' and r['bin']=='high' and r['horizon']=='all')
        e=pred-y
        diffs={'mae':abs(abs(e).mean()-old['mae']),'rmse':abs(np.sqrt(np.mean(e*e))-old['rmse']),'bias':abs(e.mean()-old['bias'])}
        assert max(diffs.values())<2e-6,(jid,diffs)
        audit.append(dict(id=jid,checkpoint=str(cp),sha256=sha(cp),high_origins=len(ix),metric_max_difference=float(max(diffs.values()))))
        predictions[variant]=pred
        if variant=='R3_RAW':
            power=data.region_power.numpy()[ix];weights=data.region_capacity.numpy()*data.region_mask.numpy();weights=weights/weights.sum()
            avg=(power*weights[None,:,None]).sum(1);latest=avg[:,-2];regional_delta=avg[:,-2]-avg[:,-6]
            gap=latest-h[:,-1]
            groups['region_minus_local']={'lower':gap<-.1,'similar':abs(gap)<=.1,'higher':gap>.1}
            groups['region_change_1h']={'down':regional_delta<-.05,'flat':abs(regional_delta)<=.05,'up':regional_delta>.05}
        del model,data;gc.collect()
        atomic_json(out/'PROGRESS.json',dict(timestamp=now(),variant=variant,seconds=time.monotonic()-began))
    # Diagnostics below use outcomes only for post hoc grouping, never for model input.
    groups['outcome_x_nwp']={f'{a}__{b}':ma&mb for a,ma in groups['outcome_from_origin'].items() for b,mb in groups['nwp_change_from_origin'].items()}
    groups['outcome_x_region']={f'{a}__{b}':ma&mb[:,None] for a,ma in groups['outcome_from_origin'].items() for b,mb in groups['region_minus_local'].items()}
    for group,bins in groups.items():
        coverage=np.zeros_like(y,dtype=int)
        for mask in bins.values():coverage+=mask[:,None] if mask.ndim==1 else mask
        assert np.all(coverage==1),group
    rows=[];paired=[];p0=predictions['N0'];e0=p0-y
    for group,bins in groups.items():
        for label,mask in bins.items():
            if mask.ndim==1:mask=np.broadcast_to(mask[:,None],y.shape)
            for horizon,lo,hi in [('all',0,24),('0_1h',0,4),('1_3h',4,12),('3_6h',12,24)]:
                valid=mask.copy();valid[:,:lo]=False;valid[:,hi:]=False
                if not valid.any():continue
                meta=dict(site=site,seed=seed,group=group,bin=label,horizon=horizon)
                for variant,pred in predictions.items():
                    values=score(pred-y,y,pred,valid);rows.append(dict(**meta,variant=variant,**values))
                    if variant!='N0':
                        update=pred-p0;eb=e0[valid];u=update[valid];ep=(pred-y)[valid]
                        cross=float(2*np.sum(eb*u));energy=float(np.sum(u*u));diff=float(np.sum(ep*ep-eb*eb))
                        assert abs(diff-cross-energy)<1e-8
                        paired.append(dict(**meta,variant=variant,n=int(valid.sum()),
                            delta_sq_sum=diff,cross_sum=cross,update_sq_sum=energy,
                            mean_update=float(u.mean()),mean_actual_change=float((y-o)[valid].mean()),
                            prediction_above1=int((pred[valid]>1).sum()),target_above1=int((y[valid]>1).sum())))
    atomic_json(out/'RESULT.json',dict(timestamp=now(),rows=rows,paired=paired,audit=audit,
        site=site,seed=seed,selection_year=2023,exploratory=True,predictions_saved=False,seconds=time.monotonic()-began))
    atomic_json(out/'SUCCESS.json',dict(timestamp=now(),seconds=time.monotonic()-began))
    print(site,seed,round(time.monotonic()-began,2),len(ix),flush=True)


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--site',type=int,required=True);a.add_argument('--seed',type=int,default=44);args=a.parse_args()
    main(args.site,args.seed)
