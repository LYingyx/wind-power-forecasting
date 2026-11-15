"""Existing prediction diagnostics; no training, no raw prediction copies, no seed means."""
from settings import *
from physical_data import FarmData
from train_physics import metrics
import numpy as np,torch,csv,gc


def run():
    torch.set_num_threads(1)
    out=RUN/'analysis';out.mkdir(exist_ok=True)
    actual=[];paired=[];bins={}
    sid=np.load(CACHE/'static/station_id.npy')
    for site in range(56):
        tr=FarmData('train',site,'BASE');va=FarmData('val',site,'BASE',stats=tr.stats)
        def descriptors(d):
            st=d.diagnostic['stability'].numpy();dy=d.diagnostic['dynamics'].numpy()
            return {'density_proxy':d.density.numpy()[...,0], 'bulk_ri_proxy':st[...,2],
                'theta_gradient_proxy':st[...,0],'pressure_tendency_hpa_h':dy[...,5],
                'temperature_advection_k_h':dy[...,4],'wind_tendency_m_s_h':dy[...,6],
                'weather_speed_change':np.hypot(d.physical.numpy()[...,1],d.physical.numpy()[...,2])-np.hypot(d.reference_physical.numpy()[:,1],d.reference_physical.numpy()[:,2])[:,None],
                'plateau_duration_fraction':d.diagnostic['state'].numpy()[...,6]}
        dt,dv=descriptors(tr),descriptors(va);masks={};bins[str(site)]={}
        for key,v in dv.items():
            q=np.unique(np.quantile(dt[key],[.25,.5,.75]));bins[str(site)][key]=q.tolist()
            ix=np.searchsorted(q,v,side='right')
            for i in range(len(q)+1):masks[(key,f'bin_{i}')]=ix==i
        duration=va.diagnostic['state'].numpy()[...,6]
        is_plateau=duration>=4/23
        low=va.h[:,-1].numpy()[:,None]<=.05
        masks[('history_plateau_state','not_sustained')]=~is_plateau
        masks[('history_plateau_state','sustained_low_power')]=is_plateau & low
        masks[('history_plateau_state','sustained_nonlow_power')]=is_plateau & ~low
        clock=(va.origin[:,None]+np.arange(1,25)[None]*900+8*3600)
        hours=clock//3600%24
        for i in range(4):masks[('local_valid_hour',f'{i*6:02d}_{(i+1)*6:02d}')]=(hours>=i*6)&(hours<(i+1)*6)
        for label,low,high in [('low',-np.inf,.1),('midlow',.1,.3),('midhigh',.3,.7),('high',.7,np.inf)]:
            masks[('current_power',label)]=np.broadcast_to(((va.h[:,-1].numpy()>=low)&(va.h[:,-1].numpy()<high))[:,None],(va.n,24))
        masks[('all','all')]=np.ones((va.n,24),bool)
        truth=va.y.numpy().astype('f8');del tr,dt;gc.collect()
        for seed in SEEDS:
            paths={'W0':OLD/f'jobs/ablation_NEUTRAL_LAST_s{site:02d}_r{seed}/val_prediction.npy',
                   'TF':OLD/f'jobs/formal_TRANSFORMER_FC_s{site:02d}_r{seed}/val_prediction.npy'}
            errors={name:np.asarray(np.load(path,mmap_mode='r'),dtype='f8')-truth for name,path in paths.items()}
            assert all(e.shape==truth.shape for e in errors.values())
            for group,label in masks:
                mask=masks[(group,label)]
                for horizon,a,b in [('all',0,24),('0_1h',0,4),('1_3h',4,12),('3_6h',12,24)]:
                    sel=mask[:,a:b];n=int(sel.sum())
                    if n==0:continue
                    vals={name:e[:,a:b][sel] for name,e in errors.items()}
                    common=dict(site=site,station_id=str(sid[site]),seed=seed,year=2023,group=group,bin=label,horizon=horizon,n=n)
                    for name,e in vals.items():
                        total_sq=float(np.square(errors[name][:,a:b]).sum())
                        actual.append(dict(**common,model=name,mae=float(abs(e).mean()),rmse=float(np.sqrt(np.square(e).mean())),bias=float(e.mean()),sq_error_sum=float(np.square(e).sum()),squared_error_fraction=float(np.square(e).sum()/total_sq),source=str(paths[name])))
                    ew,et=vals['W0'],vals['TF']
                    paired.append(dict(**common,mae_difference=float(abs(ew).mean()-abs(et).mean()),rmse_difference=float(np.sqrt(np.square(ew).mean())-np.sqrt(np.square(et).mean())),squared_error_difference=float((ew*ew-et*et).sum()),mse_difference_contribution=float((ew*ew-et*et).sum()/errors['W0'][:,a:b].size)))
        print('diagnosed site',site,flush=True)
        del va,dv,errors,truth,masks;gc.collect()
    for name,rows in [('existing_actual_by_farm_seed.csv',actual),('existing_paired_by_farm_seed.csv',paired)]:
        with (out/name).open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    atomic_json(out/'train_only_bin_edges.json',bins)
    # Every diagnostic partition must reconstruct all-sample squared error.
    import pandas as pd
    df=pd.DataFrame(paired);keys=['site','seed','horizon'];tot=df[df.group=='all'].set_index(keys).squared_error_difference
    maximum=0.
    for group,g in df[df.group!='all'].groupby('group'):
        delta=g.groupby(keys).squared_error_difference.sum()-tot
        maximum=max(maximum,float(delta.abs().max()))
    assert maximum<1e-7,maximum
    atomic_json(out/'DIAGNOSTICS_COMPLETE.json',dict(timestamp=now(),actual_rows=len(actual),paired_rows=len(paired),n_farms=56,seeds=SEEDS,year=2023,partition_max_abs_error=maximum,no_training=True,no_prediction_copies=True))

if __name__=='__main__':run()
