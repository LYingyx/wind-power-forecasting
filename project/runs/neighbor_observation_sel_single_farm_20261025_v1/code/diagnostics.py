"""Origin-available strata; train-only boundaries; no saved prediction arrays."""
from settings import *
import numpy as np,torch
from physical_data import interpolate

def descriptors(data):
    physical=data.physical.detach().cpu().numpy();speed=np.hypot(physical[...,1],physical[...,2])
    batch=data.batch(slice(None));nodes=batch['allnodes'];times=batch['all_times']
    mean=torch.as_tensor(np.load(CACHE/'static/nwp_mean.npy'),device=nodes.device)
    sd=torch.as_tensor(np.load(CACHE/'static/nwp_std.npy'),device=nodes.device)
    earlier=interpolate(nodes,times,torch.arange(1,25,device=nodes.device)/4-1)*sd+mean
    previous=np.hypot(earlier[...,1].cpu().numpy(),earlier[...,2].cpu().numpy())
    history=data.h.detach().cpu().numpy()
    return dict(nwp_speed=speed,stability_proxy=data.diagnostic['stability'].numpy()[...,2],
        forecast_wind_change_1h=speed-previous,current_power=np.broadcast_to(history[:,-1:],speed.shape),
        history_power_change_1h=np.broadcast_to((history[:,-1]-history[:,-5])[:,None],speed.shape))

def build_bins():
    from physical_data import FarmData
    torch.set_num_threads(1);bins={}
    for site in range(56):
        data=FarmData('train',site,'BASE');d=descriptors(data)
        bins[str(site)]={name:np.unique(np.quantile(d[name],[.25,.5,.75])).tolist() for name in ['nwp_speed','stability_proxy']}
        bins[str(site)].update({name:float(np.quantile(abs(d[name]),.75)) for name in ['forecast_wind_change_1h','history_power_change_1h']})
        print('train-only bins',site,flush=True)
        del data,d
    atomic_json(RUN/'analysis/train_only_bins.json',dict(timestamp=now(),source_split='train_2022',bins=bins,wind_interpretation='quartiles, not cut-in/rated turbine thresholds'))

def masks(data):
    edges=json.loads((RUN/'analysis/train_only_bins.json').read_text())['bins'][str(data.site)];d=descriptors(data);out={}
    for group in ['nwp_speed','stability_proxy']:
        ix=np.searchsorted(edges[group],d[group],side='right')
        out[group]={f'q{i+1}':ix==i for i in range(len(edges[group])+1)}
    for group in ['forecast_wind_change_1h','history_power_change_1h']:
        v=d[group];cut=edges[group]
        out[group]=dict(down=v < -cut,calm=abs(v)<=cut,up=v > cut)
    power=d['current_power'];out['current_power']=dict(low=power<.1,midlow=(power>=.1)&(power<.3),midhigh=(power>=.3)&(power<.7),high=power>=.7)
    unix=data.origin[:,None]+np.arange(1,25)[None]*900+8*3600
    month=unix.astype('datetime64[s]').astype('datetime64[M]').astype(int)%12+1
    out['season']={name:np.isin(month,months) for name,months in [('DJF',[12,1,2]),('MAM',[3,4,5]),('JJA',[6,7,8]),('SON',[9,10,11])]}
    # Outcome strata only, never used by the network or the neighbor selection.
    y=data.y.detach().cpu().numpy();h=data.h.detach().cpu().numpy()
    delta=y-np.concatenate((h[:,-4:],y[:,:20]),axis=1);cut=edges['outcome_power_change_1h']
    out['outcome_power_change_1h']=dict(down=delta < -cut,calm=abs(delta)<=cut,up=delta > cut)
    with np.load(ASSETS/'neighbor_tables.npz') as meta:near=bool(meta['near50'][data.site])
    out['neighbor_within_50km']={'yes' if near else 'no':np.ones_like(y,dtype=bool)}
    for group,parts in out.items():assert np.all(sum(parts.values())==1),group
    return out

def stratified(prediction,data):
    error=np.asarray(prediction,dtype='f8')-data.y.detach().cpu().numpy();rows=[]
    for group,parts in masks(data).items():
        for horizon,a,b in [('all',0,24),('0_1h',0,4),('1_3h',4,12),('3_6h',12,24)]:
            total_sq=0.;total_n=0
            for label,mask in parts.items():
                e=error[:,a:b][mask[:,a:b]];n=int(e.size)
                if not n:continue
                sq=float(np.square(e).sum());total_sq+=sq;total_n+=n
                rows.append(dict(group=group,bin=label,horizon=horizon,n=n,mae=float(abs(e).mean()),rmse=float(np.sqrt(sq/n)),bias=float(e.mean()),sq_sum=sq,abs_sum=float(abs(e).sum())))
            assert total_n==error[:,a:b].size and np.isclose(total_sq,np.square(error[:,a:b]).sum(),rtol=1e-12,atol=1e-9)
    return rows

if __name__=='__main__':build_bins()
