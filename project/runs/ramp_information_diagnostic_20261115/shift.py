"""Extra: (a) Pangu implied change from 1000/925 hPa wind; (b) time-shifted implied change vs W0 residual (phase-error signal). 2023, zero training."""
import numpy as np, json, sys
sys.path.insert(0,'/shared_data_ssd2/liying/wind_cluster_project_20261018/project/runs/daily_protocol_restart_20261114_v1/code')
from operational import interpolate_nodes
P='/shared_data_ssd2/liying/wind_cluster_project_20261018/project/runs'; D=f'{P}/daily_protocol_restart_20261114_v1/data'; FT=f'{P}/daily_site_finetune_20261114_v1'; A=f'{P}/daily_protocol_restart_20261114_v1/assets'
mu=np.load(f'{A}/nwp_mean.npy'); sd=np.load(f'{A}/nwp_std.npy')
CH={'10m':(1,2),'1000hPa':(10,12),'925hPa':(11,13)}
def ws(sp,qs):
    cy=np.load(f'{D}/{sp}/cycle.npy'); off=np.load(f'{D}/{sp}/offset.npy'); nodes=np.load(f'{D}/{sp}/nodes.npy',mmap_mode='r')
    out={k:np.zeros((len(cy),55,len(qs)),'f4') for k in CH}
    for s in range(55):
        n=np.asarray(nodes[cy,s])*sd+mu
        for k,(a,b) in CH.items():
            w=interpolate_nodes(n[...,[a,b]],off,qs); out[k][:,s]=np.hypot(w[...,0],w[...,1])
    return out
# query times (h): 0, and 4h target with shifts -2..+2 (needs offset+7 >= 6 -> max 6h)
qs=np.array([0.,2.,3.,4.,5.,6.])
W2=ws('train',np.array([0.,1.,2.,3.,4.,5.,6.])); W3=ws('val',qs)
y2=np.asarray(np.load(f'{D}/train/y.npy',mmap_mode='r')); y3=np.asarray(np.load(f'{D}/val/y.npy',mmap_mode='r')[:,:,15]); h3=np.asarray(np.load(f'{D}/val/h.npy',mmap_mode='r')[:,:,-1])
idx2=[3,7,11,15,19,23]  # train y leads 1..6h for curve fit
def curve(wtr,ptr):
    e=np.quantile(wtr,np.linspace(0,1,41)); b=np.clip(np.searchsorted(e,wtr)-1,0,39); m=np.array([ptr[b==i].mean() if (b==i).any() else np.nan for i in range(40)]); m=np.where(np.isnan(m),np.nanmean(ptr),m)
    return lambda w:m[np.clip(np.searchsorted(e,w)-1,0,39)]
R={}
for s in range(55):
    W0=np.mean([np.load(f'{FT}/jobs/W0_P1_FULL_s{s:02d}_r{x}/p0_prediction.npy')[:,0,15] for x in range(42,47)],0); res=y3[:,s]-W0; d=y3[:,s]-h3[:,s]; ev=np.abs(d)>.3
    for k in CH:
        f=curve(W2[k][:,s,1:].ravel(),y2[:,s][:,idx2].ravel())
        base=f(W3[k][:,s,0])
        for j,sh in zip(range(1,6),[-2,-1,0,1,2]):
            imp=f(W3[k][:,s,j])-base
            R.setdefault((k,sh,'res'),[]).append(np.corrcoef(res,imp)[0,1]); R.setdefault((k,sh,'resev'),[]).append(np.corrcoef(res[ev],imp[ev])[0,1])
            R.setdefault((k,sh,'dy'),[]).append(np.corrcoef(d,imp)[0,1])
for k in CH:
    print(k,'target 4h; implied change evaluated at 4h+shift')
    for sh in [-2,-1,0,1,2]: print(f'   shift {sh:+d}h: corr(implied Δ, actual Δ) {np.mean(R[(k,sh,"dy")]):.3f} | corr(W0 residual, implied Δ) all {np.mean(R[(k,sh,"res")]):+.3f}, on |Δ|>.3 {np.mean(R[(k,sh,"resev")]):+.3f}')
