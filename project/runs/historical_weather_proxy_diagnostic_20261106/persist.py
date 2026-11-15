"""Proxy diagnostic: multi-day persistence of weather-only mapping residuals (2022 fit, 2023 eval). CPU only."""
import numpy as np, sys
R='/shared_data_ssd2/liying/wind_cluster_project_20261018/project/runs/shared_personalized_20261104_v1'
mu=np.load(f'{R}/assets/nwp_mean.npy'); sd=np.load(f'{R}/assets/nwp_std.npy')
def load(sp):
    o=np.load(f'{R}/data/{sp}/origin.npy'); ph=np.load(f'{R}/data/{sp}/phase.npy'); cy=np.load(f'{R}/data/{sp}/cycle.npy')
    h=np.load(f'{R}/data/{sp}/h.npy',mmap_mode='r'); y=np.load(f'{R}/data/{sp}/y.npy',mmap_mode='r'); n=np.load(f'{R}/data/{sp}/nodes.npy',mmap_mode='r')
    k=np.flatnonzero(ph==0)
    p=np.asarray(h[k,:,-1])                      # power at base hour (CF)
    w=np.asarray(n[cy[k],:,0,:])*sd+mu           # as-used forecast valid at base hour
    # also 4h-ahead pair: forecast node 4 (valid b+4h) vs observed y at lead 16 (origin+4h)
    w4=np.asarray(n[cy[k],:,4,:])*sd+mu; p4=np.asarray(y[k,:,15])
    ws=np.hypot(w[...,10],w[...,12]); ws4=np.hypot(w4[...,10],w4[...,12])   # 1000 hPa wind speed
    day=((o[k]+28800)//86400)
    return p,ws,p4,ws4,day
p2,w2,p42,w42,d2=load('train'); p3,w3,p43,w43,d3=load('val')
F=p2.shape[1]; edges=np.quantile(w2,np.linspace(0,1,31),axis=0)
def curve(ws,s,P,W):
    e=edges[:,s]; b=np.clip(np.searchsorted(e,W[:,s])-1,0,29); m=np.array([np.nanmean(P[b==i,s]) if (b==i).any() else np.nan for i in range(30)])
    m=np.where(np.isnan(m),np.nanmean(P[:,s]),m); bb=np.clip(np.searchsorted(e,ws)-1,0,29); return m[bb]
out=[]
for s in range(F):
    for tag,(P2,W2,P3,W3) in {'lead0':(p2,w2,p3,w3),'lead4h':(p42,w42,p43,w43)}.items():
        f3=curve(W3[:,s],s,P2,W2); r=P3[:,s]-f3; ok=np.isfinite(r)
        days=np.unique(d3); dm=np.array([np.nanmean(r[(d3==d)&ok]) for d in days])
        ac=[np.corrcoef(dm[:-L],dm[L:])[0,1] for L in (1,2,7)]
        res={}
        for D in (3,7,14,30):
            corr=np.zeros_like(r)
            for i,d in enumerate(days):
                m=(days<d)&(days>=d-D)
                prev=np.isin(d3,days[m])&ok
                corr[d3==d]=np.nanmean(r[prev]) if prev.sum()>24 else 0.
            e0=r[ok]; e1=(r-corr)[ok]
            res[D]=(np.mean(np.abs(e1))/np.mean(np.abs(e0))-1, np.sqrt(np.mean(e1**2))/np.sqrt(np.mean(e0**2))-1)
        out.append((s,tag,ac,res))
for tag in ['lead0','lead4h']:
    rows=[o for o in out if o[1]==tag]
    ac=np.array([o[2] for o in rows]); print(tag,'daily-mean residual autocorr lag1/2/7 (median over farms):',np.round(np.nanmedian(ac,0),3))
    for D in (3,7,14,30):
        v=np.array([o[3][D] for o in rows]); print(f'  rolling {D}d bias corr: MAE {100*v[:,0].mean():+.2f}% RMSE {100*v[:,1].mean():+.2f}%  farms improved MAE {int((v[:,0]<0).sum())}/{F} RMSE {int((v[:,1]<0).sum())}/{F}')
