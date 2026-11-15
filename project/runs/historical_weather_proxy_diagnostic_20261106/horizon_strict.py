"""Strictly causal version of horizon.py: rolling bias uses only residuals whose TARGET time <= origin - 15 min
and origin >= t - D days. 2022 fit, 2023 eval, CPU only, no 2024."""
import numpy as np, os
R='/shared_data_ssd2/liying/wind_cluster_project_20261018/project/runs/shared_personalized_20261104_v1'
mu=np.load(f'{R}/assets/nwp_mean.npy'); sd=np.load(f'{R}/assets/nwp_std.npy')
def load(sp,node,L):
    o=np.load(f'{R}/data/{sp}/origin.npy'); ph=np.load(f'{R}/data/{sp}/phase.npy'); cy=np.load(f'{R}/data/{sp}/cycle.npy')
    h=np.load(f'{R}/data/{sp}/h.npy',mmap_mode='r'); y=np.load(f'{R}/data/{sp}/y.npy',mmap_mode='r'); n=np.load(f'{R}/data/{sp}/nodes.npy',mmap_mode='r')
    k=np.flatnonzero(ph==0); w=np.asarray(n[cy[k],:,node,:])*sd+mu
    return np.asarray(h[k,:,-1]), np.hypot(w[...,10],w[...,12]), np.asarray(y[k,:,L]), o[k]
def rolling(origin,target,r,valid,D,groups=None,ng=1):
    """mean residual over samples j with target_j <= t-900 and origin_j >= t-D*86400, per group of the evaluated sample."""
    corr=np.zeros(len(origin))
    g=np.zeros(len(origin),int) if groups is None else groups
    for gg in range(ng):
        m=valid&(g==gg); T=target[m]; O=origin[m]; v=r[m]
        order=np.argsort(T); T=T[order]; O=O[order]; v=v[order]
        cs=np.concatenate([[0],np.cumsum(v)]); cn=np.arange(len(T)+1)
        ev=np.flatnonzero(g==gg)
        hi=np.searchsorted(T,origin[ev]-900,side='right')
        lo=np.searchsorted(T,origin[ev]-D*86400+ (target[0]-origin[0]),side='left')  # target >= (t-D)+lead  <=> origin_j >= t-D
        cnt=cn[hi]-cn[lo]; s=cs[hi]-cs[lo]
        corr[ev]=np.where(cnt>24,s/np.maximum(cnt,1),0.)
    return corr
for hz,node,L in [(1,1,3),(2,2,7),(4,4,15),(6,6,23)]:
    p2,w2,y2,o2=load('train',node,L); p3,w3,y3,o3=load('val',node,L); F=p2.shape[1]; lead=(L+1)*900
    edges=np.quantile(w2,np.linspace(0,1,31),axis=0); out=[]
    for s in range(F):
        e=edges[:,s]; b=np.clip(np.searchsorted(e,w2[:,s])-1,0,29)
        m=np.array([np.nanmean(y2[b==i,s]) if (b==i).any() else np.nan for i in range(30)]); m=np.where(np.isnan(m),np.nanmean(y2[:,s]),m)
        c2=m[b]; c3=m[np.clip(np.searchsorted(e,w3[:,s])-1,0,29)]
        X2=np.stack([np.ones_like(c2),p2[:,s],c2,p2[:,s]*c2],1); X3=np.stack([np.ones_like(c3),p3[:,s],c3,p3[:,s]*c3],1)
        ok=np.isfinite(X2).all(1)&np.isfinite(y2[:,s]); beta=np.linalg.lstsq(X2[ok],y2[ok,s],rcond=None)[0]
        r=y3[:,s]-X3@beta; okv=np.isfinite(r); rr=np.where(okv,r,0.)
        q=np.quantile(c2,[1/3,2/3]); g=np.digitize(c3,q)
        res={}
        for name,D,grp,ng in [('uniform30',30,None,1),('cond30',30,g,3),('cond14',14,g,3),('cond60',60,g,3)]:
            corr=rolling(o3,o3+lead,rr,okv,D,grp,ng)
            e0=r[okv]; e1=(r-corr)[okv]
            res[name]=(np.mean(np.abs(e1))/np.mean(np.abs(e0))-1, np.sqrt(np.mean(e1**2))/np.sqrt(np.mean(e0**2))-1)
        out.append(res)
    for name in ['uniform30','cond14','cond30','cond60']:
        v=np.array([o[name] for o in out])
        print(f'{hz}h {name}: MAE {100*v[:,0].mean():+.2f}% RMSE {100*v[:,1].mean():+.2f}%  improved {int((v[:,0]<0).sum())}/{F}, {int((v[:,1]<0).sum())}/{F}')
