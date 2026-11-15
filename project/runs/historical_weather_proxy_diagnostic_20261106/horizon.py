"""Conditional 30d rolling bias on 'state + forecast curve' base, by horizon (1/2/4/6 h). 2022 fit, 2023 eval."""
import numpy as np
R='/shared_data_ssd2/liying/wind_cluster_project_20261018/project/runs/shared_personalized_20261104_v1'
mu=np.load(f'{R}/assets/nwp_mean.npy'); sd=np.load(f'{R}/assets/nwp_std.npy')
def load(sp,node,L):
    o=np.load(f'{R}/data/{sp}/origin.npy'); ph=np.load(f'{R}/data/{sp}/phase.npy'); cy=np.load(f'{R}/data/{sp}/cycle.npy')
    h=np.load(f'{R}/data/{sp}/h.npy',mmap_mode='r'); y=np.load(f'{R}/data/{sp}/y.npy',mmap_mode='r'); n=np.load(f'{R}/data/{sp}/nodes.npy',mmap_mode='r')
    k=np.flatnonzero(ph==0); w=np.asarray(n[cy[k],:,node,:])*sd+mu
    return np.asarray(h[k,:,-1]), np.hypot(w[...,10],w[...,12]), np.asarray(y[k,:,L]), (o[k]+28800)//86400
for hz,node,L in [(1,1,3),(2,2,7),(4,4,15),(6,6,23)]:
    p2,w2,y2,d2=load('train',node,L); p3,w3,y3,d3=load('val',node,L); F=p2.shape[1]
    edges=np.quantile(w2,np.linspace(0,1,31),axis=0); out=[]
    for s in range(F):
        e=edges[:,s]; b=np.clip(np.searchsorted(e,w2[:,s])-1,0,29)
        m=np.array([np.nanmean(y2[b==i,s]) if (b==i).any() else np.nan for i in range(30)]); m=np.where(np.isnan(m),np.nanmean(y2[:,s]),m)
        c2=m[b]; c3=m[np.clip(np.searchsorted(e,w3[:,s])-1,0,29)]
        X2=np.stack([np.ones_like(c2),p2[:,s],c2,p2[:,s]*c2],1); X3=np.stack([np.ones_like(c3),p3[:,s],c3,p3[:,s]*c3],1)
        ok=np.isfinite(X2).all(1)&np.isfinite(y2[:,s]); beta=np.linalg.lstsq(X2[ok],y2[ok,s],rcond=None)[0]
        r=y3[:,s]-X3@beta; okv=np.isfinite(r); q=np.quantile(c2,[1/3,2/3]); g=np.digitize(c3,q); corr=np.zeros_like(r); days=np.unique(d3)
        for d in days:
            prev=(d3<d)&(d3>=d-30)&okv
            for gg in range(3):
                mm=prev&(g==gg); corr[(d3==d)&(g==gg)]=np.nanmean(r[mm]) if mm.sum()>24 else 0.
        e0=r[okv]; e1=(r-corr)[okv]; out.append((np.mean(np.abs(e1))/np.mean(np.abs(e0))-1, np.sqrt(np.mean(e1**2))/np.sqrt(np.mean(e0**2))-1, np.mean(np.abs(e0))))
    v=np.array(out); print(f'{hz}h: base MAE {v[:,2].mean():.4f}; 30d conditional bias corr MAE {100*v[:,0].mean():+.2f}% RMSE {100*v[:,1].mean():+.2f}%  improved {int((v[:,0]<0).sum())}/{F}, {int((v[:,1]<0).sum())}/{F}')
