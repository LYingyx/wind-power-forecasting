"""Same proxy, but base model uses current power + forecast curve (2022 linear fit); tests whether rolling bias persists after state."""
import numpy as np
exec(open(__import__('os').path.join(__import__('os').path.dirname(__file__),'persist.py')).read().split("out=[]")[0])
out=[]
for s in range(F):
    c2=curve(w42[:,s],s,p42,w42); c3=curve(w43[:,s],s,p42,w42)
    X2=np.stack([np.ones_like(c2),p2[:,s],c2,p2[:,s]*c2],1); X3=np.stack([np.ones_like(c3),p3[:,s],c3,p3[:,s]*c3],1)
    ok2=np.isfinite(X2).all(1)&np.isfinite(p42[:,s]); beta=np.linalg.lstsq(X2[ok2],p42[ok2,s],rcond=None)[0]
    r=p43[:,s]-X3@beta; ok=np.isfinite(r); days=np.unique(d3); res={}
    for D in (7,14,30,60):
        corr=np.zeros_like(r)
        for d in days:
            prev=(d3<d)&(d3>=d-D)&ok
            corr[d3==d]=np.nanmean(r[prev]) if prev.sum()>24 else 0.
        e0=r[ok]; e1=(r-corr)[ok]
        res[D]=(np.mean(np.abs(e1))/np.mean(np.abs(e0))-1, np.sqrt(np.mean(e1**2))/np.sqrt(np.mean(e0**2))-1)
    # conditional (by forecast-curve tercile) rolling bias, 30d
    q=np.quantile(c2,[1/3,2/3]); g3=np.digitize(c3,q); corr=np.zeros_like(r)
    for d in days:
        prev=(d3<d)&(d3>=d-30)&ok
        for g in range(3):
            m=prev&(g3==g); corr[(d3==d)&(g3==g)]=np.nanmean(r[m]) if m.sum()>24 else 0.
    e1=(r-corr)[ok]; res['30c']=(np.mean(np.abs(e1))/np.mean(np.abs(r[ok]))-1, np.sqrt(np.mean(e1**2))/np.sqrt(np.mean(r[ok]**2))-1)
    out.append(res)
for D in (7,14,30,60,'30c'):
    v=np.array([o[D] for o in out]); print(f'state+curve base, 4h lead, rolling {D} bias corr: MAE {100*v[:,0].mean():+.2f}% RMSE {100*v[:,1].mean():+.2f}%  improved MAE {int((v[:,0]<0).sum())}/{F} RMSE {int((v[:,1]<0).sum())}/{F}')
