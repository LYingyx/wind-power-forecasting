"""Does the previous cycle (age 24-36h) add info at 4h lead beyond state + latest-cycle curve? 2022 fit, 2023 eval."""
import numpy as np, json
exec(open(__import__('os').path.join(__import__('os').path.dirname(__file__),'persist.py')).read().split("out=[]")[0])
S='/shared_data_ssd2/liying/wind_cluster_project_20261018/project/runs/privileged_wind_age_pilot_20261101_v1/data'
keep=np.array(json.load(open(f'{R}/assets/COHORT.json'))['original_indices'])
def old(sp):
    ph=np.load(f'{R}/data/{sp}/phase.npy'); cy=np.load(f'{R}/data/{sp}/cycle.npy'); k=np.flatnonzero(ph==0)
    n=np.load(f'{S}/age12/{sp}/future.npy',mmap_mode='r'); a0=np.load(f'{S}/age0/{sp}/future.npy',mmap_mode='r')
    w=np.asarray(n[cy[k]][:,keep,4,:])*sd+mu; w0=np.asarray(a0[cy[k]][:,keep,4,:])*sd+mu
    return np.hypot(w[...,10],w[...,12]), np.hypot(w0[...,10],w0[...,12])
o2,chk2=old('train'); o3,chk3=old('val')
print('alignment check (age0 source vs shared cache), max abs diff:', float(np.nanmax(np.abs(chk3-w43))))
gain=[]
for s in range(F):
    c2=curve(w42[:,s],s,p42,w42); c3=curve(w43[:,s],s,p42,w42); q2=curve(o2[:,s],s,p42,w42); q3=curve(o3[:,s],s,p42,w42)
    def fit(cols2,cols3):
        X2=np.stack(cols2,1); X3=np.stack(cols3,1); ok=np.isfinite(X2).all(1)&np.isfinite(p42[:,s])
        b=np.linalg.lstsq(X2[ok],p42[ok,s],rcond=None)[0]; r=p43[:,s]-X3@b; r=r[np.isfinite(r)]; return np.mean(np.abs(r)),np.sqrt(np.mean(r**2))
    one2=np.ones_like(c2); one3=np.ones_like(c3)
    base=fit([one2,p2[:,s],c2,p2[:,s]*c2],[one3,p3[:,s],c3,p3[:,s]*c3])
    lag=fit([one2,p2[:,s],c2,p2[:,s]*c2,q2,np.abs(q2-c2)],[one3,p3[:,s],c3,p3[:,s]*c3,q3,np.abs(q3-c3)])
    gain.append((lag[0]/base[0]-1,lag[1]/base[1]-1))
g=np.array(gain); print(f'+previous cycle (level + |revision|): MAE {100*g[:,0].mean():+.2f}% RMSE {100*g[:,1].mean():+.2f}%  improved {int((g[:,0]<0).sum())}/{F}, {int((g[:,1]<0).sum())}/{F}')
