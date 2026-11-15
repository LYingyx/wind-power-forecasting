"""Frozen strictly causal rolling correction; no fitting on evaluation labels."""
import numpy as np

GROUPS=[('0_1h',0,4),('1_3h',4,12),('3_6h',12,24)]
BINS=np.array([.1,.3,.7])

def correction(origin,pred,y,days=30,exclude_hours=0,binned=True):
    origin=np.asarray(origin,dtype=np.int64)
    assert np.all(np.diff(origin)>0) and pred.shape==y.shape==(len(origin),24)
    assert np.isfinite(pred).all() and np.isfinite(y).all()
    bins=np.digitize(pred,BINS);residual=y.astype('f8')-pred
    out=np.zeros(pred.shape,'f8');fallback=np.zeros(3,'i8')
    lo=np.searchsorted(origin,origin-days*86400,side='left')
    for gi,(_,a,b) in enumerate(GROUPS):
        sums=np.zeros((len(origin),4));counts=np.zeros((len(origin),4),'i8')
        for k in range(a,b):
            # Contributions are from past ORIGINS within D days whose TARGET
            # has been observed by t-delay. This remains correct across gaps.
            hi=np.searchsorted(origin,origin-900-int(exclude_hours*3600)-(k+1)*900,side='right')
            hi=np.maximum(hi,lo)
            for g in range(4):
                ok=bins[:,k]==g
                cs=np.r_[0.,np.cumsum(np.where(ok,residual[:,k],0.))]
                cn=np.r_[0,np.cumsum(ok)]
                sums[:,g]+=cs[hi]-cs[lo];counts[:,g]+=cn[hi]-cn[lo]
        su=sums.sum(1);ct=counts.sum(1)
        uniform=np.divide(su,ct,out=np.zeros_like(su),where=ct>=24)
        for k in range(a,b):
            c=counts[np.arange(len(origin)),bins[:,k]]
            v=sums[np.arange(len(origin)),bins[:,k]]
            out[:,k]=np.where(c>=24,v/np.maximum(1,c),uniform) if binned else uniform
            fallback[0]+=np.sum(c<24);fallback[1]+=np.sum((c<24)&(ct>=24));fallback[2]+=np.sum(ct<24)
    return out,dict(bin_under24=int(fallback[0]),uniform_fallback=int(fallback[1]),zero_fallback=int(fallback[2]),total=int(pred.size))

def metrics(pred,y,high):
    err=pred.astype('f8')-y
    result=dict(mae=float(abs(err).mean()),rmse=float(np.sqrt((err**2).mean())),
                lead_mae=abs(err).mean(0).tolist(),lead_mse=(err**2).mean(0).tolist(),
                high_power_rmse=float(np.sqrt((err[high]**2).mean())) if np.any(high) else None,
                high_power_count=int(np.sum(high)),count=len(y))
    result['horizons']={name:dict(mae=float(abs(err[:,a:b]).mean()),rmse=float(np.sqrt((err[:,a:b]**2).mean()))) for name,a,b in GROUPS}
    return result

def analyze_predictions(origin,pred,y,high):
    result={'raw':metrics(pred,y,high)}
    for name,days,gap,binned in [('bin30',30,0,True),('bin7',7,0,True),('bin14',14,0,True),('bin30_exclude24h',30,24,True),('uniform30',30,0,False)]:
        c,f=correction(origin,pred,y,days,gap,binned)
        result[name]=metrics(pred+c,y,high)
        result[name]['fallback']=f
    return result

def self_test():
    # Explicit brute-force test includes exact time/bin boundaries, a large gap,
    # overlapping horizons, the 24-count fallback and the 24-hour exclusion.
    rng=np.random.default_rng(4)
    origin=np.r_[np.arange(45)*900,np.arange(100,135)*900,np.arange(300,340)*900]
    pred=rng.choice([-.1,.1,.2,.3,.6,.7,.9],size=(len(origin),24));y=rng.normal(.4,.2,pred.shape)
    for days,gap in [(1,0),(1,24),(30,0),(30,24)]:
        actual,_=correction(origin,pred,y,days,gap)
        expected=np.zeros_like(actual);bins=np.digitize(pred,BINS)
        for i,t in enumerate(origin):
            for _,a,b in GROUPS:
                target=origin[:,None]+np.arange(a+1,b+1)[None]*900
                ok=(origin[:,None]>=t-days*86400)&(target<=t-900-gap*3600)
                for k in range(a,b):
                    chosen=ok&(bins[:,a:b]==bins[i,k])
                    if chosen.sum()<24:chosen=ok
                    if chosen.sum()>=24:expected[i,k]=(y[:,a:b]-pred[:,a:b])[chosen].mean()
        assert np.allclose(actual,expected,atol=1e-12)
    p2=pred.copy();y2=y.copy();p2[50:]+=100;y2[50:]-=100
    a,_=correction(origin,pred,y);b,_=correction(origin,p2,y2)
    assert np.array_equal(a[:50],b[:50]),'future labels/predictions affected earlier outputs'
    return True
