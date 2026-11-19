"""Descriptive pooled diagnostics; checkpoint selection still uses farm-macro MAE."""
import collections
import numpy as np


class Diagnostics:
    def __init__(self):
        self.bins = collections.defaultdict(lambda: np.zeros(5, dtype='f8'))
        self.diversity = np.zeros(3, dtype='f8')
        self.median_error = np.zeros(3, dtype='f8')

    def update(self, b, y, p, samples=None):
        y=y.cpu().numpy().astype('f8');p=p.cpu().numpy().astype('f8')
        h=b['h'][:, -1:].cpu().numpy().astype('f8');err=p-y;dy=y-h;dp=p-h
        phase=(b['phase'].cpu().numpy()[:, None]+np.arange(1,25)[None])%4
        masks={f'clock_{int(q)*15:02d}':phase==q for q in range(4)}
        masks.update(high_up=(h>=.75)&(dy>0), high_down=(h>=.75)&(dy<0))
        for name,a,z in [('0_1h',0,4),('1_3h',4,12),('3_6h',12,24)]:
            mask=np.zeros_like(dy,dtype=bool);mask[:,a:z]=abs(dy[:,a:z])>.3
            masks['large_change_'+name]=mask
        for name,mask in masks.items():
            self.bins[name]+=np.array([mask.sum(),abs(err[mask]).sum(),(err[mask]**2).sum(),
                                      (dp*np.sign(dy))[mask].sum(),abs(dy[mask]).sum()])
        if samples is not None:
            draws=samples.cpu().numpy().astype('f8');std=draws.std(1)
            self.diversity+=np.array([std.size,std.sum(),(std>1e-6).sum()])
            median=np.median(draws,axis=1)-y
            self.median_error+=np.array([median.size,abs(median).sum(),(median**2).sum()])

    def result(self):
        bins={}
        for name,(n,ab,sq,cap,den) in self.bins.items():
            bins[name]=dict(count=int(n),mae=ab/n if n else None,mse=sq/n if n else None,
                            capture_ratio=cap/den if den else None,
                            abs_error_sum=ab,squared_error_sum=sq,capture_numerator=cap,capture_denominator=den)
        result=dict(aggregation='pooled diagnostic cells; not farm-macro selection metrics',bins=bins)
        n,sd,positive=self.diversity
        if n:
            count,ab,sq=self.median_error
            result.update(sample_std_mean=sd/n,nonzero_sample_std_fraction=positive/n,
                          sample_median_mae=ab/count,sample_median_rmse=float(np.sqrt(sq/count)))
        return result
