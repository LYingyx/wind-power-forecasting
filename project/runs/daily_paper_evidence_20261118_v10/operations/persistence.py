"""Zero-training persistence sanity baseline, identical cohort and metric definition."""
from pathlib import Path
import sys
RUN=Path(__file__).resolve().parents[1];sys.path.insert(0,str(RUN/'code'))
from settings import atomic_json,now
import numpy as np

def main():
 for split,year in [('val',2023),('test',2024)]:
  root=RUN/'data'/split;h=np.load(root/'h.npy',mmap_mode='r');y=np.load(root/'y.npy',mmap_mode='r')
  ab=np.zeros((55,24));sq=np.zeros_like(ab);hi=np.zeros_like(ab);count=np.zeros(55);hc=np.zeros(55)
  for start in range(0,len(h),256):
   current=np.array(h[start:start+256,:,-1:],dtype='float64');truth=np.array(y[start:start+256],dtype='float64')
   err=current-truth;ab+=abs(err).sum(0);sq+=(err**2).sum(0);count+=len(err)
   mask=current[...,0]>=.75;hi+=((err**2)*mask[...,None]).sum(0);hc+=mask.sum(0)
  lm=ab/count[:,None];ms=sq/count[:,None];fm=lm.mean(1);fr=np.sqrt(ms.mean(1));hr=np.sqrt(hi.sum(1)/np.maximum(hc*24,1))
  metrics=dict(mae=float(fm.mean()),rmse=float(fr.mean()),high_power_rmse=float(hr[hc>0].mean()),
   per_farm_mae=fm.tolist(),per_farm_rmse=fr.tolist(),high_power_per_farm_rmse=hr.tolist(),high_power_counts=hc.tolist(),
   lead_mae=lm.tolist(),lead_rmse=np.sqrt(ms).tolist(),counts=count.tolist(),
   horizons={label:dict(mae=float(lm[:,a:b].mean()),rmse=float(np.sqrt(ms[:,a:b].mean(1)).mean())) for label,a,b in [('0_1h',0,4),('1_3h',4,12),('3_6h',12,24)]})
  atomic_json(RUN/'reference_metrics'/f'PERSISTENCE_{year}.json',dict(timestamp=now(),group='PERSISTENCE',seed=0,
   evaluation_year=year,metrics=metrics,deterministic=True,trained=False,description='all 24 forecasts equal the last observed power; no fitted parameters; seed0 is an index only'))
  print(year,metrics['mae'],metrics['rmse'],flush=True)
if __name__=='__main__':main()
