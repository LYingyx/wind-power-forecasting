from settings import *
import numpy as np

def main():
 # No model or weather needed; compute each farm's persistence directly.
 y=np.load(CACHE/'val/station_future.npy',mmap_mode='r');h=np.load(CACHE/'val/station_history.npy',mmap_mode='r');rows=[]
 for i in range(56):
  e=np.asarray(h[:,i,-1:,],dtype='f8')-y[:,i]
  def score(e):return dict(n=int(e.size),mae=float(abs(e).mean()),rmse=float(np.sqrt((e*e).mean())),bias=float(e.mean()))
  rows.append(dict(site=i,**score(e),horizons={name:score(e[:,a:b]) for name,a,b in [('0_1h',0,4),('1_3h',4,12),('3_6h',12,24)]}))
 atomic_json(RUN/'analysis/PERSISTENCE.json',dict(timestamp=now(),data_fingerprint=FINGERPRINT,year=2023,rows=rows,equal_farm={k:float(np.mean([r[k] for r in rows])) for k in ['mae','rmse']}))
if __name__=='__main__':main()
