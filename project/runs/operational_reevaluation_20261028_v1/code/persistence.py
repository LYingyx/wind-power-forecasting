from settings import *
import numpy as np
from fit_model import metrics

def run():
 require_test_freeze();rows=[]
 for split,year in [('val',2023),('test',2024)]:
  h=np.load(CACHE/split/'station_history.npy',mmap_mode='r');y=np.load(CACHE/split/'station_future.npy',mmap_mode='r')
  for site in range(56):
   pred=np.broadcast_to(h[:,site,-1:],y[:,site].shape);high=h[:,site,-1]>=.7
   rows.append(dict(site=site,year=year,metrics=metrics(pred,y[:,site]),high_power_rmse=metrics(pred[high],y[high,site])['rmse'] if high.any() else None))
 atomic_json(RUN/'analysis/PERSISTENCE.json',dict(timestamp=now(),rows=rows,data_fingerprint=FINGERPRINT))
if __name__=='__main__':run()
