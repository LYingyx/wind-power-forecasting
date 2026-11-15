from settings import *
import numpy as np

def masks(d):
 h=d.actual_history.cpu().numpy();y=d.y.cpu().numpy();p=h[:,-1:];shape=y.shape
 age=(d.origin-d.issue_unix)/3600
 month=(d.origin+28800).astype('datetime64[s]').astype('datetime64[M]').astype(int)%12+1
 return dict(current_power={k:np.broadcast_to(v,shape) for k,v in dict(low=p<.1,midlow=(p>=.1)&(p<.3),midhigh=(p>=.3)&(p<.7),high=p>=.7).items()},
    forecast_age={f'{a}_{b}h':np.broadcast_to(((age>=a)&(age<b))[:,None],shape) for a,b in [(12,16),(16,20),(20,24)]},
    season={k:np.broadcast_to(np.isin(month,ms)[:,None],shape) for k,ms in [('DJF',[12,1,2]),('MAM',[3,4,5]),('JJA',[6,7,8]),('SON',[9,10,11])]},
    future_ramp={k:v for k,v in dict(down=(y-p)<-.2,flat=abs(y-p)<=.2,up=(y-p)>.2).items()},
    high_power_outcome={k:v for k,v in dict(not_high=np.broadcast_to(p<.7,shape),drop=(p>=.7)&(y-p<-.2),other=(p>=.7)&(y-p>=-.2)).items()})

def stratified(pred,d):
 e=np.asarray(pred,'f8')-d.y.cpu().numpy();rows=[]
 for group,parts in masks(d).items():
  assert np.all(sum(parts.values())==1)
  for hz,a,b in [('all',0,24),('0_1h',0,4),('1_3h',4,12),('3_6h',12,24)]:
   for label,mask in parts.items():
    z=e[:,a:b][mask[:,a:b]]
    if len(z):rows.append(dict(group=group,bin=label,horizon=hz,n=int(z.size),mae=float(abs(z).mean()),rmse=float(np.sqrt((z*z).mean())),bias=float(z.mean()),sq_sum=float((z*z).sum()),abs_sum=float(abs(z).sum())))
 return rows

def daily_sums(pred,d):
 e=np.asarray(pred,'f8')-d.y.cpu().numpy();days=(d.origin+28800)//86400
 return {str(day):dict(n=int(e[days==day].size),abs_sum=float(abs(e[days==day]).sum()),sq_sum=float((e[days==day]**2).sum())) for day in np.unique(days)}
