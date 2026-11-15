"""If ERA5 up to the origin is available: does the forecast error at the origin (ERA5 - Pangu, 10 m wind at the base hour <= origin)
explain W0's remaining 0-6 h error? Own-farm and 55-farm regional mean. 2023, zero training."""
import numpy as np, json, sys
sys.path.insert(0,'/shared_data_ssd2/liying/wind_cluster_project_20261018/project/runs/daily_protocol_restart_20261114_v1/code')
P='/shared_data_ssd2/liying/wind_cluster_project_20261018/project/runs'; D=f'{P}/daily_protocol_restart_20261114_v1/data'; S=f'{P}/privileged_wind_age_pilot_20261101_v1/data'; FT=f'{P}/daily_site_finetune_20261114_v1'; A=f'{P}/daily_protocol_restart_20261114_v1/assets'
mu=np.load(f'{A}/nwp_mean.npy'); sd=np.load(f'{A}/nwp_std.npy'); emu=np.load(f'{S}/cache/static/nwp_mean.npy')[1:3]; esd=np.load(f'{S}/cache/static/nwp_std.npy')[1:3]
keep=np.array(json.load(open(f'{A}/COHORT.json'))['original_indices'])
cy=np.load(f'{D}/val/cycle.npy'); cy2=np.load(f'{S}/common/val/cycle.npy'); off=np.load(f'{D}/val/offset.npy')
nodes=np.load(f'{D}/val/nodes.npy',mmap_mode='r'); era=np.load(f'{S}/labels/val/future_uv.npy',mmap_mode='r')
y=np.asarray(np.load(f'{D}/val/y.npy',mmap_mode='r')); 
LEADS=[3,7,15,23]
# node 0 is valid at base hour (<= origin). ERA5 node j for j>0 is future -> only node 0 used as "available analysis".
pg=np.hypot(*(np.asarray(nodes[cy][:,:,0,[1,2]])*sd[[1,2]]+mu[[1,2]]).transpose(2,0,1))
er=np.asarray(era[cy2][:,keep,0])*esd+emu; er=np.hypot(er[...,0],er[...,1])
err0=er-pg; reg=err0.mean(1,keepdims=True).repeat(55,1)
# future forecast error at target (for reference: how persistent is the 10 m error?)
R={}
for s in range(55):
    W0=np.mean([np.load(f'{FT}/jobs/W0_P1_FULL_s{s:02d}_r{x}/p0_prediction.npy')[:,0] for x in range(42,47)],0)
    for L in LEADS:
        res=y[:,s,L]-W0[:,L]
        R.setdefault((L,'own'),[]).append(np.corrcoef(res,err0[:,s])[0,1]); R.setdefault((L,'reg'),[]).append(np.corrcoef(res,reg[:,s])[0,1])
        X=np.stack([np.ones(len(res)),err0[:,s],reg[:,s]],1); half=len(res)//2
        b=np.linalg.lstsq(X[:half],res[:half],rcond=None)[0]; r2=res[half:]-X[half:]@b
        R.setdefault((L,'mse_gain'),[]).append((r2**2).mean()/(res[half:]**2).mean()-1); R.setdefault((L,'mae_gain'),[]).append(np.abs(r2).mean()/np.abs(res[half:]).mean()-1)
for L in LEADS: print(f'lead {(L+1)/4:.0f}h: corr(W0 residual, own ERA5−Pangu at origin) {np.mean(R[(L,"own")]):+.3f}; regional mean {np.mean(R[(L,"reg")]):+.3f}; linear correction (fit H1 2023, eval H2) MSE {100*np.mean(R[(L,"mse_gain")]):+.2f}% MAE {100*np.mean(R[(L,"mae_gain")]):+.2f}%')
