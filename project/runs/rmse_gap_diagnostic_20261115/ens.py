import numpy as np
R='/shared_data_ssd2/liying/wind_cluster_project_20261018/project/runs/daily_site_finetune_20261114_v1'
seeds=[42,43,44,45,46]; F=55; G=[('all',0,24),('0-1h',0,4),('1-3h',4,12),('3-6h',12,24)]
res={}
for s in range(F):
    y=np.load(f'{R}/feature_cache/s{s:02d}/val/y.npy').astype('f8'); h=np.load(f'{R}/feature_cache/s{s:02d}/val/h.npy')[:,-1].astype('f8')
    P={m:np.stack([np.load(f'{R}/jobs/{m}_P1_FULL_s{s:02d}_r{sd}/p0_prediction.npy')[:,0] for sd in seeds]).astype('f8') for m in ['W0','TF']}
    cand={'W0_single':P['W0'],'TF_single':P['TF'],'W0_ens5':P['W0'].mean(0,keepdims=True),'TF_ens5':P['TF'].mean(0,keepdims=True),'W0TF_ens10':np.concatenate([P['W0'],P['TF']]).mean(0,keepdims=True)}
    for k,p in cand.items():
        e=p-y[None]
        for g,a,b in G:
            res.setdefault((k,g,'mae'),[]).append(np.abs(e[:,:,a:b]).mean(axis=(1,2)).mean())
            res.setdefault((k,g,'rmse'),[]).append(np.sqrt((e[:,:,a:b]**2).mean(axis=(1,2))).mean())
    # variance share: mean over seeds of MSE = MSE(ens) + mean seed variance
    for m in ['W0','TF']:
        e=P[m]-y[None]; v=P[m].var(0).mean(); res.setdefault((m,'var_share'),[]).append(v/(e**2).mean())
M=lambda *k: np.mean(res[k])
for g,_,_ in G:
    print(g)
    for k in ['W0_single','TF_single','W0_ens5','TF_ens5','W0TF_ens10']: print(f'   {k:11s} MAE {M(k,g,"mae"):.6f} RMSE {M(k,g,"rmse"):.6f}')
    print(f'   W0 vs TF single: {M("W0_single",g,"mae")/M("TF_single",g,"mae")-1:+.3%} / {M("W0_single",g,"rmse")/M("TF_single",g,"rmse")-1:+.3%};  ens5 vs ens5: {M("W0_ens5",g,"mae")/M("TF_ens5",g,"mae")-1:+.3%} / {M("W0_ens5",g,"rmse")/M("TF_ens5",g,"rmse")-1:+.3%}')
for m in ['W0','TF']: print(m,'seed-variance share of MSE: %.3f'%M(m,'var_share'))
