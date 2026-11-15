import numpy as np
R='/shared_data_ssd2/liying/wind_cluster_project_20261018/project/runs/daily_site_finetune_20261114_v1'
seeds=[42,43,44,45,46]; F=55; b=[0,.05,.15,.3,1.01]; nb=4
for g,a,c in [('1-3h',4,12),('3-6h',12,24)]:
    dM=np.zeros((nb,nb)); cnt=np.zeros((nb,nb)); cap={'W0':[0,0],'TF':[0,0]}; tot=0
    for s in range(F):
        y=np.load(f'{R}/feature_cache/s{s:02d}/val/y.npy')[:,a:c].astype('f8'); h=np.load(f'{R}/feature_cache/s{s:02d}/val/h.npy')[:,-1:].astype('f8')
        P={m:np.stack([np.load(f'{R}/jobs/{m}_P1_FULL_s{s:02d}_r{sd}/p0_prediction.npy')[:,0,a:c] for sd in seeds]).mean(0).astype('f8') for m in ['W0','TF']}
        dy=y-h; ia=np.clip(np.digitize(abs(dy),b)-1,0,nb-1)
        ref=(P['W0']+P['TF'])/2-h; ip=np.clip(np.digitize(abs(ref),b)-1,0,nb-1)   # predicted-change bin from the pair mean (same for both)
        dmse=(P['W0']-y)**2-(P['TF']-y)**2
        for i in range(nb):
            for j in range(nb):
                k=(ip==i)&(ia==j); dM[i,j]+=dmse[k].sum(); cnt[i,j]+=k.sum()
        tot+=y.size
        big=abs(dy)>.3
        for m in ['W0','TF']:
            cap[m][0]+=((P[m]-h)*np.sign(dy))[big].sum(); cap[m][1]+=abs(dy)[big].sum()
    print(g,'W0−TF ΔMSE (×1e-4 of per-sample MSE), rows: predicted |change| bin, cols: actual |change| bin [0,.05,.15,.3,1]')
    for i in range(nb): print('   pred',b[i],' '.join(f'{1e4*dM[i,j]/tot:+7.3f}' for j in range(nb)),'  n%:',' '.join(f'{100*cnt[i,j]/tot:5.1f}' for j in range(nb)))
    for m in ['W0','TF']: print(f'   {m}: on |actual change|>0.3, fraction of change captured in correct direction = {cap[m][0]/cap[m][1]:.3f}')
