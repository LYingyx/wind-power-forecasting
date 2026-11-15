import numpy as np, json, os
R='/shared_data_ssd2/liying/wind_cluster_project_20261018/project/runs/daily_site_finetune_20261114_v1'
seeds=[42,43,44,45,46]; F=55
G=[('0-1h',0,4),('1-3h',4,12),('3-6h',12,24)]
acc={}
def add(k,v): acc.setdefault(k,[]).append(v)
rampb=[0,.05,.15,.3,1.01]
for s in range(F):
    y=np.load(f'{R}/feature_cache/s{s:02d}/val/y.npy').astype('f8'); h=np.load(f'{R}/feature_cache/s{s:02d}/val/h.npy')[:,-1].astype('f8')
    ramp=np.abs(y-h[:,None])
    for sd in seeds:
        P={m:np.load(f'{R}/jobs/{m}_P1_FULL_s{s:02d}_r{sd}/p0_prediction.npy')[:,0].astype('f8') for m in ['W0','TF']}
        P['AVG']=(P['W0']+P['TF'])/2
        for m,p in P.items():
            e=p-y
            add((m,'mae'),abs(e).mean()); add((m,'rmse'),np.sqrt((e**2).mean()))
            add((m,'bias'),e.mean()); add((m,'q99abs'),np.quantile(abs(e),.99))
            add((m,'tail_share'),(e[abs(e)>.3]**2).sum()/(e**2).sum())
            for g,a,b in G:
                ee=e[:,a:b]; pp=p[:,a:b]; yy=y[:,a:b]
                add((m,g,'mse'),(ee**2).mean()); add((m,g,'mae'),abs(ee).mean())
                add((m,g,'predstd'),pp.std()); add((m,g,'ystd'),yy.std())
                c=np.cov(pp.ravel(),yy.ravel()); add((m,g,'slope'),c[0,1]/c[0,0])   # y on pred; <1 => over-responsive
                for i in range(4):
                    k=(ramp[:,a:b]>=rampb[i])&(ramp[:,a:b]<rampb[i+1])
                    add((m,g,'sse_ramp',i),(ee[k]**2).sum()/ee.size); add((m,g,'ae_ramp',i),abs(ee[k]).sum()/ee.size)
                hp=h>=.75; add((m,g,'hp_mse'),(ee[hp]**2).mean())
                # mean minus median residual (MAE-loss median vs RMSE mean gap), pooled
                add((m,g,'mean_minus_median'),np.mean(-ee)-np.median(-ee))
M=lambda *k: np.mean(acc[k])
print('macro MAE/RMSE: W0 %.6f/%.6f  TF %.6f/%.6f  AVG %.6f/%.6f'%(M('W0','mae'),M('W0','rmse'),M('TF','mae'),M('TF','rmse'),M('AVG','mae'),M('AVG','rmse')))
for m in ['W0','TF']: print(m,'bias %.5f  q99|e| %.4f  share of SSE from |e|>0.3: %.3f'%(M(m,'bias'),M(m,'q99abs'),M(m,'tail_share')))
for g,_,_ in G:
    print(f'== {g}: W0/TF MSE ratio {M("W0",g,"mse")/M("TF",g,"mse")-1:+.3%}, MAE ratio {M("W0",g,"mae")/M("TF",g,"mae")-1:+.3%}, highpower MSE ratio {M("W0",g,"hp_mse")/M("TF",g,"hp_mse")-1:+.3%}')
    for m in ['W0','TF']: print(f'   {m}: pred std {M(m,g,"predstd"):.4f} (obs {M(m,g,"ystd"):.4f}), slope y~pred {M(m,g,"slope"):.3f}, mean-median residual {M(m,g,"mean_minus_median"):+.4f}')
    dS=[M('W0',g,'sse_ramp',i)-M('TF',g,'sse_ramp',i) for i in range(4)]; dA=[M('W0',g,'ae_ramp',i)-M('TF',g,'ae_ramp',i) for i in range(4)]
    tS=sum(dS); tA=sum(dA)
    print('   W0-TF ΔMSE by |y-h_last| bins [0,.05,.15,.3,1]:', ' '.join(f'{x:+.2e}' for x in dS), ' total %+.2e'%tS)
    print('   W0-TF ΔMAE by same bins             :', ' '.join(f'{x:+.2e}' for x in dA), ' total %+.2e'%tA)
