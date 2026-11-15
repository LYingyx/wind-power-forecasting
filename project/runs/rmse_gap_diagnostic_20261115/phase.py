"""Error at on-the-hour vs sub-hour targets (same lead block), W0_R3 & TF_R3 shared P0, 2023, 5 seeds mean. Zero training."""
import numpy as np
R='/shared_data_ssd2/liying/wind_cluster_project_20261018/project/runs/daily_site_finetune_20261114_v1'
D='/shared_data_ssd2/liying/wind_cluster_project_20261018/project/runs/daily_protocol_restart_20261114_v1/data'
o=np.load(f'{D}/val/origin.npy'); minute=((o//60)%60)//15   # origin quarter
k=np.arange(24); tq=(minute[:,None]+k[None]+1)%4             # target quarter: 0 = on the hour
G=[('0-1h',0,4),('1-3h',4,12),('3-6h',12,24)]
acc={}
for s in range(55):
    y=np.load(f'{R}/feature_cache/s{s:02d}/val/y.npy').astype('f8'); h=np.load(f'{R}/feature_cache/s{s:02d}/val/h.npy')[:,-1].astype('f8')
    for m in ['W0','TF']:
        p=np.mean([np.load(f'{R}/jobs/{m}_P1_FULL_s{s:02d}_r{x}/p0_prediction.npy')[:,0] for x in range(42,47)],0)
        e2=(p-y)**2; ae=np.abs(p-y); dy2=(y-h[:,None])**2
        for g,a,b in G:
            for q in range(4):
                msk=(tq[:,a:b]==q)
                acc.setdefault((m,g,q,'mse'),[]).append(e2[:,a:b][msk].mean()); acc.setdefault((m,g,q,'mae'),[]).append(ae[:,a:b][msk].mean())
                acc.setdefault((g,q,'var'),[]).append(dy2[:,a:b][msk].mean())
for g,_,_ in G:
    print(g)
    for m in ['W0','TF']:
        base=np.mean(acc[(m,g,0,'mse')])
        print(f'  {m}: MSE by target minute :00/:15/:30/:45 = '+' '.join(f'{np.mean(acc[(m,g,q,"mse")]):.5f}' for q in range(4))+'   rel to :00 '+' '.join(f'{np.mean(acc[(m,g,q,"mse")])/base-1:+.2%}' for q in range(4)))
    vb=np.mean(acc[(g,0,'var')]); print('  E[(y-h)^2] (difficulty) rel to :00 '+' '.join(f'{np.mean(acc[(g,q,"var")])/vb-1:+.2%}' for q in range(4)))
