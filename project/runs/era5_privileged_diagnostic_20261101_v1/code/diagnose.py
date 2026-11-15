"""2022 fit / 2023 no-November meteorological error diagnosis. CPU only.

Deployable diagnostic regressions never receive ERA5 as an input. The explicitly
named ORACLE_PAST_ERROR model is an unavailable-information reference only.
"""
from __future__ import annotations
import csv,json,sys,time,re
from pathlib import Path
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from prepare_era5 import RUN,PROJECT,OUT,BUNDLE,STATIC,atomic,stamp,load

ASSETS=PROJECT/'runs/final_mechanism_20261030_v1/assets'
CASES=['MOS','LOCAL_RAW','LOCAL_EQ','LOCAL_REGION_RAW','LOCAL_REGION_EQ',
       'WRONGDAY_RAW','WRONGDAY_EQ','ORACLE_PAST_ERROR']

def write_csv(path,rows):
    with Path(path).open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def surface_stations():
    assert (OUT/'SURFACE_READY.json').exists()
    grid=np.load(OUT/'era5_crop.npy',mmap_mode='r');ids=load('bilinear_patch_index');weights=load('bilinear_weight')
    path=OUT/'station_surface.npy'
    dst=np.lib.format.open_memmap(path,mode='w+',dtype='f4',shape=(len(grid),56,4))
    for start in range(0,len(grid),128):
        x=np.asarray(grid[start:start+128,:4]);assert np.isfinite(x).all()
        dst[start:start+128]=(x.reshape(len(x),4,-1)[...,ids]*weights).sum(-1).transpose(0,2,1)
    dst.flush();return dst

def read_split(sp,era):
    root=BUNDLE/'cache'/sp;o=np.load(root/'origin_unix.npy');keep=o%3600==0
    if sp=='val':keep &= np.load(ASSETS/'val_keep.npy')
    rows=np.flatnonzero(keep);o=o[rows];cy=np.load(root/'weather_cycle_index.npy')[rows]
    h=np.asarray(np.load(root/'station_history.npy',mmap_mode='r')[rows],dtype='f4')
    mean,sd=load('nwp_mean'),load('nwp_std')
    fu=np.load(BUNDLE/'local_weather'/f'{sp}.npy',mmap_mode='r')[cy,:,:,:4]*sd[:4]+mean[:4]
    pa=np.load(BUNDLE/'physical'/sp/'past_weather.npy',mmap_mode='r')[cy,:,:,:4]*sd[:4]+mean[:4]
    fidx=np.load(OUT/f'{sp}_future_index.npy')[cy];pidx=np.load(OUT/f'{sp}_past_index.npy')[cy]
    ef=np.asarray(era[fidx]).transpose(0,2,1,3);ep=np.asarray(era[pidx]).transpose(0,2,1,3)
    wind=lambda w:np.hypot(w[...,1],w[...,2])
    fv,pv=wind(fu),wind(pa);ev,pe=wind(ef),wind(ep)
    issue=np.load(root/'origin_issue_unix.npy')[rows];age=(o-issue)/3600
    hour=((o+28800)%86400)/3600;doy=(o+28800).astype('datetime64[s]').astype('datetime64[D]')
    day=(doy-doy.astype('datetime64[Y]')).astype(float)/365.25
    clock=np.stack([age/24,np.sin(hour*2*np.pi/24),np.cos(hour*2*np.pi/24),np.sin(day*2*np.pi),np.cos(day*2*np.pi)],-1)
    # Six targets correspond to exactly 1..6 hours after the power origin.
    y=ev[:,:,1:7]-fv[:,:,1:7]
    alluv=np.concatenate([pa[:,:,:,1:3],fu[:,:,:1,1:3]],axis=2)
    locwind=np.linalg.norm(alluv[:,:,1:],axis=-1)
    regwind=np.linalg.norm(alluv[:,:,:-1]*.25+alluv[:,:,1:]*.75,axis=-1)
    return dict(origin=o,rows=rows,h=h,fu=fu,pa=pa,fv=fv,pv=pv,y=y,clock=clock,age=age,
                era_now=ev[:,:,0],past_error=pe-pv,local_wind=locwind,regional_wind=regwind)

def fit_curves(tr):
    curves=[];records=[]
    for s in range(56):
        v=tr['era_now'][:,s];h=tr['h'][:,s];p=h[:,-1]
        good=(p>.05)&(p<.85)&(h[:,-4:].std(1)>.002)&np.isfinite(v)&(v>0)
        if good.sum()<300:raise ValueError(f'Insufficient curve samples at site {s}')
        model=IsotonicRegression(increasing=True,out_of_bounds='clip').fit(v[good],p[good])
        axis=np.linspace(np.quantile(v[good],.01),np.quantile(v[good],.99),401)
        cf=model.predict(axis);slope=np.gradient(cf,axis);unique,idx=np.unique(cf,return_index=True)
        curves.append((unique,axis[idx],axis,slope))
        records.append(dict(site=s,samples=int(good.sum()),wind_min=float(axis[0]),wind_max=float(axis[-1]),cf_min=float(unique[0]),cf_max=float(unique[-1])))
    np.savez(OUT/'inverse_curves_2022.npz',**{f'{s}_{k}':a for s,c in enumerate(curves) for k,a in enumerate(c)})
    write_csv(OUT/'inverse_curve_coverage.csv',records);return curves

def equivalents(d,curves):
    residuals=[];masks=[]
    for role,idx in [('local',[3,7,11,15,19,23]),('regional',[2,6,10,14,18,22])]:
        eq=np.zeros((len(d['origin']),56,6),'f4');valid=np.zeros_like(eq,dtype=bool)
        for s,(power,wind,axis,slope) in enumerate(curves):
            h=d['h'][:,s];values=h[:,idx];inverse=np.interp(values,power,wind)
            dynamic=np.stack([h[:,max(0,i-3):i+1].std(1)>.002 for i in idx],1)
            good=(values>.05)&(values<.85)&(values>power[0])&(values<power[-1])&dynamic
            good &= np.interp(inverse,axis,slope)>.015
            eq[:,s]=np.where(good,inverse-d[role+'_wind'][:,s],0);valid[:,s]=good
        d[role+'_eq']=eq;d[role+'_valid']=valid

def distance_masks():
    lat=np.deg2rad(load('station_latitude').astype(float));lon=np.deg2rad(load('station_longitude').astype(float))
    a=np.sin((lat[:,None]-lat[None,:])/2)**2+np.cos(lat[:,None])*np.cos(lat[None,:])*np.sin((lon[:,None]-lon[None,:])/2)**2
    dist=2*6371*np.arcsin(np.sqrt(np.clip(a,0,1)));np.fill_diagonal(dist,np.inf)
    return [(dist>=lo)&(dist<hi) for lo,hi in [(0,20),(20,60),(60,150.000001)]]

def features(d,s,masks):
    n=len(d['origin']);base=np.concatenate([d['fu'][:,s,:,:].reshape(n,-1),d['pa'][:,s].reshape(n,-1),d['fv'][:,s],d['pv'][:,s],d['clock']],1)
    own=d['h'][:,s];eq=np.concatenate([d['local_eq'][:,s],d['local_valid'][:,s]],1)
    rawrings=[];eqrings=[];cap=load('station_capacity_mw')
    for m in masks:
        nb=np.flatnonzero(m[s]);w=cap[nb].astype(float);w=w/w.sum() if len(nb) else w
        if not len(nb):rawrings.append(np.zeros((n,12)));eqrings.append(np.zeros((n,18)));continue
        r=d['h'][:,nb][:,:,[2,6,10,14,18,22]]
        mean=(r*w[None,:,None]).sum(1);sd=np.sqrt(((r-mean[:,None])**2*w[None,:,None]).sum(1))
        rawrings.append(np.concatenate([mean,sd],1))
        good=d['regional_valid'][:,nb];res=d['regional_eq'][:,nb];ww=good*w[None,:,None];coverage=ww.sum(1)
        mu=(res*ww).sum(1)/np.maximum(coverage,1e-9)
        var=((res-mu[:,None])**2*ww).sum(1)/np.maximum(coverage,1e-9)
        eqrings.append(np.concatenate([mu,np.sqrt(var),coverage],1))
    return base,own,eq,np.concatenate(rawrings,1),np.concatenate(eqrings,1)

def donors(d,tr,seed=20261101):
    rng=np.random.default_rng(seed);ta=tr['origin']+28800;tb=d['origin']+28800
    ma=ta.astype('datetime64[s]').astype('datetime64[M]').astype(int)%12;mb=tb.astype('datetime64[s]').astype('datetime64[M]').astype(int)%12
    ha=ta%86400;hb=tb%86400;out=[]
    for t,m,h in zip(d['origin'],mb,hb):
        ii=np.flatnonzero((ma==m)&(ha==h)&(np.abs(tr['origin']-t)>=7*86400))
        assert len(ii);out.append(rng.choice(ii))
    return np.array(out)

def main():
    started=time.monotonic();era=surface_stations();tr=read_split('train',era);va=read_split('val',era)
    curves=fit_curves(tr);equivalents(tr,curves);equivalents(va,curves);masks=distance_masks()
    dt, dv=donors(tr,tr),donors(va,tr);daily=(va['origin']+28800).astype('datetime64[s]').astype('datetime64[D]');days,dayidx=np.unique(daily,return_inverse=True)
    counts=np.bincount(dayidx).astype(float);sse=np.zeros((len(CASES)+1,len(days),56,6),'f8')
    rows=[];corr=[];group_rows=[];n=len(va['origin'])
    atomic(OUT/'D2_PROTOCOL.json',dict(timestamp=stamp(),target='ERA5 minus operational Pangu wind speed magnitude at 10m',
      train_samples=len(tr['origin']),validation_samples=n,origin_frequency_minutes=60,validation_no_november=True,
      ridge_alpha=10,hyperparameters='fixed before2023 outcomes; training-only standardization',
      primary_estimand='additional reduction of held-out meteorological error MSE vs weather-only MOS',
      inverse_fit='sitewise2022 ERA5 wind -> power isotonic; CF .05-.85, trailing variation>.002; inverse slope>.015; proxy only, no perfect curtailment classification',
      regional='invert each farm first, delay15min, then capacity-weight by <20/20-60/60-150km rings; target excluded',
      wrongday='observation block only: same month/hour, joint donor day across stations from2022, >=7day displacement; train and validation mismatched',
      oracle='past ERA5 error reference is explicitly nondeployable; never included in deployable models',
      clock_age_confounding='age and local hour are deterministic under fixed08/20 availability; no separate causal age claim',test_era5_read=False))
    for s in range(56):
        bt,ot,et,rt,qt=features(tr,s,masks);bv,ov,ev,rv,qv=features(va,s,masks)
        xt=[bt,np.c_[bt,ot],np.c_[bt,et],np.c_[bt,ot,rt],np.c_[bt,et,qt],np.c_[bt,ot[dt],rt[dt]],np.c_[bt,et[dt],qt[dt]],np.c_[bt,tr['past_error'][:,s]]]
        xv=[bv,np.c_[bv,ov],np.c_[bv,ev],np.c_[bv,ov,rv],np.c_[bv,ev,qv],np.c_[bv,ot[dv],rt[dv]],np.c_[bv,et[dv],qt[dv]],np.c_[bv,va['past_error'][:,s]]]
        target=va['y'][:,s];rawmse=np.mean(target**2,0);center=np.mean((target-tr['y'][:,s].mean(0))**2,0)
        errors={};np.add.at(sse[0,:,s],dayidx,target**2)
        for c,(name,x,y) in enumerate(zip(CASES,xt,xv),1):
            assert np.isfinite(x).all() and np.isfinite(y).all()
            estimator=make_pipeline(StandardScaler(),Ridge(alpha=10)).fit(x,tr['y'][:,s]);pred=estimator.predict(y)
            se=(target-pred)**2;errors[name]=se;np.add.at(sse[c,:,s],dayidx,se)
        for j in range(6):
            for name in CASES:
                mse=float(errors[name][:,j].mean());mos=float(errors['MOS'][:,j].mean())
                rows.append(dict(site=s,lead_hours=j+1,model=name,samples=n,rmse_mps=mse**.5,
                    raw_pangu_rmse_mps=float(rawmse[j]**.5),mos_rmse_mps=mos**.5,
                    improvement_vs_raw_percent=100*(1-mse/rawmse[j]),incremental_vs_MOS_percent=100*(1-mse/mos),
                    r2_vs_train_error_mean=1-mse/center[j]))
            a=va['past_error'][:,s,-1];b=target[:,j]
            corr.append(dict(site=s,lead_hours=j+1,lag_from_last_past_hours=j+2,correlation=float(np.corrcoef(a,b)[0,1])))
        for group,labels in [('age',np.where(va['age']<16,'12-16',np.where(va['age']<20,'16-20','20-24'))),
                             ('wind',np.where(va['fv'][:,s,0]<4,'<4',np.where(va['fv'][:,s,0]<8,'4-8','>=8'))),
                             ('season',np.array(['DJF','DJF','MAM','MAM','MAM','JJA','JJA','JJA','SON','SON','SON','DJF'])[((va['origin']+28800).astype('datetime64[s]').astype('datetime64[M]').astype(int)%12)])]:
            for lab in np.unique(labels):
                m=labels==lab
                for name in ['MOS','LOCAL_REGION_RAW','LOCAL_REGION_EQ']:
                    for j in range(6):
                        mse=float(errors[name][m,j].mean());bm=float(errors['MOS'][m,j].mean())
                        group_rows.append(dict(site=s,group=group,label=str(lab),lead_hours=j+1,model=name,samples=int(m.sum()),rmse_mps=mse**.5,incremental_vs_MOS_percent=100*(1-mse/bm)))
        atomic(OUT/'STATUS_D2.json',dict(timestamp=stamp(),done=s+1,total=56,elapsed_seconds=time.monotonic()-started))
        print('diagnosed',s+1,'/56',flush=True)
    write_csv(OUT/'D2_per_farm.csv',rows);write_csv(OUT/'D2_past_future_correlations.csv',corr);write_csv(OUT/'D2_groups.csv',group_rows)
    np.savez(OUT/'D2_daily_sufficient_statistics.npz',sse=sse,counts=counts,days=days,models=np.array(['RAW']+CASES))
    summary=[];rng=np.random.default_rng(20261101)
    # Synchronized calendar-week bootstrap retains shared weather across farms.
    weeks=(days.astype(int)//7);uw=np.unique(weeks);ws=np.stack([sse[:,weeks==w].sum(1) for w in uw],1)
    draw=rng.integers(0,len(uw),size=(1000,len(uw)));ms=sse.sum(1)
    for c,name in enumerate(CASES,1):
        for label,idx in [('1-6',np.arange(6)),('1-3',np.arange(3)),('4-6',np.arange(3,6))]:
            base=ms[1,:,idx].sum(0);cur=ms[c,:,idx].sum(0);gain=100*(1-cur/base)
            bm=ws[1,:,:,idx].sum(0);cm=ws[c,:,:,idx].sum(0)
            # bm/cm are week by farm after summing lead axis.
            boots=100*(1-cm[draw].sum(1)/bm[draw].sum(1));bootmean=boots.mean(1)
            summary.append(dict(model=name,horizon=label,mean_incremental_MSE_reduction_percent=float(gain.mean()),
                median_incremental_MSE_reduction_percent=float(np.median(gain)),positive_farms=int((gain>0).sum()),
                ci95_week_bootstrap=np.quantile(bootmean,[.025,.975]).tolist()))
    atomic(OUT/'D2_SUMMARY.json',dict(timestamp=stamp(),passed=True,results=summary,
        local_equivalent_valid_fraction_train=float(tr['local_valid'].mean()),local_equivalent_valid_fraction_val=float(va['local_valid'].mean()),
        past_error_correlation_mean_by_lead=[float(np.mean([r['correlation'] for r in corr if r['lead_hours']==j])) for j in range(1,7)],
        test_era5_read=False,models_are_weather_diagnostic_regressions_not_power_forecasters=True))
    write_report(summary,tr,va)

def write_report(summary,tr,va):
    lines=['# ERA5 校正方向：D2 初步诊断',f'完成时间：{stamp()}。2022 拟合，2023 去11月评价；没有读取2024 ERA5。',
      f'只取整点功率起报样本，训练{len(tr["origin"])}、评价{len(va["origin"])}；逐站独立拟合，56站等权汇总。',
      '以下是**未来10m风速预报误差MSE相对天气-only MOS的额外下降**，不是功率MAE/RMSE提升。负值表示变差。',
      '|输入|时效|平均额外改善%|中位数%|改善站数|同步周bootstrap95%区间|','|---|---|---:|---:|---:|---|']
    for r in summary:
        ci=r['ci95_week_bootstrap'];lines.append(f'|{r["model"]}|{r["horizon"]}h|{r["mean_incremental_MSE_reduction_percent"]:.3f}|{r["median_incremental_MSE_reduction_percent"]:.3f}|{r["positive_farms"]}/56|[{ci[0]:.3f}, {ci[1]:.3f}]|')
    lines += ['', 'LOCAL/REGION_RAW使用原始功率；EQ使用逐站反曲线等效风速残差与有效掩膜；WRONGDAY把观测换为2022同月同钟点另一日。',
      'ORACLE_PAST_ERROR读取真实的过去ERA5误差，仅为不可部署的信息诊断，不能作为业务模型成绩。',
      '反曲线是粗网格风速对应的经验等效量，不是轮毂实测风；低/高功率、历史平稳与低斜率样本被屏蔽，但不能保证完全剔除限电。',
      '该线性诊断不能作为所有非线性模型无效的证明；正结果也不能直接证明功率预测改善，需要另做D3和E组。',
      'D3是使用ERA5输入的不可部署参考实验，需训练224项（两骨干×56站×2种子）；本诊断不自动启动大规模模型训练。']
    (OUT/'D2_REPORT_CN.md').write_text(re.sub(r'(?<=\|)\n\n(?=\|)','\n','\n\n'.join(lines))+'\n')
if __name__=='__main__':main()
