"""Predeclared train/validation screen. No performance threshold or 2024 tuning."""
from settings import *
from physical_data import FEATURES

DESCRIPTIONS={
 'BASE':'原固定点天气/历史功率骨干，不增加输入或结构',
 'CAP':'仅已有天气、末值和时效的额外零初始化线性投影，额外容量对照',
 'CAL':'目标当地时刻和年周期四个sin/cos特征',
 'REF':'起报天气代理14通道及历史/未来发布年龄2通道',
 'DELTA':'REF + 目标天气减起报天气；与REF比较变化量重参数化',
 'DENS':'DEM干空气压强修正后的密度与风能代理3维',
 'STAB':'位温梯度/切变/Ri/有效掩码/中性高度风代理5维',
 'DYN':'925hPa地转风/非地转差/温度平流/气压倾向/风速倾向7维',
 'STATE':'仅历史功率生成的波动、趋势、平台持续时间7维',
 'CYCLE':'同有效时间相邻发布周期天气差14维及有效掩码',
 'ALL':'CAL+REF+DELTA+DENS+STAB+DYN+STATE+CYCLE组合，不预称主方法',
 'HIST':'六个历史天气节点、对应时差及发布年龄的查询投影',
 'SHIFT_ZERO':'HIST + 恒零时间位移；保留相同位移网络，参数不活跃须注明',
 'SHIFT_FIXED':'HIST + 相同网络读取训练期平均状态，学习全站固定时间位移',
 'SHIFT_SAMPLE':'HIST + 实际历史功率/天气估计每样本[-1,1]h位移',
 'PC_INPUT':'REF+DENS仅输入对照',
 'CURVE_FREE':'PC_INPUT + 可正负分段响应曲线、衰减偏差、零初始化残差门控',
 'CURVE_BIAS':'PC_INPUT + 单调饱和分段响应曲线、衰减偏差、零初始化残差门控；不截断CF',
 'NEIGH_GENERIC':'CAL + 8个最近邻站/150km掩码的历史功率注意力，含null token',
 'NEIGH_ADVECT':'NEIGH_GENERIC同参数同信息，另加固定方向与传播时间软先验',
}

def build():
    setup();jobs=[]
    def add(stage,variant,sites,losses=['MAE'],delays=[15],capacity_control=False):
        for loss in losses:
            for site in sites:
                for seed in SEEDS:
                    for backbone in ['W0','TF']:
                        for delay in delays:
                            job=dict(stage=stage,variant=variant,site=site,seed=seed,backbone=backbone,loss=loss,delay_minutes=delay,
                                capacity_control=capacity_control,protocol='neighbor_observation' if variant.startswith('NEIGH_') else 'single_farm_only')
                            job['id']=f'{stage}_{backbone}_{variant}_{loss}_s{site:02d}_r{seed}_d{delay}'
                            if capacity_control:job['id']+='_capacity'
                            jobs.append(job)
    # Reps first for feedback, then all farms; ordering does not use outcomes.
    sites=REPRESENTATIVES+[s for s in range(56) if s not in REPRESENTATIVES]
    for variant in ['BASE','CAL','REF','DELTA','DENS','STAB','DYN','STATE','CYCLE','ALL']:
        add('input',variant,sites)
    add('loss','BASE',sites,['MSE','MIX'])
    for variant in ['CAL','REF','DELTA','DENS','STAB','DYN','STATE','CYCLE','ALL']:
        add('capacity',variant,REPRESENTATIVES,capacity_control=True)
    for variant in ['HIST','SHIFT_ZERO','SHIFT_FIXED','SHIFT_SAMPLE','PC_INPUT','CURVE_FREE','CURVE_BIAS','NEIGH_GENERIC','NEIGH_ADVECT']:
        add('mechanism',variant,REPRESENTATIVES,['MAE','MSE'])
    for variant in ['NEIGH_GENERIC','NEIGH_ADVECT']:add('latency',variant,REPRESENTATIVES,['MAE'],[0,30])
    # Launch representative coverage before the full input matrix, without selecting by results.
    jobs.sort(key=lambda s:(s['site'] not in REPRESENTATIVES, {'input':0,'loss':1,'mechanism':2,'latency':3,'capacity':1}[s['stage']]))
    assert len({s['id'] for s in jobs})==len(jobs)
    atomic_json(RUN/'manifest.json',jobs)
    atomic_json(RUN/'PROTOCOL.json',dict(version='20261024_v1',created=now(),train_year=2022,validation_year=2023,
        heldout_access=False,retrospective_reevaluation_year=2024,epochs=80,patience=10,stopping_metric='validation_MAE_for_all_losses',
        checkpoint_selection=['MAE','RMSE'],batch=256,precision='FP32_no_TF32',optimizer='AdamW',weight_decay=1e-4,clip_norm=1,
        w0_lr=2e-4,tf_lr=1e-3,extra_lr=2e-4,schedule='cosine80',
        losses={'MAE':'mean(abs(error))','MSE':'mean(error**2)','MIX':'0.5*MAE+0.5*MSE/0.1; fixed CF scale, not an improvement threshold'},
        seeds=SEEDS,representatives=REPRESENTATIVES,n_jobs=len(jobs),independent_farm_parameters=True,
        neighbor_k=8,neighbor_radius_km=150,default_neighbor_delay_minutes=15,latency_sensitivity_minutes=[0,30],
        report='actual results for each farm and seed; paired strict differences, no farm/seed averaging or percent cutoff',
        selection='retain all outcomes; exploratory Pareto and per-farm seed-consistency ranking on validation only; freeze separately before 2024',
        storage='reuse old grids; compact cycle features; at most two distinct selected epochs, one temporary resumable state; no predictions',
        descriptions=DESCRIPTIONS))
    print('jobs',len(jobs))
if __name__=='__main__':build()
