import json,re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from prepare_era5 import OUT,stamp

def main():
    primary=json.loads((OUT/'D2_SUMMARY.json').read_text());robust=json.loads((OUT/'D2_ROBUSTNESS_SUMMARY.json').read_text())
    pre=json.loads((OUT/'D2_PROTOCOL.json').read_text());status=json.loads((OUT/'STATUS_pressure.json').read_text())
    ready=(OUT/'COMPLETENESS_AUDIT.json').exists()
    at=lambda src,m,h:next(x for x in src['results'] if x['model']==m and x['horizon']==h)
    items=[('本站原始功率，限制训练范围外推',robust,'LOCAL_RAW_CLIPPED'),('本站+区域原始功率，限制外推',robust,'LOCAL_REGION_RAW_CLIPPED'),
           ('本站+区域等效风速',primary,'LOCAL_REGION_EQ'),('异日功率负对照，限制外推',robust,'WRONGDAY_RAW_CLIPPED'),('真实过去ERA5误差（不可部署）',primary,'ORACLE_PAST_ERROR')]
    lines=['# ERA5特权校正：方案评审与已执行诊断',f'更新时间：{stamp()}。',
      '**结论：值得完成参考标签准备；现有证据只支持谨慎探索短时校正，不支持直接启动约4500项训练，也不支持把它当作已验证的3–6h修正方案。**',
      '## 数据与完成状态',
      f'2022拟合、2023去11月评价；每小时起报，分别{pre["train_samples"]}和{pre["validation_samples"]}个样本/站，56站。未读取2024 ERA5。',
      'ERA5裁剪为41.50–47.25°N、120.75–126.75°E，0.25°、24×25。地面四通道来自surface_new，高空十通道来自年份GRIB；按变量、气压层、有效时间和坐标元数据选取。三个跨年/季节时刻与盘古0h场的14通道逐元素完全一致。',
      'D1标签及审计全部完成。' if ready else f'地面四通道16700小时已完成；D2与范围漂移复查已完成。高空十通道采样为{status["done"]}/16700，仍在后台运行；随后自动生成future/past与15维天气SEL并审计。当前进度以PIPELINE_STATUS.json为准。',
      '## D2：预报误差有持续性，但不等于功率能够观测该误差',
      '过去最后一小时误差与未来1、3、6h误差的平均相关分别为0.484、0.210、0.086。知道真实过去ERA5误差时，MOS之外1–6h气象MSE还可下降6.783%，但这条输入业务上不可得。',
      '原始Pangu的10m风速RMSE逐站平均约0.648m/s（1h）到0.715m/s（6h）；天气-only MOS后为0.619到0.690m/s。',
      '以下数值为**风速误差MSE相对天气-only MOS的额外降低百分比，正值为改善**；不是功率MAE/RMSE。每站先聚合样本与所列时效，再对56站等权平均。',
      '|诊断输入|1–3h|4–6h|1–6h|全段改善站数|','|---|---:|---:|---:|---:|']
    for label,src,m in items:
        a,b,c=[at(src,m,h) for h in ['1-3','4-6','1-6']]
        lines.append(f'|{label}|{a["mean_incremental_MSE_reduction_percent"]:+.3f}%|{b["mean_incremental_MSE_reduction_percent"]:+.3f}%|{c["mean_incremental_MSE_reduction_percent"]:+.3f}%|{c["positive_farms"]}/56|')
    lines+=['## 为什么必须保留范围漂移检查',
      '最初不限制外推时，本站原始功率全段平均为−0.055%（52/56改善），本站+区域为−1.137%。单个龙湖风电场的2022归一化功率最大仅0.226，2023达到0.968；其线性校正误差严重外推。运行容量/状态变化是候选解释，尚未凭运维记录确认。',
      '补充分析仅按2022训练输入的逐维最小/最大值限制外推，不删除任何站点或样本，不用2023确定阈值。原始结果完整保留；这属于看过2023后补充的选择年分析。',
      '限制外推后，本站原始功率1–6h的额外MSE降低0.867%，同步周bootstrap95%区间[0.342%,1.301%]；但4–6h只有0.062%，区间[−0.247%,0.306%]。区域输入没有平均增量，反曲线也未胜出；反演可靠条件在2023仅覆盖约19.6%的站点历史节点。',
      '## 对方案的具体修改',
      '1. ERA5称再分析参考，不称真实轮毂天气。撤回“ERA5初始场令功率映射误差近零”的解释。',
      '2. 保留天气-only MOS、原始功率、等效风速、异日错配，不能仅比较“校正前后”就归因于实时观测。',
      '3. 先逐站反演再分环聚合；必须带可靠性与训练范围外推标记。运行容量或功率标定变化应独立审计。',
      '4. 若后续做小试，优先本站原始功率驱动的短时风校正，使用随时效收缩的修正与MOS对照。暂不主张区域“校正观测网”。只校正u/v时仍需说明标量功率不能直接识别风向，并保持SEL空间导数的一致性。',
      '5. D3需224项训练，是ERA5输入的不可部署参考，不能称严格上限；本轮没有启动。先验3%只是投入经验值。D2线性负结果不能排除所有非线性形式，但不支持立即放大到约4500项。',
      '6. 2024模型推理不读ERA5；未来若做天气诊断，应在锁定并导出预测后用独立离线程序读标签。当前没有任何2024 ERA5读取。',
      '原方案已更新：project/ERA5_PRIVILEGED_CORRECTION_PLAN_20261101_CN.md。采样代码、诊断、补充分析、逐站CSV、同步周统计均在本run中。']
    if (OUT/'PRESSURE_VERSION_AUDIT.json').exists():
        version=json.loads((OUT/'PRESSURE_VERSION_AUDIT.json').read_text())
        lines += [f'高空档案版本补充：最终版ERA5为{version["final_frames"]}小时，ERA5T为{version["era5t_frames"]}小时；后者不是最终版，已输出future/past逐节点pressure_final_mask，后续训练必须披露并按需做排除敏感性。此次地面风D2不受高空版本影响。']
    (OUT/'REPORT_CN.md').write_text(re.sub(r'(?<=\|)\n\n(?=\|)','\n','\n\n'.join(lines))+'\n')
    z=np.load(OUT/'D2_daily_sufficient_statistics.npz');zr=np.load(OUT/'D2_robustness_daily.npz');base=z['sse'][1].sum(0)
    fig,ax=plt.subplots(1,2,figsize=(11,4.1))
    colors=['#2379b8','#dc8330','#677d36','#8b59ab']
    plot=[('Local power, bounded',zr['sse'][0]),('Local + regional, bounded',zr['sse'][1]),('Equivalent wind',z['sse'][5]),('Oracle past error',z['sse'][8])]
    for (label,se),col in zip(plot,colors):
        gain=100*(1-se.sum(0)/base);ax[0].plot(range(1,7),gain.mean(0),marker='o',label=label,color=col)
    for idx,(label,se) in enumerate(plot[:2]):
        gain=100*(1-se.sum((0,2))/base.sum(1));ax[1].plot(np.sort(gain),label=label,color=colors[idx])
    for a in ax:a.axhline(0,color='black',lw=.8);a.grid(alpha=.2);a.legend(fontsize=8)
    ax[0].set(xlabel='Lead (h)',ylabel='Weather MSE reduction vs MOS (%)',title='2023, November excluded')
    ax[1].set(xlabel='Farm rank (sorted separately)',ylabel='Weather MSE reduction, 1–6 h (%)',title='All 56 farms retained')
    fig.tight_layout();fig.savefig(OUT/'D2_evidence.png',dpi=180);fig.savefig(OUT/'D2_evidence.pdf');plt.close(fig)
if __name__=='__main__':main()
