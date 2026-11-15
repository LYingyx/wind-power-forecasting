import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from prepare_age import OUT,RUN,PARENT,PROJECT,atomic,stamp

def main():
    result=json.loads((OUT/'SUMMARY.json').read_text());cohort=json.loads((OUT/'COMMON_COHORT.json').read_text())
    def get(age,model,h='1-6'):
        return next(x for x in result['results'] if x['shift_hours']==age and x['model']==model and x['horizon']==h)
    paired=lambda age,model,h:next(x for x in result['paired_age_effects'] if x['shift_hours']==age and x['model']==model and x['horizon']==h)
    a=get(12,'LOCAL','4-6');d=paired(12,'LOCAL','4-6');control=get(12,'WRONG_LOCAL','4-6')
    supported=(a['ci95_week_bootstrap'][0]>0 and d['ci95_paired_week_bootstrap'][0]>0 and
               d['absolute_ci95_paired_week_bootstrap'][0]>0 and
               a['raw_macro_rmse_mps']>get(0,'LOCAL','4-6')['raw_macro_rmse_mps'] and
               a['extra_MSE_reduction_percent']>control['extra_MSE_reduction_percent'])
    verdict='再增加12h预报年龄后，本站功率对远时效气象误差的额外校正价值上升，支持小规模后续验证。' if supported else '再增加12h预报年龄后，尚未同时建立远时效正增益及其随年龄增加的稳定证据，不宜据此承诺机制有效。'
    lines=['# 盘古预报再老化12/24小时：配对D2诊断',f'完成时间：{stamp()}。',f'**{verdict}**',
      '## 设定',
      '这里的老化是把盘古起报时刻向前移动12或24小时，并取相同有效时刻的预报；不是错移ERA5标签，也不是重新使用GFS初始场运行盘古。',
      f'2022拟合，2023去11月评价，56站独立岭回归、alpha=10。整点起报样本：训练{cohort["splits"]["train"]["retained"]}、评价{cohort["splits"]["val"]["retained"]}；三种年龄同样本同功率同ERA5标签。2024未读取。',
      '气象输入为同期次的过去6个及未来8个小时节点，msl/u10/v10/t2m；观测为本站已知历史与延迟15min的三圈邻站历史。原始功率通路按2022训练范围限制外推，另外完整报告不限制外推的敏感性结果。',
      '## 天气误差及观测校正收益',
      '以下MSE增益均相对各年龄单独拟合的天气-only MOS；正值表示额外降低。它们是10m风速气象误差，不是功率MAE或RMSE。RMSE先逐站计算再等权平均。',
      '|预报年龄|原始风速RMSE m/s，1–6h|MOS后RMSE|本站功率额外MSE改善，1–3h|本站功率额外MSE改善，4–6h|本站功率额外MSE改善，1–6h|',
      '|---|---:|---:|---:|---:|---:|']
    for age in [0,12,24]:
        q=get(age,'LOCAL');n=get(age,'LOCAL','1-3');f=get(age,'LOCAL','4-6')
        lines.append(f'|{12+age}–{24+age}h|{q["raw_macro_rmse_mps"]:.4f}|{q["mos_macro_rmse_mps"]:.4f}|{n["extra_MSE_reduction_percent"]:+.3f}%|{f["extra_MSE_reduction_percent"]:+.3f}%|{q["extra_MSE_reduction_percent"]:+.3f}%|')
    lines+=['','## 增加12小时的关键检验',
      f'24–36h年龄下，本站功率对4–6h气象误差MSE额外降低{a["extra_MSE_reduction_percent"]:.3f}%，{a["positive_farms"]}/56站改善，同步周bootstrap95%区间{a["ci95_week_bootstrap"]}。',
      f'相对12–24h年龄，该收益变化{d["change_in_gain_percentage_points"]:.3f}个百分点，配对同步周95%区间{d["ci95_paired_week_bootstrap"]}。同月同钟点异日功率对照为{control["extra_MSE_reduction_percent"]:.3f}%。',
      '绝对MSE降低量也报告在SUMMARY.json，避免仅因分母改变而宣称价值增加。所有年龄使用相同日历周重采样索引，保留全站共同天气相关性。',
      '## 区域、反曲线与不可部署参考',
      '|预报年龄|本站+区域，1–6h|本站+区域，4–6h|本站等效风速，1–6h|区域等效风速，1–6h|已知真实过去误差，1–6h|',
      '|---|---:|---:|---:|---:|---:|']
    for age in [0,12,24]:
        vals=[get(age,'LOCAL_REGION'),get(age,'LOCAL_REGION','4-6'),get(age,'LOCAL_EQ'),get(age,'REGION_EQ'),get(age,'ORACLE_PAST_ERROR')]
        lines.append(f'|{12+age}–{24+age}h|'+ '|'.join(f'{x["extra_MSE_reduction_percent"]:+.3f}%' for x in vals)+'|')
    lines+=['','## 能写与不能写的结论',
      '本实验直接检验的是更旧的ERA5初始化盘古预报。即使观测增益随年龄上升，也只能说明该档案中的年龄扰动具有此性质，不能推广为任何低质量预报都更可校正。GFS初始场误差的空间结构、风向与系统位置偏差可能不同。',
      '所有可部署诊断回归都只在2022用ERA5当目标训练，2023前向输入没有ERA5；ORACLE_PAST_ERROR是单独标记的不可部署信息参考。没有训练最终天气校正网络或W0/TF功率模型，因此还不能宣称功率MAE/RMSE提高。',
      '若该配对检验成立，下一步先用现有更旧预报做小规模端到端功率校正与天气-only MOS、无ERA5辅助监督、异日观测对照；通过后再投入GFS初始化的正式验证。若不成立，线性诊断仍不是所有非线性形式无效的证明，但不支持大规模投入。',
      '此前ERA5高空采样遇到expver0005后已恢复并标注ERA5T/最终版掩膜；本轮复用已完成的地面标签，故其结果不依赖高空ERA5T。']
    # Keep table lines adjacent; separate prose paragraphs for readable Markdown.
    text=''
    for i,line in enumerate(lines):text+=('' if i==0 else '\n' if line.startswith('|') and lines[i-1].startswith('|') else '\n\n')+line
    (OUT/'REPORT_CN.md').write_text(text+'\n')
    verdict_data=dict(timestamp=stamp(),additional12h_far_horizon_supported=supported,local_far_gain=a,paired_age_effect=d,wrongday_far_control=control,
        conclusion=verdict,test_read=False,no_power_training=True)
    atomic(OUT/'VERDICT.json',verdict_data)
    fig,axes=plt.subplots(1,3,figsize=(14,4.2))
    ages=np.array([18,30,42]);labels=['12–24 h','24–36 h','36–48 h']
    axes[0].plot(ages,[get(x,'RAW')['macro_rmse_mps'] for x in [0,12,24]],'o-',label='Raw Pangu')
    axes[0].plot(ages,[get(x,'MOS')['macro_rmse_mps'] for x in [0,12,24]],'o-',label='Weather-only MOS')
    for h,color in [('1-3','#267ab2'),('4-6','#d67e24')]:
        values=[get(x,'LOCAL',h) for x in [0,12,24]];y=np.array([x['extra_MSE_reduction_percent'] for x in values]);ci=np.array([x['ci95_week_bootstrap'] for x in values]).T
        axes[1].errorbar(ages,y,yerr=np.maximum(np.stack([y-ci[0],ci[1]-y]),0),fmt='o-',label=f'Lead {h} h',color=color,capsize=4)
        axes[2].plot(ages,[x['absolute_MSE_reduction'] for x in values],'o-',label=f'Lead {h} h',color=color)
    for ax in axes:ax.set_xticks(ages,labels);ax.set_xlabel('Forecast age');ax.grid(alpha=.25);ax.legend(fontsize=8)
    axes[0].set_ylabel('10 m wind RMSE (m/s), lead 1–6 h')
    axes[1].axhline(0,color='black',lw=.8);axes[1].set_ylabel('Local-power MSE reduction vs MOS (%)')
    axes[2].axhline(0,color='black',lw=.8);axes[2].set_ylabel('Absolute MSE reduction vs MOS (m²/s²)')
    fig.suptitle('Paired 2023 samples; fit on 2022; November excluded',fontsize=12);fig.tight_layout()
    fig.savefig(OUT/'forecast_age_effect.png',dpi=180);fig.savefig(OUT/'forecast_age_effect.pdf');plt.close(fig)
    atomic(OUT/'SUCCESS.json',dict(timestamp=stamp(),passed=True,report=str(OUT/'REPORT_CN.md'),test_read=False))
    active_path=PROJECT/'ACTIVE_EXPERIMENT.json';active=json.loads(active_path.read_text())
    if active.get('run')==str(RUN):
        active.update(timestamp=stamp(),stage='completed_paired_forecast_age_diagnostic',audit=str(OUT/'COMPLETENESS_AUDIT.json'),report=str(OUT/'REPORT_CN.md'))
        atomic(active_path,active)
if __name__=='__main__':main()
