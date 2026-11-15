from settings import *
import pandas as pd

def table(columns,rows):
    return '| '+' | '.join(columns)+' |\n| '+' | '.join(['---']*len(columns))+' |\n'+''.join('| '+' | '.join(map(str,r))+' |\n' for r in rows)+'\n'

def write():
    out=RUN/'analysis';check=json.loads((out/'DIAGNOSTICS_COMPLETE.json').read_text())
    actual=pd.read_csv(out/'existing_actual_by_farm_seed.csv',float_precision='round_trip')
    paired=pd.read_csv(out/'existing_paired_by_farm_seed.csv',float_precision='round_trip')
    overall=actual[(actual.group=='all')&(actual.horizon=='all')]
    maximum=0.
    for row in overall.itertuples():
        old=json.loads(Path(row.source).with_name('metrics.json').read_text())['val']
        maximum=max(maximum,abs(row.mae-old['mae']),abs(row.rmse-old['rmse']))
    assert maximum<1e-7,maximum
    d=paired[paired.horizon=='all'];rows=[];summary=[]
    for seed in SEEDS:
        s=d[d.seed==seed];all=s[s.group=='all'];worse=set(all[all.rmse_difference>0].site)
        power=s[(s.group=='current_power')&s.site.isin(worse)]
        biggest=power.loc[power.groupby('site').squared_error_difference.idxmax()] if len(power) else power
        summary.append([seed,56,int((all.mae_difference<0).sum()),int((all.rmse_difference<0).sum()),len(worse),int((biggest.bin=='high').sum())])
        for group,label in [('bulk_ri_proxy','bin_3'),('local_valid_hour','18_24'),('current_power','high'),('history_plateau_state','sustained_nonlow_power')]:
            g=s[(s.group==group)&(s.bin==label)]
            rows.append([seed,group,label,len(g),int((g.squared_error_difference>0).sum()),int((g.squared_error_difference<0).sum())])
    report='''# 新增强化方向的零训练诊断

使用已有W0（NEUTRAL_LAST）和Transformer的2023验证预测：56站、seeds42/43。该报告没有训练新模型，没有复制预测，也没有读取2024。分箱只由2022确定；指标在每个风场×seed内的样本/时效上计算，不跨风场或seed平均。

零训练诊断可以定位错误分布，不能证明物理因果或预告新模型收益。两种旧模型都曾按2023 MAE选择epoch，因此这些结果属于开发期分析。

## 每个seed的总体比较

'''
    report+=table(['seed','风场数','W0 MAE更好','W0 RMSE更好','W0 RMSE较差','较差站中高功率是最大恶化来源'],summary)
    report+='高功率定义为起报观测CF≥0.7。最后一列按可加的平方误差恶化总量选最大来源，既考虑误差大小，也考虑样本数量；不能以某个分组RMSE高就说它解释了总体差距。高功率、夜间等单一归因目前都不充分。\n\n'
    report+='## 预先关注的场景\n\n'
    report+=table(['seed','分组','区间','有样本风场数','W0平方误差较高','W0平方误差较低'],rows)
    report+='最高Ri代理四分位组（bin_3）的差距较一致，值得保留STAB/CAL对照。它不是实测稳定度，风速、季节、功率状态和运行因素仍可能共同影响分组结果。不同分组存在重叠，不能将计数相加。\n\n'
    report+='平台状态完全基于历史：连续至少4个15min小变化间隔（阈值为各站训练期历史差分绝对值第10百分位）；再区分当前CF≤0.05与>0.05。不把该标签命名为已确认限电。持续时间的四分位边界在不少风场退化为0，因此另列显式平台/非平台类别，避免把两者压进同一分箱。\n\n'
    report+='## 既有代表站实际值\n\n'
    values=[]
    for seed in SEEDS:
        for site in REPRESENTATIVES:
            w=overall[(overall.site==site)&(overall.seed==seed)&(overall.model=='W0')].iloc[0]
            t=overall[(overall.site==site)&(overall.seed==seed)&(overall.model=='TF')].iloc[0]
            values.append([int(w.station_id),seed,*[f'{x:.9f}' for x in [w.mae,t.mae,w.rmse,t.rmse,w.bias,t.bias]]])
    report+=table(['station_id','seed','W0 MAE','TF MAE','W0 RMSE','TF RMSE','W0 bias','TF bias'],values)
    report+='## 文件与校验\n\n'
    report+=f"逐模型分组原值共{check['actual_rows']}行，配对共{check['paired_rows']}行。各完整分区重构总平方误差差的最大绝对误差为{check['partition_max_abs_error']:.3g}；与旧metrics逐模型总MAE/RMSE对照的最大差为{maximum:.3g}。\n\n"
    report+='- [逐风场逐seed原值](existing_actual_by_farm_seed.csv)\n- [逐风场逐seed配对与平方误差贡献](existing_paired_by_farm_seed.csv)\n- [训练期分箱边界](train_only_bin_edges.json)\n- [诊断验收](DIAGNOSTICS_COMPLETE.json)\n'
    (out/'DIAGNOSTIC_REPORT_CN.md').write_text(report)
    atomic_json(out/'DIAGNOSTIC_METRIC_REPLAY.json',dict(timestamp=now(),max_abs_error=maximum,passed=True))
    print('WROTE',out/'DIAGNOSTIC_REPORT_CN.md')
if __name__=='__main__':write()
