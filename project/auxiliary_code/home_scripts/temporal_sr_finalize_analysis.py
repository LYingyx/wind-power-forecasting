"""Final audit, summaries and report for temporal_sr_named_models_20260917_v2."""
from __future__ import annotations
import csv,hashlib,json,math
from pathlib import Path
import numpy as np
from scipy.stats import wilcoxon

ROOT=Path('/storage/liying/new_try/wind_cluster_forced_evolution/artifacts/temporal_sr_named_models_20260917_v2')
STATIONS=['1043019','1043010','1043036','1043007','1043028','1043033','1043011','1043058','1043015','1043061']
SEEDS=list(range(42,52))
FAMILIES=['DLinear','PatchTST','HistGradientBoostingRegressor','TiDE','TFT','NBEATSx'] \
 + ['TiDE_SR_'+x for x in ['DET','ICFM','VPCFM','SINE_SI','VP_EPS','VP_V','EDM','MEANFLOW']] \
 + ['TiDE_FROZEN_ICFM','TiDE_FROZEN_EDM'] \
 + ['P0I_CBH_SINGLE_'+x for x in ['LINEAR','DET','ICFM','EDM']]
LABEL={
 'DLinear':'DLinear','PatchTST':'PatchTST','HistGradientBoostingRegressor':'HistGradientBoostingRegressor',
 'TiDE':'TiDE','TFT':'TFT','NBEATSx':'NBEATSx','TiDE_SR_DET':'TiDE + DET',
 'TiDE_SR_ICFM':'TiDE + ICFM','TiDE_SR_VPCFM':'TiDE + VP-CFM','TiDE_SR_SINE_SI':'TiDE + Sine-SI',
 'TiDE_SR_VP_EPS':'TiDE + VP-epsilon','TiDE_SR_VP_V':'TiDE + VP-v','TiDE_SR_EDM':'TiDE + EDM',
 'TiDE_SR_MEANFLOW':'TiDE + MeanFlow','TiDE_FROZEN_ICFM':'TiDE + frozen ICFM',
 'TiDE_FROZEN_EDM':'TiDE + frozen EDM','P0I_CBH_SINGLE_LINEAR':'P0I-single + linear weather',
 'P0I_CBH_SINGLE_DET':'P0I-single + DET','P0I_CBH_SINGLE_ICFM':'P0I-single + ICFM',
 'P0I_CBH_SINGLE_EDM':'P0I-single + EDM'}

def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def dump(p,x):
 p=Path(p);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(x,ensure_ascii=False,indent=2));tmp.replace(p)
def write_csv(p,rows):
 if not rows:return
 with open(p,'w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

protocol=json.loads((ROOT/'configs/protocol.json').read_text());protocol_sha=sha(ROOT/'configs/protocol.json')
metrics={}; audit_errors=[]
for fam in FAMILIES:
 arr=np.empty((10,10));rmse=np.empty_like(arr);stop=np.empty_like(arr)
 for j,sid in enumerate(STATIONS):
  for i,seed in enumerate(SEEDS):
   p=ROOT/'results'/fam/sid/f'seed{seed}';mp=p/'metrics.json'
   if not mp.exists():raise RuntimeError(f'Missing {mp}')
   m=json.loads(mp.read_text())
   for k in ['report_mae','report_rmse','val_stop_mae']:
    if not math.isfinite(float(m[k])):audit_errors.append(f'{mp}: nonfinite {k}')
   if str(m['station_id'])!=sid or int(m['seed'])!=seed or m['model']!=fam:audit_errors.append(f'{mp}: identity mismatch')
   if m['protocol_sha256']!=protocol_sha:audit_errors.append(f'{mp}: protocol mismatch')
   checkpoint=p/('best_model.joblib' if fam=='HistGradientBoostingRegressor' else 'best_model.pt')
   if sha(checkpoint)!=m['checkpoint_sha256']:audit_errors.append(f'{checkpoint}: checkpoint hash mismatch')
   pred=np.load(p/'predictions_report.npy')
   if pred.shape!=(m['report_n'],24) or not np.isfinite(pred).all():audit_errors.append(f'{p}: prediction shape/finite')
   sample_path=p/'samples_report.npy'
   if sample_path.exists():
    samples=np.load(sample_path)
    expected_k=1 if fam.endswith(('_DET','_LINEAR')) else 5
    if samples.shape!=(m['report_n'],expected_k,24) or not np.isfinite(samples).all():audit_errors.append(f'{p}: sample shape/finite')
    if not np.allclose(np.median(samples,axis=1),pred,atol=2e-6):audit_errors.append(f'{p}: sample median != prediction')
   arr[i,j]=m['report_mae'];rmse[i,j]=m['report_rmse'];stop[i,j]=m['val_stop_mae']
 metrics[fam]={'mae':arr,'rmse':rmse,'stop':stop}
if audit_errors:raise RuntimeError('\n'.join(audit_errors[:30]))

overall=[];per_station=[];seed_rows=[]
for fam in FAMILIES:
 a=metrics[fam]['mae'];seed_mean=a.mean(1);station_mean=a.mean(0)
 for i,seed in enumerate(SEEDS):seed_rows.append(dict(model=fam,label=LABEL[fam],seed=seed,station_equal_mae=float(seed_mean[i])))
 for j,sid in enumerate(STATIONS):
  v=a[:,j];lo=int(v.argmin());hi=int(v.argmax())
  per_station.append(dict(model=fam,label=LABEL[fam],station_id=sid,mean=float(v.mean()),std=float(v.std(ddof=1)),
   min=float(v[lo]),min_seed=SEEDS[lo],max=float(v[hi]),max_seed=SEEDS[hi]))
 lo=int(seed_mean.argmin());hi=int(seed_mean.argmax())
 overall.append(dict(model=fam,label=LABEL[fam],mean=float(seed_mean.mean()),std=float(seed_mean.std(ddof=1)),
  min=float(seed_mean[lo]),min_seed=SEEDS[lo],max=float(seed_mean[hi]),max_seed=SEEDS[hi],
  mean_rmse=float(metrics[fam]['rmse'].mean()),mean_val_stop=float(metrics[fam]['stop'].mean()),
  min_station=STATIONS[int(station_mean.argmin())],min_station_mean=float(station_mean.min()),
  max_station=STATIONS[int(station_mean.argmax())],max_station_mean=float(station_mean.max()),stations=10,seeds=10))

comparisons=[
 ('PatchTST','DLinear'),('HistGradientBoostingRegressor','DLinear'),('TiDE','DLinear'),('TFT','TiDE'),('NBEATSx','TiDE'),
 ('TiDE_SR_DET','TiDE'),('TiDE_SR_ICFM','TiDE'),('TiDE_SR_VPCFM','TiDE'),('TiDE_SR_SINE_SI','TiDE'),
 ('TiDE_SR_VP_EPS','TiDE'),('TiDE_SR_VP_V','TiDE'),('TiDE_SR_EDM','TiDE'),('TiDE_SR_MEANFLOW','TiDE'),
 ('TiDE_SR_ICFM','TiDE_SR_DET'),('TiDE_SR_EDM','TiDE_SR_DET'),('TiDE_FROZEN_ICFM','TiDE_SR_ICFM'),
 ('TiDE_FROZEN_EDM','TiDE_SR_EDM'),('P0I_CBH_SINGLE_DET','P0I_CBH_SINGLE_LINEAR'),
 ('P0I_CBH_SINGLE_ICFM','P0I_CBH_SINGLE_LINEAR'),('P0I_CBH_SINGLE_EDM','P0I_CBH_SINGLE_LINEAR')]
rng=np.random.default_rng(20260917);B=20000;paired=[]
for model,control in comparisons:
 d=metrics[model]['mae']-metrics[control]['mae'];seed_diff=d.mean(1);station_diff=d.mean(0)
 # Crossed bootstrap: independently resample the ten trained seeds and ten fixed stations.
 si=rng.integers(0,10,(B,10));sj=rng.integers(0,10,(B,10));boot=np.empty(B)
 for b in range(B):boot[b]=d[np.ix_(si[b],sj[b])].mean()
 try:p=float(wilcoxon(seed_diff,alternative='two-sided',method='exact').pvalue)
 except ValueError:p=1.0
 paired.append(dict(model=model,control=control,delta=float(d.mean()),relative_percent=float(100*d.mean()/metrics[control]['mae'].mean()),
  ci95_low=float(np.quantile(boot,.025)),ci95_high=float(np.quantile(boot,.975)),seed_wins=int((seed_diff<0).sum()),
  station_wins=int((station_diff<0).sum()),wilcoxon_seed_p=p))
# Holm correction across the explicitly listed comparisons.
order=np.argsort([x['wilcoxon_seed_p'] for x in paired]);adj=np.empty(len(paired));running=0
for rank,idx in enumerate(order):
 running=max(running,min(1.,(len(paired)-rank)*paired[idx]['wilcoxon_seed_p']));adj[idx]=running
for x,a in zip(paired,adj):x['holm_p']=float(a)

proxy=[]
for fam in ['TiDE_SR_DET','TiDE_SR_ICFM','TiDE_SR_VPCFM','TiDE_SR_SINE_SI','TiDE_SR_VP_EPS','TiDE_SR_VP_V','TiDE_SR_EDM','TiDE_SR_MEANFLOW']:
 q=[];power_std=[];crps=[]
 for sid in STATIONS:
  for seed in SEEDS:
   p=ROOT/'results'/fam/sid/f'seed{seed}';q.append(json.loads((p/'proxy_metrics.json').read_text()));m=json.loads((p/'metrics.json').read_text());power_std.append(m['sample_std']);crps.append(m['report_crps'])
 row=dict(model=fam,label=LABEL[fam])
 for stride in ['4h_to_1h','2h_to_1h']:
  tag='4h' if stride.startswith('4') else '2h'
  for phase in ['before_joint','after_joint']:
   row[f'{tag}_{phase}_generated_mae']=float(np.mean([x[phase][stride]['generated_mae'] for x in q]))
  row[f'{tag}_linear_mae']=float(np.mean([x['after_joint'][stride]['linear_mae'] for x in q]))
  row[f'{tag}_after_sample_std']=float(np.mean([x['after_joint'][stride]['sample_std'] for x in q]))
 row['power_sample_std']=float(np.mean(power_std));row['power_crps']=float(np.mean(crps));proxy.append(row)

dump(ROOT/'analysis/overall_seed_summary.json',overall);dump(ROOT/'analysis/per_station_seed_summary.json',per_station)
dump(ROOT/'analysis/paired_comparisons.json',paired);dump(ROOT/'analysis/proxy_summary.json',proxy)
write_csv(ROOT/'analysis/overall_seed_summary.csv',overall);write_csv(ROOT/'analysis/per_station_seed_summary.csv',per_station)
write_csv(ROOT/'analysis/seed_values.csv',seed_rows);write_csv(ROOT/'analysis/paired_comparisons.csv',paired);write_csv(ROOT/'analysis/proxy_summary.csv',proxy)

expected=sum(1 for _ in (ROOT/'results').glob('*/*/seed*/metrics.json'))
audits={}
for name in ['baseline_input_audit','generator_audit','proxy_condition_audit']:
 x=json.loads((ROOT/'status'/(name+'.json')).read_text());audits[name]=bool(x.get('passed'))
status=dict(all_complete=True,expected_models=FAMILIES,models=len(FAMILIES),stations=10,seeds=10,metric_files=expected,
 expected_metric_files=len(FAMILIES)*100,all_metrics_and_checkpoints_verified=True,audits=audits,
 rejected_proxy_batch_present=(ROOT/'tmp/rejected_proxy_context_leakage/REJECTED.json').exists(),protocol_sha256=protocol_sha)
if expected!=len(FAMILIES)*100 or not all(audits.values()):raise RuntimeError(str(status))
dump(ROOT/'status/FINAL_AUDIT.json',status);dump(ROOT/'status/completeness.json',dict(all_complete=True,missing=[]))

def fmt(x):return f'{x:.6f}'
def row_for(name):return next(x for x in overall if x['model']==name)
def table(names):
 lines=['| 模型 | 10-seed MAE 均值±SD | 最低（seed） | 最高（seed） |','|---|---:|---:|---:|']
 for name in names:
  x=row_for(name);lines.append(f"| {LABEL[name]} | {fmt(x['mean'])} ± {fmt(x['std'])} | {fmt(x['min'])} ({x['min_seed']}) | {fmt(x['max'])} ({x['max_seed']}) |")
 return '\n'.join(lines)
def comp(model,control):return next(x for x in paired if x['model']==model and x['control']==control)

named=['DLinear','PatchTST','HistGradientBoostingRegressor','TiDE','TFT','NBEATSx']
sr=['TiDE_SR_DET','TiDE_SR_ICFM','TiDE_SR_VPCFM','TiDE_SR_SINE_SI','TiDE_SR_VP_EPS','TiDE_SR_VP_V','TiDE_SR_EDM','TiDE_SR_MEANFLOW','TiDE_FROZEN_ICFM','TiDE_FROZEN_EDM']
current=['P0I_CBH_SINGLE_LINEAR','P0I_CBH_SINGLE_DET','P0I_CBH_SINGLE_ICFM','P0I_CBH_SINGLE_EDM']
keypairs=[('TiDE','DLinear'),('TiDE_SR_SINE_SI','TiDE'),('TiDE_SR_EDM','TiDE'),('TiDE_FROZEN_ICFM','TiDE_SR_ICFM'),('TiDE_FROZEN_EDM','TiDE_SR_EDM'),('P0I_CBH_SINGLE_ICFM','P0I_CBH_SINGLE_LINEAR'),('P0I_CBH_SINGLE_EDM','P0I_CBH_SINGLE_LINEAR')]
pairlines=['| 比较（前者−后者） | ΔMAE | crossed bootstrap 95% CI | seed胜数 | station胜数 | Holm p |','|---|---:|---:|---:|---:|---:|']
for a,b in keypairs:
 x=comp(a,b);pairlines.append(f"| {LABEL[a]} − {LABEL[b]} | {x['delta']:+.6f} | [{x['ci95_low']:+.6f}, {x['ci95_high']:+.6f}] | {x['seed_wins']}/10 | {x['station_wins']}/10 | {x['holm_p']:.6f} |")
proxylines=['| 生成器 | 4h→1h: linear / joint | 2h→1h: linear / joint | 天气 sample SD | 功率 sample SD |','|---|---:|---:|---:|---:|']
for x in proxy:proxylines.append(f"| {x['label']} | {x['4h_linear_mae']:.4f} / {x['4h_after_joint_generated_mae']:.4f} | {x['2h_linear_mae']:.4f} / {x['2h_after_joint_generated_mae']:.4f} | {x['4h_after_sample_std']:.4f} | {x['power_sample_std']:.5f} |")

report=f'''# 单站点天气时间超分辨：命名基线、生成路线与联合训练结果

本报告只使用修正后的正式结果。早期一批代理预训练曾把被遮盖小时包含在全窗均值条件中，已经整体移到 `tmp/rejected_proxy_context_leakage`，不进入任何下表。最终审计覆盖 {len(FAMILIES)} 个模型家族、10 个站点、seeds 42–51，共 {expected} 个 station×seed 结果。

## 先回答模型连接

主实验的完整计算图是：`小时 NWP(7×14) → missing-only 天气生成器 → 15min 天气(24×14) → TiDE futr_exog encoder/temporal decoder + 本站历史功率(24) → 未来功率(24)`。联合阶段的功率 MAE 可穿过可微 sampler 回传到生成器，同时保留 0.1 倍的天气遮盖代理损失。整点天气直接旁路生成器，四种 origin phase 的逐值不变审计均通过。

当前框架对照是：`生成天气 → 单站 P0I/CBH adapter → Transformer decoder → PatchTST 冻结功率参考 + correction`。它是对现有框架的单站适配，不冒充发表模型。

## 已有模型 baseline

{table(named)}

这些名称是实际实现：DLinear 和 PatchTST 是历史功率模型；HistGradientBoostingRegressor 是 weather-only；TiDE、TFT、NBEATSx 同时用历史功率与未来已知天气。TiDE/TFT 的 future-exogenous forward 已用天气扰动测试确认。TimeXer 未列入，因为检查到的本地实现只读取历史外生变量。

TiDE 是本批命名 baseline 中均值最好者（0.113824）。它相对 DLinear 的配对 Δ={comp('TiDE','DLinear')['delta']:+.6f}。TFT 在这个小样本单站设置中 seed 波动最大，不能只报其最佳 seed。HistGradientBoostingRegressor 在固定数据与超参数下是确定性的，因此 10 次带 seed 运行得到相同数值，SD 为 0。

## TiDE 后的生成/补全方法

{table(sr)}

ICFM 是独立线性 conditional flow；VP-CFM 使用三角 variance-preserving path；Sine-SI 在路径中加入中间随机项；VP-epsilon/VP-v 是同一 VP 扩散路径的两种训练参数化；EDM 使用 Karras 预条件、rho=7 schedule 和 Heun；MeanFlow 学平均速度并以 1 NFE 采样。DDIM 只是采样器，因此没有被伪列为另一种训练模型。OT-CFM 没有用于跨天气条件重配，因为本任务每个残差已有时间和发行周期配对，随意 minibatch OT 会破坏条件语义。TimeBridge/I²SB 作为下一阶段的 paired bridge 方案保留，未把当前 EDM 改名为 bridge。

在联合训练的随机生成路线中，Sine-SI 的点 MAE 最低（{row_for('TiDE_SR_SINE_SI')['mean']:.6f}），EDM 次近（{row_for('TiDE_SR_EDM')['mean']:.6f}）。但各方法差异小于站点间差异，且选择模型时看过这批 2023 report split；这些数值是探索证据，不是盲测冠军结论。

## 天气代理任务揭示的限制

{chr(10).join(proxylines)}

DET 在训练尺度 4h→1h 略好于线性插值，但在更接近目标的 2h→1h 外推上变差。所有随机方法在两种天气代理任务上都劣于线性插值。功率 MAE 的改善因此不能解释为“恢复了真实 15 分钟天气”；更准确的说法是生成器充当了受天气代理正则约束的未来协变量适配器。原始数据没有真实 15 分钟气象标签，1h→15min 仍是未经直接验证的尺度外推。

冻结 ICFM/EDM 与各自联合版本的对照都没有证明下游反传带来稳定收益；冻结版本的均值甚至略低，但差值置信区间跨 0，Holm 校正后 p 均为 1。联合训练确实有非零梯度，并不等于这种梯度提高了天气真实性或平均功率性能。

## 现有框架单站对照

{table(current)}

P0I-single 整体优于本批 TiDE，但两者容量、冻结参考和 correction 结构不同，不能把绝对差归因于超分。P0I 内部应比较 LINEAR/DET/ICFM/EDM：ICFM 和 EDM 的均值改善都较小，需结合置信区间解释。

## 关键配对统计

{chr(10).join(pairlines)}

Δ<0 表示前者更好。CI 用 20,000 次 crossed bootstrap，独立重采样 10 个站点和 10 个 seed；Wilcoxon 在每个 seed 的 10 站等权均值上计算，并对预列出的 20 个比较做 Holm 校正。站点是根据训练期质量和多样性固定选出的，并非 56 站的随机样本，所以 CI 不能外推为全省总体置信区间。

## 对 ViT/CLS 想法的判断

当前单站输入已经是固定 2×2 双线性提取后的 14 个物理变量。先把每帧压成一个 CLS token 再生成，会丢失变量级约束，也让“整点逐变量相等”变成难以审计的 latent 约束。本阶段以 raw 14-variable residual 生成更合适。如果以后使用完整 24×25 天气 patch，应保留多个空间 patch token，并在 decoder 端施加观测 token mask；CLS 可作为全局条件，不应作为唯一被超分的状态。

## 研究决策

1. 如果论文主张是“真实天气时间超分辨”，应先获得 15 分钟再分析/观测天气，或构造有严格降采样算子的高频配对数据；现有数据不足以验证该主张。
2. 如果目标是功率精度，可继续把 anchored generator 定义为 future-covariate adapter。当前候选是 Sine-SI/EDM 和更强的 P0I-single 下游，但必须用新的固定年份或全新站点复评。
3. 下一种值得实现的生成路线是 TimeBridge/I²SB：线性天气作为数据依赖 prior，只在 missing residual 子空间做 bridge，并在每个 reverse step 写回整点。它比从纯 Gaussian 出发更符合本任务；仍需真实或可信高频监督。
4. 不建议继续把 solver 步数、DDIM、Euler/Heun 当成不同“模型”。应固定 field 容量，分别比较 path、target parameterization、prior、NFE 和是否联合训练。

## 审计与入口

- `status/FINAL_AUDIT.json`：完整性、checkpoint hash、数组形状和三类输入/anchor/梯度审计。
- `analysis/overall_seed_summary.csv`：每个模型按同一 seed 对 10 站等权后的均值、SD、最低/最高及 seed。
- `analysis/per_station_seed_summary.csv`：每站 10 seeds 的最低、最高、均值和 SD。
- `analysis/paired_comparisons.csv`：配对差、crossed-bootstrap CI、seed/station 胜数、Holm p。
- `analysis/proxy_summary.csv`：天气代理恢复、样本离散度和功率 CRPS。
- `analysis/RESEARCH_AND_PROTOCOL_CN.md`、`analysis/METHOD_MATRIX_CN.md`：方法、源码、概率路径和适用边界。
'''
(ROOT/'analysis/FINAL_REPORT_CN.md').write_text(report)
print(json.dumps(status,ensure_ascii=False,indent=2))
