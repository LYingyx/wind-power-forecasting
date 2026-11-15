from pathlib import Path
import json,collections,hashlib,datetime
R=Path(__file__).resolve().parent;P=R.parents[1]
s=json.loads((R/'LIVE.json').read_text());hosts=collections.Counter();counts=collections.defaultdict(dict)
for c in s['cards']:
 hosts[c['host']]+=c['workers']
 if c['workers']:counts[c['host']][c['gpu']]=c['workers']
rows=[]
for h in ['v1006001','v1006002','local']:rows.append(f'|{h}|{hosts[h]}|'+', '.join(f'GPU{g}: {n}' for g,n in sorted(counts[h].items()))+'|')
text=f'''# 2026-11-14 同卡并发扩容与实测

最新快照：{s['timestamp']}。**当前 {s['active']} 个正式训练进程，失败 {len(s['failed'])} 项**。本次检查起点22:20为22并发（本机10、V1006001为9、V1006002为3）；期间调参完成会自然解锁任务，因此进程数增长不能全部归因于调度修改。

|机器|实际工作进程|逐卡并发|
|---|---:|---|
'''+ '\n'.join(rows)+'''

## 具体修改

1. 解除补充1291项清单“等共享180项和微调1100项全部结束”的调度屏障。现在每项只等待自己的新协议调参/OOF参考，未改变实验清单、种子、输入、训练预算、模型代码或数据。
2. 同时运行的两个调度器使用不同GPU池。共享及后续微调：本机0–7、V1006001的1–7、V1006002的0/1/3/4/5；补充消融：V1006001的0、V1006002的2/6/7。可用性仍按实际外部负载和显存检查，池中有卡不代表立即占用。
3. 修复“显存已分配但整卡持续空闲、pmon没有进程SM读数”导致永不准入：首个试运行要求连续三次整卡利用率不高于10%，且满足显存预留；其他进程真实活动出现后会让本轮任务退让。
4. 由历史单次最快step改为最近多个独立窗口的中位数，过滤末尾过短窗口。每次增一个进程，至少45秒校准；单次写盘/检查点尖峰不触发错误降并发。
5. 每卡上限提高到V100 **10**、4090 **6**，实际并发由显存、CPU/RAM和实测吞吐决定。只使用自身进程的卡，即使GPU利用率读数高，也可进行一次受控扩容试验，避免用“GPU忙”直接判定多进程无收益。
6. 扩容保护：相比此前较低并发，持续每进程降速超过50%，或本次增进程使原进程持续降速超过35%，或估计总吞吐增幅不足2%，三个新测量窗口后自动撤回一个本轮进程。回退冷却180秒；始终预留至少max(4GiB,20%)显存。短期试验阶段可能暂时越过降速阈值，需看后续是否回退，不能把单个瞬时比值称为稳态。

“估计总吞吐增幅”用同一在运行任务的扩容前后更新耗时配对推算；混合模型时是调度近似量，不是完整实验完成率。完整epoch包含验证耗时，后续ETA应使用新的并行阶段日志估算。

## 已完成验证

- [资源策略检查](CHECKS.json)：闲置占显存、外部忙碌、显存不足、外部恢复计算、单次耗时尖峰、持续降速回退。
- 两台V100分别完成补充队列GPU数值检查和数据/代码哈希核验，正式任务已经产生新协议progress记录。
- 共享训练的三个旧调度器通过正常信号请求保存模型/optimizer/RNG检查点后退出；新调度器从断点恢复，没有强杀工作进程。
- 共享主实验与补充消融的模型/数据代码、每项manifest保持原哈希；仅调度、准入、阶段激活元信息和对应冻结记录修订。
- 旧串行ETA已明确标记过时；不能按进程数增加比例直接缩短原ETA。

## 证据与维护入口

- [实时两队列资源快照](LIVE.json)、[连续测量](measurements.jsonl)、[监视日志](monitor.log)。监视脚本每20秒记录一次，持续40分钟；常驻调度器之后仍持续根据吞吐调节。
- [操作修改及哈希](REVISION.json)、[断点恢复记录](RESTART.json)、[修改前状态](BEFORE_STATUS.json)。
- [共享当前状态](../../runs/daily_protocol_restart_20261114_v1/STATUS.json)、[消融当前状态](../../runs/daily_ablation_restart_20261114_v1/STATUS.json)、[补充队列监督状态](../../runs/daily_ablation_restart_20261114_v1/operations/QUEUE_STATUS.json)。
- [多轮实验入口](../../ACTIVE_EXPERIMENTS.json)。微调仍等待其新协议共享底座完成，之后与补充消融在各自GPU池并行。
'''
(R/'REPORT_CN.md').write_text(text)
# Compare only scientifically meaningful files to the original frozen entry.
checks=[]
for name,oldfile,nowfile in [
 ('daily_protocol_restart_20261114_v1','daily_protocol_restart_20261114_v1_SCIENCE_FREEZE.json.before','SCIENCE_FREEZE.json'),
 ('daily_ablation_restart_20261114_v1','daily_ablation_restart_20261114_v1_PLAN_FREEZE.json.before','SCIENCE_FREEZE.json')]:
 run=P/'runs'/name;old=json.loads((R/oldfile).read_text())['files'];latest=json.loads((run/nowfile).read_text())['files'];scientific=[f for f in old if f.startswith(('code/','data/','assets/')) or f=='manifest.json']
 for f in scientific:assert hashlib.sha256((run/f).read_bytes()).hexdigest()==old[f],f
 for f,h in latest.items():assert hashlib.sha256((run/f).read_bytes()).hexdigest()==h,f
 checks.append(dict(run=name,scientific_files_unchanged=len(scientific),current_freeze_verified=len(latest)))
(R/'VERIFIED.json').write_text(json.dumps(dict(timestamp=datetime.datetime.now().astimezone().isoformat(),passed=True,active=s['active'],failures=s['failed'],runs=checks),ensure_ascii=False,indent=2)+'\n')
print('report',s['timestamp'],'active',s['active'],'hosts',dict(hosts),'science verified')
