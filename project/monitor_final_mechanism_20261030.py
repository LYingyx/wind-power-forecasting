"""Keep one human-readable report current through training, freeze, test and audit."""
from pathlib import Path
import json,time,os,datetime,fcntl,statistics
P=Path(__file__).resolve().parent;R=P/'runs/final_mechanism_20261030_v1';T=P/'runs/final_mechanism_test_20261030_v1';REPORT=P/'RESULTS_AND_NEXT_EXPERIMENTS_20261030_CN.md'
MARK='\n## 自动更新：新一轮实际结果\n'
def now():return datetime.datetime.now().astimezone()
def read(p,default=None):
 try:return json.loads(p.read_text())
 except (OSError,ValueError):return default

def atomic(p,v):
 t=p.with_name(p.name+f'.{os.getpid()}.tmp');t.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n');os.replace(t,p)

def loop():
 lock=(R/'operations/report_monitor.lock').open('a+')
 try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 except BlockingIOError:return
 seen={};refs=set(read(R/'REUSED_REFERENCES.json',{}))
 while True:
  try:
   stamp=now();s=read(R/'STATUS.json',{});remote=read(R/'operations/REMOTE_STATUS.json',{});test=read(T/'operations/FOLLOWUP_STATUS.json',{})
   for f in (R/'jobs').glob('*/SUCCESS.json'):
    if f.parent.name in refs or f.parent.name in seen:continue
    m=read(f.parent/'metrics.json')
    if m:seen[f.parent.name]=dict(finished=m['finished'],seconds=m['seconds'],epochs=m['epochs'])
   times=[datetime.datetime.fromisoformat(x['finished']) for x in seen.values()];recent=sum((stamp-t).total_seconds()<=600 for t in times)
   elapsed=(stamp-min(times)).total_seconds() if times else 0
   window=min(600,max(elapsed,60));rate=recent/window*60
   left=3360-len(seen);estimate=left/rate if rate else None
   progress=dict(timestamp=stamp.isoformat(),new_training_completed=len(seen),new_training_total=3360,reused=1120,failed=s.get('failed'),recent_jobs_per_minute=rate,estimated_training_minutes_remaining=estimate,estimate_note='Throughput extrapolation only; excludes follow-up inference; adapts to external workloads',hosts={h:dict(active=len(d.get('active',[])),connected=d.get('connected'),error=d.get('error')) for h,d in remote.get('hosts',{}).items()},test_status=test)
   atomic(R/'operations/LIVE_PROGRESS.json',progress)
   lines=[MARK,f'更新：{stamp:%Y-%m-%d %H:%M:%S %Z}。新增训练{len(seen)}/3360，复用1120，失败{s.get("failed",0)}。2024状态：{test.get("state","等待")}。']
   if estimate is not None and left:lines.append(f'近段吞吐外推：训练约剩{estimate/60:.1f}小时（尚未计后续推理，随资源变化）。')
   for title,path in [('2023验证',R/'analysis/MECHANISM_RESULTS.json'),('2024探索性复评',T/'analysis/MECHANISM_2024_RESULTS.json')]:
    result=read(path,{})
    lines+=['',f'### {title}','','|骨干|种子数|完整配对风场|MAE交互/E11|RMSE交互/E11|MAE/RMSE正交互风场|判定|','|---|---:|---:|---:|---:|---:|---|']
    for x in result.get('analyses',[]):
     if x['complete']:
      a=x['measures']['all']['mae'];b=x['measures']['all']['rmse']
      lines.append(f'|{x["backbone"]}|{len(x["seeds"])}|56|{100*a["interaction_relative_to_E11"]:.3f}%|{100*b["interaction_relative_to_E11"]:.3f}%|{a["positive_farms"]}/{b["positive_farms"]}|{"通过" if x["passed"] else "未通过"}|')
     else:lines.append(f'|{x["backbone"]}|{len(x["seeds"])}|{x["paired_farms"]}|待齐|待齐|待齐|待齐|')
    if not result:lines.append('|—|—|—|等待|等待|—|—|')
   audit=read(T/'operations/COMPLETE_AUDIT.json',{});complete=audit.get('passed',False)
   if complete:
    v=read(R/'analysis/MECHANISM_RESULTS.json',{});q=read(T/'analysis/MECHANISM_2024_RESULTS.json',{})
    a={x['backbone']:x for x in v['analyses'] if len(x['seeds'])==10};b={x['backbone']:x for x in q['analyses']}
    lines+=['','最终判读：']
    for name in ['W0','TF']:
     if a[name]['passed'] and b[name]['passed']:lines.append(f'- {name}：最终模型的预测交互在2023和2024均通过预设门槛。')
     else:lines.append(f'- {name}：2023通过={a[name]["passed"]}，2024通过={b[name]["passed"]}；不能主张该骨干跨年一致成立。')
    lines+=['','全部3360项新训练、1120项参照及4480项2024推理已审计。结论仍限定在保留真实当前功率锚点与区域实况的预测交互；2024属于探索性复评。']
   text=REPORT.read_text().split(MARK)[0]+'\n'.join(lines)+'\n';temp=REPORT.with_suffix('.tmp');temp.write_text(text);os.replace(temp,REPORT)
   active=read(P/'ACTIVE_EXPERIMENT.json',{})
   if active.get('run')==str(R):
    active.update(stage='complete' if complete else 'test_evaluation' if test.get('state')=='RUNNING' else 'training',new_completed=len(seen),updated=stamp.isoformat(),summary=str(REPORT))
    atomic(P/'ACTIVE_EXPERIMENT.json',active)
   atomic(R/'operations/REPORT_MONITOR.json',dict(timestamp=stamp.isoformat(),pid=os.getpid(),complete=complete,report=str(REPORT)))
   if complete:return
   (R/'operations/REPORT_MONITOR_ERROR.json').unlink(missing_ok=True)
  except Exception as e:atomic(R/'operations/REPORT_MONITOR_ERROR.json',dict(timestamp=now().isoformat(),error=repr(e)))
  time.sleep(60)

if __name__=='__main__':loop()
