"""Persist both concurrent queues' measured throughput and resource decisions."""
from pathlib import Path
import json,time,datetime,collections,os
P=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
NAMES=['daily_protocol_restart_20261114_v1','daily_site_finetune_20261114_v1','daily_ablation_restart_20261114_v1']

def snapshot():
 result=dict(timestamp=datetime.datetime.now().astimezone().isoformat(),runs={},cards=[],active=0,failed=[],completed=0)
 for name in NAMES:
  r=P/'runs'/name;p=r/'STATUS.json'
  if not p.exists():continue
  s=json.loads(p.read_text());a=s['active'] if not (r/'COMPLETE.json').exists() else []
  result['runs'][name]=dict(timestamp=s['timestamp'],completed=s['completed'],planned=s['planned'],active=len(a),pending=s['pending'],failed=s['failed'],errors=s.get('errors',{}))
  result['completed']+=s['completed'];result['active']+=len(a);result['failed'] += [name+'/'+j for j in s['failed']]
  states=[(h,v['status']) for h,v in s.get('hosts',{}).items()]
  local=r/'schedulers/status.json'
  if local.exists():states.append(('local',json.loads(local.read_text())))
  for host,st in states:
   counts=collections.Counter(x['gpu'] for x in st.get('active',[]))
   for gpu,d in st.get('decisions',{}).items():
    e=d.get('efficiency',{});g=st.get('gpus',{}).get(gpu,{})
    result['cards'].append(dict(run=name,host=host,gpu=gpu,workers=counts[gpu],utilization=g.get('utilization'),free_mib=g.get('free_mib'),reason=d['reason'],slowdown=e.get('slowdown_ratio'),incremental_slowdown=e.get('paired_incremental_slowdown'),aggregate_gain=e.get('estimated_aggregate_gain'),step_seconds=e.get('per_job_step_seconds',{}),cap=e.get('cap'),waiting=e.get('waiting_for_step_calibration')))
 return result
if __name__=='__main__':
 for i in range(120):
  try:
   s=snapshot();(OUT/'LIVE.json').write_text(json.dumps(s,ensure_ascii=False,indent=2)+'\n')
   with (OUT/'measurements.jsonl').open('a') as f:f.write(json.dumps(s,ensure_ascii=False)+'\n')
   print(s['timestamp'],'active',s['active'],'completed',s['completed'],'failed',s['failed'],'cards',[(x['host'],x['gpu'],x['workers'],x['slowdown'],x['aggregate_gain']) for x in s['cards'] if x['workers']],flush=True)
  except Exception as exc:print(repr(exc),flush=True)
  time.sleep(20)
