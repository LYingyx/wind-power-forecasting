"""Calibrate per-update throughput, not full long shared-training epochs."""
import json,time,datetime,statistics
from pathlib import Path
class EfficiencyGovernor:
    def __init__(self,jobs,ceiling=6):self.jobs=Path(jobs);self.ceiling=ceiling;self.cards={}
    def control(self,gpu,items,decision):
        now=time.time();count=len(items);s=self.cards.setdefault(gpu,dict(count=0,changed=now,cap=self.ceiling,until=0,baseline={},seen={},bad=0,fingerprint=None))
        if count!=s['count']:s.update(count=count,changed=now,bad=0,fingerprint=None,seen={})
        if now>s['until'] and s['cap']<self.ceiling:s['cap']+=1;s['until']=now+600
        ratios=[];fresh=[];waiting=False
        for jid,item in items:
            try:r=json.loads((self.jobs/jid/'progress.json').read_text())
            except (OSError,ValueError):waiting=True;continue
            stamp=datetime.datetime.fromisoformat(r['timestamp']).timestamp()
            if r['pid']!=item['p'].pid or stamp<s['changed']+10:waiting=True;continue
            seen=s['seen'].setdefault(jid,set());seen.add(r['completed_steps']);waiting|=len(seen)<2
            cost=r['seconds_per_step'];fresh.append((jid,r['completed_steps']))
            prior=s['baseline'].get(jid)
            if prior and prior['count']<count:ratios.append(cost/prior['cost'])
            if prior is None or count<prior['count'] or (prior['count']==count and cost<prior['cost']):s['baseline'][jid]=dict(cost=cost,count=count)
        ratio=statistics.median(ratios) if ratios else None
        if tuple(fresh)!=s['fingerprint']:
            s['bad']=s['bad']+1 if ratio is not None and ratio>1.5 else 0;s['fingerprint']=tuple(fresh)
        decision['efficiency']=dict(cap=s['cap'],workers=count,slowdown_ratio=ratio,max_slowdown=1.5,waiting_for_step_calibration=waiting)
        if decision.get('yield_count'):return decision
        if count>1 and s['bad']>=2:
            s.update(cap=count-1,until=now+600,bad=0);return dict(decision,launch=False,yield_count=1,reason='per_process_slowdown_above_1p5')
        if count>=s['cap']:return dict(decision,launch=False,reason='efficiency_cap')
        if count and (waiting or now-s['changed']<30):return dict(decision,launch=False,reason='measuring_step_throughput')
        return decision
