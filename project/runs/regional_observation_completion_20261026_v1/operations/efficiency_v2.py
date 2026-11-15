"""Operational admission only: matched-job epoch timing, no change to training math."""
from pathlib import Path
import datetime,json,statistics,time

class EfficiencyGovernor:
    def __init__(self,jobs,ceiling=12,clock=time.time):
        self.jobs=Path(jobs);self.ceiling=ceiling;self.clock=clock;self.cards={}
        self.warmup=60.;self.slowdown_limit=1.50;self.cooldown=600.

    def control(self,gpu,items,decision):
        now=self.clock();ids=tuple(sorted(jid for jid,_ in items));count=len(ids)
        state=self.cards.setdefault(gpu,dict(ids=(),changed=now,cap=self.ceiling,
            until=0.,baseline={},bad=0,last_fingerprint=None))
        if ids!=state['ids']:
            state.update(ids=ids,changed=now,bad=0,last_fingerprint=None)
        if now>=state['until'] and state['cap']<self.ceiling:
            # At most one additional slot per ten-minute probation interval.
            state['cap']+=1;state['until']=now+self.cooldown
        eligible=[];ratios=[];fingerprint=[];waiting=False
        for jid,_ in items:
            path=self.jobs/jid/'history.json'
            if not path.exists():
                # Reference replays have no epoch loop; ordinary resource warmup applies.
                try:task=json.loads((self.jobs/jid/'spec.json').read_text())['task']
                except (OSError,ValueError,KeyError):task='train'
                waiting |= task=='train'
                continue
            try:history=json.loads(path.read_text())
            except (OSError,ValueError):waiting=True;continue
            rows=[r for r in history if r['epoch']>=2 and
                  datetime.datetime.fromisoformat(r['timestamp']).timestamp()>state['changed']+15]
            if len(rows)<3:waiting=True;continue
            recent=rows[-3:];cost=statistics.median(r['seconds'] for r in recent)
            if cost<=0:waiting=True;continue
            eligible.append(jid);fingerprint.append((jid,recent[-1]['epoch']))
            prior=state['baseline'].get(jid)
            if prior and prior['count']<count:
                ratios.append(cost/prior['seconds'])
            if prior is None or count<prior['count'] or (count==prior['count'] and cost<prior['seconds']):
                state['baseline'][jid]=dict(count=count,seconds=cost)
        ratio=statistics.median(ratios) if ratios else None
        fresh=tuple(fingerprint)!=state['last_fingerprint']
        if fresh:
            state['bad']=state['bad']+1 if ratio is not None and ratio>self.slowdown_limit else 0
            state['last_fingerprint']=tuple(fingerprint)
        info=dict(cap=state['cap'],matched_jobs=len(ratios),slowdown_ratio=ratio,
                  max_slowdown_ratio=self.slowdown_limit,stable_seconds=now-state['changed'],
                  calibration_jobs=len(eligible),waiting_for_epoch_calibration=waiting)
        decision['efficiency']=info
        if decision.get('yield_count',0):return decision
        if count>1 and state['bad']>=2:
            state['cap']=max(1,count-1);state['until']=now+self.cooldown;state['bad']=0
            info['cap']=state['cap']
            return dict(decision,launch=False,yield_count=1,reason='matched_job_epoch_slowdown')
        if count>state['cap']:
            return dict(decision,launch=False,yield_count=count-state['cap'],reason='efficiency_cap')
        if count>=state['cap']:
            return dict(decision,launch=False,reason='efficiency_cap')
        if count and (now-state['changed']<self.warmup or waiting):
            return dict(decision,launch=False,reason='measuring_worker_efficiency')
        return decision
