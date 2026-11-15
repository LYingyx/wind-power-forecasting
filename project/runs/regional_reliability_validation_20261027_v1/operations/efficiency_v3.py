"""Refill proven parallel slots; only expansion needs fresh timing calibration."""
from pathlib import Path
import datetime,json,statistics,time

class EfficiencyGovernor:
    def __init__(self,jobs,ceiling=12,clock=time.time):
        self.jobs=Path(jobs);self.ceiling=ceiling;self.clock=clock;self.cards={}
        self.warmup=20.;self.slowdown_limit=1.5;self.cooldown=600.

    def control(self,gpu,items,decision):
        now=self.clock();ids=tuple(sorted(jid for jid,_ in items));count=len(ids)
        state=self.cards.setdefault(gpu,dict(ids=(),count=0,changed=now,cap=self.ceiling,
            safe_slots=1,until=0.,baseline={},bad=0,last_fingerprint=None))
        if count!=state['count']:
            state.update(count=count,changed=now,bad=0,last_fingerprint=None)
        # A completed job/replacement is not a reason to forget proven concurrency.
        state['ids']=ids
        if now>=state['until'] and state['cap']<self.ceiling:
            state['cap']+=1;state['until']=now+self.cooldown
        eligible=[];ratios=[];fingerprint=[];waiting=False
        for jid,_ in items:
            path=self.jobs/jid/'history.json'
            if not path.exists():
                try:task=json.loads((self.jobs/jid/'spec.json').read_text())['task']
                except (OSError,ValueError,KeyError):task='train'
                waiting |= task=='train'
                continue
            try:history=json.loads(path.read_text())
            except (OSError,ValueError):waiting=True;continue
            rows=[r for r in history if r['epoch']>=2 and
                  datetime.datetime.fromisoformat(r['timestamp']).timestamp()>state['changed']+5]
            if len(rows)<2:waiting=True;continue
            recent=rows[-3:];cost=statistics.median(r['seconds'] for r in recent)
            if cost<=0:waiting=True;continue
            eligible.append(jid);fingerprint.append((jid,recent[-1]['epoch']))
            prior=state['baseline'].get(jid)
            if prior and prior['count']<count:ratios.append(cost/prior['seconds'])
            if prior is None or count<prior['count'] or (count==prior['count'] and cost<prior['seconds']):
                state['baseline'][jid]=dict(count=count,seconds=cost)
        ratio=statistics.median(ratios) if ratios else None
        fresh=tuple(fingerprint)!=state['last_fingerprint']
        if fresh:
            state['bad']=state['bad']+1 if ratio is not None and ratio>self.slowdown_limit else 0
            state['last_fingerprint']=tuple(fingerprint)
        info=dict(cap=state['cap'],safe_slots=state['safe_slots'],matched_jobs=len(ratios),
            slowdown_ratio=ratio,max_slowdown_ratio=self.slowdown_limit,
            stable_seconds=now-state['changed'],calibration_jobs=len(eligible),
            waiting_for_epoch_calibration=waiting)
        decision['efficiency']=info
        # Existing external-workload/memory releases always take priority.
        if decision.get('yield_count',0):return decision
        if count>1 and state['bad']>=2:
            state['cap']=max(1,count-1);state['safe_slots']=min(state['safe_slots'],state['cap'])
            state['until']=now+self.cooldown;state['bad']=0
            info.update(cap=state['cap'],safe_slots=state['safe_slots'])
            return dict(decision,launch=False,yield_count=1,reason='matched_job_epoch_slowdown')
        if count>state['cap']:
            return dict(decision,launch=False,yield_count=count-state['cap'],reason='efficiency_cap')
        if count>=state['cap']:return dict(decision,launch=False,reason='efficiency_cap')
        if not waiting and now-state['changed']>=self.warmup and (ratio is None or ratio<=self.slowdown_limit):
            state['safe_slots']=max(state['safe_slots'],min(count,state['cap']))
            info['safe_slots']=state['safe_slots']
        if count<state['safe_slots']:
            # Preserve every compute/VRAM/CPU/RAM veto from the outer scheduler.
            if decision['launch']:decision['reason']='refill_proven_parallel_slot'
            return decision
        if count and (now-state['changed']<self.warmup or waiting):
            return dict(decision,launch=False,reason='measuring_higher_concurrency_efficiency')
        return decision
