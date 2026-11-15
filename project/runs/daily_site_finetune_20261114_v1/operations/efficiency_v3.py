"""Robust paired update-cost windows and bounded, reversible concurrency trials."""
import json,time,datetime,statistics
from pathlib import Path

class EfficiencyGovernor:
    def __init__(self,jobs,ceiling=8):
        self.jobs=Path(jobs);self.ceiling=ceiling;self.cards={}
    def control(self,gpu,items,decision):
        now=time.time();members=tuple(sorted((j,it['p'].pid) for j,it in items));count=len(items)
        s=self.cards.setdefault(gpu,dict(members=(),count=0,changed=now,cap=self.ceiling,until=0,windows={},baseline={},reference={},bad=0,fingerprint=None,trial_from=0))
        if members!=s['members']:
            previous=s['count'];reference={j:statistics.median(v[-5:]) for j,v in s['windows'].items() if len(v)>=3}
            s.update(members=members,count=count,changed=now,windows={},reference=reference if count>previous else {},trial_from=previous if count>previous else 0,bad=0,fingerprint=None)
        if now>s['until'] and s['cap']<self.ceiling:s['cap']+=1;s['until']=now+180
        paired=[];absolute=[];fingerprint=[];ready=True
        for jid,it in items:
            try:r=json.loads((self.jobs/jid/'progress.json').read_text())
            except (OSError,ValueError):ready=False;continue
            stamp=datetime.datetime.fromisoformat(r['timestamp']).timestamp()
            if r['pid']!=it['p'].pid or stamp<s['changed']+10 or now-stamp>120:ready=False;continue
            key=(jid,it['p'].pid,r['epoch'],r['completed_steps']);fingerprint.append(key)
            w=s['windows'].setdefault(jid,[]);seen=s.setdefault('seen',{})
            if seen.get(jid)!=key and r.get('chunk_steps',25)>=10:
                w.append(r['seconds_per_step']);del w[:-5];seen[jid]=key
            if len(w)<3:ready=False;continue
            cost=statistics.median(w);prior=s['baseline'].get(jid)
            if prior is None or count<prior['count']:s['baseline'][jid]=dict(cost=cost,count=count)
            elif count==prior['count']:prior['cost']=.8*prior['cost']+.2*cost
            prior=s['baseline'][jid]
            if count>prior['count']:absolute.append(cost/prior['cost'])
            if jid in s['reference']:paired.append(cost/s['reference'][jid])
        ratio=statistics.median(absolute) if absolute else None
        marginal=statistics.median(paired) if paired else None
        gain=count/s['trial_from']/marginal if marginal and s['trial_from'] else None
        bad=ready and ((ratio is not None and ratio>1.5) or (marginal is not None and marginal>1.35) or (gain is not None and gain<1.02))
        if tuple(fingerprint)!=s['fingerprint']:
            s['bad']=s['bad']+1 if bad else 0;s['fingerprint']=tuple(fingerprint)
        decision['efficiency']=dict(version=4,cap=s['cap'],workers=count,slowdown_ratio=ratio,paired_incremental_slowdown=marginal,estimated_aggregate_gain=gain,max_slowdown=1.5,max_incremental_slowdown=1.35,waiting_for_step_calibration=not ready,stable_seconds=round(now-s['changed'],1),per_job_step_seconds={j:statistics.median(v) for j,v in s['windows'].items() if len(v)>=3})
        if decision.get('yield_count'):return decision
        if count>1 and s['bad']>=3:
            s.update(cap=count-1,until=now+180,bad=0)
            return dict(decision,launch=False,yield_count=1,reason='sustained_paired_throughput_regression')
        if count>=s['cap']:return dict(decision,launch=False,reason='efficiency_cap')
        if count and (not ready or now-s['changed']<45):return dict(decision,launch=False,reason='measuring_robust_step_windows')
        return decision
