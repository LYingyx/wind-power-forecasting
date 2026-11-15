"""Admission decisions based on compute and memory, including idle allocated GPUs."""
from dataclasses import dataclass
import statistics

@dataclass(frozen=True)
class Policy:
    poll_seconds:int=10
    idle_samples:int=3
    max_workers_per_gpu:int=12
    external_compute_limit:int=20
    total_compute_target:int=92
    external_yield_threshold:int=40
    initial_worker_mib:int=2048
    minimum_reserve_mib:int=4096
    reserve_fraction:float=.20
    warmup_seconds:int=40
    yield_cooldown_seconds:int=120
    allow_shared_compute:bool=False


def decide(samples,own_count,starting_count,recent_launch_age,estimated_worker_mib,policy=Policy()):
    """Return an explanation as well as admission; thresholds concern resources only."""
    if len(samples)<policy.idle_samples:return dict(launch=False,yield_count=0,reason='collecting_resource_samples')
    sample=samples[-1]
    reserve=max(policy.minimum_reserve_mib,int(sample['total_mib']*policy.reserve_fraction))
    budget=max(1024,int(estimated_worker_mib))
    # Newly started processes may not yet appear in NVML; reserve their full budgets.
    usable=sample['free_mib']-reserve-starting_count*budget
    capacity=max(0,int(usable//budget))
    recent=samples[-policy.idle_samples:]
    foreign_present=any(s['other_pids'] for s in recent)
    unknown=any(s['other_pids'] and not s['process_utilization_known'] for s in recent)
    external=max(s['external_sm'] if s['process_utilization_known'] else s['utilization'] for s in recent) if foreign_present else 0
    utilization=statistics.mean(s['utilization'] for s in recent)
    base=dict(reserve_mib=reserve,worker_budget_mib=budget,additional_memory_slots=capacity,external_sm_peak=external,total_utilization_mean=utilization,yield_count=0)
    if sample['free_mib']<reserve:
        return dict(base,launch=False,**{'yield_count':min(own_count, max(1,int((reserve-sample['free_mib'])//budget)+1))},reason='release_our_memory_for_reserve') if own_count else dict(base,launch=False,reason='memory_reserve')
    # A genuine rise in another process' compute causes OUR jobs to checkpoint and release.
    if not policy.allow_shared_compute and own_count and foreign_present and not unknown and sample['external_sm']>policy.external_yield_threshold:
        base['yield_count']=own_count
        return dict(base,launch=False,reason='other_workload_became_active')
    if own_count>=policy.max_workers_per_gpu:return dict(base,launch=False,reason='worker_safety_cap')
    if capacity<1:return dict(base,launch=False,reason='insufficient_memory_headroom')
    if not policy.allow_shared_compute and foreign_present and unknown:
        return dict(base,launch=False,reason='cannot_attribute_other_process_compute')
    if not policy.allow_shared_compute and foreign_present and external>policy.external_compute_limit:
        return dict(base,launch=False,reason='other_workload_compute_busy')
    if starting_count or recent_launch_age<policy.warmup_seconds:
        return dict(base,launch=False,reason='waiting_for_worker_warmup')
    if utilization>=policy.total_compute_target:
        return dict(base,launch=False,reason='compute_target_reached')
    return dict(base,launch=True,reason='shared_compute_headroom' if policy.allow_shared_compute else 'spare_compute_and_memory')
