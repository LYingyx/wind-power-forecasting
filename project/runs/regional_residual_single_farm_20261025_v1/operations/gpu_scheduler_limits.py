"""Original frozen GPU scheduler with cgroup-aware RAM admission only."""
import sys,argparse
from resource_limits import effective_memory_available
import resource_scheduler as scheduler
from gpu_policy import Policy
scheduler.cpu_memory_available=effective_memory_available
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--max-workers-per-gpu',type=int,default=12);p.add_argument('--allow-shared-compute',action='store_true');p.add_argument('--warmup-seconds',type=int,default=20);a=p.parse_args()
 scheduler.run(Policy(max_workers_per_gpu=a.max_workers_per_gpu,allow_shared_compute=a.allow_shared_compute,warmup_seconds=a.warmup_seconds))
