"""Keep original GPU idle monitoring, enter the memory-aware scheduler on admission."""
from pathlib import Path
import os
import bootstrap_remote as bootstrap
from resource_limits import effective_memory_available
original=os.execv
original_decide=bootstrap.decide
def guarded_decide(*args,**kwargs):
    decision=original_decide(*args,**kwargs);free,total=effective_memory_available()
    if free<max(8192,total*.15)+3072:decision.update(launch=False,reason='host_or_cgroup_ram_reserve')
    return decision
bootstrap.decide=guarded_decide

def execute(python,args):
    args=[str(Path(__file__).with_name('gpu_scheduler_limits.py')) if x.endswith('/resource_scheduler.py') else x for x in args]
    original(python,args)
bootstrap.os.execv=execute
if __name__=='__main__':bootstrap.main()
