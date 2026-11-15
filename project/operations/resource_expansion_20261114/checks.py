from pathlib import Path
import importlib.util,tempfile,json,datetime,types,os
ROOT=Path(__file__).resolve().parent

def module(name,file):
 import sys
 s=importlib.util.spec_from_file_location(name,file);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
policy=module('resource_policy_v4',ROOT/'gpu_policy_v4.py');governor=module('resource_efficiency_v4',ROOT/'efficiency_v4.py')
base=dict(total_mib=16384,free_mib=14500,utilization=0,other_pids=[99],external_sm=0,process_utilization_known=False)
def decide(**kw):
 b=dict(base,**kw);return policy.decide([b]*3,0,0,100,2048)
assert decide()['launch']
assert not decide(utilization=60)['launch']
assert not decide(free_mib=4000)['launch']
assert not decide(external_sm=60,process_utilization_known=True)['launch']
x=policy.decide([dict(base,utilization=98)]*3,2,0,100,2048);assert not x['yield_count']
x=policy.decide([dict(base,external_sm=65,process_utilization_known=True)]*3,2,0,100,2048);assert x['yield_count']==2
x=policy.decide([dict(base,free_mib=3000)]*3,2,0,100,2048);assert x['yield_count']>0 and not x['launch']
clock=[1000.];governor.time=types.SimpleNamespace(time=lambda:clock[0])
with tempfile.TemporaryDirectory() as root:
 g=governor.EfficiencyGovernor(root,8);jobs=[('a',dict(p=types.SimpleNamespace(pid=1)))]
 def step(costs,delta=15):
  clock[0]+=delta
  for i,(j,item) in enumerate(jobs):
   p=Path(root)/j;p.mkdir(exist_ok=True);(p/'progress.json').write_text(json.dumps(dict(pid=item['p'].pid,timestamp=datetime.datetime.fromtimestamp(clock[0],datetime.timezone.utc).isoformat(),epoch=1,completed_steps=int(clock[0]),seconds_per_step=costs[i],chunk_steps=25)))
  return g.control('0',jobs,dict(launch=True,yield_count=0,reason='spare_compute_and_memory'))
 for _ in range(5):r=step([1.])
 assert r['launch']
 jobs.append(('b',dict(p=types.SimpleNamespace(pid=2))))
 for costs in [[1.1,1.1],[1.1,1.1],[8.,1.1],[1.1,1.1],[1.1,1.1]]:
  r=step(costs);assert r.get('yield_count',0)==0,'a single checkpoint spike must not reduce concurrency'
 assert r['launch'] and r['efficiency']['estimated_aggregate_gain']>1.5
 yielded=False
 for _ in range(8):
  r=step([2.,2.]);yielded|=bool(r['yield_count'])
 assert yielded and r['efficiency']['cap']<=1,'sustained slowdown must roll back'
 # A short terminal progress chunk is not a valid throughput sample.
print('resource checks passed: idle allocation, busy/low-memory rejection, foreign activation, isolated spike, sustained regression')
(ROOT/'CHECKS.json').write_text(json.dumps(dict(timestamp=datetime.datetime.now().astimezone().isoformat(),passed=True,checks=['idle_allocated_first_trial','busy_unknown_block','preserved_memory_reserve','foreign_activity_yield','own_busy_no_false_foreign_yield','single_spike_no_backoff','sustained_slowdown_backoff','aggregate_throughput_gain']),indent=2)+'\n')
