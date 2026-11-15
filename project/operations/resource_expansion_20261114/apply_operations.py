from pathlib import Path
import json,hashlib,shutil,datetime
P=Path(__file__).resolve().parents[2];R=P/'operations/resource_expansion_20261114';names=['daily_protocol_restart_20261114_v1','daily_site_finetune_20261114_v1','daily_ablation_restart_20261114_v1'];changes=[]
def edit(p,value):
    old=p.read_bytes() if p.exists() else None
    if old==value.encode():return
    backup=R/'before'/p.relative_to(P);backup.parent.mkdir(parents=True,exist_ok=True)
    if old is not None:backup.write_bytes(old)
    p.write_text(value);changes.append(dict(path=str(p.relative_to(P)),before=hashlib.sha256(old).hexdigest() if old else None,after=hashlib.sha256(p.read_bytes()).hexdigest()))
def js(p,x):edit(p,json.dumps(x,indent=2,ensure_ascii=False)+'\n')
for name in names:
    r=P/'runs'/name;op=r/'operations'
    for dst,src in [('efficiency_v3.py','efficiency_v4.py'),('gpu_policy.py','gpu_policy_v4.py')]:edit(op/dst,(R/src).read_text())
    for fn,ceiling in [('LOCAL_ADMISSION.json',6),('REMOTE_ADMISSION.json',10)]:
        s=json.loads((op/fn).read_text());s['max_workers_per_gpu']=ceiling;js(op/fn,s)
    pools=dict(local=list(range(8)),v1006001=list(range(1,8)),v1006002=[0,1,3,4,5]) if name!=names[2] else dict(local=[],v1006001=[0],v1006002=[2,6,7])
    js(op/'RESOURCE_POOLS.json',pools)
    p=op/'scheduler.py';s=p.read_text()
    s=s.replace("def run(gpu_ids,remote=False):", "def run(gpu_ids,remote=False):\n    poolfile=RUN/'operations/RESOURCE_POOLS.json'\n    hostkey={'iipserver1':'v1006001','iipserver2':'v1006002'}.get(socket.gethostname(),'local')\n    if poolfile.exists():gpu_ids=[str(g) for g in json.loads(poolfile.read_text())[hostkey]]\n    if not gpu_ids:return")
    s=s.replace("ordered_gpus=sorted(gpu_ids,key=lambda g:sum(x['gpu']==g for x in running.values())) if not remote else gpu_ids", "ordered_gpus=sorted(gpu_ids,key=lambda g:sum(x['gpu']==g for x in running.values()))")
    s=s.replace("if len(items)==1 and not g['other_pids'] and not starting and decision['reason']=='compute_target_reached' and decision.get('additional_memory_slots',0)>0 and time.monotonic()-last[gpu]>25:","if 1<=len(items)<ceiling and not g['other_pids'] and not starting and decision['reason']=='compute_target_reached' and decision.get('additional_memory_slots',0)>0 and time.monotonic()-last[gpu]>60:")
    s=s.replace("reason='two_worker_throughput_trial'","reason='bounded_additional_worker_trial'")
    # Configured width128 peaks are measured under2GiB; retain >=4GiB card reserve
    # plus actual observed process memory. Avoid denying a3GiB idle remainder.
    s=s.replace("4096 if cfg['width']==128 else 2560","3072 if cfg['width']==128 else 2048")
    edit(p,s)
    p=op/'supervise.py';s=p.read_text();old="args=['--gpus','0,1,2,3,4,5,6,7']+([] if host=='local' else ['--remote']) if name=='scheduler' else []"
    new="pools=json.loads((RUN/'operations/RESOURCE_POOLS.json').read_text())\n                ids=pools[host]\n                if name=='scheduler' and not ids:continue\n                args=['--gpus',','.join(map(str,ids))]+([] if host=='local' else ['--remote']) if name=='scheduler' else []"
    assert old in s,name;edit(p,s.replace(old,new))
# Dispatch the existing scientific matrix as dependencies become ready.
r=P/'runs'/names[2];p=r/'operations/collect.py';s=p.read_text();s=s.replace('def collect():','''def collect():
    # A source config is published only after all of its tune seeds finish.
    source=PROJECT/'runs/daily_protocol_restart_20261114_v1'
    for selected in (source/'assets/selected').glob('*.json'):
        dest=ASSETS/'selected'/selected.name
        if dest.exists():assert sha(dest)==sha(selected), 'Source selection changed'
        else:atomic_json(dest,json.loads(selected.read_text()))
''')
# atomic_json reformats source JSON identically, but preserve exact bytes for its provenance hash.
s=s.replace("else:atomic_json(dest,json.loads(selected.read_text()))", "else:\n            import shutil\n            dest.parent.mkdir(parents=True,exist_ok=True);tmp=dest.with_suffix('.copy.tmp');shutil.copy2(selected,tmp);os.replace(tmp,dest)")
edit(p,s)
p=r/'operations/deploy.py';s=p.read_text()
start="    for dep in [source,PROJECT/'runs/daily_site_finetune_20261114_v1']:\n        assert (dep/'COMPLETE.json').exists(),str(dep)\n        assert json.loads((dep/'PROTOCOL.json').read_text())['preprocess_fingerprint']==json.loads((RUN/'PROTOCOL.json').read_text())['preprocess_fingerprint']"
replace="    assert (source/'data/READY.json').exists()\n    assert json.loads((source/'PROTOCOL.json').read_text())['preprocess_fingerprint']==json.loads((RUN/'PROTOCOL.json').read_text())['preprocess_fingerprint']"
assert start in s;s=s.replace(start,replace)
s=s.replace("for family in ['W0_R3','TF_BASE','TF_R3','TIDE_R3','TFT_R3']:shutil.copy2(source/f'assets/selected/{family}.json',selected/f'{family}.json')", "for family in ['W0_R3','TF_BASE','TF_R3','TIDE_R3','TFT_R3']:\n        p=source/f'assets/selected/{family}.json'\n        if p.exists():shutil.copy2(p,selected/f'{family}.json')")
s=s.replace("for seed in range(42,47):\n p=r/'data'/f'reference_r{{seed}}';meta=json.loads((p/'READY.json').read_text())", "for marker in (r/'data').glob('reference_r*/READY.json'):\n p=marker.parent;meta=json.loads(marker.read_text())")
s=s.replace("'LOCAL_ADMISSION.json','REMOTE_ADMISSION.json']", "'LOCAL_ADMISSION.json','REMOTE_ADMISSION.json','RESOURCE_POOLS.json']")
s=s.replace("choices=[x for x in choices if x[1]>8192 and x[2]<80]", "pools=json.loads((r/'operations/RESOURCE_POOLS.json').read_text());allowed=pools[{host!r}]\nchoices=[x for x in choices if x[0] in allowed and x[1]>4096 and x[2]<50]")
# selected configs arrive causally during training; record their hash in each resolved spec.
s=s.replace("p.is_file() and '__pycache__' not in p.parts", "p.is_file() and '__pycache__' not in p.parts and 'selected' not in p.parts")
s=s.replace("'DATA_FREEZE.json','EXECUTION_PLAN_CN.md']]", "'DATA_FREEZE.json','EXECUTION_PLAN_CN.md']]+[RUN/'operations'/n for n in ['LOCAL_ADMISSION.json','REMOTE_ADMISSION.json','RESOURCE_POOLS.json']]")
edit(p,s)
p=r/'operations/watch.py';s=p.read_text().replace("['daily_protocol_restart_20261114_v1','daily_site_finetune_20261114_v1']","['daily_protocol_restart_20261114_v1']").replace("if not (p/'COMPLETE.json').exists()","if not (p/'data/READY.json').exists()").replace("stage='waiting_for_prior_stages'","stage='waiting_for_verified_data'");edit(p,s)
# The future fine-tune deployment must copy the resource-pool configuration too.
p=P/'runs'/names[1]/'operations/deploy.py';s=p.read_text().replace("'LOCAL_ADMISSION.json','REMOTE_ADMISSION.json']", "'LOCAL_ADMISSION.json','REMOTE_ADMISSION.json','RESOURCE_POOLS.json']");edit(p,s)
# Update ONLY operational activation metadata; all per-job scientific specs remain byte-identical.
p=r/'PROTOCOL.json';v=json.loads(p.read_text());v['dependencies']=['verified daily data','per-family selected config when available','per-seed OOF reference when required'];v['dispatch']='concurrent with shared training and personalization on disjoint GPU pools; per-job dependencies unchanged';js(p,v)
js(R/'REVISION.json',dict(timestamp=datetime.datetime.now().astimezone().isoformat(),reason='User requested higher concurrency with measured contention limits',changes=changes,model_training_data_or_manifest_changed=False,activation='per-job dependencies instead of whole-run barriers'))
print('operational files changed',len(changes))
