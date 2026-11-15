import os,json,hashlib,collections,time
from pathlib import Path
S=Path('/shared_data_ssd2/liying/new_try/wind_cluster_forced_evolution')
P=Path('/shared_data_ssd2/liying/wind_cluster_project_20261018/project')
T=P/'integration/new_try_merge_20261018'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
# Retain final predictions and best checkpoints; omit transient training artifacts and duplicate environments.
skipdirs={'.migration_env','__pycache__','.git','logs','tmp','smoke','val_smoke','benchmark','archive','gpu_locks','.cache'}
skipnames={'history.json','baseline_history.json','FAILED.json'}
counts=collections.Counter(); sizes=collections.Counter();conflicts=[];start=time.time()
with (T/'PLAN.jsonl').open('w') as out,(T/'EXCLUDED.jsonl').open('w') as exc:
 for root,dirs,files in os.walk(S):
  dirs[:]=sorted(d for d in dirs if d not in skipdirs)
  for name in sorted(files):
   src=Path(root)/name; rel=src.relative_to(S);st=src.stat()
   if name in skipnames or src.suffix in {'.log','.pyc','.pid','.lock'} or name.startswith('events.out.tfevents'):
    exc.write(json.dumps({'path':str(rel),'bytes':st.st_size,'reason':'transient_training_output'})+'\n');counts['excluded_files']+=1;sizes['excluded_files']+=st.st_size;continue
   dst=P/rel;row={'path':str(rel),'bytes':st.st_size,'mtime_ns':st.st_mtime_ns,'action':'new'}
   if dst.exists():
    row['sha256']=sha(src)
    old=sha(dst) if dst.stat().st_size==st.st_size else None
    if old==row['sha256'] and not dst.is_symlink():row['action']='identical'
    elif old==row['sha256'] and dst.is_symlink():row['action']='materialize_symlink'
    else:
     row['action']='replace_preserve_old';row['old_sha256']=old or sha(dst);conflicts.append(row)
   counts[row['action']]+=1;sizes[row['action']]+=st.st_size
   out.write(json.dumps(row)+'\n')
  if time.time()-start>30:
   print('progress',sum(counts.values()),dict(counts),flush=True);start=time.time()
summary={'source':str(S),'destination':str(P),'counts':dict(counts),'GiB':{k:round(v/2**30,4) for k,v in sizes.items()},'conflicts':conflicts,'excluded_directories':sorted(skipdirs)}
(T/'PLAN_SUMMARY.json').write_text(json.dumps(summary,indent=2))
print(json.dumps({k:v for k,v in summary.items() if k!='conflicts'},indent=2));print('conflicts',len(conflicts),collections.Counter(Path(x['path']).suffix for x in conflicts))
