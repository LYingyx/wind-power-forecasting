import os,json,hashlib,time,shutil,collections
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
S=Path('/shared_data_ssd2/liying/new_try/wind_cluster_forced_evolution')
P=Path('/shared_data_ssd2/liying/wind_cluster_project_20261018/project')
T=P/'integration/new_try_merge_20261018'
B=P/'provenance/before_new_try_merge_20261018'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def transfer(row):
 rel=Path(row['path']);src=S/rel;dst=P/rel
 before=src.stat()
 if before.st_size!=row['bytes'] or before.st_mtime_ns!=row['mtime_ns']:raise RuntimeError(f'Source changed: {rel}')
 if row['action']=='identical':return row
 if row['action']=='replace_preserve_old':
  backup=B/rel;backup.parent.mkdir(parents=True,exist_ok=True)
  if not backup.exists():shutil.copy2(dst,backup)
  if sha(backup)!=row['old_sha256']:raise RuntimeError(f'Backup checksum mismatch: {rel}')
 elif row['action']=='materialize_symlink':row['previous_symlink']=os.readlink(dst)
 dst.parent.mkdir(parents=True,exist_ok=True)
 temporary=dst.with_name(dst.name+'.merge-incoming')
 h=hashlib.sha256()
 with src.open('rb') as inp,temporary.open('xb') as out:
  for block in iter(lambda:inp.read(4<<20),b''):out.write(block);h.update(block)
 digest=h.hexdigest()
 if row.get('sha256',digest)!=digest or sha(temporary)!=digest:raise RuntimeError(f'Copy checksum mismatch: {rel}')
 after=src.stat()
 if (after.st_size,after.st_mtime_ns)!=(before.st_size,before.st_mtime_ns):raise RuntimeError(f'Source changed during copy: {rel}')
 shutil.copystat(src,temporary);os.replace(temporary,dst)
 row['sha256']=digest
 return row
if not (T/'PLAN_SUMMARY.json').exists():raise RuntimeError('Plan is not complete')
if (T/'COPY_VERIFIED.json').exists():raise RuntimeError('Merge already committed')
rows=[json.loads(l) for l in (T/'PLAN.jsonl').open()]
# No active incoming jobs. Archive queues as evidence; never resume them.
counts=collections.Counter();sizes=collections.Counter();last=time.time()
with (T/'FILES_SHA256.jsonl').open('w') as out,ThreadPoolExecutor(max_workers=8) as pool:
 for row in pool.map(transfer,rows):
  out.write(json.dumps(row)+'\n');counts[row['action']]+=1;sizes[row['action']]+=row['bytes']
  if time.time()-last>30:
   out.flush();print('copied/verified',sum(counts.values()),dict(counts),flush=True);last=time.time()
report={'source':str(S),'destination':str(P),'files':sum(counts.values()),'counts':dict(counts),'bytes':dict(sizes),'all_selected_files_sha256_verified':True,'source_preserved':True,'conflict_backup':str(B)}
(T/'COPY_VERIFIED.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
