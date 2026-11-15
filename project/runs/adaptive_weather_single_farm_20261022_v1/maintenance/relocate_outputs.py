"""Merge interrupted relocation; retain one durable byte-identical file."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
import hashlib,json,os,shutil,time,datetime
RUN=Path(__file__).resolve().parents[1]
LOCAL=Path('/home/liying/var/adaptive_weather_single_farm_20261022_v1_recovery')
DEST=Path('/shared_data_ssd3/liying/adaptive_weather_single_farm_20261022_v1/evaluation')
SOURCES=[RUN/'evaluation',LOCAL/'evaluation']
inventory=json.loads((LOCAL/'evaluation_inventory_before.json').read_text())
DEST.mkdir(parents=True,exist_ok=True)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def move(rel):
 candidates=[p/rel for p in SOURCES if (p/rel).is_file()]
 dest=DEST/rel;dest.parent.mkdir(parents=True,exist_ok=True)
 if candidates:
  src=candidates[0];assert src.stat().st_size==inventory[rel]
  expected=sha(src)
  if not dest.exists() or sha(dest)!=expected:shutil.copy2(src,dest)
  with dest.open('rb') as f:os.fsync(f.fileno())
  assert sha(dest)==expected
  for f in candidates:
   assert sha(f)==expected;f.unlink()
 else:
  assert dest.stat().st_size==inventory[rel];expected=sha(dest)
 return dict(path=rel,bytes=dest.stat().st_size,sha256=expected)
count=total=0;start=last=time.monotonic()
with (RUN/'maintenance/relocation_manifest.jsonl').open('w') as log,ThreadPoolExecutor(max_workers=8) as pool:
 for fut in as_completed([pool.submit(move,k) for k in inventory]):
  row=fut.result();log.write(json.dumps(row)+'\n');count+=1;total+=row['bytes']
  if time.monotonic()-last>15:
   log.flush();print(count,'/',len(inventory),'GiB',round(total/2**30,2),'seconds',round(time.monotonic()-start),flush=True);last=time.monotonic()
 log.flush();os.fsync(log.fileno())
assert count==len(inventory)
assert {str(p.relative_to(DEST)):p.stat().st_size for p in DEST.rglob('*') if p.is_file()}==inventory
for source in SOURCES:
 for d,dirs,files in os.walk(source,topdown=False):
  assert not files;Path(d).rmdir()
(RUN/'evaluation').symlink_to(DEST,target_is_directory=True)
result=dict(timestamp=datetime.datetime.now().astimezone().isoformat(),files=count,bytes=total,all_sha256_verified=True,physical_path=str(DEST),original_path=str(RUN/'evaluation'),local_partial_copy_removed=True)
(RUN/'STORAGE_RELOCATION.json').write_text(json.dumps(result,indent=2)+'\n');print(result,flush=True)
