"""Lossless post-evaluation compaction; scientific computation stays frozen."""
from pathlib import Path
import sys,os,json,hashlib,fcntl,datetime
import numpy as np
RUN=Path(__file__).resolve().parents[1]
def sha(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def atomic_json(path,obj):
 temp=path.with_name(path.name+f'.{os.getpid()}.tmp')
 temp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');os.replace(temp,path)
def link_replace(source,target):
 if os.path.samefile(source,target):return False
 tmp=target.with_name(target.name+f'.{os.getpid()}.linktmp')
 tmp.unlink(missing_ok=True);os.link(source,tmp);os.replace(tmp,target);return True
def _compact_locked(folder):
 folder=Path(folder)
 if not (folder/'SUCCESS.json').exists():return None
 marker=folder/'STORAGE_COMPACTED.json'
 previous=json.loads(marker.read_text()) if marker.exists() else {}
 if previous.get('version',0)>=2:return previous
 metrics=json.loads((folder/'metrics.json').read_text())
 manifest=previous.get('attention',{});saved=previous.get('bytes_saved',0);deduplicated=previous.get('duplicate_arrays_linked',0)
 # Upgrade the completed lossless pass without re-reading already verified large arrays.
 if previous.get('version')==1 and previous.get('passed'):
  removed=0
  for label,m in metrics['interventions'].items():
   f=folder/f'{label}_metrics.json'
   if f.exists():
    assert json.loads(f.read_text())==m
    saved+=f.stat().st_size;f.unlink();removed+=1
  previous.update(version=2,bytes_saved=saved,redundant_variant_metrics_removed=removed,timestamp=datetime.datetime.now().astimezone().isoformat())
  atomic_json(marker,previous);return previous
 # Original and intervention arrays are kept, with exactly equal arrays sharing an inode.
 canonical={}
 for label,m in metrics['interventions'].items():
  f=folder/f'{label}_prediction.npy';h=sha(f)
  assert h==m['prediction_sha256'],f'Prediction checksum mismatch: {f}'
  if h in canonical:
   size=f.stat().st_size
   if link_replace(canonical[h],f):saved+=size;deduplicated+=1
  else:canonical[h]=f
 # Variant JSON files are resume checkpoints; the final metrics file contains each one verbatim.
 duplicate_metrics=0
 for label,m in metrics['interventions'].items():
  f=folder/f'{label}_metrics.json'
  if f.exists():
   assert json.loads(f.read_text())==m,f'Variant metrics mismatch: {f}'
   saved+=f.stat().st_size;f.unlink();duplicate_metrics+=1
 shared=RUN/'evaluation/_shared';shared.mkdir(exist_ok=True)
 for f in folder.glob('donors_*.npy'):
  h=sha(f);target=shared/f'donors_{h}.npy'
  try:os.link(f,target)
  except FileExistsError:pass
  assert sha(target)==h
  size=f.stat().st_size
  if link_replace(target,f):saved+=size;deduplicated+=1
 original=folder/'attention_diagnostics.npy';compressed=folder/'attention_diagnostics.npz'
 if original.exists():
  arr=np.load(original);h=sha(original);size=original.stat().st_size
  temp=compressed.with_name(compressed.name+f'.{os.getpid()}.tmp')
  with temp.open('wb') as stream:
   np.savez_compressed(stream,diagnostics=arr);stream.flush();os.fsync(stream.fileno())
  with np.load(temp) as check:assert np.array_equal(check['diagnostics'],arr)
  os.replace(temp,compressed)
  manifest=dict(original_sha256=h,compressed_sha256=sha(compressed),shape=list(arr.shape),dtype=str(arr.dtype),lossless_roundtrip_verified=True)
  # Record the verified replacement before removing the redundant uncompressed array.
  atomic_json(folder/'ATTENTION_STORAGE.json',manifest)
  original.unlink();saved+=size-compressed.stat().st_size
 removed=0
 for f in folder.glob('*.tmp'):
  saved+=f.stat().st_size;f.unlink();removed+=1
 result=dict(timestamp=datetime.datetime.now().astimezone().isoformat(),version=2,passed=True,bytes_saved=saved,
             duplicate_arrays_linked=deduplicated,obsolete_temporary_files_removed=removed,attention=manifest,
             redundant_variant_metrics_removed=previous.get('redundant_variant_metrics_removed',0)+duplicate_metrics,
             all_prediction_checksums_unchanged=True)
 atomic_json(marker,result);return result
def compact(folder):
 folder=Path(folder)
 if not (folder/'SUCCESS.json').exists():return None
 with (folder/'.storage.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  return _compact_locked(folder)
if __name__=='__main__':
 from concurrent.futures import ThreadPoolExecutor,as_completed
 import time
 start=last=time.monotonic();count=total=0
 folders=[p.parent for p in (RUN/'evaluation').glob('*/SUCCESS.json')]
 with ThreadPoolExecutor(max_workers=4) as pool:
  for future in as_completed([pool.submit(compact,p) for p in folders]):
   result=future.result();count+=1;total+=result['bytes_saved']
   if time.monotonic()-last>15:
    print('COMPACTED',count,'/',len(folders),'saved_GiB',round(total/2**30,3),flush=True);last=time.monotonic()
 atomic_json(RUN/'STORAGE_COMPACTION.json',dict(passed=True,completed_evaluations=count,bytes_saved=total,timestamp=datetime.datetime.now().astimezone().isoformat()))
 print('COMPLETE',count,'saved_GiB',total/2**30,flush=True)
