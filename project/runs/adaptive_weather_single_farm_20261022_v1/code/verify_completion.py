from config import *
from data import score
import sqlite3,io

def main():
 db=sqlite3.connect(f'file:{RUN}/queue.sqlite?mode=ro',uri=True);rows=db.execute('SELECT id,kind,status,spec FROM jobs').fetchall();db.close()
 assert len(rows)==15856 and all(r[2]=='done' for r in rows)
 train_count=eval_count=pred_count=compressed_attention_count=0
 for jid,kind,status,raw in rows:
  s=json.loads(raw);folder=RUN/('jobs' if kind=='train' else 'evaluation')/jid;m=json.loads((folder/'metrics.json').read_text());split='val' if kind=='train' else 'test'
  site=s['site'];y=np.load(OLD/f'prepared/{split}/{site:02d}/target.npy',mmap_mode='r');h=np.load(OLD/f'prepared/{split}/{site:02d}/history.npy',mmap_mode='r')
  if kind=='train':
   assert 'test' not in m and sha(folder/'best.pt')==m['checkpoint_sha256'];train_count+=1;entries=[('val',m['val'])]
  else:
   assert m['selection_freeze_sha256']==sha(RUN/'ALL_SELECTIONS_FROZEN.json');eval_count+=1;entries=list(m['interventions'].items())
   compact=json.loads((folder/'STORAGE_COMPACTED.json').read_text());assert compact['passed'] and compact['all_prediction_checksums_unchanged']
   if (folder/'ATTENTION_STORAGE.json').exists():
    record=json.loads((folder/'ATTENTION_STORAGE.json').read_text());archive=folder/'attention_diagnostics.npz'
    assert sha(archive)==record['compressed_sha256']
    with np.load(archive) as z:
     arr=z['diagnostics'];assert list(arr.shape)==record['shape'] and str(arr.dtype)==record['dtype']
     buf=io.BytesIO();np.save(buf,arr);assert hashlib.sha256(buf.getbuffer()).hexdigest()==record['original_sha256']
    compressed_attention_count+=1
  for label,metrics in entries:
   p=np.load(folder/f'{label}_prediction.npy',mmap_mode='r');assert p.shape==y.shape and np.isfinite(p).all()
   error=np.asarray(p,dtype=np.float64)-np.asarray(y,dtype=np.float64)
   assert abs(float(np.abs(error).mean())-metrics['mae'])<1e-12 and abs(float(np.sqrt((error*error).mean()))-metrics['rmse'])<1e-12
   if label.startswith(('weather_shuffle','weight_shuffle')):
    seed=int(label.rsplit('_',1)[1]);donor=np.load(folder/f'donors_{seed}.npy');assert np.array_equal(np.sort(donor),np.arange(len(donor))) and np.all(donor!=np.arange(len(donor)))
   pred_count+=1
 hashes=json.loads((RUN/'TRAINING_CODE_FROZEN.json').read_text());assert all(sha(RUN/p)==v for p,v in hashes.items())
 assert train_count==8016 and eval_count==7840
 atomic_json(RUN/'COMPLETION_AUDIT.json',dict(passed=True,train_jobs=train_count,evaluation_jobs=eval_count,prediction_arrays=pred_count,all_mae_rmse_recomputed=True,training_hashes_verified=True,compressed_attention_arrays_verified=compressed_attention_count,completed_at=now()))

if __name__=='__main__':main()
