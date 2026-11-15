"""Cache exactly the shared model's inputs, restricted to one target station."""
from settings import *
from data import SharedData
import numpy as np
import torch
import fcntl

class StationData:
    def __init__(self,split,shared_spec,station,smoke=False):
        assert split in ('train','val')
        self.station=int(station);self.farms=1
        folder=RUN/'feature_cache'/f's{station:02d}'/split;folder.mkdir(parents=True,exist_ok=True)
        ready=folder/'READY.json';source_hash=sha(RUN/'SOURCE_DATA_FREEZE.json')
        with (folder/'build.lock').open('a+') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            if not ready.exists():
                source=SharedData(split,shared_spec);n=len(source.rows);arrays={}
                assert 0<=station<source.farms and source.reference is None
                for start in range(0,n,1024):
                    ix=np.arange(start,min(start+1024,n))
                    b,y,si,ti=source.batch(ix*source.farms+station,'cpu')
                    assert (si==station).all() and np.array_equal(ti,ix)
                    values={k:v.numpy() for k,v in b.items()};values['y']=y.numpy()
                    for key,v in values.items():
                        if key not in arrays:arrays[key]=np.lib.format.open_memmap(folder/(key+'.npy'),mode='w+',dtype=v.dtype,shape=(n,*v.shape[1:]))
                        assert np.isfinite(v).all(),key
                        arrays[key][start:start+len(ix)]=v
                for arr in arrays.values():arr.flush()
                origin=np.asarray(source.a['origin']);years=(origin+28800).astype('datetime64[s]').astype('datetime64[Y]').astype(int)+1970
                assert (years==(2022 if split=='train' else 2023)).all()
                if split=='val':assert not (((origin+28800).astype('datetime64[s]').astype('datetime64[M]').astype(int)%12+1)==11).any()
                atomic_json(ready,dict(timestamp=now(),source_hash=source_hash,station=station,split=split,n=n,keys=list(arrays),passed=True))
                del source,arrays
            meta=json.loads(ready.read_text());assert meta['source_hash']==source_hash and meta['station']==station and meta['passed']
        self.a={k:np.load(folder/(k+'.npy'),mmap_mode='r') for k in meta['keys']}
        self.rows=np.arange(meta['n']) if not smoke else np.linspace(0,meta['n']-1,128,dtype=int)
        self.n=len(self.rows)

    def batch(self,indices,device='cpu'):
        ix=self.rows[np.asarray(indices)]
        b={k:torch.as_tensor(np.array(v[ix]),device=device) for k,v in self.a.items() if k!='y'}
        y=torch.as_tensor(np.array(self.a['y'][ix]),device=device)
        assert (b['site']==self.station).all()
        return b,y,np.zeros(len(ix),dtype=int),ix
