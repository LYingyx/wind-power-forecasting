"""Exact dependency versions and deterministic CPU solver/data parity."""
from pathlib import Path
import sys,json,importlib.metadata as metadata
RUN=Path(__file__).resolve().parents[1];sys.path.insert(0,str(RUN/'code'))
from relocate import activate
activate(RUN)
from settings import atomic_json,now
from physical_data import FarmData
from residual_inputs import mapping_history
import numpy as np,torch
from sklearn.linear_model import MultiTaskElasticNet

def check():
    expected={'numpy':'1.26.4','scipy':'1.11.4','scikit-learn':'1.7.2','joblib':'1.5.2','threadpoolctl':'3.6.0'}
    versions={k:metadata.version(k) for k in expected};assert versions==expected
    torch.set_num_threads(1);data=FarmData('train',6,'N0',limit=128)
    mapped=mapping_history('train',6,np.arange(128))[0]
    x=np.concatenate((data.h.numpy(),data.x.numpy()[:,3],mapped[:,-5:-1]),1).astype('f8');x=(x-x.mean(0))/np.maximum(x.std(0),1e-4)
    # Fixed, well-conditioned random design exercises the same Cython solver builds.
    rng=np.random.default_rng(20261025);xx=rng.normal(size=(400,32));yy=xx[:,:12]@rng.normal(size=(12,24))+.01*rng.normal(size=(400,24))
    model=MultiTaskElasticNet(alpha=.003,l1_ratio=.8,max_iter=3000,tol=1e-4).fit(xx,yy);assert model.n_iter_<3000
    value=dict(timestamp=now(),versions=versions,feature_mean=x.mean(0).tolist(),feature_std=x.std(0).tolist(),mapped_summary=[float(mapped.mean()),float(mapped.std())],solver_coef=model.coef_.tolist(),solver_intercept=model.intercept_.tolist(),n_iter=int(model.n_iter_))
    reference=RUN/'operations/CPU_RUNTIME_REFERENCE.json'
    if '--make' in sys.argv:atomic_json(reference,value)
    else:
        ref=json.loads(reference.read_text());errors={k:float(np.max(np.abs(np.asarray(value[k])-np.asarray(ref[k])))) for k in ['feature_mean','feature_std','mapped_summary','solver_coef','solver_intercept']}
        assert all(v<1e-10 for v in errors.values()),errors
        atomic_json(RUN/'operations/CPU_RUNTIME_READY.json',dict(timestamp=now(),passed=True,versions=versions,maximum_differences=errors,n_iter=value['n_iter']))
        print(json.dumps(errors))
if __name__=='__main__':check()
