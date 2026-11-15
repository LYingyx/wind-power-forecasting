"""Reconstruct shared Transformer for the fixed case origins, FP32 CPU only."""
import sys,json,hashlib
import numpy as np
import pandas as pd
import torch
from audit_tables import ROOT,OUT,ART,save
sys.path.insert(0,str(ROOT/'runtime'))
from archive_loader import load_module

def main():
    torch.set_num_threads(2)
    m=load_module('review_shared_train',ART/'shared_all_station_baselines_20260922_v1/code/train_one.py')
    cases=pd.read_csv(OUT/'tables/forecast_case_selection.csv')
    ids=(cases.origin_index*56+cases.station_index).to_numpy(dtype=np.int64)
    data=m.SourceSplit('test',True);batch=data.batch(ids,torch.device('cpu'))
    assert np.array_equal(np.asarray(data.origins[cases.origin_index]),cases.origin_unix)
    rows=[];provenance=[]
    for seed in range(42,52):
        path=ART/f'shared_all_station_baselines_20260922_v1/results/main/TRANSFORMER_FC/seed{seed}/best_model.pt'
        cp=torch.load(path,map_location='cpu',weights_only=False)
        model=m.build_model('TRANSFORMER_FC',64);model.load_state_dict(cp['model_state'],strict=True);model.eval()
        with torch.inference_mode():values=m.predict(model,batch).numpy()
        assert values.shape==(len(cases),24) and np.isfinite(values).all()
        for i,case in cases.iterrows():
            for j in range(24):rows.append(dict(case_id=case.case_id,model='Shared_Transformer_FP32_case',seed=seed,lead_hours=(j+1)/4,value_cf=float(values[i,j])))
        provenance.append(dict(seed=seed,checkpoint=str(path.relative_to(ROOT)),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    save(pd.DataFrame(rows),'forecast_cases_transformer',False)
    (OUT/'case_inference.json').write_text(json.dumps(dict(precision='FP32 CPU; archival baseline metrics were BF16 CUDA or unrecorded CPU',
        purpose='descriptive case curves only; not substituted for formal metrics',cases=len(cases),seeds=10,checkpoints=provenance),indent=2)+'\n')
    print('Computed',len(cases),'cases x 10 seeds; no training.')

if __name__=='__main__':main()
