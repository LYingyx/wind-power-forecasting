"""Inspect actual inputs to condition_projection in the archived FP16 models."""
import sys,json,hashlib
import numpy as np
import torch
from audit_tables import ROOT,OUT,ART,save
import pandas as pd
sys.path.insert(0,str(ROOT/'runtime'))
from wind_paths import select_source,rebase_tree
select_source('cbh')
from wind_forcing.data import ForecastDataset
from wind_forcing.data.dataset import ForecastCollator
from wind_forcing.models import build_forecast_model

def main():
    torch.set_num_threads(2)
    data=ForecastDataset(ROOT/'data/model_cache_pad5_f8','val',preload_weather=False,verbose=False)
    batch=ForecastCollator(data)([100,200])
    rows=[]
    for variant in ['CBH','CB','CH','CNEUTRAL']:
        path=ART/f'p0i_condition_ablation_completion_20261014_v1/results/{variant}_seed42/best_model.pt'
        cp=torch.load(path,map_location='cpu',weights_only=False)
        cfg=rebase_tree(cp['model_config'])
        model=build_forecast_model(cfg,data.static,history_steps=24,forecast_steps=24)
        model.load_state_dict(cp['model_state'],strict=True);model.eval()
        raw=[]
        hook=model.condition_projection[0].register_forward_pre_hook(lambda module,args:raw.append(args[0].detach().cpu()))
        with torch.inference_mode():model(**batch)
        hook.remove();v=raw[0].numpy();assert v.shape[-1]==67
        rows.append(dict(variant=variant,condition_mode=cfg['condition_mode'],summary_abs_max=float(np.max(np.abs(v[...,:64]))),
            baseline_abs_max=float(np.max(np.abs(v[...,64]))),increment_abs_max=float(np.max(np.abs(v[...,65]))),
            lead_abs_max=float(np.max(np.abs(v[...,66]))),checkpoint=str(path.relative_to(ROOT)),
            checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    d=pd.DataFrame(rows);save(d,'condition_actual_inputs')
    assert d.set_index('variant').loc['CB','increment_abs_max']>0
    assert d.set_index('variant').loc['CB','summary_abs_max']==0
    assert d.set_index('variant').loc['CH','baseline_abs_max']==0
    assert d.set_index('variant').loc['CH','increment_abs_max']==0
    data.close();print(d.to_string(index=False),flush=True)

if __name__=='__main__':main()
