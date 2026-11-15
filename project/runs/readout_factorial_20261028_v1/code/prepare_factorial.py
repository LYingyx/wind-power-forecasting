from settings import *
from operational import november_exclusions
import numpy as np

def run():
    setup()
    origin=np.load(CACHE/'val/origin_unix.npy')
    reasons=november_exclusions(origin,2023)
    keep=~np.logical_or.reduce(list(reasons.values()))
    np.save(ASSETS/'val_keep.npy',keep)
    atomic_json(ASSETS/'VALIDATION_MASK_AUDIT.json',dict(year=2023,total=len(origin),kept=int(keep.sum()),removed=int((~keep).sum()),reasons={k:int(v.sum()) for k,v in reasons.items()},overlapping_reason_counts=True,mask_sha256=sha(ASSETS/'val_keep.npy'),source_origins_sha256=sha(CACHE/'val/origin_unix.npy'),training_unchanged=True,test_accessed=False))
    configurations=[('W0_H32', 'W0',32,False),('W0_H64','W0',64,False),('W0_O_H32','W0',32,True),('W0_O_H64','W0',64,True),('TF_R3_RAW','TF',64,True)]
    specs=[]
    # Interleave all arms, stations and seeds for paired progress; no early stopping across arms.
    for seed in SEEDS:
        for site in range(56):
            for name,backbone,width,direct in configurations:
                specs.append(dict(id=f'{name}_s{site:02d}_r{seed}',task='train',configuration=name,backbone=backbone,variant='R3_RAW',head_width=width,head_current_power=direct,site=site,seed=seed,loss='MAE',data_fingerprint=FINGERPRINT,validation_mask_sha256=sha(ASSETS/'val_keep.npy')))
    assert len(specs)==2800 and len({s['id'] for s in specs})==2800
    for name in ['manifest.json','screen_manifest.json','candidate_manifest.json']:atomic_json(RUN/name,specs)
    atomic_json(RUN/'PROTOCOL.json',dict(timestamp=now(),data_fingerprint=FINGERPRINT,train_year=2022,selection_year=2023,excluded_validation_month=11,validation_mask_sha256=sha(ASSETS/'val_keep.npy'),validation_rows=int(keep.sum()),seeds=SEEDS,farms=56,training_jobs=len(specs),arms=[x[0] for x in configurations],primary_checkpoint='2023_excluding_November_MAE',loss='MAE',epochs=80,patience=10,batch_size=256,learning_rate=.0002,weight_decay=.0001,selection=dict(references=['W0_H32','TF_R3_RAW'],candidates=['W0_H64','W0_O_H32','W0_O_H64'],require_both_metrics_relative_improvement=.001,require_each_metric_farm_wins=29,maximum_high_power_RMSE_increase=.005,maximum_0_1h_metric_increase_vs_W0=.001,tiebreak='lowest macro2023 MAE among candidates passing all criteria',aggregation='within farm average10 seeds, then equal farm average',all_arms_reported=True),inference_limits=['No equivalence claim from failed superiority','10 seeds quantify fitting variability, not interannual uncertainty','2024 has already informed this experiment: any future2024 reevaluation is exploratory','Output head is not the only W0/TF architecture difference'],test_policy='No2024 evaluation or fitting in this run; selected model and references are frozen before a separate follow-up',initialization='Original W0 initialization preserved; common modules and shared head coordinates identical across W0 cells; final output layer zero for all arms'))
    print(json.dumps(dict(jobs=len(specs),validation_kept=int(keep.sum()),validation_removed=int((~keep).sum()))))

if __name__=='__main__':run()
