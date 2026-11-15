"""Create a fresh personalization matrix using only completed daily-protocol models."""
from settings import *
import shutil
import numpy as np
import torch


def main():
    shared=PROJECT/'runs/daily_protocol_restart_20261114_v1'
    assert (shared/'COMPLETE.json').exists()
    ready=json.loads((shared/'data/READY.json').read_text())
    assert ready['protocol']=='daily_08_BJT_latency15h' and ready['regional_delay_minutes']==0
    fingerprint=ready['preprocess_fingerprint']
    if not DATA.exists():DATA.symlink_to('../daily_protocol_restart_20261114_v1/data',target_is_directory=True)
    assert DATA.resolve()==(shared/'data').resolve()
    shutil.copytree(shared/'assets',ASSETS,dirs_exist_ok=True)
    for name in ['DATA_FREEZE.json','PROTOCOL.json','QC_RULES.json','SCIENCE_FREEZE.json']:
        shutil.copy2(shared/name,RUN/('SOURCE_'+name))
    cohort=json.loads((ASSETS/'COHORT.json').read_text());sources={}
    for family in ['W0','TF']:
        for seed in SEEDS:
            source=shared/'jobs'/f'BASE_{family}_R3_r{seed}'
            assert (source/'SUCCESS.json').is_file()
            m=json.loads((source/'metrics.json').read_text())
            assert not m['smoke_only'] and m['spec']['preprocess_fingerprint']==fingerprint
            assert sha(source/'best.pt')==m['checkpoint_sha256']
            ck=torch.load(source/'best.pt',map_location='cpu',weights_only=False)
            assert ck['normalization_sha256']==sha(ASSETS/'normalization.json')
            assert ck['cohort_sha256']==sha(ASSETS/'COHORT.json')
            shutil.copy2(source/'best.pt',RUN/f'pretrained/{family}_r{seed}.pt')
            shutil.copy2(source/'metrics.json',RUN/f'pretrained/{family}_r{seed}.metrics.json')
            sources[family,seed]=m
    specs=[]
    for station,station_id in enumerate(cohort['station_ids']):
        for seed in SEEDS:
            for mode in ['P1_FULL','P2_HEAD']:
                for family in ['W0','TF']:
                    m=sources[family,seed]
                    specs.append(dict(id=f'{family}_{mode}_s{station:02d}_r{seed}',stage='personalization',group=family+'_'+mode,family=family,mode=mode,station=station,station_id=station_id,seed=seed,config=m['spec']['config'],pretrained=f'pretrained/{family}_r{seed}.pt',source_sha256=m['checkpoint_sha256'],epochs=30,patience=6,finetune_lr=2e-5,weight_decay=1e-4,dependencies=[f'pretrained/{family}_r{seed}.pt'],preprocess_fingerprint=fingerprint))
    atomic_json(RUN/'manifest.json',specs)
    atomic_json(RUN/'PROTOCOL.json',dict(timestamp=now(),weather_protocol=ready['protocol'],preprocess_fingerprint=fingerprint,source_run=shared.name,regional_delay_minutes=0,farms=cohort['count'],seeds=SEEDS,train_year=2022,selection_year=2023,epochs=30,patience=6,finetune_lr=2e-5,uniform_epochs=[1,6],validation_selection='per-station2023MAE including epoch0; separately report fixedE1 andE6',test2024_read=False,new_farm_transfer_claim=False,previous_protocol_results_reused=False,jobs=len(specs)))
    print('Daily personalization jobs',len(specs),flush=True)


if __name__=='__main__':main()
