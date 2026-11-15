from settings import *
import shutil

def run():
    setup();parent=json.loads((PARENT/'manifest.json').read_text())
    refs=[s for s in parent if s['configuration'] in ['W0_H32','TF_R3_RAW']]
    specs=[]
    for seed in SEEDS:
        for site in range(56):
            for backbone in ['W0','TF']:
                ref=next(s for s in refs if s['backbone']==backbone and s['site']==site and s['seed']==seed)
                specs.append(ref)
                for cell in ['H1W0','H0W1','H0W0']:
                    c=backbone+'_'+cell
                    specs.append(dict(ref,id=f'{c}_s{site:02d}_r{seed}',configuration=c,variant='R3_'+cell))
    assert len(specs)==4480 and len(refs)==1120
    ledger={}
    for s in refs:
        source=PARENT/'jobs'/s['id'];dest=JOBS/s['id'];dest.mkdir(exist_ok=True)
        assert json.loads((source/'spec.json').read_text())==s and (source/'SUCCESS.json').exists()
        m=json.loads((source/'metrics.json').read_text())
        for selection in m['selections'].values():assert sha(source/selection['checkpoint'])==selection['sha256']
        files=['spec.json','normalization.json','metrics.json','diagnostics.json','daily.json','history.json','selection.json','FIT_COMPLETE.json','SUCCESS.json']+sorted({x['checkpoint'] for x in m['selections'].values()})
        for name in files:
            if not (dest/name).exists():os.link(source/name,dest/name)
        ledger[s['id']]=dict(source=str(source),metrics_sha256=sha(source/'metrics.json'),spec_sha256=sha(source/'spec.json'))
    atomic_json(RUN/'REUSED_REFERENCES.json',ledger)
    for name in ['manifest.json','candidate_manifest.json','screen_manifest.json']:atomic_json(RUN/name,specs)
    protocol=dict(timestamp=now(),data_fingerprint=FINGERPRINT,train_year=2022,selection_year=2023,excluded_validation_month=11,validation_mask_sha256=sha(ASSETS/'val_keep.npy'),validation_rows=29695,seeds=SEEDS,farms=56,total_jobs=4480,new_training_jobs=3360,reused_reference_jobs=1120,primary_checkpoint='2023_noNov_MAE',loss='MAE',batch_size=256,epochs=80,patience=10,lr=.0002,weight_decay=.0001,comparison='2x2 history_real x weather_real on W0+SEL+R3 and TF+SEL+R3',fixed_history='2022 mean24-step history; recompute STATE and all history-dependent queries',fixed_weather='2022 mean weather trajectory; zero normalized weather-derived SEL; replace physical wind entering neighbor attention query',held_constant='real current power in additive output anchor, actual regional observations and rings, topology, phase, all hyperparameters and architecture',interpretation='Predictive history-weather interaction conditional on actual anchor and regional observations; neither causal identification nor complete removal of all state information',interaction_formula='(E_H1W0-E_H1W1)-(E_H0W0-E_H0W1)',mechanism_gate=dict(both_metrics=True,relative_to_E11_min=.001,positive_farms_min=29,real_history_weather_gain_positive=True),report_seed_counts=[2,10],seed_expansion='All10 seeds predeclared including negative outcomes; first2 seeds reported as provisional only',test_policy='Freeze all8 cells after complete2023 audit; then exploratory2024 replay of all8 cells regardless of result, no candidate selection',no_new_architecture=True)
    atomic_json(RUN/'PROTOCOL.json',protocol)
    print(json.dumps(dict(new_training=3360,reused=1120,total=4480)))

if __name__=='__main__':run()
