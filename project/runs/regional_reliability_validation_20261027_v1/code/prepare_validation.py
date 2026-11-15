"""Prepare static geometry controls, causal diagnostic labels and immutable manifests."""
from settings import *
from validation_design import *
import numpy as np
from scipy.optimize import linear_sum_assignment

def main():
    setup();tables={v:np.zeros((10,56,55),dtype=bool) for v in MATCH+FULL};audit=[]
    with np.load(ASSETS/'topology.npz') as z:
        for seed in range(42,52):
            for site in range(56):
                valid=z['mask'][site];dist=z['geometry'][site,:,2];cap=z['capacity'][site]
                pools=[np.flatnonzero(valid&(dist>=lo)&(dist<hi)) for lo,hi in [(0,20),(20,60),(60,150.0001)]]
                k=min(4,*map(len,pools));rng=np.random.default_rng(20261027+1000*site+seed)
                # One common near-ring target capacity distribution; no power data used.
                selected=rng.choice(pools[0],k,replace=False) if k else np.array([],dtype=int)
                targets=np.log(cap[selected]) if k else np.array([])
                for v,pool in zip(MATCH,[*pools,np.flatnonzero(valid)]):
                    if k:
                        costs=abs(targets[:,None]-np.log(cap[pool])[None,:])+rng.uniform(0,1e-7,(k,len(pool)))
                        row,col=linear_sum_assignment(costs);chosen=pool[col]
                        tables[v][seed-42,site,chosen]=True
                        audit.append(dict(site=site,seed=seed,variant=v,k=k,
                            ids=z['ids'][site,chosen].tolist(),distance_mean=float(dist[chosen].mean()),
                            log_capacity_mean_error=float(abs(targets[row]-np.log(cap[chosen])).mean())))
                for v,pool in zip(FULL,[*pools,np.flatnonzero(valid)]):tables[v][seed-42,site,pool]=True
                assert all(tables[v][seed-42,site].sum()==k for v in MATCH)
    np.savez_compressed(ASSETS/'matched_subsets.npz',**tables)
    atomic_json(ASSETS/'MATCHING_AUDIT.json',dict(timestamp=now(),selection='geometry/nameplate capacity only; equal weights',
        maximum_k=4,common_support_sites=int((tables['MATCH_INNER'][0].sum(1)>0).sum()),
        no_near_ring='All four MATCH summaries empty, N3 retained; excluded from distance contrasts',rows=audit))
    # Precomputed labels let every reference/candidate share identical strata.
    with np.load(ASSETS/'topology.npz') as z:
        for split in ['train','val']:
            power=np.load(CACHE/split/'station_history.npy',mmap_mode='r');labels=np.empty((len(power),56,2),dtype='i1')
            for site in range(56):
                w=z['capacity'][site]*z['mask'][site];w=w/w.sum()
                delta=((power[:,z['ids'][site],-2]-power[:,z['ids'][site],-6])*w[None]).sum(1)
                gap=(power[:,z['ids'][site],-2]*w[None]).sum(1)-power[:,site,-1]
                labels[:,site,0]=np.where(delta<-.05,0,np.where(delta>.05,2,1))
                labels[:,site,1]=np.where(gap<-.1,0,np.where(gap>.1,2,1))
            np.save(ASSETS/f'validation_labels_{split}.npy',labels)
    specs=[];reuse=[]
    sites=REPRESENTATIVES+[s for s in range(56) if s not in REPRESENTATIVES]
    for site in sites:
        for seed in [42,43]:
            for config in ['W0',TF]:
                for v in REFERENCES+(['R3_MEAN_DELTA'] if config=='W0' else []):
                    jid=f'{config}_{v}_s{site:02d}_r{seed}';source=D_RUN/'jobs'/jid
                    assert (source/'SUCCESS.json').exists()
                    old=json.loads((source/'metrics.json').read_text())
                    if old.get('task')=='diagnose_reference':
                        source_job=old['source_job'];source_run=old['source_run']
                    else:source_job=jid;source_run=D_RUN.name
                    item=dict(id=jid,stage='REFERENCE',task='diagnose_reference',configuration=config,
                        backbone='W0' if config=='W0' else 'TF',variant=v,site=site,seed=seed,
                        source_run=source_run,source_job=source_job)
                    specs.append(item)
                for v in W0_NEW if config=='W0' else TF_NEW:specs.append(spec(config,v,site,seed))
    assert len({s['id'] for s in specs})==len(specs)
    atomic_json(RUN/'manifest.json',specs)
    # Predeclared superset for the conditional eight-seed replication, never activated early.
    variants={c:REFERENCES+(W0_NEW+['R3_MEAN_DELTA'] if c=='W0' else TF_NEW) for c in ['W0',TF]}
    extension=[spec(c,v,s,r) for r in range(44,52) for s in sites for c in ['W0',TF] for v in variants[c] if v not in REFERENCES]
    atomic_json(RUN/'candidate_extension_manifest.json',extension)
    atomic_json(RUN/'PROTOCOL.json',dict(timestamp=now(),fit_year=2022,selection_year=2023,open_2024=False,
        farms=56,screen_seeds=[42,43],replication_seeds=list(range(44,52)),training=sum(s['task']=='train' for s in specs),
        replays=sum(s['task']=='diagnose_reference' for s in specs),geometry_arms=REMOVE+MATCH+FULL,
        geometry_control='N3 fixed for matched/full controls; 4 shared slots; equal weights; log capacity matching; 41-site common support',
        deletion='EXCLUDE20_KEEP masks <20km in both routes without replacement; REFILL picks next nearest 8 and masks <20km everywhere',
        gates=dict(arms=GATES,bounded='output interpolation weight in [0,1]; prediction itself not clipped',
            formula='local + amplitude * exp(-lead/tau) * (regional-local); independent LEAD and STATE bypasses',
            state='own CF/trend, region CF/trend/spread/gap, NWP target speed and change from origin',
            dropout='both shared-backbone forwards use same dropout draw; local path is jointly trained, not a frozen N0 checkpoint',
            controls='Each gate has own C2; identical allocated parameters across 4 gates; G_FREE against old RAW tests implementation parity'),
        simplification='R3_MEAN_DELTA with own C2 and both W0/TF; reuse old W0 42/43 weights',
        training_settings=dict(epochs=80,patience=10,batch=256,loss='MAE',learning_rate=.0002,TF_dropout=.05,
            TF_zero_output=True,precision='FP32, no TF32',selection='same MAE checkpoint for all primary metrics'),
        replication_rule='See frozen SELECTION_RULES.json; complete screen and paired 56 farms first; no 2024',
        interpretation='2023 selected-year tests, not independent-year confirmation; outcome labels are diagnostic only',
        storage='Reuse references, <=2 selected checkpoints, no prediction arrays',
        hosts=['local4090','v1006001','v1006002']))
    atomic_json(RUN/'SELECTION_RULES.json',dict(
        primary='G_STATE_LEAD',secondary=['G_LEAD','G_STATE','R3_MEAN_DELTA'],
        performance_gate=dict(overall_vs_N0_N3_both_metrics_max_percent=-.1,improved_farms_each_metric_min=29,
            overall_vs_RAW_both_metrics_max_percent=.1,short_0_3_vs_RAW_each_horizon_both_metrics_max_percent=.1,
            high_all_and_3_6_RMSE_vs_N0_and_N3_max_percent=.1,
            beats_own_C2_both_metrics=True,
            benefit='either overall both metrics <=-0.1% vs RAW, or high3_6 both metrics <=-0.1% vs RAW'),
        gate_replication='If a gate passes on a backbone, repeat all four gates and their C2 on that backbone, preserving component contrasts',
        simplification_replication='If MEAN_DELTA passes, repeat it and its C2 on that backbone',
        removal_replication='If KEEP or REFILL has overall both metrics <=+0.1% vs RAW and high all/3_6 RMSE <=+0.1% vs RAW, repeat both deletion arms; report noninferiority CI separately, point screen is not proof of equivalence',
        geometry_replication='On common-support farms, if any matched distance arm beats another by >=0.1% in both metrics with majority wins, or FULL beats MATCH within any ring by same criterion, repeat all MATCH and FULL arms',
        diagnostic_extension='Geometry/deletion reproduction validates explanations; does not promote to final predictive model without full C2/TF tests',
        comparisons='Seeds within farm, then equal farms; paired 10000-farm bootstrap for intervals, spatial dependence limits inference',
        no_optional_2024=True))
    print(json.dumps(dict(tasks=len(specs),train=sum(s['task']=='train' for s in specs),replay=sum(s['task']=='diagnose_reference' for s in specs),extension_superset=len(extension))))

if __name__=='__main__':main()
