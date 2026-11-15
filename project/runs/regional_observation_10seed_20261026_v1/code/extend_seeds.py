"""Predeclared, conditional extension; not a new search on 2024."""
from settings import *
import numpy as np
TF='TF_z1_lr0.0002_dp0.05'
CANDIDATES=['R3_RAW','R3_ALLPOOL']
NEW_SEEDS=list(range(44,52))

def spec_for(config,variant,site,seed):
    spec=dict(id=f'{config}_{variant}_s{site:02d}_r{seed}',stage=variant if config=='W0' else 'T_'+variant,
              task='train',configuration=config,backbone='W0' if config=='W0' else 'TF',variant=variant,
              site=site,seed=seed,loss='MAE',delay_minutes=15,capacity_control=False,
              protocol='independent_target_regional_raw_completion_SEL_2023_10seed')
    if config==TF:spec.update(tf_zero_init=True,tf_lr=2e-4,tf_dropout=.05)
    return spec

def select_arms(recommendation):
    assert recommendation['complete'] and recommendation['open_2024'] is False
    arms=[]
    for config in ['W0',TF]:
        # Paired references are repeated too: all comparisons use the same ten seeds.
        arms.extend((config,v) for v in ['N0','N3'])
        for variant in CANDIDATES:
            gate=recommendation['gates'][config+'/'+variant]
            if gate['passes']:
                assert gate['dual'] and gate['beats_control'] and gate['high_power_rmse_percent']<=.1
                arms.extend((config,v) for v in [variant,'C2_'+variant])
    return arms

def build_manifest(arms):
    return [spec_for(c,v,s,r) for r in NEW_SEEDS for s in range(56) for c,v in arms]

def prepare():
    setup()
    all_arms=[(c,v) for c in ['W0',TF] for v in ['N0','N3','R3_RAW','C2_R3_RAW','R3_ALLPOOL','C2_R3_ALLPOOL']]
    atomic_json(RUN/'candidate_manifest.json',build_manifest(all_arms))
    protocol=dict(timestamp=now(),parent=str(SCREEN_RUN),fit_year=2022,selection_year=2023,
        open_2024=False,seeds=list(range(42,52)),reuse_seeds=[42,43],new_seeds=NEW_SEEDS,
        trigger='Complete D audit AND all 56 farms x 2 paired seeds; no representative-site early selection',
        eligible_candidates=CANDIDATES,backbones=['W0',TF],
        candidate_rule='D frozen gate: MAE and RMSE >=0.1% vs both N0/N3; >=29/56 farms per metric; beat own C2 on both; high-power RMSE vs N3 <=+0.1%',
        references=['N0','N3'],controls='Each admitted candidate brings its own C2 control to ten seeds',
        references_always_extended=True,min_new_fits=1792,max_new_fits=5376,
        excluded='Diagnostic ring/latency/SELF arms are not primary candidates in the frozen D protocol; do not promote partial or exploratory winners',
        reporting='Separate selection seeds 42/43, new replication seeds 44..51, and combined ten; within-farm seed average then equal farms; MAE-selected weights for both metrics',
        interpretation='Additional optimization seeds on the same selected year; not an independent-year confirmation',
        scheduling='D first; audited handoff to one queue per host; GPU/CPU/cgroup RAM + matched-job epoch efficiency guard',
        storage='Reuse original 42/43 metrics by source references, no duplicate weights; <=2 selected checkpoints; no prediction arrays')
    atomic_json(RUN/'PROTOCOL.json',protocol)
    train_times=np.load(CACHE/'train/origin_unix.npy');local=train_times+8*3600
    month=local.astype('datetime64[s]').astype('datetime64[M]').astype(int)%12+1
    slot=local%86400//900;day=local//86400
    pools={(m,s):np.flatnonzero((month==m)&(slot==s)) for m in range(1,13) for s in range(96)}
    checks={}
    for split in ['train','val']:
        times=np.load(CACHE/split/'origin_unix.npy');local=times+8*3600
        months=local.astype('datetime64[s]').astype('datetime64[M]').astype(int)%12+1
        slots=local%86400//900;days=local//86400
        for seed in SEEDS:
            rng=np.random.default_rng(20261025+seed);out=np.empty(len(times),dtype='i8')
            for i,(m,s,d) in enumerate(zip(months,slots,days)):
                pool=pools[int(m),int(s)]
                if split=='train':pool=pool[np.abs(day[pool]-d)>=2]
                assert len(pool)>0
                out[i]=pool[rng.integers(len(pool))]
            assert np.array_equal(month[out],months) and np.array_equal(slot[out],slots)
            assert np.all(abs(day[out]-days)>=2)
            if split=='val':assert np.all(train_times[out]<times)
            path=ASSETS/f'mismatch_{split}_r{seed}.npy'
            if seed in [42,43]:assert np.array_equal(np.load(path),out),'Donor generation drift'
            else:np.save(path,out)
            checks[f'{split}/{seed}']=dict(n=len(out),sha256=sha(path),donor_year=2022)
    atomic_json(ASSETS/'SEED_EXTENSION_CHECKS.json',dict(timestamp=now(),passed=True,checks=checks,open_2024=False))
    initializers={}
    for site in range(56):
        for seed in NEW_SEEDS:
            path=OLD/f'jobs/reference_B0_s{site:02d}_r{seed}/best.pt';assert path.exists()
            initializers[str(path)]=sha(path)
    atomic_json(RUN/'INITIALIZER_FREEZE.json',dict(timestamp=now(),files=initializers))
    print('Prepared conditional superset: 5376 possible new fits; 448 initialization hashes; donors 42..51')

if __name__=='__main__':prepare()
