"""Meaningful synthetic checks; never included in scientific result tables."""
from common import *
from analyze import paired_statistics,complete,selected_baseline

def main():
    index={};records=[]
    for family,error in [('A',.105),('B',.1)]:
        for site in range(56):
            for seed in range(42,47):
                daily={str(d):dict(n=3,abs_sum=error*3*24,sq_sum=error**2*3*24) for d in range(20000,20028)}
                index[family,site,seed]=dict(test=dict(mae=error,daily_utc=daily))
    stats=paired_statistics(index,'A','B',list(range(42,47)))
    for name in ['delta_mae','farm_seed_ci95_low','farm_seed_ci95_high','temporal_7day_ci95_low','temporal_7day_ci95_high']:
        assert abs(stats[name]-.005)<1e-10,(name,stats[name])
    assert stats['farms_left_better']==0 and not stats['equivalent_at_margin']
    same=paired_statistics(index,'A','A',list(range(42,47)))
    assert same['delta_mae']==0 and same['equivalent_at_margin']
    # The deliberately best test model is NOT the deliberately best validation model.
    candidates=FAMILIES+['B0','PERSISTENCE']
    for site in range(56):
        for family in candidates:
            for seed in ([None] if family=='PERSISTENCE' else SEEDS):
                records.append(dict(family=family,site=site,seed=seed,val=dict(mae=.05 if family=='GRU_H' else .1),
                    test=dict(mae=.01 if family=='TFT_FC' else .2)))
    selection,chosen=selected_baseline(records,set(candidates))
    assert len(selection)==56 and len(chosen)==560
    assert all(r['family']=='GRU_H' for r in selection)
    assert complete(records,'GRU_H')
    incomplete=[r for r in records if not (r['family']=='GRU_H' and r['site']==55 and r['seed']==51)]
    assert not complete(incomplete,'GRU_H')
    assert selected_baseline(records,set(candidates)-{'TFT_FC'})==([],[])
    atomic_json(RUN/'ANALYSIS_PREFLIGHT.json',dict(passed=True,tests=['constant_effect_bootstrap_both_axes_and_time_blocks',
        'identical_predictions_equivalence','baseline_selection_ignores_test_ranking','incomplete_cohort_excluded']))
    print('ANALYSIS_PREFLIGHT_PASSED')

if __name__=='__main__':main()
