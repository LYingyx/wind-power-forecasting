"""Independent review of completed 2023 screening; never opens model/data caches."""
from pathlib import Path
import datetime
import hashlib
import json
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
PROJECT = OUT.parents[1]
SCREEN = PROJECT / 'runs/regional_reliability_validation_20261027_v1'
TF = 'TF_z1_lr0.0002_dp0.05'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    manifest = json.loads((SCREEN / 'manifest.json').read_text())
    audit = json.loads((SCREEN / 'COMPLETION_AUDIT.json').read_text())
    assert audit['passed'] and audit['tasks'] == len(manifest) == 4256
    overall = pd.read_csv(SCREEN / 'analysis/overall_per_farm_seed.csv')
    diag = pd.read_csv(SCREEN / 'analysis/scenarios_per_farm_seed.csv')
    key = ['configuration', 'variant', 'site', 'seed', 'horizon']
    assert not overall.duplicated(key).any()
    assert not diag.duplicated(key + ['group', 'bin']).any()
    assert set(overall.seed) == {42, 43}
    assert (overall.groupby(['configuration', 'variant', 'horizon']).size() == 112).all()
    table = overall.set_index(key)
    receipts = []
    for spec in manifest:
        job = SCREEN / 'jobs' / spec['id']
        assert (job / 'SUCCESS.json').is_file()
        raw = json.loads((job / 'metrics.json').read_text())
        assert not raw.get('smoke_only', False)
        selected = raw['selections']['mae']
        for horizon, value in [('all', selected['metrics']), *selected['metrics']['horizons'].items()]:
            row = table.loc[tuple(spec[k] for k in key[:-1]) + (horizon,)]
            for metric in ['mae', 'rmse', 'bias', 'n']:
                assert np.isclose(row[metric], value[metric], rtol=1e-12, atol=1e-14), (spec['id'], horizon, metric)
            assert row.epoch == selected['epoch']
        receipts.append(dict(id=spec['id'], metrics_sha256=sha(job / 'metrics.json')))
    print('All 4256 raw metrics reproduce all 17024 overall/horizon CSV rows.', flush=True)
    pd.DataFrame(receipts).to_csv(OUT / 'raw_metrics_audit.csv', index=False)

    recommendation = json.loads((SCREEN / 'analysis/SELECTION_RECOMMENDATION.json').read_text())
    common = recommendation['common_support_sites']
    assert len(common) == 41
    overall['group'] = 'overall'
    overall['bin'] = 'all'
    groups = ['current_power', 'high_outcome_from_origin', 'high_outcome_x_nwp', 'high_region_trend', 'ring0_presence']
    frame = pd.concat([overall, diag[diag.group.isin(groups)]], ignore_index=True)
    axes = ['configuration', 'variant', 'horizon', 'group', 'bin']
    farm = frame.groupby(axes + ['site'])[['mae', 'rmse', 'bias', 'n']].mean().sort_index()
    counts = frame.groupby(axes + ['site']).seed.nunique()
    assert (counts == 2).all()
    results = []
    cache = {}

    def effect(config, variant, reference, horizon='all', group='overall', label='all', support='all'):
        args = (config, variant, reference, horizon, group, label, support)
        if args in cache:
            return cache[args]
        left = farm.loc[(config, variant, horizon, group, label)]
        right = farm.loc[(config, reference, horizon, group, label)]
        if support == 'common':
            left = left.loc[left.index.intersection(common)]
            right = right.loc[right.index.intersection(common)]
        pair = left.join(right, lsuffix='_a', rsuffix='_b', how='inner')
        assert len(pair) and np.array_equal(pair.n_a, pair.n_b)
        n = len(pair)
        draw = np.random.default_rng(20261027).integers(0, n, (10000, n))
        value = dict(zip(['configuration', 'variant', 'reference', 'horizon', 'group', 'bin', 'support'], args))
        value['farms'] = n
        for metric in ['mae', 'rmse']:
            a = pair[metric + '_a'].to_numpy()
            b = pair[metric + '_b'].to_numpy()
            boots = 100 * (a[draw].mean(1) / b[draw].mean(1) - 1)
            value[metric + '_percent'] = float(100 * (a.mean() / b.mean() - 1))
            value[metric + '_wins'] = int((a < b).sum())
            value[metric + '_ties'] = int((a == b).sum())
            for q, suffix in [(.025, 'ci_low'), (.975, 'ci_high'), (.95, 'upper95')]:
                value[metric + '_' + suffix] = float(np.quantile(boots, q))
        value['bias_change'] = float((pair.bias_a - pair.bias_b).mean())
        results.append(value)
        cache[args] = value
        return value

    candidates = ['G_FREE', 'G_LEAD', 'G_STATE', 'G_STATE_LEAD', 'R3_MEAN_DELTA']
    for config in ['W0', TF]:
        for variant in ['R3_RAW', 'N3', 'R3_ALLPOOL'] + candidates:
            references = ['N0', 'N3', 'R3_RAW', 'G_FREE']
            if variant in candidates:
                references.append('C2_' + variant)
            if variant == 'G_STATE_LEAD':
                references.append('G_LEAD')
            for ref in dict.fromkeys(references):
                if variant == ref:
                    continue
                for horizon in ['all', '0_1h', '1_3h', '3_6h']:
                    effect(config, variant, ref, horizon)
                    effect(config, variant, ref, horizon, 'current_power', 'high')
            for group, labels in [('high_outcome_from_origin', ['drop_gt20', 'drop_5to20', 'steady5', 'rise_gt5']),
                                  ('high_outcome_x_nwp', ['drop_gt20__down']),
                                  ('high_region_trend', ['down', 'flat', 'up'])]:
                for label in labels:
                    for ref in ['N0', 'R3_RAW']:
                        if variant != ref:
                            effect(config, variant, ref, '3_6h', group, label)
    for variant in ['EXCLUDE20_KEEP', 'EXCLUDE20_REFILL']:
        for ref in ['R3_RAW', 'N0']:
            for horizon in ['all', '0_1h', '1_3h', '3_6h']:
                effect('W0', variant, ref, horizon)
                effect('W0', variant, ref, horizon, 'current_power', 'high')
            for label in ['absent', 'present']:
                effect('W0', variant, ref, 'all', 'ring0_presence', label)
    match = ['MATCH_INNER', 'MATCH_MIDDLE', 'MATCH_OUTER', 'MATCH_RANDOM']
    for variant in match:
        for ref in match:
            if variant != ref:
                for horizon in ['all', '0_1h', '1_3h', '3_6h']:
                    effect('W0', variant, ref, horizon, support='common')
    for variant, ref in zip(['FULL_INNER', 'FULL_MIDDLE', 'FULL_OUTER', 'FULL_ALL'], match):
        for horizon in ['all', '0_1h', '1_3h', '3_6h']:
            effect('W0', variant, ref, horizon, support='common')
    output = pd.DataFrame(results)
    output.to_csv(OUT / 'independent_effects.csv', index=False)
    auto = pd.read_csv(SCREEN / 'analysis/equal_farm_effects.csv')
    keys = ['configuration', 'variant', 'reference', 'horizon', 'group', 'bin', 'support']
    verified = 0
    for metric in ['mae', 'rmse']:
        matched = output.merge(auto[auto.metric == metric], on=keys, suffixes=('_review', '_auto'), validate='one_to_one')
        assert np.allclose(matched[metric + '_percent'], matched.effect_percent, rtol=1e-10, atol=1e-10)
        assert np.array_equal(matched[metric + '_wins'], matched.wins)
        assert np.allclose(matched[metric + '_ci_low'], matched.ci_low, atol=1e-9)
        assert np.allclose(matched[metric + '_ci_high'], matched.ci_high, atol=1e-9)
        verified += len(matched)

    # Additive MSE accounting uses each farm's total high-power count as denominator.
    # This is a separate statistic from the equal-farm RMSE ratios above.
    accounting = []
    tail = diag[diag.horizon == '3_6h']
    for config in ['W0', TF]:
        for variant, ref in [('R3_RAW', 'N0'), ('G_LEAD', 'R3_RAW'), ('G_STATE_LEAD', 'R3_RAW'), ('R3_MEAN_DELTA', 'R3_RAW')]:
            a = tail[(tail.configuration == config) & (tail.variant == variant)]
            b = tail[(tail.configuration == config) & (tail.variant == ref)]
            high_a = a[(a.group == 'current_power') & (a.bin == 'high')]
            high_b = b[(b.group == 'current_power') & (b.bin == 'high')]
            totals = high_a.merge(high_b, on=['site', 'seed'], suffixes=('_a', '_b'), validate='one_to_one')
            assert np.array_equal(totals.n_a, totals.n_b)
            totals['mse_change'] = (totals.sq_sum_a - totals.sq_sum_b) / totals.n_a
            net = float(totals.groupby('site').mse_change.mean().mean())
            parts = a[(a.group == 'high_outcome_from_origin') & (a.bin != 'not_high')].merge(
                b[(b.group == 'high_outcome_from_origin') & (b.bin != 'not_high')],
                on=['site', 'seed', 'bin'], suffixes=('_a', '_b'), validate='one_to_one')
            parts = parts.merge(totals[['site', 'seed', 'n_a']].rename(columns={'n_a': 'high_n'}), on=['site', 'seed'], validate='many_to_one')
            parts['contribution'] = (parts.sq_sum_a - parts.sq_sum_b) / parts.high_n
            parts['sample_share'] = parts.n_a / parts.high_n
            # A farm with no events in a bin contributes zero, not a missing farm.
            # Reindex before both seed and farm averaging to preserve additivity.
            bins = ['drop_gt20', 'drop_5to20', 'steady5', 'rise_gt5']
            complete = pd.MultiIndex.from_tuples(
                [(int(s), int(t), label) for s, t in totals[['site', 'seed']].itertuples(index=False, name=None) for label in bins],
                names=['site', 'seed', 'bin'])
            contributions = parts.set_index(['site', 'seed', 'bin'])[['contribution', 'sample_share']].reindex(complete, fill_value=0)
            summary = contributions.groupby(['bin', 'site']).mean().groupby('bin').mean()
            assert np.isclose(summary.contribution.sum(), net, rtol=1e-10, atol=1e-14)
            for label, row in summary.iterrows():
                accounting.append(dict(configuration=config, variant=variant, reference=ref, bin=label,
                    net_mse_change=net, mse_contribution=float(row.contribution),
                    fraction_of_net=float(row.contribution / net) if net else None,
                    sample_share=float(row.sample_share)))
    pd.DataFrame(accounting).to_csv(OUT / 'high_tail_mse_accounting.csv', index=False)
    result = dict(timestamp=datetime.datetime.now().astimezone().isoformat(), passed=True,
        tasks_verified=len(manifest), csv_overall_rows_verified=len(overall), independent_comparisons=len(output),
        automatic_metric_rows_crosschecked=verified, screen_audit=audit,
        input_hashes={name: sha(SCREEN / 'analysis' / name) for name in ['overall_per_farm_seed.csv', 'scenarios_per_farm_seed.csv', 'equal_farm_effects.csv']},
        open_2024=False, limitation='2023 selection evidence; paired farm bootstrap is not spatial-block inference or multiplicity correction; two seeds are not ten-seed replication')
    (OUT / 'REVIEW_AUDIT.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
