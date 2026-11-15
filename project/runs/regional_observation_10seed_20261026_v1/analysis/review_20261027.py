"""Independent completion review; reads existing 2023 metrics only."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import hashlib
import json
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
RUN = OUT.parent
PROJECT = RUN.parent.parent
D = PROJECT / 'runs/regional_observation_completion_20261026_v1'
TF = 'TF_z1_lr0.0002_dp0.05'
COHORTS = {'selection_2': [42, 43], 'new_8': list(range(44, 52)), 'all_10': list(range(42, 52))}
KEY = ['configuration', 'variant', 'site', 'seed', 'horizon']


def read_json(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    activation = read_json(RUN / 'ACTIVATION.json')
    for name, key in [('manifest.json', 'manifest_sha256'), ('REUSED_SOURCES.json', 'reused_sources_sha256')]:
        assert sha(RUN / name) == activation[key], name
    assert sha(D / 'analysis/D_SELECTION_RECOMMENDATION.json') == activation['source_recommendation_sha256']
    assert sha(D / 'COMPLETION_AUDIT.json') == activation['source_audit_sha256']
    new = read_json(RUN / 'manifest.json')
    reused = read_json(RUN / 'REUSED_SOURCES.json')
    assert len(new) == 5376 and len(reused) == 1344
    entries = reused + [dict(spec=s, path=str(RUN / 'jobs' / s['id'])) for s in new]
    rows = []
    for entry in entries:
        path = Path(entry['path'])
        assert (path / 'SUCCESS.json').exists(), path
        if 'metrics_sha256' in entry:
            assert sha(path / 'metrics.json') == entry['metrics_sha256'], path
        s = entry['spec']
        score = read_json(path / 'metrics.json')['selections']['mae']['metrics']
        for horizon, m in [('all', score), *score['horizons'].items()]:
            rows.append({**{k: s[k] for k in KEY[:-1]}, 'horizon': horizon,
                         **{k: m[k] for k in ['n', 'mae', 'rmse']}})
    raw = pd.DataFrame(rows)
    assert not raw.duplicated(KEY).any()
    assert len(raw) == 6720 * 4
    for _, g in raw.groupby(['configuration', 'variant', 'horizon']):
        assert set(g.site) == set(range(56)) and len(g) == 560
        assert g.groupby('site').seed.apply(lambda s: set(s) == set(range(42, 52))).all()
    assert set(map(tuple, raw[['configuration', 'variant']].drop_duplicates().to_numpy())) == set(map(tuple, activation['arms']))
    published = pd.read_csv(OUT / 'overall_per_farm_seed.csv')
    check = raw.merge(published, on=KEY, suffixes=('_raw', '_csv'), validate='one_to_one')
    assert len(check) == len(raw) == len(published)
    for m in ['n', 'mae', 'rmse']:
        np.testing.assert_allclose(check[m + '_raw'], check[m + '_csv'], rtol=0, atol=1e-12)
    high_parts = []
    for chunk in pd.read_csv(OUT / 'scenarios_per_farm_seed.csv', chunksize=100000):
        high_parts.append(chunk[(chunk['group'] == 'current_power') & (chunk.bin == 'high')])
    high = pd.concat(high_parts, ignore_index=True)
    assert not high.duplicated(KEY).any()
    assert len(high) == len(raw)
    effects = pd.read_csv(OUT / 'paired_effects.csv')

    def paired(table, cohort, cc, cv, rc, rv, horizon):
        f = table[(table.seed.isin(COHORTS[cohort])) & (table.horizon == horizon)]
        a = f[(f.configuration == cc) & (f.variant == cv)]
        b = f[(f.configuration == rc) & (f.variant == rv)]
        g = a.merge(b, on=['site', 'seed'], suffixes=('_a', '_b'), validate='one_to_one')
        assert len(g) == 56 * len(COHORTS[cohort]) and (g.n_a == g.n_b).all()
        return g

    validated = 0
    for table, data in [('overall', raw), ('scenarios', high)]:
        subset = effects[effects.table == table]
        if table == 'scenarios':
            subset = subset[(subset['group'] == 'current_power') & (subset.bin == 'high')]
        for r in subset.itertuples():
            g = paired(data, r.cohort, r.candidate_configuration, r.candidate_variant,
                       r.reference_configuration, r.reference_variant, r.horizon)
            m = r.metric
            farm = g.groupby('site')[[m + '_a', m + '_b']].mean()
            a, b = farm[m + '_a'], farm[m + '_b']
            value = 100 * (a.mean() / b.mean() - 1)
            assert abs(value - r.effect_percent) < 1e-9
            assert int((a < b).sum()) == r.improved_farms and r.farms == 56
            validated += 1

    gates = {}
    for cohort in COHORTS:
        gates[cohort] = {}
        for config in ['W0', TF]:
            for variant in ['R3_RAW', 'R3_ALLPOOL']:
                outcomes = {}
                for ref in ['N0', 'N3', 'C2_' + variant]:
                    g = paired(raw, cohort, config, variant, config, ref, 'all')
                    outcomes[ref] = {}
                    for m in ['mae', 'rmse']:
                        farm = g.groupby('site')[[m + '_a', m + '_b']].mean()
                        a, b = farm[m + '_a'], farm[m + '_b']
                        outcomes[ref][m] = dict(effect_percent=float(100 * (a.mean() / b.mean() - 1)),
                                               improved_farms=int((a < b).sum()))
                h = paired(high, cohort, config, variant, config, 'N3', 'all')
                h = h.groupby('site')[['rmse_a', 'rmse_b']].mean()
                hp = float(100 * (h.rmse_a.mean() / h.rmse_b.mean() - 1))
                passes = all(outcomes[ref][m]['effect_percent'] <= -.1 and outcomes[ref][m]['improved_farms'] >= 29
                             for ref in ['N0', 'N3'] for m in ['mae', 'rmse'])
                passes &= all(outcomes['C2_' + variant][m]['effect_percent'] < 0 for m in ['mae', 'rmse']) and hp <= .1
                gates[cohort][config + '/' + variant] = dict(comparisons=outcomes, high_power_rmse_vs_N3_percent=hp, passes=bool(passes))

    farm_rows, seed_rows = [], []
    edges = [(c, v, c, r) for c in ['W0', TF] for v in ['R3_RAW', 'R3_ALLPOOL']
             for r in ['N0', 'N3', 'C2_' + v]] + [('W0', 'R3_RAW', TF, 'R3_RAW')]
    for cc, cv, rc, rv in edges:
        g = paired(raw, 'all_10', cc, cv, rc, rv, 'all')
        meta = dict(candidate_configuration=cc, candidate_variant=cv, reference_configuration=rc, reference_variant=rv)
        for unit, output in [('site', farm_rows), ('seed', seed_rows)]:
            for value, values in g.groupby(unit):
                row = dict(**meta, **{unit: int(value)})
                for m in ['mae', 'rmse']:
                    a, b = values[m + '_a'].mean(), values[m + '_b'].mean()
                    row.update({m + '_candidate': a, m + '_reference': b, m + '_percent': 100 * (a / b - 1)})
                row['both_improved'] = bool(row['mae_percent'] < 0 and row['rmse_percent'] < 0)
                output.append(row)
    pd.DataFrame(farm_rows).to_csv(OUT / 'REVIEW_PER_FARM_20261027.csv', index=False)
    pd.DataFrame(seed_rows).to_csv(OUT / 'REVIEW_PER_SEED_20261027.csv', index=False)
    high_summary = effects[(effects.table == 'scenarios') & (effects.cohort == 'all_10') &
                           (effects['group'] == 'current_power') & (effects.bin == 'high') &
                           (effects.horizon == 'all')]
    high_summary.to_csv(OUT / 'REVIEW_HIGH_POWER_20261027.csv', index=False)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharex=True)
    for ax, metric in zip(axes, ['mae', 'rmse']):
        for config, label, color in [('W0', 'W0', '#1967a3'), (TF, 'TF*', '#cc5b28')]:
            f = effects[(effects.table == 'leads') & (effects.cohort == 'all_10') &
                        (effects.candidate_configuration == config) & (effects.reference_configuration == config) &
                        (effects.candidate_variant == 'R3_RAW') & (effects.reference_variant == 'N3') & (effects.metric == metric)].sort_values('lead_hours')
            assert len(f) == 24
            ax.plot(f.lead_hours, -f.effect_percent, label=label, color=color, linewidth=2)
        ax.axhline(0, color='0.6', linewidth=.8)
        ax.axvline(1, color='0.8', linestyle=':', linewidth=.8)
        ax.axvline(3, color='0.8', linestyle=':', linewidth=.8)
        ax.set(xlabel='Forecast horizon (hours)', ylabel=metric.upper() + ' reduction vs N3 (%)', xlim=(.25, 6))
        ax.grid(alpha=.2)
        ax.legend(frameon=False)
    fig.suptitle('R3_RAW vs 8-neighbor pooling | 2023 | 56 farms, 10 seeds')
    fig.tight_layout()
    fig.savefig(OUT / 'REVIEW_LEAD_EFFECTS_20261027.png', dpi=180)
    fig.savefig(OUT / 'REVIEW_LEAD_EFFECTS_20261027.svg')
    plt.close(fig)
    audit = dict(timestamp=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(), passed=True,
                 new_jobs=5376, reused_jobs=1344, complete_model_farm_seed_cells=6720,
                 raw_metric_rows_verified=len(raw), recomputed_effect_rows=validated,
                 unique_arms=12, farms=56, seeds=list(range(42, 52)),
                 metrics_selected_by='mae', source_activation_hashes_verified=True,
                 reused_metrics_hashes_verified=1344, gates=gates,
                 D_audit=read_json(D / 'COMPLETION_AUDIT.json'),
                 extension_audit=read_json(RUN / 'COMPLETION_AUDIT.json'),
                 older_ARX_status=read_json(PROJECT / 'runs/regional_residual_single_farm_20261025_v1/STATUS.json'),
                 scope='2023 metrics only; no new training or raw weather/power access; not an E selection freeze')
    (OUT / 'REVIEW_AUDIT_20261027.json').write_text(json.dumps(audit, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps({k: audit[k] for k in ['passed', 'new_jobs', 'reused_jobs', 'raw_metric_rows_verified', 'recomputed_effect_rows']}, ensure_ascii=False))
    print('gates', {cohort: {k: v['passes'] for k, v in g.items()} for cohort, g in gates.items()})


if __name__ == '__main__':
    main()
