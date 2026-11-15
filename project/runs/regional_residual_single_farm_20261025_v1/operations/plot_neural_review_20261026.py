"""Plot completed neural results; no ARX or test-year data involved."""
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

folder = Path(__file__).resolve().parents[1] / 'analysis'
data = pd.read_csv(folder / 'lead_per_farm_seed.csv')
fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharex=True)
rows = []
for metric, ax in zip(['mae', 'rmse'], axes):
    ref = data[(data.configuration == 'W0') & (data.variant == 'N3')]
    for variant in ['R1', 'R3', 'R3_RAW', 'R4']:
        candidate = data[(data.configuration == 'W0') & (data.variant == variant)]
        paired = candidate.merge(ref, on=['site', 'seed', 'lead_hours'],
                                 suffixes=('_candidate', '_reference'), validate='one_to_one')
        assert len(paired) == 56 * 2 * 24
        assert (paired.n_candidate == paired.n_reference).all()
        farm = paired.groupby(['site', 'lead_hours'])[[metric+'_candidate', metric+'_reference']].mean()
        means = farm.groupby('lead_hours').mean()
        effect = 100 * (means[metric+'_candidate'] / means[metric+'_reference'] - 1)
        ax.plot(effect.index, effect, label=variant, lw=1.7)
        rows.extend(dict(variant=variant, metric=metric, lead_hours=float(h),
                         effect_percent=float(v), farms=56) for h, v in effect.items())
    ax.axhline(0, color='black', lw=.8)
    ax.axhline(-.1, color='gray', lw=.7, ls=':')
    ax.set(xlabel='Forecast lead (h)', ylabel=metric.upper()+' change vs W0 N3 (%)',
           xlim=(.25, 6))
    ax.grid(alpha=.2); ax.legend(frameon=False)
fig.suptitle('2023 screening: W0 regional observations and residual variants', fontsize=11)
fig.tight_layout()
fig.savefig(folder / 'NEURAL_LEAD_EFFECTS_20261026.png', dpi=200)
fig.savefig(folder / 'NEURAL_LEAD_EFFECTS_20261026.pdf')
plt.close(fig)
pd.DataFrame(rows).to_csv(folder / 'NEURAL_LEAD_EFFECTS_20261026.csv', index=False)
