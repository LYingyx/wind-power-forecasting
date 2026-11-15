"""Standalone research figure from audited, aggregate screening metrics."""
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parent
data = pd.read_csv(root / 'independent_effects.csv')
fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True, constrained_layout=True)
variants = [('R3_RAW', 'Original regional model'), ('G_LEAD', 'Lead decay'),
            ('G_STATE_LEAD', 'State + lead decay'), ('R3_MEAN_DELTA', 'Mean + change only')]
colors = ['#bd4940', '#2166ac', '#6b4c9a', '#25826b']
horizons = ['0_1h', '1_3h', '3_6h']
for ax, config, name in zip(axes, ['W0', 'TF_z1_lr0.0002_dp0.05'], ['W0 + SEL', 'Tuned Transformer + SEL']):
    for (variant, label), color in zip(variants, colors):
        rows = data[(data.configuration == config) & (data.variant == variant) &
                    (data.reference == 'N0') & (data.group == 'current_power') &
                    (data.bin == 'high')].set_index('horizon').loc[horizons]
        ax.plot(range(3), rows.rmse_percent, marker='o', label=label, color=color, linewidth=1.8)
    ax.axhline(0, color='#444444', linewidth=.8)
    ax.set_xticks(range(3), ['0–1 h', '1–3 h', '3–6 h'])
    ax.set_title(name)
    ax.set_xlabel('Forecast horizon')
    ax.grid(axis='y', alpha=.2)
axes[0].set_ylabel('RMSE change vs own-farm baseline (%)\nNegative = improvement')
axes[1].legend(fontsize=8, loc='lower right')
fig.suptitle('High origin power (CF ≥ 0.7): 2023 screening, 56 farms, 2 seeds', fontsize=11)
fig.savefig(root / 'high_power_horizon.png', dpi=200)
fig.savefig(root / 'high_power_horizon.svg')
