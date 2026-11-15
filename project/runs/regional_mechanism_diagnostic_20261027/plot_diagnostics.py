from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent
a=pd.read_csv(HERE/'HIGH_POWER_EFFECTS.csv')
order=['0_1h','1_3h','3_6h'];labels=['0-1 h','1-3 h','3-6 h']
colors={'N3':'#777777','R3_RAW':'#d55e00','R3_ALLPOOL':'#0072b2'}
fig,axes=plt.subplots(1,2,figsize=(10,3.8),sharey=True,layout='constrained')
for ax,config,title in zip(axes,['W0','TF_z1_lr0.0002_dp0.05'],['W0','Tuned Transformer']):
    for variant in colors:
        z=a[(a.configuration==config)&(a.variant==variant)&(a.reference=='N0')].set_index('horizon').loc[order]
        ax.plot(labels,z.rmse_percent,'o-',label=variant,color=colors[variant],lw=2)
    ax.axhline(0,color='black',lw=.8);ax.grid(alpha=.2);ax.set_title(title);ax.set_xlabel('Forecast horizon')
axes[0].set_ylabel('High-origin-power RMSE change vs N0 (%)')
axes[1].legend(frameon=False)
fig.suptitle('2023; 56 farms; 10 seeds per farm; origin CF >= 0.7\nNegative = improvement; N0 = backbone + SEL',fontsize=11)
for ext in ['png','svg']:fig.savefig(HERE/f'HIGH_POWER_HORIZONS.{ext}',dpi=180)
plt.close(fig)
if not (HERE/'REPLAY_GROUP_EFFECTS.csv').exists():raise SystemExit(0)
r=pd.read_csv(HERE/'REPLAY_GROUP_EFFECTS.csv')
fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
outcomes=['drop_gt20','drop_5to20','steady_5','rise_gt5']
names=['Drop >0.20','Drop 0.05-0.20','Within +/-0.05','Rise >0.05']
z=r[(r.variant=='R3_RAW')&(r.group=='outcome_from_origin')&(r.horizon=='3_6h')].set_index('bin').loc[outcomes]
axes[0].bar(names,z.fraction_of_high*100,color='#999999');axes[0].set_ylabel('High-power forecast cells (%)')
axes[0].set_title('Observed future change relative to origin')
axes[1].bar(names,z.delta_sq_contribution*1e4,color=['#d55e00' if v>0 else '#0072b2' for v in z.delta_sq_contribution])
axes[1].set_ylabel('Contribution to high-power MSE change\n(x 1e-4; positive = worse)')
axes[1].set_title('R3_RAW vs N0: exact additive decomposition')
for ax in axes:
    ax.axhline(0,color='black',lw=.8);ax.tick_params(axis='x',rotation=20);ax.grid(axis='y',alpha=.2)
fig.suptitle('Exploratory replay: 2023, 56 farms, seeds 44/45, 3-6 h\nOutcome bins are diagnostic only, unavailable at forecast origin',fontsize=11)
for ext in ['png','svg']:fig.savefig(HERE/f'HIGH_POWER_OUTCOME_DECOMPOSITION.{ext}',dpi=180)
