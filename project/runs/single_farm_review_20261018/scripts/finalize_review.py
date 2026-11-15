"""Publish compact paper tables, statistical checks, and exportable figures."""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from audit_tables import ROOT,OUT,ART,save,bootstrap,summary

COLORS={'E2C':'#086f83','E0':'#83a8b3','TRANSFORMER_FC':'#d98c36','Persistence':'#92969c',
        'GRU_H':'#8b75a9','E2C_T5':'#086f83','Pretrained_Transformer':'#d98c36','T5_r16':'#83a8b3','Scratch_dev_selected':'#8b75a9'}
LABEL={'E2C':'E2C-CBH (+ regional history)','E0':'E0-CBH (+ regional history)',
       'TRANSFORMER_FC':'Shared Transformer','GRU_H':'Shared GRU (history)',
       'E2C_T5':'E2C adaptation','T5_r16':'CBH adaptation','Pretrained_Transformer':'Pretrained Transformer',
       'Scratch_dev_selected':'Scratch: dev-selected family','Persistence':'Persistence'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                     'axes.labelcolor':'#253445','text.color':'#253445','axes.titleweight':'bold',
                     'axes.grid':True,'grid.alpha':.18,'grid.linewidth':.5,'savefig.dpi':220,'pdf.fonttype':42})

def read(name,**kw):return pd.read_csv(OUT/'tables'/f'{name}.csv',**kw)
def figure(fig,name):
    fig.savefig(OUT/'figures'/f'{name}.png',bbox_inches='tight')
    fig.savefig(OUT/'figures'/f'{name}.pdf',bbox_inches='tight')
    plt.close(fig)

def crossed_blocks(blocks,split):
    d=blocks[blocks.split.eq(split)&blocks.model.isin(['E0','E2C'])]
    w=d.pivot(index=['seed','station','block'],columns='model',values='mae').sort_index()
    delta=(w.E2C-w.E0).to_numpy().reshape(10,56,-1)
    ns,nf,nb=delta.shape
    counts=d.groupby('block').origins.first().sort_index().to_numpy(float)
    rng=np.random.default_rng(20261018);draws=[]
    for _ in range(40):
        sw=rng.multinomial(ns,np.ones(ns)/ns,size=250)/ns
        fw=rng.multinomial(nf,np.ones(nf)/nf,size=250)/nf
        bw=rng.multinomial(nb,np.ones(nb)/nb,size=250)*counts
        a=np.einsum('rs,sfw->rfw',sw,delta)
        a=np.einsum('rf,rfw->rw',fw,a)
        draws.extend((a*bw).sum(1)/bw.sum(1))
    lo,hi=np.quantile(draws,[.025,.975])
    return dict(split=split,comparison='E2C-E0',delta=float((delta.mean((0,1))*counts).sum()/counts.sum()),
                ci_low=lo,ci_high=hi,bootstrap_replicates=10000,unit='crossed seeds, farms, common nonoverlapping 7-day blocks; valid-origin weights')

def main():
    pred=read('prediction_station_seed',dtype={'station':str})
    shared=read('shared_baselines_station_seed',dtype={'station':str})
    two=read('sharing_2x2_station_seed',dtype={'station':str})
    names={'SC':'CBH_SC','SL':'CBH_SL','IC':'CBH_IC','IL':'CBH_IL'}
    two['model']=two.model.map(names)
    merged=pd.concat([pred,shared,two],ignore_index=True)
    merged['context']=np.where(merged.model.isin(['E0','E2C','CBH_SC','CBH_IC']),'regional history','local history / weather only')
    merged['training']=np.where(merged.model.isin(['CBH_IC','CBH_IL']),'independent',np.where(merged.model.eq('Persistence'),'none','shared across source56'))
    save(merged,'source_full_year_station_seed',False)
    sm=summary(merged,['model','split','context','training'])
    sm['precision']=np.where(sm.model.isin(['E0','E2C']),'FP32',np.where(sm.model.eq('Persistence'),'deterministic','archived BF16 / see protocol'))
    save(sm,'source_full_year_system_comparison')
    # Compact table keeps local-only and context-assisted panels explicit.
    paper=[]
    for model in ['Persistence','GRU_H','LSTM_H','PATCHTST_H','ITRANSFORMER_H','MODERNTCN_H','CNN_W','VIT_W','SWIN_W','DLINEAR_FC','NBEATSX_FC','TCN_FC','TFT_FC','TIMEXER_FC','TIDE_FC','TRANSFORMER_FC','CBH_SL','CBH_IL','CBH_SC','CBH_IC','E0','E2C']:
        g=sm[sm.model.eq(model)].set_index('split')
        r={'model':model,'training':g.loc['test','training'],'context':g.loc['test','context']}
        for split in ['val','test']:
            r[f'{split}_mae']=g.loc[split,'mae'];r[f'{split}_seed_sd']=g.loc[split,'seed_sd']
            r[f'{split}_macro_rmse']=g.loc[split,'rmse_macro'];r[f'{split}_macro_r2']=g.loc[split,'r2_macro']
        paper.append(r)
    save(pd.DataFrame(paper),'paper_source_main')
    per=merged.groupby(['station','split','model']).mae.mean().unstack('model').reset_index()
    save(per,'source_all56_per_farm',False)
    reprs=read('source56_representatives',dtype={'station_id':str})
    compact=reprs[['station_id','station_name','capacity_mw','mean_cf','ramp_1h_q90','represented_farms']].merge(
        per[per.split.eq('test')][['station','E0','E2C','CBH_SL','CBH_IL','TRANSFORMER_FC']],left_on='station_id',right_on='station').drop(columns='station')
    compact['E2C_minus_Transformer']=compact.E2C-compact.TRANSFORMER_FC
    compact['local_CBH_minus_Transformer']=compact.CBH_SL-compact.TRANSFORMER_FC
    save(compact,'paper_source_representatives_2024')
    ps=per[per.split.eq('test')]
    contrasts=[]
    for split,g in per.groupby('split'):
        for l,r in [('E2C','TRANSFORMER_FC'),('E2C','E0'),('CBH_SL','TRANSFORMER_FC'),('CBH_SL','CBH_IL')]:
            contrasts.append(dict(split=split,comparison=f'{l}-{r}',unit='farm_after_seed_mean',**bootstrap(g[l]-g[r])))
    save(pd.DataFrame(contrasts),'source_main_paired_contrasts')
    blocks=read('prediction_week_station_seed',dtype={'station':str})
    save(pd.DataFrame([crossed_blocks(blocks,s) for s in ['val','test']]),'source_e2c_week_bootstrap')
    # Cross-check prior seasonal report and current prediction recomputation.
    old=pd.read_csv(ART/'final_e2c_extensions_20260930_v1/analysis/source_station_horizon_season.csv',dtype={'station_id':str})
    new=read('scenario_per_station',dtype={'station':str})
    seasonal=new[new.dimension.eq('season_origin_UTC')].pivot(index=['split','station','group'],columns='model',values='mae').reset_index()
    joined=seasonal.merge(old[old.dimension.eq('season')],left_on=['split','station','group'],right_on=['split','station_id','group'])
    max_season_diff=max((joined.E2C-joined.E2C_mae).abs().max(),(joined.E0-joined.E0_mae).abs().max())
    assert max_season_diff<2e-7,max_season_diff
    # Coverage is valid-cache coverage, not 365 days of uninterrupted observations.
    coverage=[]
    for split in ['train','val','test']:
        o=np.load(ROOT/f'data/model_cache_pad5_f8/{split}/origin_unix.npy');t=pd.to_datetime(o,unit='s',utc=True).tz_convert('Asia/Shanghai')
        for month in sorted(set(t.month)):
            coverage.append(dict(split=split,month_local=int(month),origins=int((t.month==month).sum()),
                first_local=str(t.min()),last_local=str(t.max()),max_gap_days=float(np.diff(o).max()/86400)))
    save(pd.DataFrame(coverage),'valid_origin_coverage')
    # Compact target table and fixed medoid details.
    target=read('target20_main_comparison')
    wide=target.pivot(index='budget',columns='model',values='mae').loc[['7d','30d','90d','180d','full']].reset_index()
    chosen=read('target20_scratch_dev_selected')[['budget','selected_family']]
    wide=wide.merge(chosen,on='budget');save(wide,'paper_target_learning_curve')
    treps=read('target20_representatives',dtype={'station_id':str})
    tper=read('target20_per_farm_budget',dtype={'station':str})
    tw=tper.pivot(index=['station','budget'],columns='model',values='mae').reset_index()
    tw=treps[['station_id','station_name','capacity_mw','represented_farms']].merge(tw,left_on='station_id',right_on='station').drop(columns='station')
    save(tw,'paper_target_representatives')
    # 1. Selection: location and training descriptors.
    sdesc=read('source56_descriptors',dtype={'station_id':str});tdesc=read('target20_descriptors',dtype={'station_id':str})
    fig,axes=plt.subplots(1,2,figsize=(11,4.4),constrained_layout=True)
    for d,label,color,marker in [(sdesc,'Source farms (2022 descriptors)','#086f83','o'),(tdesc,'Held-out targets (2023 descriptors)','#d98c36','s')]:
        for ax,x,y in [(axes[0],'longitude','latitude'),(axes[1],'mean_cf','ramp_1h_q90')]:
            ax.scatter(d[x],d[y],s=25,c=color,alpha=.25,marker=marker,label=label)
            chosen=d[d.representative]
            ax.scatter(chosen[x],chosen[y],s=80,edgecolor='#253445',lw=.8,c=color,marker=marker)
            for _,r in chosen.iterrows():ax.annotate(str(r.station_id),(r[x],r[y]),xytext=(4,5),textcoords='offset points',fontsize=7)
    axes[0].set(xlabel='Longitude',ylabel='Latitude',title='Farm coverage and fixed medoids')
    axes[1].set(xlabel='Training-period mean capacity factor',ylabel='Training |1-hour change|: 90th percentile',title='Selection uses no prediction errors')
    axes[0].legend(fontsize=8,loc='upper right');figure(fig,'01_farm_selection')
    # 2. Source result: matched evaluation, different input/training packages flagged.
    models=['GRU_H','CNN_W','TFT_FC','TIMEXER_FC','TIDE_FC','TRANSFORMER_FC','CBH_SL','E0','E2C']
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),sharey=True,constrained_layout=True)
    for ax,split in zip(axes,['val','test']):
        g=sm[sm.split.eq(split)].set_index('model').loc[models]
        colors=['#b9c5d1']*6+['#48688f','#83a8b3','#086f83']
        ax.barh(np.arange(len(g)),g.mae,xerr=g.seed_sd.fillna(0),color=colors,capsize=2)
        ax.set_yticks(np.arange(len(g)),['GRU (history only)','CNN (weather only)','TFT','TimeXer','TiDE','Transformer','CBH: local-only input','E0: regional history','E2C: regional history'])
        ax.set(xlim=(0,.135),xlabel='Capacity-factor MAE; whiskers = seed SD',title='2023 validation' if split=='val' else '2024 cross-year evaluation')
        ax.invert_yaxis() if split=='val' else None
        for i,v in enumerate(g.mae):ax.text(v+.0015,i,f'{v:.4f}',va='center',fontsize=8)
    figure(fig,'02_source_baseline_comparison')
    # 3. All lead horizons, common 56-farm evaluation.
    leads=pd.concat([read('prediction_lead_seed'),read('shared_baselines_lead_seed')])
    save(leads,'all_models_lead_seed',False)
    fig,axes=plt.subplots(1,2,figsize=(10,4),sharey=True,constrained_layout=True)
    for ax,split in zip(axes,['val','test']):
        for model in ['Persistence','GRU_H','TRANSFORMER_FC','E0','E2C']:
            g=leads[leads.split.eq(split)&leads.model.eq(model)].groupby('lead_hours').mae.agg(['mean','std'])
            ax.plot(g.index,g['mean'],label=LABEL[model],c=COLORS[model],lw=2 if model=='E2C' else 1.4)
        ax.set(xlabel='Forecast lead (hours)',ylabel='Mean per-farm CF MAE',title='2023' if split=='val' else '2024',xlim=(.25,6))
    axes[1].legend(fontsize=7,loc='lower right');figure(fig,'03_forecast_horizons')
    # 4. Descriptive scene contrasts, retaining all farms.
    sc=read('scenario_e2c_minus_e0');dims=['season_origin_UTC','origin_hour_Asia_Shanghai','observed_power_at_origin','observed_1h_ramp_train_q90']
    orders=[['DJF','MAM','JJA','SON'],['00-06','06-12','12-18','18-24'],['CF<0.1','0.1<=CF<0.3','0.3<=CF<0.7','CF>=0.7'],['past_1h_fall','past_1h_moderate','past_1h_rise']]
    titles=['Meteorological season (origin UTC)','Origin time (Asia/Shanghai)','Observed power at forecast origin','Observed past 1h ramp (training threshold)']
    fig,axes=plt.subplots(2,2,figsize=(11,7),constrained_layout=True)
    for ax,dim,order,title in zip(axes.flat,dims,orders,titles):
        for split,offset,color,label in [('val',-.12,'#83a8b3','2023'),('test',.12,'#086f83','2024')]:
            g=sc[sc.dimension.eq(dim)&sc.split.eq(split)].set_index('group').loc[order]
            vals=g.delta.to_numpy()*1000
            ax.errorbar(np.arange(len(order))+offset,vals,yerr=np.vstack([vals-g.ci_low.to_numpy()*1000,g.ci_high.to_numpy()*1000-vals]),fmt='o',capsize=3,color=color,label=label)
        ax.axhline(0,c='#667384',lw=.8);ax.set_xticks(np.arange(len(order)),[x.replace('past_1h_','') for x in order],fontsize=8)
        ax.set(title=title,ylabel='E2C - E0 MAE (x 1,000)')
    axes[0,0].legend();figure(fig,'04_scenario_contrasts')
    # 5. Transfer with dev-frozen scratch family and a strong pretrained comparison.
    budgets=['7d','30d','90d','180d','full'];x=np.arange(5)
    fig,axes=plt.subplots(1,2,figsize=(11,4.3),constrained_layout=True)
    for model in ['Scratch_dev_selected','T5_r16','E2C_T5','Pretrained_Transformer']:
        g=target[target.model.eq(model)].set_index('budget').loc[budgets]
        axes[0].plot(x,g.mae,'o-',label=LABEL[model],color=COLORS[model],lw=1.7)
    axes[0].set_xticks(x,['7d','30d','90d','180d','Full 2023']);axes[0].set(xlabel='Adaptation training budget*',ylabel='2024 H2: equal-farm CF MAE',title='20 held-out farms; common 5 seeds')
    axes[0].legend(fontsize=8)
    con=read('target20_paired_contrasts');g=con[con.comparison.eq('E2C_T5-Pretrained_Transformer')].set_index('budget').loc[budgets]
    axes[1].errorbar(x,g.delta*1000,yerr=np.vstack([(g.delta-g.ci_low)*1000,(g.ci_high-g.delta)*1000]),fmt='o',capsize=4,color='#086f83')
    axes[1].axhline(0,c='#667384',lw=.8);axes[1].set_xticks(x,['7d','30d','90d','180d','Full'])
    axes[1].set(ylabel='E2C - pretrained Transformer MAE (x 1,000)',title='Paired farm-bootstrap 95% intervals',xlabel='Positive values favor Transformer')
    fig.supxlabel('* Full 2024 H1 labels were used for checkpoint selection. Historical precision differs across transfer arms.',fontsize=8)
    figure(fig,'05_transfer_learning_curve')
    # 6. Forecast examples use a fixed seed, not ensemble scoring or best-case selection.
    cases=pd.concat([read('forecast_cases'),read('forecast_cases_transformer')])
    meta=read('forecast_case_selection',dtype={'station_id':str})
    save(cases,'all_forecast_cases',False)
    for group,farm_list in [('06_source_forecast_examples',reprs.station_id.tolist()),('07_failure_forecast_examples',['1043056'])]:
        fig,axes=plt.subplots(len(farm_list),2,figsize=(12,2.4*len(farm_list)),squeeze=False,constrained_layout=True)
        for i,station in enumerate(farm_list):
            for j,kind in enumerate(['ordinary','recent_ramp']):
                ax=axes[i,j];caseid=f'{station}_{kind}';g=cases[cases.case_id.eq(caseid)]
                if len(g)==0:continue
                observed=g[g.model.isin(['Observed_history','Observed_future'])].sort_values('lead_hours')
                ax.plot(observed.lead_hours,observed.value_cf,c='#172d40',lw=1.8,label='Observed')
                for model,label,color in [('E2C','E2C, seed 42','#086f83'),('E0','E0, seed 42','#83a8b3'),('Shared_Transformer_FP32_case','Transformer, seed 42 (FP32)','#d98c36'),('Persistence','Persistence','#92969c')]:
                    h=g[g.model.eq(model)]
                    line=h[h.seed.eq(42)] if model!='Persistence' else h
                    ax.plot(line.lead_hours,line.value_cf,c=color,lw=1.5,label=label,ls='--' if model=='Persistence' else '-')
                    if model=='E2C':
                        band=h.groupby('lead_hours').value_cf.quantile([.1,.9]).unstack()
                        ax.fill_between(band.index,band[.1],band[.9],color=color,alpha=.13)
                ax.axvline(0,c='#77869a',lw=.8,ls=':');m=meta[meta.case_id.eq(caseid)].iloc[0]
                ax.set(title=f'{station} | {kind.replace("_"," ")} | {str(m.origin_local)[:16]}',xlabel='Hours from forecast origin',ylabel='Capacity factor',xlim=(-2,6))
        handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncol=3,fontsize=8)
        fig.suptitle('Fixed-rule descriptive cases; E2C shading = 10-90% seed variation, not predictive uncertainty',fontsize=11)
        figure(fig,group)
    # 8. Fixed target medoids deliberately include the transfer failure regime.
    fig,axes=plt.subplots(2,2,figsize=(10,7),constrained_layout=True)
    for ax,station in zip(axes.flat,treps.station_id):
        for model in ['Scratch_dev_selected','T5_r16','E2C_T5','Pretrained_Transformer']:
            g=tper[tper.station.eq(station)&tper.model.eq(model)].set_index('budget').loc[budgets]
            ax.plot(x,g.mae,'o-',color=COLORS[model],label=LABEL[model],lw=1.5)
        ax.set_xticks(x,['7d','30d','90d','180d','Full']);ax.set(title=f'Training-descriptor medoid: {station}',ylabel='2024 H2 CF MAE')
    axes[0,0].legend(fontsize=7);figure(fig,'08_target_medoid_learning_curves')
    audit={'seasonal_rows_checked':len(joined),'max_difference_vs_archived_seasonal_report':float(max_season_diff),
           'figures':8,'primary_statistic':'mean single-model error over seeds, equal farms; no ensemble metric',
           'macro_rmse_definition':'mean across farms of sqrt(within-farm MSE), then mean seeds; differs from sqrt(mean farm MSE)',
           'case_lines':'seed42 fixed; E2C/E0 saved predictions, Transformer new FP32 CPU inference, no performance-based origin selection'}
    (OUT/'final_verification.json').write_text(json.dumps(audit,indent=2)+'\n');print(json.dumps(audit,indent=2))

if __name__=='__main__':main()
