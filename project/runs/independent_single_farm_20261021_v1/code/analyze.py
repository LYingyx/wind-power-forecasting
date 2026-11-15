"""Automatic complete-cohort tables, paired uncertainty and publication plots.

Partial progress exports raw rows only. No incomplete model is promoted to a
main-table comparison. Selection uses validation only; plots never select cases
by prediction error. Bootstrap draws are fixed and paired across comparisons.
"""
from common import *
from train import metrics
import collections,csv,datetime,sqlite3,fcntl
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT=RUN/'analysis'
EXPECTED={f:10 for f in ['B0','PERSISTENCE','E2C_CBH','SELECTED_BASELINE']+FAMILIES+['E0']}
EXPECTED.update({f:5 for f in ARMS if f!='E0'})
CONTRASTS=[('CB','CNEUTRAL'),('E2C_CBH','CH'),('CH','CNEUTRAL'),('E2C_CBH','CB'),
           ('DIRECT','E2C_CBH'),('CNEUTRAL','NEUTRAL_LAST'),('E2C_CBH','E0'),('E2C_CBH','CLIMATOLOGY')]

def export(name,rows,columns=None):
    path=OUT/f'{name}.csv';temp=path.with_suffix('.tmp')
    pd.DataFrame(rows,columns=columns).to_csv(temp,index=False);os.replace(temp,path)

def figure_save(fig,name):
    fig.tight_layout();fig.savefig(OUT/'figures'/f'{name}.pdf',bbox_inches='tight')
    fig.savefig(OUT/'figures'/f'{name}.png',dpi=240,bbox_inches='tight');plt.close(fig)

def persistence():
    dest=OUT/'persistence_metrics.json'
    if dest.exists():return json.loads(dest.read_text())
    result=[]
    for site in range(56):
        record=dict(family='PERSISTENCE',site=site,seed=None,stage='deterministic',id=f'persistence_s{site:02d}',
                    station_id=str(static()['station_id'][site]),seconds_training=0,trainable_parameters=0,total_parameters=0,active_gradient_parameters=0)
        for split in ['val','test']:
            d=RUN/'prepared'/split/f'{site:02d}';h=np.load(d/'history.npy',mmap_mode='r');y=np.load(d/'target.npy',mmap_mode='r')
            record[split]=metrics(np.broadcast_to(h[:,-1:],y.shape),y,h,site,split)
        result.append(record)
    atomic_json(dest,result);return result

def load_records():
    db=sqlite3.connect(RUN/'queue.sqlite');rows=db.execute("SELECT id FROM jobs WHERE status='done' AND stage!='tune' AND seed>=42").fetchall();db.close()
    records=[]
    for (jid,) in rows:
        m=json.loads((RUN/'jobs'/jid/'metrics.json').read_text());assert not m['smoke_only'];records.append(m)
    return records

def freeze_cases():
    dest=OUT/'case_selection.csv'
    if dest.exists():return pd.read_csv(dest,dtype={'station_id':str})
    src=ROOT/'runs/single_farm_review_20261018/tables/forecast_case_selection.csv'
    df=pd.read_csv(src,dtype={'station_id':str});rows=df.to_dict('records');ids=list(map(str,static()['station_id']))
    # Existing windows are retained exactly. Only missing failure-farm cases are added.
    for station in ['1043056','1043061']:
        site=ids.index(station);h=np.load(RUN/'prepared/test'/f'{site:02d}'/'history.npy',mmap_mode='r')
        origin=np.load(RUN/'prepared/test/origin_unix.npy');local=pd.to_datetime(origin,unit='s',utc=True).tz_convert('Asia/Shanghai')
        desc=json.loads((RUN/'prepared/train'/f'{site:02d}'/'train_descriptors.json').read_text());q=desc['ramp_q90']
        change=h[:,-1]-h[:,-5];current=h[:,-1]
        for kind in ['ordinary','recent_ramp']:
            if any(str(r['station_id'])==station and r['kind']==kind for r in rows):continue
            mask=((current>=.1)&(current<.7)&(abs(change)<=q)) if kind=='ordinary' else abs(change)>q
            selected=np.flatnonzero(mask&(local.hour==6)&(local.minute==0)&(local.month==4));fallback=False
            if not len(selected):selected=np.flatnonzero(mask&(local.hour==6)&(local.minute==0));fallback=True
            if not len(selected):continue
            ix=int(selected[0]);rows.append(dict(case_id=f'{station}_{kind}',station_id=station,kind=kind,source='prespecified_failure_farm_history_rule',
                origin_index=ix,station_index=site,origin_unix=int(origin[ix]),origin_local=str(local[ix]),current_cf=float(current[ix]),
                observed_ramp_1h=float(change[ix]),train_ramp_q90=q,fallback_full_year=fallback))
    df=pd.DataFrame(rows);df.to_csv(dest,index=False);return df

def complete(records,family):
    selected=[m for m in records if m['family']==family]
    count=collections.Counter(m['site'] for m in selected)
    return len(count)==56 and all(n==(1 if family=='PERSISTENCE' else EXPECTED[family]) for n in count.values())

def indexed(records):
    d={}
    for m in records:
        for seed in SEEDS if m['family']=='PERSISTENCE' else [m['seed']]:d[m['family'],m['site'],seed]=m
    return d

def paired_statistics(index,left,right,seeds):
    a=np.array([[index[left,s,r]['test']['mae'] for r in seeds] for s in range(56)])
    b=np.array([[index[right,s,r]['test']['mae'] for r in seeds] for s in range(56)])
    diff=a-b;rng=np.random.default_rng(20261021)
    fs=rng.integers(0,56,size=(5000,56));rs=rng.integers(0,len(seeds),size=(5000,len(seeds)))
    draws=np.array([diff[f[:,None],r[None,:]].mean() for f,r in zip(fs,rs)])
    low,high=np.quantile(draws,[.025,.975]);l90,h90=np.quantile(draws,[.05,.95]);margin=PROTOCOL['equivalence_margin_cf']
    # Joint-date bootstrap preserves simultaneous cross-farm weather dependence.
    day_dict=index[left,0,seeds[0]]['test']['daily_utc'];days=np.arange(min(map(int,day_dict)),max(map(int,day_dict))+1)
    weights=np.array([day_dict.get(str(d),{}).get('n',0) for d in days],dtype=float)
    daily=np.zeros(len(days))
    for site in range(56):
        for seed in seeds:
            aa=index[left,site,seed]['test']['daily_utc'];bb=index[right,site,seed]['test']['daily_utc']
            for k,day in enumerate(days):
                if weights[k]:daily[k]+=(aa[str(day)]['abs_sum']-bb[str(day)]['abs_sum'])/24
    daily/=56*len(seeds)
    blocks=[]
    for _ in range(2000):
        starts=rng.integers(0,len(days),size=(len(days)+6)//7)
        ix=((starts[:,None]+np.arange(7))%len(days)).flatten()[:len(days)]
        blocks.append(daily[ix].sum()/weights[ix].sum())
    blo,bhi=np.quantile(blocks,[.025,.975])
    return dict(left=left,right=right,n_farms=56,n_seeds=len(seeds),left_mae=float(a.mean()),right_mae=float(b.mean()),
        delta_mae=float(diff.mean()),relative_improvement_percent=float((b.mean()-a.mean())/b.mean()*100),
        farm_seed_ci95_low=float(low),farm_seed_ci95_high=float(high),temporal_7day_ci95_low=float(blo),temporal_7day_ci95_high=float(bhi),
        seed_delta_std=float(diff.mean(0).std(ddof=1)),farms_left_better=int((diff.mean(1)<0).sum()),
        equivalence_margin_cf=margin,ci90_low=float(l90),ci90_high=float(h90),equivalent_at_margin=bool(l90>-margin and h90<margin),
        interpretation='conditional regional sample; interval crossing zero does not establish equivalence')

def disaggregated_contrasts(index,left,right,seeds):
    """Paired descriptive intervals, with the same available farms in both arms."""
    result=[];rng=np.random.default_rng(20261021)
    keys=[('lead',k) for k in range(24)]+[('scenario',k) for k in index[left,0,seeds[0]]['test']['scenarios']]
    for kind,key in keys:
        differences=[];count=0
        for site in range(56):
            values=[];n=0
            for seed in seeds:
                a=index[left,site,seed]['test'];b=index[right,site,seed]['test']
                if kind=='lead':values.append(a['lead_mae'][key]-b['lead_mae'][key]);n=a['n_origins']
                else:
                    aa=a['scenarios'][key];bb=b['scenarios'][key];assert aa['n_origins']==bb['n_origins']
                    if aa['n_origins']:values.append(aa['mae']-bb['mae']);n=aa['n_origins']
            if len(values)==len(seeds):differences.append(values);count+=n
        if not differences:continue
        d=np.asarray(differences);fs=rng.integers(0,len(d),size=(2000,len(d)));rs=rng.integers(0,len(seeds),size=(2000,len(seeds)))
        means=np.array([d[f[:,None],r[None,:]].mean() for f,r in zip(fs,rs)]);low,high=np.quantile(means,[.025,.975])
        result.append(dict(left=left,right=right,kind=kind,group=(key+1)/4 if kind=='lead' else key,
            n_farms=len(d),n_seeds=len(seeds),n_farm_origin_pairs=count,delta_mae=float(d.mean()),ci95_low=float(low),ci95_high=float(high),
            future_information_diagnostic_only=key=='future_ramp_diagnostic_only'))
    return result

def selected_baseline(records,ready):
    needed=set(FAMILIES+['B0','PERSISTENCE'])
    if not needed.issubset(ready):return [],[]
    ix=indexed(records);selection=[];chosen=[]
    for site in range(56):
        scores={f:float(np.mean([ix[f,site,r]['val']['mae'] for r in SEEDS])) for f in sorted(needed)}
        family=min(scores,key=lambda f:(scores[f],f))
        selection.append(dict(site=site,station_id=str(static()['station_id'][site]),family=family,validation_mae=scores[family],selection='2023_mean_over_10_seeds'))
        for seed in SEEDS:
            m=dict(ix[family,site,seed]);m.update(family='SELECTED_BASELINE',seed=seed,selected_family=family);chosen.append(m)
    return selection,chosen

def make_cases(cases,index,ready,selection):
    if not {'E2C_CBH','TRANSFORMER_FC','B0'}.issubset(ready):return
    choices={r['site']:r['family'] for r in selection};rows=[]
    for case in cases.to_dict('records'):
        site=int(case['station_index']);oi=int(case['origin_index']);h=np.load(RUN/'prepared/test'/f'{site:02d}'/'history.npy',mmap_mode='r')[oi]
        truth=np.load(RUN/'prepared/test'/f'{site:02d}'/'target.npy',mmap_mode='r')[oi]
        families=list(dict.fromkeys(['B0','E2C_CBH','TRANSFORMER_FC',choices.get(site,'TRANSFORMER_FC')]))
        fig,ax=plt.subplots(figsize=(7.2,3.4));leads=np.arange(1,25)/4
        ax.plot(np.arange(-23,1)/4,h,color='.55',lw=1,label='Observed history');ax.plot(leads,truth,color='black',lw=1.8,label='Observed future')
        for family in families:
            if family=='PERSISTENCE':arr=np.full((10,24),h[-1])
            else:arr=np.stack([np.load(RUN/'jobs'/index[family,site,r]['id']/'test_prediction.npy',mmap_mode='r')[oi] for r in SEEDS])
            mean=arr.mean(0);sd=arr.std(0,ddof=1);ax.plot(leads,mean,lw=1.5,label=family)
            if family=='E2C_CBH':ax.fill_between(leads,mean-sd,mean+sd,alpha=.15,label='E2C-CBH ±1 seed s.d.')
            for k in range(24):rows.append(dict(case_id=case['case_id'],station_id=case['station_id'],origin_unix=case['origin_unix'],family=family,
                        lead_hours=float(leads[k]),truth=float(truth[k]),prediction_mean=float(mean[k]),prediction_seed_std=float(sd[k])))
        ax.axvline(0,color='.5',ls='--',lw=.7);ax.set(xlabel='Hours relative to issuance',ylabel='Power / rated capacity',title=f"Farm {case['station_id']} | {case['kind']}")
        ax.legend(fontsize=7,ncol=2,frameon=False);figure_save(fig,'case_'+case['case_id'])
    export('forecast_cases',rows)

def main():
    OUT.mkdir(exist_ok=True);(OUT/'figures').mkdir(exist_ok=True)
    lock=open(OUT/'analysis.lock','w');fcntl.flock(lock,fcntl.LOCK_EX)
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42})
    records=load_records();cases=freeze_cases()
    selections=[]
    for path in sorted((RUN/'selected').glob('*.json')):
        selected=json.loads(path.read_text())
        selections.append(dict(family=selected['family'],selected_config_index=selected['selected_config_index'],**selected['config'],
            selection_split=selected['selection_split'],validation_mae=selected['validation_macro_mae'][str(selected['selected_config_index'])]))
    export('selected_hyperparameters',selections)
    db=sqlite3.connect(RUN/'queue.sqlite');search_rows=db.execute("SELECT id FROM jobs WHERE status='done' AND (stage='tune' OR (stage='reference' AND seed<42))").fetchall();db.close()
    search_cost=[]
    for (jid,) in search_rows:
        m=json.loads((RUN/'jobs'/jid/'metrics.json').read_text())
        search_cost.append({k:m.get(k) for k in ['id','family','stage','site','seed','seconds_training','epochs_run','peak_reserved_mib']})
    export('validation_search_cost',search_cost)
    # Deterministic baseline is generated once; it never drives hyperparameter tuning.
    records+=persistence();ready={f for f in EXPECTED if complete(records,f)}
    selection,chosen=selected_baseline(records,ready);records+=chosen
    if chosen:ready.add('SELECTED_BASELINE');export('validation_selected_baselines',selection)
    index=indexed(records);raw=[];leadrows=[];scenario=[];cost=[]
    for m in records:
        if m['family']=='SELECTED_BASELINE':continue
        cost.append({k:m.get(k) for k in ['id','family','site','seed','seconds_training','trainable_parameters','total_parameters','active_gradient_parameters','peak_reserved_mib','best_epoch','epochs_run']})
    db=sqlite3.connect(RUN/'queue.sqlite')
    timings={jid:dict(worker_elapsed_seconds=(datetime.datetime.fromisoformat(end)-datetime.datetime.fromisoformat(start)).total_seconds(),attempts=attempts)
             for jid,start,end,attempts in db.execute("SELECT id,started,finished,attempts FROM jobs WHERE status='done' AND started IS NOT NULL AND finished IS NOT NULL")}
    db.close()
    for row in cost:row.update(timings.get(row['id'],{}))
    cost_lookup={r['id']:r for r in cost}
    for row in cost:
        if row['family']=='E2C_CBH':
            reference=cost_lookup.get(ref_id(row['site'],row['seed']),{})
            row['two_stage_training_seconds']=row['seconds_training']+reference.get('seconds_training',0)
            row['two_stage_worker_elapsed_seconds']=row.get('worker_elapsed_seconds',0)+reference.get('worker_elapsed_seconds',0)
    for m in records:
        for split in ['val','test']:
            if split not in m:continue
            v=m[split];base=dict(family=m['family'],station_id=m['station_id'],site=m['site'],seed=m['seed'],split=split,complete_family=m['family'] in ready)
            raw.append(dict(**base,**{k:v[k] for k in ['mae','rmse','bias','n_origins','outside_0_1_fraction']}))
            for k in range(24):leadrows.append(dict(**base,lead_hours=(k+1)/4,mae=v['lead_mae'][k],rmse=v['lead_rmse'][k]))
            for key,value in v['scenarios'].items():scenario.append(dict(**base,scenario=key,**value))
    export('metrics_per_farm_seed',raw);export('lead_per_farm_seed',leadrows);export('scenario_per_farm_seed',scenario);export('training_cost',cost)
    mainrows=[];farmrows=[];leadmeans=[];scenariomeans=[]
    for family in sorted(ready):
        seeds=SEEDS[:EXPECTED[family]]
        for split in ['val','test']:
            seed_mae=[];seed_rmse=[]
            for seed in seeds:
                seed_mae.append(np.mean([index[family,s,seed][split]['mae'] for s in range(56)]))
                seed_rmse.append(np.mean([index[family,s,seed][split]['rmse'] for s in range(56)]))
            mainrows.append(dict(family=family,display_name=family+' (adapted)' if family in FAMILIES else family,split=split,farms=56,seeds=1 if family=='PERSISTENCE' else len(seeds),macro_mae=float(np.mean(seed_mae)),macro_rmse=float(np.mean(seed_rmse)),
                        seed_mae_std=float(np.std(seed_mae,ddof=1)) if family!='PERSISTENCE' else 0.,official_reproduction=False if family in FAMILIES else None))
            for site in range(56):
                farmrows.append(dict(family=family,split=split,site=site,station_id=str(static()['station_id'][site]),station_name=str(static()['station_name'][site]),
                    mae=float(np.mean([index[family,site,r][split]['mae'] for r in seeds])),rmse=float(np.mean([index[family,site,r][split]['rmse'] for r in seeds]))))
            lead=np.array([index[family,s,r][split]['lead_mae'] for s in range(56) for r in seeds]).mean(0)
            for k in range(24):leadmeans.append(dict(family=family,split=split,lead_hours=(k+1)/4,mae=float(lead[k])))
            keys=index[family,0,seeds[0]][split]['scenarios'].keys()
            for key in keys:
                vals=[index[family,s,r][split]['scenarios'][key]['mae'] for s in range(56) for r in seeds]
                valid=[v for v in vals if v is not None]
                scenariomeans.append(dict(family=family,split=split,scenario=key,macro_mae=float(np.mean(valid)) if valid else None,valid_farm_seed_cells=len(valid)))
    export('main_table_complete_only',mainrows);export('farm_table_complete_only',farmrows);export('lead_table_complete_only',leadmeans);export('scenario_table_complete_only',scenariomeans)
    bands=[]
    for family in ready:
        for split in ['val','test']:
            values=[r['mae'] for r in leadmeans if r['family']==family and r['split']==split]
            for label,lo,hi in [('0_1h',0,4),('1_3h',4,12),('3_6h',12,24)]:bands.append(dict(family=family,split=split,band=label,mae=float(np.mean(values[lo:hi]))))
    export('horizon_bands_complete_only',bands)
    contrasts=[]
    if 'E2C_CBH' in ready:
        pairs=CONTRASTS+[('E2C_CBH',f) for f in FAMILIES+['B0','PERSISTENCE','SELECTED_BASELINE']]
        for left,right in pairs:
            if left in ready and right in ready:contrasts.append(paired_statistics(index,left,right,SEEDS[:min(EXPECTED[left],EXPECTED[right])]))
    export('paired_contrasts_complete_only',contrasts)
    disaggregated=[]
    if 'E2C_CBH' in ready:
        for opponent in ['TRANSFORMER_FC','SELECTED_BASELINE']:
            if opponent in ready:disaggregated+=disaggregated_contrasts(index,'E2C_CBH',opponent,SEEDS)
    export('paired_horizon_scenario_intervals_complete_only',disaggregated)
    mechanism=[r for r in contrasts if (r['left'],r['right']) in CONTRASTS]
    if mechanism:
        fig,ax=plt.subplots(figsize=(7.2,max(2.5,.42*len(mechanism))))
        yy=np.arange(len(mechanism));xx=np.array([r['delta_mae'] for r in mechanism])
        lo=np.array([r['farm_seed_ci95_low'] for r in mechanism]);hi=np.array([r['farm_seed_ci95_high'] for r in mechanism])
        # Separate points and intervals; percentile intervals need not contain the point estimate.
        ax.hlines(yy,lo,hi,color='#397ca8',lw=1.5);ax.scatter(xx,yy,color='#397ca8',s=22,zorder=3)
        ax.set_yticks(yy,[r['left']+' − '+r['right'] for r in mechanism],fontsize=8)
        ax.axvline(0,color='.4',ls='--',lw=.8);ax.invert_yaxis();ax.set_xlabel('Paired MAE difference (95% farm/seed interval)')
        figure_save(fig,'matched_ablation_effects')
    if {'E2C_CBH','TRANSFORMER_FC'}.issubset(ready):
        selected=[r for r in scenariomeans if r['split']=='test' and r['family'] in ['E2C_CBH','TRANSFORMER_FC','SELECTED_BASELINE']]
        frame=pd.DataFrame(selected);pivot=frame.pivot(index='scenario',columns='family',values='macro_mae')
        pivot=pivot.drop(index='future_ramp_diagnostic_only',errors='ignore')
        order=sorted(pivot.index,key=lambda s:({'season':0,'local':1,'current':2,'observed':3}.get(s.split('_')[0],4),s))
        pivot=pivot.loc[order];fig,ax=plt.subplots(figsize=(8,max(4,len(pivot)*.27)))
        y=np.arange(len(pivot));colors={'E2C_CBH':'#397ca8','TRANSFORMER_FC':'#cb7553','SELECTED_BASELINE':'#779c66'}
        for k,family in enumerate(pivot.columns):ax.barh(y+(k-(len(pivot.columns)-1)/2)*.23,pivot[family],height=.22,color=colors[family],label=family)
        ax.set_yticks(y,pivot.index,fontsize=7);ax.invert_yaxis();ax.set_xlabel('Macro MAE within observable scenario')
        ax.legend(frameon=False,fontsize=7,loc='lower right');figure_save(fig,'observable_scenarios')
    if len(ready)>1:
        df=pd.DataFrame(leadmeans);fig,ax=plt.subplots(figsize=(7.2,4))
        show=['E2C_CBH','B0','TRANSFORMER_FC','GRU_H','CNN_W','SELECTED_BASELINE']
        for family in show:
            a=df[(df.family==family)&(df.split=='test')]
            if len(a):ax.plot(a.lead_hours,a.mae,label=family,lw=1.6)
        ax.set(xlabel='Forecast lead (hours)',ylabel='Macro MAE (capacity factor)');ax.legend(frameon=False,fontsize=8);figure_save(fig,'lead_mae')
    if {'E2C_CBH','SELECTED_BASELINE'}.issubset(ready):
        delta=np.array([np.mean([index['E2C_CBH',s,r]['test']['mae']-index['SELECTED_BASELINE',s,r]['test']['mae'] for r in SEEDS]) for s in range(56)])
        fig,ax=plt.subplots(figsize=(11,3.8));ax.bar(range(56),delta,color=np.where(delta<0,'#397ca8','#cb7553'))
        ax.set_xticks(range(56),list(map(str,static()['station_id'])),rotation=90,fontsize=6);ax.axhline(0,color='.25',lw=.7)
        ax.set(ylabel='MAE: E2C-CBH minus selected baseline',xlabel='All 56 farms (fixed station order)');figure_save(fig,'all_farm_differences')
        # Representative/known-failure table is a subset of the complete all-farm table.
        selected_ids=set(REPRESENTATIVES+['1043056','1043061'])
        export('representatives_and_failures',[r for r in farmrows if r['station_id'] in selected_ids])
    make_cases(cases,index,ready,selection)
    costs=pd.DataFrame(cost)
    if len(costs):
        aggregated=costs.groupby('family',dropna=False).agg(completed_training_stages=('id','count'),wall_seconds_sum=('seconds_training','sum'),
                  median_wall_seconds=('seconds_training','median'),trainable_parameters=('trainable_parameters','median'),active_parameters=('active_gradient_parameters','median')).reset_index()
        export('cost_summary',aggregated.to_dict('records'))
    status=json.loads((RUN/'STATUS.json').read_text()) if (RUN/'STATUS.json').exists() else {}
    counts=status.get('counts',{});done=counts.get('done',0);all_done=done==9500
    lines=['# 独立单风场实验自动报告','',f"更新时间：{datetime.datetime.now().astimezone().isoformat()}",
           f"\n训练阶段完成 {done}/9500；运行 {counts.get('running',0)}；失败 {counts.get('failed',0)}。",
           '\n**'+('全部训练阶段完成。' if all_done else '当前仍为中间状态，不得作为完整对照结论。')+'**',
           '\n只有 56 站及预定全部种子齐全的家族进入正式汇总表。完整家族：'+', '.join(sorted(ready)),
           '\n本轮任务是各站独立训练、本站输入的 E2C-CBH；所有命名基线为文档注明的适配实现。2024 是跨年复评。',
           '\n入口文件：','',
           '- `main_table_complete_only.csv`：完整模型主表；`farm_table_complete_only.csv`：全部风场。',
           '- `paired_contrasts_complete_only.csv`：匹配效应量、风场/种子区间、7 日块区间及探索性等价分析。',
           '- `lead_table_complete_only.csv`、`horizon_bands_complete_only.csv`：提前期结果。',
           '- `scenario_table_complete_only.csv`：季节、当地时段、当前功率、已观测爬坡及未来诊断分组。',
           '- `validation_selected_baselines.csv`：只按 2023 选择的逐站基线；有完整候选集时生成。',
           '- `case_selection.csv`、`forecast_cases.csv`、`figures/`：固定案例和 PDF/PNG 图。',
           '- `training_cost.csv`：分阶段成本；主模型部署训练成本需加同站同种子的 B0。并发墙钟耗时不是独占 GPU 活跃时间。',
           '\n置信区间跨零不等于排除了某种机制；仅 90% 区间完全落在 ±0.001 CF 内时，报告该探索性容差下的等价。不得将事后未来爬坡诊断表述为可部署条件。']
    if mainrows:
        lines+=['\n完整家族 2024 主表（MAE）：','']
        for row in sorted([r for r in mainrows if r['split']=='test'],key=lambda r:r['macro_mae']):lines.append(f"- {row['family']}: {row['macro_mae']:.6f}，seed std {row['seed_mae_std']:.6f}")
    if contrasts:
        lines+=['\n预先指定比较的结果：','']
        for r in contrasts:lines.append(f"- {r['left']} − {r['right']}: ΔMAE={r['delta_mae']:+.6f}，95% 风场/种子区间 [{r['farm_seed_ci95_low']:+.6f}, {r['farm_seed_ci95_high']:+.6f}]；{r['farms_left_better']}/56 站前者更好。")
    evidence=['# 三项贡献的本轮证据状态','',f'训练完成 {done}/9500。只有完整家族的比较可以进入下列判断。',
              '\n基线是任务适配实现；不得把此报告表述为官方 SOTA 复现。区域样本上的统计区间不支持全球泛化声明。']
    contrasts_by_pair={(r['left'],r['right']):r for r in contrasts}
    def describe(pair):
        r=contrasts_by_pair.get(pair)
        if r is None:return '尚无完整匹配比较。'
        direction='前者平均误差较低' if r['delta_mae']<0 else '前者平均误差较高或相同'
        support='两个 95% 区间均低于零' if r['farm_seed_ci95_high']<0 and r['temporal_7day_ci95_high']<0 else '现有区间不一致支持前者更优'
        return f"ΔMAE={r['delta_mae']:+.6f}，{direction}，{r['farms_left_better']}/56 站前者更好；{support}。"
    evidence+=['\n## 方法证据','', 'E2C-CBH 对验证集逐站选出的基线：'+describe(('E2C_CBH','SELECTED_BASELINE')),
        'E2C-CBH 对历史参考 B0：'+describe(('E2C_CBH','B0')),
        '\n## 受控归因','',
        '显式轨迹条件（CB−CNEUTRAL）：'+describe(('CB','CNEUTRAL')),
        '摘要条件之外加入轨迹（CBH−CH）：'+describe(('E2C_CBH','CH')),
        '实际天气相对训练均值天气（CBH−CLIMATOLOGY）：'+describe(('E2C_CBH','CLIMATOLOGY')),
        '直接输出相对参考修正（DIRECT−CBH）：'+describe(('DIRECT','E2C_CBH'))]
    direct=contrasts_by_pair.get(('DIRECT','E2C_CBH'))
    if direct:
        evidence.append('直接输出与参考修正：'+('90% 区间完全位于预设探索性 ±0.001 CF 容差内。' if direct['equivalent_at_margin'] else '未满足预设探索性等价判据；区间跨零不能写成输出形式无作用。'))
    evidence+=['\n这些对照估计训练协议下的增量。轨迹是历史的确定性表征；中性臂输出端仍使用 B0；DIRECT 同时改变初始输出及优化问题。因此不把差异直接写成物理或纯函数空间因果结论。',
       '\n## 适用范围','',
       '查看全部 56 站、24 个时效、可观测场景以及龙湖/余字失败站表。历史与天气曲线是否交叉、优势是否随提前期增加，应逐项依据表格描述；不默认成立。',
       '典型案例线为 10 个独立种子预测的均值，阴影为种子间标准差，不是预测置信区间。未来真值爬坡分组只能作事后诊断。']
    (OUT/'CONTRIBUTION_EVIDENCE_CN.md').write_text('\n'.join(evidence)+'\n')
    (OUT/'REPORT_CN.md').write_text('\n'.join(lines)+'\n')
    atomic_json(OUT/'ANALYSIS_STATUS.json',dict(training_complete=all_done,completed_stages=done,complete_families=sorted(ready),
                code_sha256=sha(Path(__file__)),tables=len(list(OUT.glob('*.csv'))),figures=len(list((OUT/'figures').glob('*.pdf')))))
    print('ANALYZED',done,sorted(ready),flush=True)

if __name__=='__main__':main()
