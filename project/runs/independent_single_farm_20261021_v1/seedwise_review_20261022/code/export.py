#!/usr/bin/env python3
"""Reproducible per-seed export. Never averages results across seeds or farms."""
import csv, gzip, hashlib, html, json, math, shutil, sqlite3, time, zipfile
from collections import Counter
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import xlsxwriter
from catalog import CATALOG, ORDER, BASELINES, DESIGNS, COMPARISONS

O=Path(__file__).resolve().parents[1]; R=O.parent; P=R.parents[1]; T=O/'tables'
for d in [T,O/'farms',O/'legacy']: d.mkdir(exist_ok=True)
def log(s): print(datetime.now().isoformat(timespec='seconds'),s,flush=True)
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(2**20),b''): h.update(b)
    return h.hexdigest()
def save(df,name): df.to_csv(T/name,index=False,float_format='%.17g',encoding='utf-8-sig')
def md(df):
    def val(v):
        if v is None or pd.isna(v): return '未运行'
        if isinstance(v,(float,np.floating)): return format(v,'.17g')
        return str(v).replace('|','/').replace('\n',' ')
    return '| '+' | '.join(map(str,df.columns))+' |\n|'+'|'.join(['---']*len(df.columns))+'|\n'+'\n'.join('| '+' | '.join(val(v) for v in row)+' |' for row in df.itertuples(index=False,name=None))+'\n'

static=P/'data/model_cache_pad5_f8/static'
names=np.load(static/'station_name.npy',allow_pickle=True).astype(str)
ids=np.load(static/'station_id.npy',allow_pickle=True).astype(str)
caps=np.load(static/'station_capacity_mw.npy').astype(float)
finish=json.loads((R/'FINISHED.json').read_text()); audit=json.loads((R/'COMPLETION_AUDIT.json').read_text())
assert finish['all_experiments_complete'] and audit['passed']
db=sqlite3.connect(f'file:{R}/queue.sqlite?mode=ro',uri=True)
jobs=db.execute('SELECT id, status FROM jobs ORDER BY id').fetchall();db.close()
assert len(jobs)==9500 and all(s=='done' for _,s in jobs)
configs=pd.read_csv(R/'analysis/selected_hyperparameters.csv').drop(columns=['validation_mae'])
selected=pd.read_csv(R/'analysis/validation_selected_baselines.csv',dtype={'station_id':str})[['site','station_id','family']].rename(columns={'family':'actual_family'})
selected['station_name']=selected.site.map(lambda i:names[i])
save(configs,'selected_hyperparameters.csv');save(selected,'selected_family_by_farm.csv')
catalog=pd.DataFrame(CATALOG);save(catalog,'architecture_dictionary.csv')
metadata=[]; overall=[]; source_hashes=[]; counts=Counter(); histories=0
identity=['source_job_id','family','stage','site','station_id','station_name','seed']
log('Reading all 9500 completed jobs; exporting exact lead/scenario/day/epoch records')
with ExitStack() as stack:
    def writer(filename,fields,compressed=False):
        f=stack.enter_context(gzip.open(T/filename,'wt',encoding='utf-8-sig',newline='',compresslevel=3) if compressed else open(T/filename,'w',encoding='utf-8-sig',newline=''))
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();return w
    leads=writer('lead_results.csv.gz',identity+['split','year','lead_step','lead_hours','n_origins','mae','rmse'],True)
    scenarios=writer('scenario_results.csv.gz',identity+['split','year','scenario','n_origins','mae','rmse','bias'],True)
    daily=writer('daily_results_utc.csv.gz',identity+['split','year','utc_day','date_utc','n_origins','n_forecast_values','abs_error_sum','squared_error_sum','mae','rmse'],True)
    history=writer('training_epoch_results.csv.gz',identity+['epoch','train_mae','val_mae','best_val_mae','best_epoch','stale','seconds','lr','timestamp'],True)
    def consume(m,source):
        site=m['site']; cfg=m.get('config',{})
        ident=dict(source_job_id=m['id'],family=m['family'],stage=m['stage'],site=site,station_id=ids[site],station_name=names[site],seed=m.get('seed'))
        meta={**ident,**cfg,**{k:m.get(k) for k in ['best_epoch','epochs_run','trainable_parameters','total_parameters','active_gradient_parameters','seconds_training','precision','completed_at','peak_reserved_mib','gpu_name','checkpoint_sha256','reference_sha256']},'metrics_path':str(source.relative_to(P)),'checkpoint_path':str((source.parent/'best.pt').relative_to(P)) if m['stage']!='deterministic' else '', 'capacity_mw':caps[site]}
        metadata.append(meta)
        for split,year in [('val',2023),('test',2024)]:
            if split not in m: continue
            x=m[split]; key={**ident,'split':split,'year':year}
            overall.append({**meta,'split':split,'year':year,**{k:x.get(k) for k in ['n_origins','mae','rmse','bias','outside_0_1_fraction']}})
            counts['overall']+=1
            for k,(mae,rmse) in enumerate(zip(x['lead_mae'],x['lead_rmse']),1):
                leads.writerow({**key,'lead_step':k,'lead_hours':k/4,'n_origins':x['n_origins'],'mae':mae,'rmse':rmse});counts['lead']+=1
            for scenario,v in x['scenarios'].items():
                scenarios.writerow({**key,'scenario':scenario,**v});counts['scenario']+=1
            for day,v in x['daily_utc'].items():
                n=v['n']*24
                daily.writerow({**key,'utc_day':day,'date_utc':str(np.datetime64(int(day),'D')),'n_origins':v['n'],'n_forecast_values':n,
                                'abs_error_sum':v['abs_sum'],'squared_error_sum':v['sq_sum'],'mae':v['abs_sum']/n,'rmse':math.sqrt(v['sq_sum']/n)})
                counts['daily']+=1
        return ident
    for i,(job,_) in enumerate(jobs):
        path=R/'jobs'/job/'metrics.json';raw=path.read_bytes();m=json.loads(raw)
        assert m['id']==job and not m['smoke_only'] and (path.parent/'SUCCESS.json').exists()
        assert str(m['station_id'])==ids[m['site']]
        ident=consume(m,path)
        source_hashes.append(dict(source_job_id=job,metrics_path=str(path.relative_to(P)),metrics_sha256=hashlib.sha256(raw).hexdigest(),checkpoint_sha256=m['checkpoint_sha256']))
        hp=path.parent/'history.json';hist=json.loads(hp.read_text()); assert len(hist)==m['epochs_run']
        for h in hist:history.writerow({**ident,**h});histories+=1
        if (i+1)%1000==0:log(f'{i+1}/9500 jobs exported')
    for m in json.loads((R/'analysis/persistence_metrics.json').read_text()):consume(m,R/'analysis/persistence_metrics.json')

df=pd.DataFrame(overall);meta=pd.DataFrame(metadata)
for d in [df,meta]: d['seed']=d['seed'].astype('Int64')
assert len(df)==18572 and len(meta)==9556
assert df.duplicated(['source_job_id','split']).sum()==0
trained=df[df.stage!='deterministic'].copy(); deterministic=df[df.stage=='deterministic'].copy()
tune=trained[trained.seed.isin([40,41])].copy(); main=trained[trained.seed.isin(range(42,52))].copy()
assert len(trained)==18460 and len(tune)==540 and len(main)==17920 and len(deterministic)==112
assert set(tune.split)=={'val'}
e10=main.merge(selected[['site','actual_family']],on='site').query('family == actual_family').copy()
e10['family']='SELECTED_BASELINE';e10['row_kind']='derived_reuse';assert len(e10)==1120
main['actual_family']=main.family; main['row_kind']='trained'
deterministic['actual_family']='PERSISTENCE';deterministic['row_kind']='deterministic'
view=pd.concat([main,e10,deterministic],ignore_index=True)
save(trained,'all_trained_results.csv');save(tune,'all_tuning_results.csv');save(deterministic,'deterministic_results.csv')
save(e10,'selected_baseline_resolved.csv');save(view,'formal_and_ablation_results.csv');save(meta,'job_configs_and_costs.csv')
save(pd.DataFrame(source_hashes),'source_metrics_hashes.csv')
coverage=meta.groupby(['stage','family'],dropna=False).agg(jobs=('source_job_id','size'),farms=('site','nunique'),seeds=('seed',lambda s:','.join(str(int(x)) for x in sorted(s.dropna().unique())))).reset_index()
save(coverage,'coverage.csv')
log('Computing strictly matched per-seed comparisons without tolerances')
pair_rows=[]
keys=['site','station_id','station_name','seed','split','year']
for cid,label,candidate,reference in COMPARISONS:
    a=view[view.family==candidate][keys+['mae','rmse','source_job_id','actual_family']]
    b=view[view.family==reference][keys+['mae','rmse','source_job_id','actual_family']]
    join=keys if reference!='PERSISTENCE' else [k for k in keys if k!='seed']
    if reference=='PERSISTENCE': b=b.drop(columns='seed')
    z=a.merge(b,on=join,suffixes=('_candidate','_reference'),validate='one_to_one' if reference!='PERSISTENCE' else 'many_to_one')
    z.insert(0,'comparison_id',cid);z.insert(1,'design_cn',label);z['candidate']=candidate;z['reference']=reference
    for metric in ['mae','rmse']:
        gain=z[metric+'_reference']-z[metric+'_candidate'];z[metric+'_gain']=gain
        z[metric+'_direction']=np.select([gain>0,gain<0],['improved','worsened'],default='tie')
    pair_rows.append(z)
pairs=pd.concat(pair_rows,ignore_index=True);save(pairs,'paired_results.csv')
assert not pairs.duplicated(['comparison_id','site','seed','split']).any()
def direction_counts(data,group):
    rows=[]
    for key,g in data.groupby(group,dropna=False,sort=False):
        if not isinstance(key,tuple):key=(key,)
        for metric in ['mae','rmse']:
            x=g[metric+'_gain']
            rows.append(dict(zip(group,key),metric=metric,n_pairs=len(x),improved=int((x>0).sum()),worsened=int((x<0).sum()),ties=int((x==0).sum()),unavailable=int(x.isna().sum())))
    return pd.DataFrame(rows)
group=['comparison_id','design_cn','candidate','reference','split','year']
seed_counts=direction_counts(pairs,group+['seed']); total_counts=direction_counts(pairs,group)
save(seed_counts,'improvement_counts_by_seed.csv');save(total_counts,'improvement_counts_all_pairs.csv')
farm_counts=direction_counts(pairs,group+['site','station_id','station_name']);save(farm_counts,'improvement_counts_by_farm.csv')
cross=pairs[pairs.split=='val'].merge(pairs[pairs.split=='test'],on=['comparison_id','site','seed'],suffixes=('_2023','_2024'),validate='one_to_one')
cross_rows=[]
for _,r in cross.iterrows():
    for metric in ['mae','rmse']:
        x,y=r[metric+'_direction_2023'],r[metric+'_direction_2024']
        category='both_improved' if x==y=='improved' else '2023_only' if x=='improved' else '2024_only' if y=='improved' else 'neither_improved'
        cross_rows.append(dict(comparison_id=r.comparison_id,design_cn=r.design_cn_2023,site=r.site,station_id=r.station_id_2023,station_name=r.station_name_2023,seed=int(r.seed),metric=metric,
                               candidate=r.candidate_2023,reference=r.reference_2023,gain_2023=r[metric+'_gain_2023'],gain_2024=r[metric+'_gain_2024'],direction_2023=x,direction_2024=y,category=category))
cross_df=pd.DataFrame(cross_rows);save(cross_df,'cross_year_paired_results.csv')
cross_counts=cross_df.groupby(['comparison_id','design_cn','metric','category'],sort=False).size().reset_index(name='n_pairs');save(cross_counts,'cross_year_direction_counts.csv')

log('Exporting lead/scenario matched contrasts and per-seed counts')
detail_counts={}
for detail,axis in [('lead','lead_step'),('scenario','scenario')]:
    detail_df=pd.read_csv(T/f'{detail}_results.csv.gz',dtype={'station_id':str,'seed':'Int64'},float_precision='round_trip')
    detail_df=detail_df[detail_df.seed.isin(range(42,52))]
    contrasts=[]
    dk=['site','station_id','station_name','seed','split','year',axis]
    for cid,label,can,ref in DESIGNS:
        cols=dk+['mae','rmse','source_job_id','n_origins']
        z=detail_df[detail_df.family==can][cols].merge(detail_df[detail_df.family==ref][cols],on=dk,suffixes=('_candidate','_reference'),validate='one_to_one')
        assert (z.n_origins_candidate==z.n_origins_reference).all()
        z['comparison_id']=cid;z['design_cn']=label;z['candidate']=can;z['reference']=ref
        for metric in ['mae','rmse']:z[metric+'_gain']=z[metric+'_reference']-z[metric+'_candidate']
        contrasts.append(z)
    detail_pairs=pd.concat(contrasts,ignore_index=True);save(detail_pairs,f'{detail}_paired_results.csv.gz')
    dc=direction_counts(detail_pairs,['comparison_id','design_cn','candidate','reference','split','year',axis,'seed'])
    save(dc,f'{detail}_improvement_counts_by_seed.csv');detail_counts[detail]=dc
    del detail_df,contrasts,detail_pairs

log('Exporting fixed forecast windows from individual prediction arrays')
cases=pd.read_csv(R/'analysis/case_selection.csv',dtype={'station_id':str});save(cases,'case_selection.csv')
case_rows=[]
for site,cs in cases.groupby('station_index'):
    y=np.load(R/f'prepared/test/{site:02d}/target.npy',mmap_mode='r');h=np.load(R/f'prepared/test/{site:02d}/history.npy',mmap_mode='r')
    model_rows=view[(view.site==site)&(view.split=='test')]
    for _,r in model_rows.iterrows():
        pred=None if r.family=='PERSISTENCE' else np.load(R/'jobs'/r.source_job_id/'test_prediction.npy',mmap_mode='r')
        for _,c in cs.iterrows():
            k=int(c.origin_index)
            for lead in range(24):
                p=float(h[k,-1] if pred is None else pred[k,lead]);truth=float(y[k,lead])
                case_rows.append(dict(case_id=c.case_id,case_kind=c.kind,case_selection_source=c.source,site=int(site),station_id=r.station_id,station_name=r.station_name,
                    seed=None if pd.isna(r.seed) else int(r.seed),family=r.family,actual_family=r.actual_family,source_job_id=r.source_job_id,origin_index=k,origin_local=c.origin_local,
                    lead_step=lead+1,lead_hours=(lead+1)/4,observed=truth,prediction=p,error=p-truth,absolute_error=abs(p-truth)))
case_df=pd.DataFrame(case_rows);case_df['seed']=case_df.seed.astype('Int64');save(case_df,'forecast_cases_per_seed.csv.gz')

log('Generating 56 farm tables, workbook, and offline browser')
wides={}
for split,year in [('val',2023),('test',2024)]:
    for metric in ['mae','rmse']:
        wide=view[(view.split==split)&view.seed.notna()].pivot(index=['site','station_id','station_name','family','actual_family'],columns='seed',values=metric).reindex(columns=range(42,52)).reset_index()
        wide.columns=[str(x) for x in wide.columns];wide['sort']=wide.family.map({f:i for i,f in enumerate(ORDER)})
        wide=wide.sort_values(['site','sort']).drop(columns='sort');wides[f'{year}_{metric.upper()}']=wide
        save(wide,f'{year}_{metric}_by_seed.csv')

architecture_md='''# 实验缩写与实际结构（2026-10-21 独立训练轮次）

下列名称依据本轮冻结源码，不依据名称推测官方结构。所有命名基线都是任务适配版，不是官方复现。

## 记号与共同协议

h：本站过去24个15分钟功率点；o：最后观测；M：历史编码token；m：M沿token维的均值；b：同站同seed冻结B0的24步预测；Δb=b−[o,b前23步]；c：decoder读出。输出24步（0.25–6小时），功率以本站容量归一化为CF。

2022训练，2023选模型，2024跨年复评（非未查看的盲测）。各站模型独立训练；FP32且无TF32；AdamW+余弦退火，最多80epoch，patience=10，batch=256，梯度裁剪1，无输出后处理裁剪。实际width/lr/weight_decay、参数量、最优epoch见逐job配置表；调参width可能不同于正式配置。

E2C的15维物理柱坐标依次为：msl,u10,v10,t2m,高程,u1000−u10,v1000−v10,θ1000,q1000,z1000/g−高程,u925−u1000,v925−v1000,θ925−θ1000,q925−q1000,(z925−z1000)/g，其中θ925=t925×(1000/925)^0.2854。坐标使用训练统计标准化。E2C使用的原有统计是训练期统计，不代表所有预处理统计都按站重新估计。

## 缩写结构字典

'''+md(catalog[['family','name_cn','inputs','condition','output','structure','caveat','source']])+'''
## 正式超参数

'''+md(configs)+'''
B0固定d=64、lr=0.0002、weight_decay=0.0001；所有CBH消融继承E2C_CBH正式配置。DLINEAR_FC的width字段不参与模型结构。所有调参试验的实际配置另见all_tuning_results.csv。

total_parameters是模块全部参数；trainable_parameters是requires_grad参数；active_gradient_parameters是训练结束时实际存在梯度的参数数目。CBH检查点含冻结B0及未激活源模块，三项不可混称“有效参数量”。seconds_training仅对应该job阶段，CBH阶段不含前置B0成本；不能把它当总部署成本。

注意：DIRECT/NEUTRAL_LAST保存的*_correction.npy定义为最终预测−B0，并非这里的原始读出c；解读输出消融请按上述公式。
'''
(O/'ARCHITECTURES_CN.md').write_text(architecture_md)
for site in range(56):
    text=f'# {ids[site]} {names[site]}：逐随机种子实测结果\n\n容量 {caps[site]:.17g} MW；误差单位CF；越小越好。所有数字保留源浮点精度，无跨seed/跨站平均。未运行保持空缺。\n\n'
    text+='[返回总报告](../REPORT_CN.md) · [结构字典](../ARCHITECTURES_CN.md) · [完整CSV](../tables/formal_and_ablation_results.csv)\n\n'
    for name,wide in wides.items():
        w=wide[wide.site==site].drop(columns=['site','station_id','station_name'])
        text+=f'## {name}\n\n'+md(w)+'\n'
    text+='## 确定性持续性基线（无seed）\n\n'+md(deterministic[deterministic.site==site][['year','mae','rmse']])
    (O/'farms'/f'{ids[site]}_{names[site]}.md').write_text(text)

# Historical catalogs are linked as a separate protocol index, never merged into this round.
legacy=[]
old=pd.read_csv(P/'results/EXPERIMENT_INDEX.csv').fillna('')
for r in old.to_dict('records'):
    legacy.append(dict(**r,entry=str(P/'PROJECT_HANDOFF_CN.md'),protocol_note='历史协议；共享/迁移/旧独立试验不可与本轮直接混合。原逐run索引含站点汇总量，不作本次逐站逐seed实测表。'))
legacy_df=pd.DataFrame(legacy);save(legacy_df,'historical_experiment_index.csv')
historical_raw=['prediction_station_seed.csv','scenario_station_seed.csv','condition_ablation_station_seed.csv','matched_output_station_seed.csv','independent_baselines_pilot_station_seed.csv','sharing_2x2_station_seed.csv','source_full_year_station_seed.csv','target20_common_seed_metrics.csv','shared_baselines_station_seed.csv']
legacy_text='# 历史结果入口（不同协议单独保留）\n\n本次完整重制的是2026-10-21起的9500个独立单场训练阶段。历史39个实验/资料目录已在原项目索引中登记；这里提供入口，不把其共享模型、迁移预算和旧独立试验误记为本轮结果，也不把原报告均值替代逐seed值。\n\n'
legacy_text+=f'- [原始逐run索引]({P}/results/RUN_METRICS.csv)：保留seed与指标来源，但部分指标本身是多站汇总；不能冒充单站值。\n- [历史模型结构与协议说明]({P}/PROJECT_HANDOFF_20261019_ARCHIVE_CN.md)\n- [39目录索引](../tables/historical_experiment_index.csv)\n\n以下是已有逐站逐seed历史表；列名与协议以各文件为准：\n\n'
for fn in historical_raw: legacy_text+=f'- [{fn}]({P}/runs/single_farm_review_20261018/tables/{fn})\n'
(O/'legacy/README_CN.md').write_text(legacy_text)

guide=pd.DataFrame([
 ('覆盖','9500训练阶段；18460条训练job×split结果；112条确定性结果；E10另1120条复用结果，不新增训练。'),
 ('逐种子','正式42–51；CB/CH/CNEUTRAL/DIRECT/NEUTRAL_LAST/CLIMATOLOGY仅42–46；调参40/41只评2023。'),
 ('改善定义','reference−candidate>0即改善；<0退化；==0相同；不使用百分比门槛或容差。'),
 ('指标','CF单位MAE/RMSE；每个值仅对该次预测样本/时效求误差；不跨种子或风场平均。'),
 ('精度','Excel显示12位小数，可查单元格完整精度；CSV保留17位有效数字；方向在格式化前判断。'),
 ('结构','参看结构字典及ARCHITECTURES_CN.md；配置成本按每个job记录；基线为任务适配版。'),
 ('E10','SELECTED_BASELINE为原2023协议选定的每站基线。actual_family和source_job_id指出实际结构及结果来源。旧选型用跨seed均值，本次不重选、不导出该均值。'),
 ('分组明细','lead/scenario/day/epoch/固定窗口逐seed预测在tables/*.csv.gz中，均无跨seed平均。daily按UTC，scenario小时/季节按UTC+8。'),
 ('参数成本','total/trainable/active_gradient含义不同；CBH训练阶段时间不含前置B0。'),
 ('历史范围','历史39目录见historical_experiment_index.csv及legacy/README_CN.md，与当前独立轮次分开。'),
 ('完成审计',json.dumps(audit,ensure_ascii=False)),
],columns=['项目','说明'])

# Direct xlsxwriter row writing preserves types and is fast with constant_memory.
book=xlsxwriter.Workbook(O/'EXPERIMENT_RESULTS_PER_SEED_20261022.xlsx',{'constant_memory':True,'strings_to_urls':False,'nan_inf_to_errors':False})
header=book.add_format({'bold':True,'bg_color':'#17324D','font_color':'white','text_wrap':True,'valign':'top'})
num=book.add_format({'num_format':'0.000000000000'});integer=book.add_format({'num_format':'0'})
wrap=book.add_format({'text_wrap':True,'valign':'top'})
def sheet(name,data):
    ws=book.add_worksheet(name);ws.freeze_panes(1,0);ws.write_row(0,0,list(data.columns),header);ws.set_row(0,32)
    for j,col in enumerate(data.columns):
        width=20
        if col in ['structure','caveat','inputs','说明','protocol_note']:width=85
        elif any(s in str(col) for s in ['path','source','design','family']):width=32
        elif str(col).isdigit():width=21
        ws.set_column(j,j,width)
    for i,row in enumerate(data.itertuples(index=False,name=None),1):
        for j,v in enumerate(row):
            if v is None or pd.isna(v):continue
            if isinstance(v,(int,np.integer)):ws.write_number(i,j,int(v),integer)
            elif isinstance(v,(float,np.floating)):ws.write_number(i,j,float(v),num)
            else:ws.write_string(i,j,str(v),wrap if str(data.columns[j]) in ['structure','caveat','inputs','说明','protocol_note'] else None)
    ws.autofilter(0,0,len(data),len(data.columns)-1)
for name,data in [('阅读说明',guide),('结构字典',catalog),('正式超参数',configs),('实验覆盖',coverage),('正式消融原始值',main),('调参原始值',tune),('确定性持续性',deterministic),('E10实际模型及原值',e10),('逐job配置与成本',meta),*wides.items(),('配对原值与差额',pairs),('逐seed提升次数',seed_counts),('全部配对提升次数',total_counts),('逐场提升次数',farm_counts),('跨年配对方向',cross_df),('跨年方向计数',cross_counts),('分时效逐seed次数',detail_counts['lead']),('分场景逐seed次数',detail_counts['scenario']),('历史实验入口',legacy_df)]:sheet(name,data)
book.close()

payload=view[['site','station_id','station_name','family','actual_family','seed','year','mae','rmse','source_job_id']].copy()
records=json.loads(payload.to_json(orient='records',double_precision=15))
# Build JSON from native values for exact JS double round-trip, not pandas' decimal formatting.
records=[{k:(None if pd.isna(v) else int(v) if k in ['site','seed','year'] else v) for k,v in r.items()} for r in payload.to_dict('records')]
web=r'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>逐种子实验结果</title><style>
body{font:15px system-ui;margin:32px;color:#17324d;background:#f5f8fb}h1{font-size:25px}select{padding:8px;margin-right:16px}table{border-collapse:collapse;background:white;width:100%}th,td{border:1px solid #d7e0e8;padding:9px;text-align:right;white-space:nowrap}th{background:#17324d;color:white;position:sticky;top:0}td:first-child{text-align:left;cursor:pointer}tr:hover{background:#edf5ff}p{max-width:1100px;line-height:1.7}a{color:#166aa1}#structure{white-space:pre-wrap;background:white;padding:18px;margin:18px 0;line-height:1.8}.scroll{overflow:auto}button{padding:8px} .small{font-size:13px;color:#506578}</style>
<h1>56个独立风场 · 逐随机种子实际结果</h1><p>每格为该风场、该随机种子的实际误差。没有跨种子或风场平均；未运行显示“—”。点击模型名称查看实际结构。确定性持续性单独显示，无seed。</p>
<p><a href="REPORT_CN.md">审查报告</a> · <a href="EXPERIMENT_RESULTS_PER_SEED_20261022.xlsx">下载Excel</a> · <a href="ARCHITECTURES_CN.md">完整结构说明</a> · <a href="tables/paired_results.csv">配对原值与差额</a></p>
<label>风场 <select id="farm"></select></label><label>年份 <select id="year"><option>2024</option><option>2023</option></select></label><label>误差 <select id="metric"><option value="mae">MAE</option><option value="rmse">RMSE</option></select></label>
<div id="structure">选择一个模型查看结构。</div><div class="scroll"><table id="grid"></table></div><p id="persist"></p><p class="small">显示17位有效数字；原值可追溯至source_job_id。2024为跨年复评；E10只是原2023验证协议选定基线的实际逐seed结果。Excel和CSV还包含全部调参、配置与成本、改善/退化计数。这里只浏览正式结果。</p><script>
const DATA=__DATA__,CAT=__CAT__,ORDER=__ORDER__;
const el=x=>document.getElementById(x);for(let i=0;i<56;i++){const r=DATA.find(x=>x.site===i);el('farm').add(new Option(r.station_id+' '+r.station_name,i));}
function render(){const rows=DATA.filter(r=>r.site===Number(el('farm').value)&&r.year===Number(el('year').value));const met=el('metric').value;let t='<tr><th>模型（点击查看结构）</th>'+Array.from({length:10},(_,i)=>'<th>seed '+(42+i)+'</th>').join('')+'</tr>';
for(const f of ORDER.filter(f=>f!=='PERSISTENCE')){const rs=rows.filter(r=>r.family===f);let label=f;if(f==='SELECTED_BASELINE')label+=' → '+rs[0].actual_family;t+='<tr><td data-family="'+f+'">'+label+'</td>'+Array.from({length:10},(_,i)=>{const r=rs.find(r=>r.seed===42+i);return '<td title="'+(r?r.source_job_id:'未运行')+'">'+(r?r[met].toPrecision(17):'—')+'</td>';}).join('')+'</tr>';}
el('grid').innerHTML=t;el('persist').textContent='PERSISTENCE（无seed）：'+met.toUpperCase()+' = '+rows.find(r=>r.family==='PERSISTENCE')[met].toPrecision(17);
document.querySelectorAll('[data-family]').forEach(x=>x.onclick=()=>{let f=x.dataset.family;let c=CAT.find(c=>c.family===f);let txt=f+'：'+c.name_cn+'\n输入：'+c.inputs+'\n结构：'+c.structure+'\n条件：'+c.condition+'\n输出：'+c.output+'\n解释：'+c.caveat;if(f==='SELECTED_BASELINE'){let actual=rows.find(r=>r.family===f).actual_family;let a=CAT.find(c=>c.family===actual);txt+='\n本站实际网络 '+actual+'：'+a.structure;}el('structure').textContent=txt;});}
['farm','year','metric'].forEach(x=>el(x).onchange=render);render();</script></html>'''
web=web.replace('__DATA__',json.dumps(records,ensure_ascii=False)).replace('__CAT__',json.dumps(CATALOG,ensure_ascii=False)).replace('__ORDER__',json.dumps(ORDER))
(O/'index.html').write_text(web)

def countrow(cid,year,metric):return total_counts[(total_counts.comparison_id==cid)&(total_counts.year==year)&(total_counts.metric==metric)].iloc[0]
design_summary=[]
for cid,label,can,ref in DESIGNS:
    v=countrow(cid,2023,'mae');t=countrow(cid,2024,'mae');q=countrow(cid,2024,'rmse')
    design_summary.append({'设计':label,'候选':can,'对照':ref,'2023 MAE 提升/退化/相同':f'{v.improved}/{v.worsened}/{v.ties}',
                           '2024 MAE 提升/退化/相同':f'{t.improved}/{t.worsened}/{t.ties}','2024 RMSE 提升/退化/相同':f'{q.improved}/{q.worsened}/{q.ties}'})
report='''# 当前独立单风场实验：逐种子结果与设计审查

更新：2026-10-22。本报告完整覆盖本轮9500个训练阶段，所有结果已完成。全量表格保留每个风场、每个seed的实际值；不对seed或风场取均值。

## 直接打开

- [Excel总表：实际结果、结构、配对、逐seed计数](EXPERIMENT_RESULTS_PER_SEED_20261022.xlsx)
- [离线交互结果浏览器](index.html)：按风场/年份/指标切换，点击缩写查看结构
- [完整模型结构字典](ARCHITECTURES_CN.md) / [所有表格的字段说明](tables/README_CN.md)
- [全部训练job×split原始值](tables/all_trained_results.csv) / [所有调参试验](tables/all_tuning_results.csv)
- [逐job配置、最优epoch、参数量、时间和源路径](tables/job_configs_and_costs.csv)
- [配对候选/对照实际值及差额](tables/paired_results.csv)
- [逐seed改善/退化/相同次数](tables/improvement_counts_by_seed.csv) / [逐场次数](tables/improvement_counts_by_farm.csv)
- [逐时效原始结果](tables/lead_results.csv.gz) / [逐场景原始结果](tables/scenario_results.csv.gz) / [逐日原始结果（UTC）](tables/daily_results_utc.csv.gz)
- [分时效配对原值](tables/lead_paired_results.csv.gz) / [分场景配对原值](tables/scenario_paired_results.csv.gz)
- [固定窗口逐seed预测与真值](tables/forecast_cases_per_seed.csv.gz) / [每次训练逐epoch记录](tables/training_epoch_results.csv.gz)
- [跨年逐配对方向](tables/cross_year_paired_results.csv) / [历史39目录与旧逐seed结果入口](legacy/README_CN.md)

## 统计口径

每条结果是一个具体风场、具体模型配置、具体随机种子、具体年份的实测MAE/RMSE，单位CF（本站容量归一化功率），越低越好。MAE/RMSE按预测样本定义计算，用户要求的“不用平均值”落实为不跨seed/风场合并结果；模型内部历史摘要、气候均值输入不属于结果平均。

差额统一为**对照误差−候选误差**，严格大于0即改善，小于0为退化，等于0为相同。无百分比门槛、无0.001容差、无显著性筛选。方向基于未舍入数据判断；CSV以17位有效数字保存源浮点数。Excel默认显示12位小数，Markdown/浏览器显示17位有效数字。

次数的分母是匹配的“风场×seed”对，不是独立样本数，也不是独立风场数。同一风场不同seed相关，计数不能解释为统计显著性。每个seed分别对56个风场计数，完整实测值在总表和下方56个风场表中。

## 完成与覆盖

9500/9500训练阶段完成；最终核验时间2026-10-21 23:03:42（UTC+8）。已核验9500权重、19032预测数组、5648修正数组，包含哈希/形状/重算MAE/参考模型一致性；本次仅做结果整理，没有新训练。

- B0参考572次：正式560次、调参参考12次。
- 调参528次：11家族×4配置×6站×2个seed（40、41）；只有2023结果。
- 正式6160次：11家族×56站×10个seed（42–51）。
- 消融2240次：6种消融各56站×5个seed（42–46），E0为56站×10个seed。
- 合计18460条训练job×split实测值（9500验证、8960跨年复评）；持续性另112条（56站×2年），没有seed。
- E10/SELECTED_BASELINE另1120条为既有基线结果复用，不计新增训练。保留原2023选型，展示actual_family和源job；原协议用验证均值选型，本次不导出该均值、不根据2024改选。

“全部”指本次已完成的独立单场轮次，包含调参和参考模型。历史共享/迁移/旧独立试验单独提供入口；没有把不同协议混成新结果。未运行的消融seed 47–51明确留空，不补值。此次不新增R²等未审计主指标。

## 哪些设计可能带来提升

以下所有正改善均保留；同时列出退化，避免把局部效果写成普遍结论。D05/D06与D09/D10为反向读取同一批配对，不是新增独立证据。

'''+md(pd.DataFrame(design_summary))+'''
1. **使用逐次变化的天气预报具有最一致的支持。** E2C_CBH相对CLIMATOLOGY在2024有271/280个MAE配对和273/280个RMSE配对改善。这里对照仍保留天气通路，只把天气换成本站训练期逐lead均值，因此支持的是天气的逐次变化信息。
2. **参考轨迹作为条件有价值，但依赖其他条件。** CB相对CNEUTRAL有170/280个MAE改善；完整模型相对CH有168/280个改善。说明即使已在输出使用b，额外把b/Δb输入adapter也可改善部分运行；仍有退化配对。
3. **额外历史摘要并非总有必要。** 去掉条件摘要的CB相对完整模型改善158/280；只加摘要的CH相对CNEUTRAL改善139/280。完整历史始终进入decoder，故只能讨论额外摘要通路，不能得出“不需要历史”的结论。
4. **输出方式值得继续简化。** DIRECT相对完整模型改善176/280；NEUTRAL_LAST相对CNEUTRAL改善187/280。直接输出同时改变初始化/优化起点；末值锚点对照只在中性条件下成立，不能外推成所有条件下B0都无用。
5. **E2C与E0均有可保留的改善案例。** E2C相对E0有271/560个MAE改善，但289个退化；RMSE改善240/560。反向用E0有289/560个MAE、320/560个RMSE改善。E2C改变了坐标、地形、非线性、深度及参数量，当前只能报告组合效应。
6. **不同指标需分别判断。** 完整模型对Transformer_FC的MAE改善345/560，但RMSE改善173/560；不能把MAE优势直接表述为所有误差指标优势。所有命名基线对比完整保留，不只展示有利对象。

后续可验证CB+DIRECT或新的条件/锚点组合，但本轮没有这些组合实验，不能把两个单独改善相加当作已证实收益。改善案例允许作为候选设计证据；2024已查看，用2024探索得到的新选择需要再验证。论文可以据此讨论天气信息、轨迹条件及输出参数化的适用边界，而不预设完整模型必须胜出。

## 2024每个seed分别有多少MAE配对改善

每个有值单元格分母均为56个风场；缺失表示该消融没有运行该seed。原始误差不在本表聚合，参见逐站表。

'''
sub=seed_counts[(seed_counts.year==2024)&(seed_counts.metric=='mae')&seed_counts.comparison_id.str.startswith('D')]
tab=sub.pivot(index=['comparison_id','design_cn'],columns='seed',values='improved').reindex(columns=range(42,52)).reset_index();tab.columns=[str(c) for c in tab.columns]
report+=md(tab)+'\n## 基线逐配对结果（2024）\n\n'
base=total_counts[(total_counts.year==2024)&total_counts.comparison_id.str.startswith('B')][['candidate','reference','metric','n_pairs','improved','worsened','ties']]
report+=md(base)+'\n## 同一配对在2023与2024的方向\n\n'
report+='验证期提升不自动等于跨年提升；下表按同一风场同一seed核对，两年均改善、仅2023改善、仅2024改善、两年都未改善。原值和相同情况保存在跨年配对CSV中。\n\n'
cc=cross_counts[(cross_counts.metric=='mae')&cross_counts.comparison_id.str.startswith('D')].pivot(index=['comparison_id','design_cn'],columns='category',values='n_pairs').fillna(0).astype(int).reset_index()
report+=md(cc)+'\n## 固定代表站的实际逐seed示例：1043008 大岗\n\n'
report+='大岗来自训练期选定代表站，不按本次提升大小选择。这里展示2024；所有模型、2023及其他55站见下方链接。数字不是平均值。\n\n'
ex=view[(view.site==6)&(view.year==2024)&view.family.isin(['B0','E2C_CBH','CB','CH','CNEUTRAL','DIRECT','NEUTRAL_LAST','E0','CLIMATOLOGY','TRANSFORMER_FC','SELECTED_BASELINE'])]
report+=md(ex[['family','actual_family','seed','mae','rmse']].sort_values(['family','seed']))
report+='''
## 场景、时段与原始预测的解释

逐时效包含24个lead；场景包含当地时区UTC+8的四季、00–06/06–12/12–18/18–24四时段、当前CF分箱（<0.1、[0.1,0.3)、[0.3,0.7)、≥0.7）、最近1小时已观测爬坡（阈值为本站训练期q90）及其补集。future_ramp_diagnostic_only按未来标签定义（未来某步1小时变化≥0.2），只能用于事后诊断。各场景并不互斥，不能把所有场景计数相加当作新增样本。

daily_results_utc按UTC日期，n_forecast_values=n_origins×24；MAE=abs_error_sum/n_forecast_values，RMSE=sqrt(squared_error_sum/n_forecast_values)。这仍是每个模型每个seed自己的日结果，没有跨seed合并。

16个预测窗口沿用已登记的case_selection，不按本次效果重挑；六个训练描述量代表站加龙湖/余字案例。龙湖来源标为posthoc_failure，不能当预注册案例。已从每个job的test_prediction.npy重新导出逐seed轨迹，没有复用旧的跨seed平均曲线。预测数组保持原样。

## 56个风场完整实测表

'''
farm_index=pd.DataFrame([dict(station_id=ids[i],station_name=names[i],capacity_mw=caps[i],table=f'[{ids[i]} {names[i]}](farms/{ids[i]}_{names[i]}.md)') for i in range(56)])
report+=md(farm_index)+'''\n## 可追溯与复现

来源是jobs/<id>/metrics.json，数值沿用完成审计核验的float64预测误差；不混用训练epoch中float32误差归约得到的val_mae。训练epoch值单独保留在training_epoch_results.csv.gz中。每个原始结果含源job、指标路径、权重路径、配置及哈希；source_metrics_hashes.csv记录本次读取的源指标哈希。

导出代码：[export.py](code/export.py)、[catalog.py](code/catalog.py)。在项目环境运行`.runtime/chronos/bin/python runs/independent_single_farm_20261021_v1/seedwise_review_20261022/code/export.py`即可重建。导出不修改训练代码、预测、队列或原有分析结果。最终检查见VALIDATION.json，文件指纹见MANIFEST.json。\n'''
(O/'REPORT_CN.md').write_text(report)

log('Validating export coverage, source equality and workbook structure')
# Exact source values already held as Python floats; round-trip CSV uses round_trip parser.
roundtrip=pd.read_csv(T/'all_trained_results.csv',float_precision='round_trip',dtype={'station_id':str})
assert np.array_equal(roundtrip.mae.to_numpy(),trained.mae.to_numpy())
assert np.array_equal(roundtrip.rmse.to_numpy(),trained.rmse.to_numpy())
assert np.array_equal(roundtrip.bias.to_numpy(),trained.bias.to_numpy())
assert counts['lead']==18572*24 and counts['scenario']==18572*15
assert (seed_counts.n_pairs==56).all()
assert (seed_counts.improved+seed_counts.worsened+seed_counts.ties==seed_counts.n_pairs).all()
assert len(list((O/'farms').glob('*.md')))==56
assert set(view[view.family=='CB'].seed)==set(range(42,47))
assert set(view[view.family=='E0'].seed)==set(range(42,52))
assert len(case_df)==16*(160+10+1)*24 # 160 real models/seeds +10 derived E10 +1 deterministic per window
import openpyxl
wb=openpyxl.load_workbook(O/'EXPERIMENT_RESULTS_PER_SEED_20261022.xlsx',read_only=True,data_only=True)
sheet_rows={s.title:s.max_row-1 for s in wb};assert sheet_rows['正式消融原始值']==17920 and sheet_rows['调参原始值']==540
wb.close()
validation=dict(passed=True,created_at=datetime.now(timezone.utc).isoformat(),trained_jobs=9500,trained_split_rows=len(trained),tuning_split_rows=len(tune),formal_ablation_split_rows=len(main),deterministic_rows=len(deterministic),derived_selected_rows=len(e10),
                detailed_rows=dict(counts),training_epoch_rows=histories,case_prediction_rows=len(case_df),paired_result_rows=len(pairs),farm_markdown_files=56,
                csv_mae_rmse_bias_exact_roundtrip=True,duplicate_primary_keys=0,no_test_tuning=True,no_fabricated_seeds=True,paired_seed_counts_all_56=True,
                workbook_sheet_rows=sheet_rows,source_completion_audit=audit,aggregate_across_seeds_or_farms=False,improvement_rule='reference - candidate > 0; no tolerance or percentage threshold')
(O/'VALIDATION.json').write_text(json.dumps(validation,ensure_ascii=False,indent=2))
manifest=[]
for path in sorted(O.rglob('*')):
    if path.is_file() and path.name not in ['MANIFEST.json','export.log','RESULTS_BUNDLE.zip'] and '__pycache__' not in path.parts:
        manifest.append(dict(path=str(path.relative_to(O)),bytes=path.stat().st_size,sha256=sha(path)))
(O/'MANIFEST.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
with zipfile.ZipFile(O/'RESULTS_BUNDLE.zip','w',compression=zipfile.ZIP_DEFLATED,compresslevel=3) as z:
    for entry in manifest:z.write(O/entry['path'],arcname=entry['path'])
    z.write(O/'MANIFEST.json',arcname='MANIFEST.json')
log(json.dumps(validation,ensure_ascii=False));log('DONE')
