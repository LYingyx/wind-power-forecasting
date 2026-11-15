"""Deterministic, CPU sparse direct ARX; same 2022 fitting / 2023 selection."""
from settings import *
from physical_data import FarmData
from residual_inputs import neighbor_arrays
from train_physics import metrics
from diagnostics import stratified
from sklearn.linear_model import MultiTaskElasticNet
import numpy as np,torch,time,warnings

def features(data,limit=None):
    selected=[0,3,7,11,15,19,23]
    local=np.concatenate([data.h.numpy(),data.weather.numpy()[:,selected].reshape(data.n,-1),data.x.numpy()[:,selected].reshape(data.n,-1)],1)
    p,m,ids,geo,mask,cap=neighbor_arrays(data.split,data.site,'R3',42,limit)
    p=p[:,mask,-5:-1].reshape(data.n,-1);m=m[:,mask,-5:-1].reshape(data.n,-1)
    return dict(LOCAL=local,REG_RAW=np.concatenate([local,p],1),REG_RES=np.concatenate([local,p-m],1),REG_PAIR=np.concatenate([local,p,m],1))

def run(spec,device='cpu',smoke=False):
    assert device=='cpu';setup();torch.set_num_threads(1);start=time.monotonic()
    out=(RUN/'smoke_jobs' if smoke else JOBS)/spec['id'];out.mkdir(parents=True,exist_ok=True)
    if (out/'SUCCESS.json').exists():return
    atomic_json(out/'spec.json',spec);limit=256 if smoke else None
    train=FarmData('train',spec['site'],'N0',limit=limit);val=FarmData('val',spec['site'],'N0',stats=train.stats,limit=limit)
    xtrain=features(train,limit);xval=features(val,limit);target=train.y.numpy()-train.h.numpy()[:,-1:]
    selections={};details=[];arrays={};curves={}
    for name in xtrain:
        x=xtrain[name].astype('f8');mu=x.mean(0);sd=x.std(0);active=sd>1e-6;sd=np.maximum(sd,1e-6)
        x=np.asfortranarray(((x-mu)/sd)[:,active]);xv=np.asfortranarray(((xval[name]-mu)/sd)[:,active]);best=float('inf');curve=[]
        model=MultiTaskElasticNet(alpha=.003,l1_ratio=.8,max_iter=3000,tol=1e-4,warm_start=True,selection='cyclic')
        for alpha in [.003,.001,.0003]:
            model.set_params(alpha=alpha);model.fit(x,target)
            if model.n_iter_>=model.max_iter:
                model.set_params(max_iter=30000);model.fit(x,target)
            assert model.n_iter_<model.max_iter,(spec['id'],name,alpha,'not converged')
            pred=model.predict(xv)+val.h.numpy()[:,-1:];score=metrics(pred,val.y.numpy())
            curve.append(dict(alpha=alpha,n_iter=int(model.n_iter_),mae=score['mae'],rmse=score['rmse']))
            if score['mae']<best:
                best=score['mae'];chosen=dict(alpha=alpha,coef=model.coef_.copy(),intercept=model.intercept_.copy(),score=score,pred=pred)
        arrays.update({name+'_coef':chosen['coef'],name+'_intercept':chosen['intercept'],name+'_mean':mu,name+'_std':sd,name+'_active':active})
        selections[name]=dict(epoch=0,alpha=chosen['alpha'],metrics=chosen['score'],nonzero=int(np.count_nonzero(chosen['coef'])),active_features=int(active.sum()))
        details.extend(dict(family=name,**row) for row in stratified(chosen['pred'],val));curves[name]=curve
        print(spec['id'],name,chosen['alpha'],chosen['score']['mae'],flush=True)
    np.savez_compressed(out/'coefficients.npz',**arrays)
    atomic_json(out/'diagnostics.json',dict(year=2023,rows=details));atomic_json(out/'alpha_selection.json',curves)
    atomic_json(out/'metrics.json',dict(**spec,finished=now(),seconds=time.monotonic()-start,smoke_only=smoke,selections=selections,artifact_sha256=sha(out/'coefficients.npz'),n_train=train.n,n_val=val.n))
    atomic_json(out/'SUCCESS.json',dict(timestamp=now(),smoke_only=smoke))
