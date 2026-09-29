import json, time, numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from core import *
df=load()
tr,te=train_test_split(df,test_size=0.2,stratify=df.optimization_class,random_state=42)
f=Fold(tr); ytr=f.label(tr); Xtr=f.features(tr)
S={'xgb':dict(n_estimators=[100,200,400,600],max_depth=[2,3,4,6],learning_rate=[.03,.05,.08,.1,.2],subsample=[.7,.85,1.0],colsample_bytree=[.6,.8,1.0],reg_lambda=[.5,1,2,5]),
'lgbm':dict(n_estimators=[100,200,400,600],num_leaves=[7,15,31,63],learning_rate=[.03,.05,.08,.1,.2],min_child_samples=[10,20,40],colsample_bytree=[.6,.8,1.0]),
'mlp':dict(hidden_layer_sizes=[(64,),(128,),(128,64),(128,64,32),(256,128)],learning_rate_init=[.0005,.001,.003],alpha=[1e-5,1e-4,1e-3,1e-2],batch_size=[32,64,128]),
'rf':dict(n_estimators=[300,500],max_features=['sqrt','log2',0.5],min_samples_leaf=[1,2,4],max_depth=[None,10,20]),
'et':dict(n_estimators=[300,500],max_features=['sqrt','log2',0.5],min_samples_leaf=[1,2,4],max_depth=[None,10,20]),
'svm':dict(C=[0.3,1,3,10,30,100,300],gamma=['scale',0.003,0.01,0.03,0.1]),
'gb':dict(n_estimators=[100,200,400],max_depth=[2,3,4],learning_rate=[.05,.1,.2],subsample=[.8,1.0]),
'ada':dict(n_estimators=[100,200,400],learning_rate=[.1,.3,.5,1.0],max_depth=[1,2,3])}
out={}; res=[]
for name,space in S.items():
    t=time.time()
    if name=='ada':
        # tune depth manually via wrapper grid
        best=None
        import itertools, random
        rnd=random.Random(42); cand=[dict(n_estimators=a,learning_rate=b,max_depth=c) for a in space['n_estimators'] for b in space['learning_rate'] for c in space['max_depth']]
        cand=rnd.sample(cand,min(30,len(cand)))
        from sklearn.model_selection import cross_val_score
        for c in cand:
            PARAMS['ada']=c
            s=cross_val_score(Pipeline([('s',StandardScaler()),('m',base('ada'))]),Xtr,ytr,cv=StratifiedKFold(3,shuffle=True,random_state=42),scoring='f1_macro',n_jobs=2).mean()
            res.append(dict(family=name,params=str(c),score=s))
            if best is None or s>best[0]: best=(s,c)
        out[name]=best[1]; print(name,best,round(time.time()-t),flush=True); continue
    rs=RandomizedSearchCV(Pipeline([('s',StandardScaler()),('m',base(name))]),{f'm__{k}':v for k,v in space.items()},n_iter=30,
        cv=StratifiedKFold(3,shuffle=True,random_state=42),scoring='f1_macro',random_state=42,n_jobs=2)
    rs.fit(Xtr,ytr)
    out[name]={k[3:]:v for k,v in rs.best_params_.items()}
    for p,s in zip(rs.cv_results_['params'],rs.cv_results_['mean_test_score']): res.append(dict(family=name,params=str({k[3:]:v for k,v in p.items()}),score=s))
    print(name,out[name],round(rs.best_score_,4),round(time.time()-t),flush=True)
json.dump(out,open('hpo_selected.json','w'),indent=1,default=str)
import pandas as pd; pd.DataFrame(res).to_csv('hpo_results.csv',index=False)
json.dump({k:{kk:str(vv) for kk,vv in v.items()} for k,v in S.items()},open('hpo_space.json','w'),indent=1)
