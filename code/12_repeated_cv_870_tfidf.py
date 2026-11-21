import pandas as pd,numpy as np,sys,time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold,train_test_split
I=['ig_outgroup','dehumanization','grievance','mobilization','threat']
G=np.round(np.arange(.05,.9501,.01),2)
g=pd.read_csv('/mnt/user-data/outputs/muril_colab_inputs/gold_corpus_final.csv')
dbl=g[g.split.isin(['iaa','mined_320','solo2_200_relabelled'])].reset_index(drop=True); sol=g[~g.split.isin(['iaa','mined_320','solo2_200_relabelled'])].reset_index(drop=True)
assert len(dbl)==870 and len(sol)==650
def f1v(y,p):
    tp=(y&p).sum(0);fp=((1-y)&p).sum(0);fn=(y&(1-p)).sum(0);d=2*tp+fp+fn
    return np.where(d>0,2*tp/np.maximum(d,1),0)
def run(extra,R=5,K=5):
    key=(dbl[I].sum(1)>0).astype(int)+2*(dbl.split=='iaa')
    out=[]
    for r in range(R):
        oof=np.zeros((len(dbl),5),int); sk=StratifiedKFold(K,shuffle=True,random_state=100+r)
        for tr,te in sk.split(dbl,key):
            T=pd.concat([dbl.iloc[tr],sol]) if extra else dbl.iloc[tr]
            fit,val=train_test_split(T,test_size=.2,random_state=r,stratify=(T[I].sum(1)>0))
            v=TfidfVectorizer(analyzer='char',ngram_range=(2,5),min_df=2,max_features=150000,sublinear_tf=True)
            Xf=v.fit_transform(fit.text);Xv=v.transform(val.text);Xt=v.transform(dbl.iloc[te].text)
            for j,i in enumerate(I):
                m=LogisticRegression(class_weight='balanced',solver='liblinear',max_iter=3000,random_state=42).fit(Xf,fit[i].astype(int))
                pv=m.predict_proba(Xv)[:,1];pt=m.predict_proba(Xt)[:,1];yv=val[i].astype(int).values
                bt,bf=0.5,-1
                for t in G:
                    f=f1v(yv[:,None],(pv>=t).astype(int)[:,None])[0]
                    if f>bf:bt,bf=t,f
                oof[te,j]=(pt>=bt).astype(int)
        out.append(oof); print('extra',extra,'repeat',r,round(f1v(dbl[I].values.astype(int),oof).mean(),3),flush=True)
    return np.array(out)
res={}
for extra in (False,True): res['with650' if extra else 'dbl_only']=run(extra)
np.savez('/tmp/claude-0/cv/oof.npz',**res)
