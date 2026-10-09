"""CBIO033 miRNA lung cancer re-test on GEO GSE137140. Written with PREREG.md before any model is scored."""
import gzip,re,json,sys,os,hashlib,collections
import numpy as np
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score,accuracy_score,balanced_accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
D=os.path.join(os.path.dirname(__file__),'..','data'); OUT=os.path.join(os.path.dirname(__file__),'..','results')
def log(*a): print(*a,flush=True)
# ---------- load
meta=collections.defaultdict(list); ids=[]; X=[]
with gzip.open(f'{D}/GSE137140_series_matrix.txt.gz','rt') as f:
    tab=False
    for l in f:
        if l.startswith('!Sample_title'): meta['title']=[x.strip('"') for x in l.rstrip('\n').split('\t')[1:]]
        elif l.startswith('!Sample_characteristics_ch1'):
            v=[x.strip('"') for x in l.rstrip('\n').split('\t')[1:]]; meta[v[0].split(':')[0]]=[x.split(': ',1)[1] if ': ' in x else '' for x in v]
        elif l.startswith('!series_matrix_table_begin'): tab=True; hdr=next(f).rstrip('\n').split('\t')[1:]
        elif l.startswith('!series_matrix_table_end'): break
        elif tab: p=l.rstrip('\n').split('\t'); ids.append(p[0].strip('"')); X.append([float(x) if x not in ('','null','NA') else np.nan for x in p[1:]])
X=np.array(X,dtype=float).T  # samples x miRNA
log('matrix',X.shape,'nan',int(np.isnan(X).sum()))
assert not np.isnan(X).any()
ds=np.array(meta['disease state']); prefix=np.array([re.match(r'[A-Za-z]+',t).group(0) for t in meta['title']])
sex=np.array(meta['Sex']); age=np.array([float(a) if re.fullmatch(r'[\d.]+',a) else np.nan for a in meta['age']])
pre=ds=='Lung cancer, pre-operation'; ctl=ds=='Non-cancer control'; post=ds=='Lung cancer, post-operation'
use=np.where(pre|ctl)[0]; y_all=pre.astype(int)
Xu=X[use]; yu=y_all[use]; pu=prefix[use]; su=sex[use]; au=age[use]
res={'D1':{'n_pre':int(pre.sum()),'n_ctl':int(ctl.sum()),'n_post':int(post.sum()),'n_mirna':X.shape[1],
 'prefix_by_class':{p:{c:int(((prefix==p)&(ds==c)).sum()) for c in set(ds) if ((prefix==p)&(ds==c)).sum()} for p in sorted(set(prefix))},
 'age_mean':{c:float(np.nanmean(age[ds==c])) for c in set(ds)},'age_missing':int(np.isnan(age).sum()),
 'sex':{c:dict(collections.Counter(sex[ds==c])) for c in set(ds)}}}
log('D1',json.dumps(res['D1']))
def rf(seed=0,n=300): return RandomForestClassifier(n_estimators=n,max_features='sqrt',n_jobs=2,random_state=seed)
def bh(p):
    n=len(p); o=np.argsort(p); q=np.empty(n); r=p[o]*n/(np.arange(n)+1); r=np.minimum.accumulate(r[::-1])[::-1]; q[o]=np.minimum(r,1); return q
def stats_train(Xt,yt):
    a=Xt[yt==1]; b=Xt[yt==0]
    fc=2.0**(a.mean(0)-b.mean(0))
    u=stats.mannwhitneyu(a,b,axis=0).pvalue
    ranks=stats.rankdata(np.vstack([a,b]),axis=0); auc=(ranks[:len(a)].sum(0)-len(a)*(len(a)+1)/2)/(len(a)*len(b))
    return fc,bh(u),np.abs(auc-0.5)
EDGES={'abs':(0.5,0.9,1.1,2.0),'alt1':(0.6,0.95,1.05,1.0/0.6),'alt2':(0.7,0.9,1.1,1.0/0.7)}
def bins(fc,q,edges=EDGES['abs']):
    lo,mdlo,mdhi,hi=edges; sig=q<0.05
    HD=sig&((fc<=lo)|(fc>=hi)); MD=sig&(((fc>lo)&(fc<=mdlo))|((fc>=mdhi)&(fc<hi))); ND=(fc>mdlo)&(fc<mdhi)
    return {'HD':HD,'MD':MD,'ND':ND}
def pick(mask,sc,k=50):
    idx=np.where(mask)[0]; return idx[np.argsort(-sc[idx],kind='stable')[:k]]
def metr(yt,pr,pp): return {'acc':accuracy_score(yt,pr),'bacc':balanced_accuracy_score(yt,pr),'auc':roc_auc_score(yt,pp)}
def agg(L): 
    out={}
    for k in L[0]: v=np.array([m[k] for m in L]); out[k]=[float(v.mean()),float(v.std(ddof=1)) if len(v)>1 else 0.0]
    return out
def folds(n,y,seed=0): return list(StratifiedKFold(5,shuffle=True,random_state=seed).split(np.zeros(n),y))
def cv_bins(Xs,ys,which=('HD','MD','ND'),edges=EDGES['abs'],fold_seed=0,k=50,shuf=False,rngseed=0,keep=None):
    F=folds(len(ys),ys,fold_seed); per={w:[] for w in which}; sizes={w:[] for w in which}
    for tr,te in F:
        fc,q,sc=stats_train(Xs[tr],ys[tr]); B=bins(fc,q,edges)
        for w in which:
            idx=pick(B[w],sc,k); sizes[w].append([int(B[w].sum()),len(idx)])
            if len(idx)==0: continue
            ytr=ys[tr].copy()
            if shuf: ytr=np.random.RandomState(rngseed).permutation(ytr)
            m=rf().fit(Xs[tr][:,idx],ytr); per[w].append(metr(ys[te],m.predict(Xs[te][:,idx]),m.predict_proba(Xs[te][:,idx])[:,1]))
    return {w:agg(per[w]) if per[w] else None for w in which},sizes
# ---------- D2
log('D2'); r2,sz=cv_bins(Xu,yu); res['D2']={'metrics':r2,'bin_sizes_per_fold[total,selected]':sz,
 'gate':{'HD>=0.95':r2['HD']['acc'][0]>=0.95,'MD>=0.90':r2['MD']['acc'][0]>=0.90,'ND>=0.85':r2['ND']['acc'][0]>=0.85}}
log(json.dumps(res['D2']['metrics']),res['D2']['gate'])
# ---------- D3 leakage
log('D3'); fc,q,sc=stats_train(Xu,yu); B=bins(fc,q); sel={w:pick(B[w],sc) for w in B}
F=folds(len(yu),yu); r3={}
for w,idx in sel.items():
    L=[]
    for tr,te in F:
        m=rf().fit(Xu[tr][:,idx],yu[tr]); L.append(metr(yu[te],m.predict(Xu[te][:,idx]),m.predict_proba(Xu[te][:,idx])[:,1]))
    r3[w]=agg(L)
res['D3']={'metrics':r3,'full_bin_sizes':{w:int(B[w].sum()) for w in B},'delta_acc_vs_D2':{w:r3[w]['acc'][0]-r2[w]['acc'][0] for w in r3}}
res['D3']['gate_leakage_matters']=any(v>=0.02 for v in res['D3']['delta_acc_vs_D2'].values()); log(json.dumps(res['D3']))
# ---------- D4
log('D4'); xs=np.isin(pu,['XA','XB']); Xab=Xu[xs&(yu==0)]; lab=(pu[xs&(yu==0)]=='XB').astype(int)
La=[]
for tr,te in folds(len(lab),lab):
    m=rf().fit(Xab[tr],lab[tr]); La.append(roc_auc_score(lab[te],m.predict_proba(Xab[te])[:,1]))
d4a=float(np.mean(La))
cm=pu[yu==0]; Xc=Xu[yu==0]; ids_c=np.unique(cm); ymul=np.searchsorted(ids_c,cm); Lm=[]
for tr,te in StratifiedKFold(5,shuffle=True,random_state=0).split(Xc,ymul):
    m=rf().fit(Xc[tr],ymul[tr]); Lm.append(accuracy_score(ymul[te],m.predict(Xc[te])))
maj=float(np.bincount(ymul).max()/len(ymul))
groups={'XA':['XA'],'XB':['XB'],'PR+SA+BC':['PR','SA','BC']}; r4c={}
for g,pf in groups.items():
    sel_=(yu==1)|np.isin(pu,pf); r,_=cv_bins(Xu[sel_],yu[sel_],which=('HD','MD')); r4c[g]={w:r[w] for w in r}
rng_=lambda w:max(r4c[g][w]['auc'][0] for g in r4c)-min(r4c[g][w]['auc'][0] for g in r4c)
res['D4']={'a_XA_vs_XB_control_only_AUROC':d4a,'a_batch_effect_present':d4a>=0.60,'b_cohort_identity_acc':float(np.mean(Lm)),'b_majority_baseline':maj,
 'c_per_control_group':r4c,'c_auc_range':{w:float(rng_(w)) for w in ('HD','MD')},'c_stable':bool(all(rng_(w)<=0.03 for w in ('HD','MD')))}
log(json.dumps(res['D4']))
# ---------- D5
log('D5'); ok=~np.isnan(au); A=np.c_[au[ok],(su[ok]=='Male').astype(float)]; Ay=yu[ok]; La=[]
for tr,te in folds(len(Ay),Ay):
    sc_=StandardScaler().fit(A[tr]); m=LogisticRegression().fit(sc_.transform(A[tr]),Ay[tr]); La.append(roc_auc_score(Ay[te],m.predict_proba(sc_.transform(A[te]))[:,1]))
d5a=float(np.mean(La))
rs=np.random.RandomState(0); cases=[i for i in np.where(ok&(yu==1))[0]]; rs.shuffle(cases); used=set(); pairs=[]
ctl_idx=np.where(ok&(yu==0))[0]
for i in cases:
    c=[j for j in ctl_idx if j not in used and su[j]==su[i] and abs(au[j]-au[i])<=3]
    if c: j=min(c,key=lambda j:(abs(au[j]-au[i]),j)); used.add(j); pairs.append((i,j))
mi=np.array([p[0] for p in pairs]+[p[1] for p in pairs]); Xm=Xu[mi]; ym=yu[mi]
r5,_=cv_bins(Xm,ym,which=('HD','MD'))
res['D5']={'age_sex_only_AUROC':d5a,'n_matched_pairs':len(pairs),'matched_age_mean':{'LC':float(au[[p[0] for p in pairs]].mean()),'ctl':float(au[[p[1] for p in pairs]].mean())},
 'matched_metrics':r5,'auc_drop_vs_D2':{w:r2[w]['auc'][0]-r5[w]['auc'][0] for w in ('HD','MD')}}
res['D5']['confound_material']=bool(d5a>=0.80 or any(v>=0.05 for v in res['D5']['auc_drop_vs_D2'].values())); log(json.dumps(res['D5']))
# ---------- D6
log('D6'); F=folds(len(yu),yu); aucs=[];accs=[]
for d in range(10):
    rs=np.random.RandomState(d); L=[]
    for tr,te in F:
        idx=rs.choice(Xu.shape[1],50,replace=False); m=rf().fit(Xu[tr][:,idx],yu[tr]); L.append(metr(yu[te],m.predict(Xu[te][:,idx]),m.predict_proba(Xu[te][:,idx])[:,1]))
    a=agg(L); aucs.append(a['auc'][0]); accs.append(a['acc'][0])
res['D6']={'random50_auc_mean':float(np.mean(aucs)),'random50_auc_sd_across_draws':float(np.std(aucs,ddof=1)),'random50_acc_mean':float(np.mean(accs)),
 'MD_auc':r2['MD']['auc'][0],'MD_beats_random_by_0.05':bool(r2['MD']['auc'][0]-np.mean(aucs)>=0.05)}; log(json.dumps(res['D6']))
# ---------- D7
log('D7'); res['D7']={}
for nm,e in EDGES.items():
    r,_=cv_bins(Xu,yu,which=('MD',),edges=e); res['D7'][nm]={'edges':e,'MD':r['MD']}
log(json.dumps(res['D7']))
# ---------- D8
log('D8'); import gzip as gz
seqs={};name=None
for l in gz.open(f'{D}/mirbase.fasta.gz','rt'):
    if l[0]=='>':
        mm=re.search(r'\b(hsa-\S+)\s*$',l.strip()); name=mm.group(1) if mm else None
        if name: seqs.setdefault(name,'')
    elif name: seqs[name]+=l.strip().upper().replace('U','T')
plat={}
for l in open(f'{D}/GPL21263.txt'):
    if l.startswith('MIMAT'): a=l.rstrip('\n').split('\t'); plat[a[0]]=a[2]
mapped=np.array([i for i,m in enumerate(ids) if plat.get(m) in seqs and len(seqs[plat[m]])>0])
def feats(s):
    n=len(s); f=[n,(s.count('G')+s.count('C'))/n]; f+=[s.count(b)/n for b in 'ACGT']
    f+=[sum(1 for i in range(n-1) if s[i:i+2]==d)/(n-1) for d in [a+b for a in 'ACGT' for b in 'ACGT']]
    seed=s[1:8]; f+=[seed.count(b)/max(len(seed),1) for b in 'ACGT']; return f
FM={i:feats(seqs[plat[ids[i]]]) for i in mapped}
def knee(Fm):
    Z=StandardScaler().fit_transform(np.array(Fm)); ks=list(range(2,61)); inert=[KMeans(k,n_init=10,random_state=0).fit(Z).inertia_ for k in ks]
    x=np.array(ks,float); y=np.array(inert); xn=(x-x[0])/(x[-1]-x[0]); yn=(y-y[-1])/(y[0]-y[-1]); dist=np.abs(xn+yn-1)/np.sqrt(2); return ks[int(np.argmax(dist))]
pool=[i for w in ('HD','MD') for i in sel[w] if i in FM]; pool=list(dict.fromkeys(pool))
kreal=knee([FM[i] for i in pool]); rk=[]
for d in range(20):
    ri=np.random.RandomState(100+d).choice(mapped,100,replace=False); rk.append(knee([FM[i] for i in ri]))
lo,hi=np.percentile(rk,[5,95])
res['D8']={'n_mapped':int(len(mapped)),'n_array_mirna':len(ids),'pool_size_after_mapping':len(pool),'knee_real':int(kreal),'knee_random_pools':[int(k) for k in rk],'random_5_95_pct':[float(lo),float(hi)],
 'knee_in_28_34':28<=kreal<=34,'real_outside_central90_random':bool(kreal<lo or kreal>hi),'note':'pool selected on all data (descriptive)'}
res['D8']['reproduced']=bool(res['D8']['knee_in_28_34'] and res['D8']['real_outside_central90_random']); log(json.dumps(res['D8']))
# ---------- D9
log('D9'); fc,q,sc=stats_train(Xu,yu); B=bins(fc,q); res['D9']={}
for w in ('HD','MD'):
    idx=pick(B[w],sc); m=rf().fit(Xu[:,idx],yu); res['D9'][w]={'post_pred_cancer_frac':float(m.predict(X[post][:,idx]).mean()),'train_fit_pre_pred_cancer_frac':float(m.predict(X[pre][:,idx]).mean()),'post_mean_proba':float(m.predict_proba(X[post][:,idx])[:,1].mean())}
log(json.dumps(res['D9']))
# ---------- D10
log('D10'); perm=[]
for p in range(5):
    r,_=cv_bins(Xu,yu,which=('MD',),shuf=True,rngseed=p); perm.append(r['MD']['auc'][0])
res['D10']={'perm_MD_auc':perm,'perm_mean_auc':float(np.mean(perm)),'sane':bool(abs(np.mean(perm)-0.5)<=0.03),'feature_count_curve_MD':{}}
for k in (5,10,50,200):
    r,_=cv_bins(Xu,yu,which=('MD',),k=k); res['D10']['feature_count_curve_MD'][k]=r['MD']
log(json.dumps(res['D10']))
json.dump(res,open(f'{OUT}/results033.json','w'),indent=1,default=float); log('done')
