import numpy as np, pandas as pd, sys, time
sys.path.insert(0,'.'); import trace_audit as T
from scipy import stats
N=3000; n=np.arange(1,N+1); cap=40-12*(n/N)**1.35
import os; QUICK=os.environ.get('TRACE_QUICK')=='1'
R=100 if QUICK else 1000; r=750; t0=time.time()
# (a) regenerated GMP nulls: 12 operating channels per replicate (V,I,T,8 cells, imbalance)
fp=np.zeros(12); fw=0; fp_naive=np.zeros(12); fw_naive=0
for s in range(R):
    rng=np.random.RandomState(1000+s)
    V=rng.normal(25.6,.12,N); I=rng.uniform(-40,40,N); Tt=rng.normal(28,1.8,N); C=rng.normal(3.2,.008,(N,8))
    X=np.column_stack([V,I,Tt,C,C.max(1)-C.min(1)])
    p=np.array([T.tts_test(X[:,j],cap,r)[1] for j in range(12)])
    pn=np.array([stats.spearmanr(X[:,j],cap)[1] for j in range(12)])
    fp+=p<=0.05; fp_naive+=pn<=0.05
    fw+=np.any(T.holm(p)<=0.05); fw_naive+=np.any(T.holm(pn)<=0.05)
print('GMP-null per-channel FPR TTS',np.round(fp/R,3),'mean',fp.mean()/R,'FWER(Holm)',fw/R)
print('GMP-null per-channel FPR naive',np.round(fp_naive/R,3),'mean',fp_naive.mean()/R,'FWER',fw_naive/R)
# (b) autocorrelated nulls: AR(1) with phi, independent of capacity
rows=[]
for phi in [0.0,0.9,0.99,0.995]:
    a=b=0
    for s in range(R):
        rng=np.random.default_rng(5000+s)
        e=rng.normal(size=N+2000); x=np.zeros_like(e)
        for i in range(1,len(e)): x[i]=phi*x[i-1]+e[i]
        x=x[2000:]
        a+=T.tts_test(x,cap,r)[1]<=0.05
        b+=stats.spearmanr(x,cap)[1]<=0.05
    rows.append(dict(phi=phi,FPR_TTS=a/R,FPR_naive=b/R))
    print(rows[-1],round(time.time()-t0))
pd.DataFrame(rows).to_csv('outputs/out_typeI_ar.csv',index=False)
pd.DataFrame(dict(channel=['V','I','T']+[f'C{i}' for i in range(1,9)]+['Imb'],FPR_TTS=fp/R,FPR_naive=fp_naive/R)).to_csv('outputs/out_typeI_gmp.csv',index=False)
open('outputs/out_typeI_fwer.txt','w').write(f'{fw/R},{fw_naive/R}')
