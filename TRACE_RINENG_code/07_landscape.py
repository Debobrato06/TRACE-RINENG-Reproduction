"""Reliability landscape: MDE80 of the beyond-index test over record length n and feature autocorrelation phi;
null p-value calibration; shift-statistic landscape on B0005."""
import numpy as np, pandas as pd, sys, json, time
sys.path.insert(0,'.'); import trace_audit as T
from scipy import stats
def ar(n,phi,rng):
    e=rng.normal(size=n+300); x=np.zeros_like(e)
    for i in range(1,len(e)): x[i]=phi*x[i-1]+e[i]
    return x[300:]
t0=time.time(); rows=[]
ns=[100,150,250,500,1000,2000]; phis=[0.0,0.5,0.8,0.95]; rhos=np.array([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]); R=200; m=5; a=0.05/m
for n in ns:
    r=int(min(max(n//4,int(np.ceil((m/0.05-1)/2))),np.floor(0.4*n)))
    for phi in phis:
        pw=[]
        for rho in rhos:
            h=0
            for s in range(R):
                rng=np.random.default_rng(hash((n,phi,s))%2**32)
                ey=ar(n,0.7,rng); ex=ar(n,phi,rng)       # target residual AR(0.7) as in NASA capacity residuals
                h+=T.tts_test(T.inject(ex,ey,rho),ey,r)[1]<=a
            pw.append(h/R)
        rows.append(dict(n=n,phi=phi,r=r,MDE80=T.mde(rhos,pw),**{f'p{x:.1f}':y for x,y in zip(rhos,pw)}))
    print(n,round(time.time()-t0),flush=True)
L=pd.DataFrame(rows); L.to_csv('outputs/out_mde_surface.csv',index=False); print(L[['n','phi','r','MDE80']].to_string())
# null p-value calibration (P-P): beyond-index TTS vs naive, n=168, phi=0.9
pt,pn=[],[]
for s in range(2000):
    rng=np.random.default_rng(90000+s); ey=ar(168,0.7,rng); ex=ar(168,0.9,rng)
    pt.append(T.tts_test(ex,ey,67)[1]); pn.append(stats.spearmanr(ex,ey)[1])
pd.DataFrame(dict(p_tts=pt,p_naive=pn)).to_csv('outputs/out_null_pvalues.csv',index=False)
print('KS uniform (TTS is conservative/discrete):',np.mean(np.array(pt)<=0.05),np.mean(np.array(pn)<=0.05))
# shift landscape for B0005
f=pd.read_csv('data/nasa_cycle_features.csv'); h=f[f.battery==5]; k=h.cycle.values.astype(float)
_,ry=T.crossfit_trend(k,h.capacity.values); r=67; out={}
for c in ['V_600s','V_mean','T_max','T_start','I_mean']:
    _,rx=T.crossfit_trend(k,h[c].values)
    out[c]=list(T._shift_corr(stats.rankdata(rx),stats.rankdata(ry),r))
json.dump(out,open('outputs/out_shift_landscape_B0005.json','w'))
