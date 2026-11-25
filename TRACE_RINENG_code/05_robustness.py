import numpy as np, pandas as pd, sys
sys.path.insert(0,'.'); import trace_audit as T
f=pd.read_csv('data/nasa_cycle_features.csv'); nf=['ambient_T','I_mean','t_discharge','t_Tmax','V_mean','V_600s','T_max','T_start']
base=pd.read_csv('outputs/out_audit_nasa.csv').set_index(['battery','feature'])['class']
rows=[]
orig=T.crossfit_trend
for kn in [6,9,12]:
    for rfrac in ['default',0.25,0.33]:
        T.crossfit_trend=lambda k,x,n_knots=None,_kn=kn: orig(k,x,_kn)
        _old=T.determinism_index
        for b,h in f.groupby('battery'):
            h=h.reset_index(drop=True); n=len(h)
            r=None if rfrac=='default' else int(rfrac*n)
            a=T.run_audit(h,'cycle','capacity',nf,B=499,r=r)
            for _,w in a.iterrows():
                rows.append(dict(knots=kn,r=rfrac,battery=f'B{b:04d}',feature=w.feature,cls=w['class'],base=base[(f'B{b:04d}',w.feature)]))
T.crossfit_trend=orig
d=pd.DataFrame(rows); d['agree']=d.cls==d.base
print(d.groupby(['knots','r']).agree.mean().unstack())
print(d[~d.agree].groupby(['battery','feature']).cls.apply(lambda s: ','.join(s)).to_string())
d.to_csv('outputs/out_robust_nasa.csv',index=False)
