import numpy as np, pandas as pd, sys, json
sys.path.insert(0,'.'); import trace_audit as T
from scipy import stats
pd.set_option('display.width',250); pd.set_option('display.max_columns',30)
g=pd.read_csv('data/GMP.csv')
cells=[f'Cell{i}_V' for i in range(1,9)]
g['Imbalance(V)']=g[cells].max(1)-g[cells].min(1)
feats=['SOH(%)','RUL(cycles)','DoD(%)','PackID','Voltage(V)','Current(A)','Temperature(C)']+cells+['Imbalance(V)']
a=T.run_audit(g,'CycleNumber','Capacity(Ah)',feats,B=999)
print(a.round(4).to_string()); print(a.attrs)
a.to_csv('outputs/out_audit_gmp.csv',index=False); json.dump({k:(v if not isinstance(v,(np.floating,np.bool_)) else float(v)) for k,v in a.attrs.items()},open('outputs/out_audit_gmp_attrs.json','w'),default=float)
f=pd.read_csv('data/nasa_cycle_features.csv')
nf=['ambient_T','I_mean','t_discharge','t_Tmax','V_mean','V_600s','T_max','T_start']
allc=[]
for b,h in f.groupby('battery'):
    h=h.reset_index(drop=True)
    a=T.run_audit(h,'cycle','capacity',nf,B=999)
    a.insert(0,'battery',f'B{b:04d}'); a['eta_target']=a.attrs['eta_target']; a['R2_index']=a.attrs['ceiling']['R2_index']; a['r']=a.attrs['r']
    allc.append(a); print(a.round(4).to_string())
A=pd.concat(allc); A.to_csv('outputs/out_audit_nasa.csv',index=False)
# (no pooling of p-values across cells: the four records are audited and reported separately)
