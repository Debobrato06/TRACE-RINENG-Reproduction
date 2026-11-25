"""Step 8 - coordinates of every MEASURED audited feature in TRACE's 3-D audit space
(log10 eta, |r_res|, beyond-index statistic S0) with its class, for Fig. 9a:
GMP columns, NASA features and the 16 charge features of the 374 public cells (no planted features).
Run after 01_audits.py and 11_external.py."""
import numpy as np, pandas as pd
rows = []
ag = pd.read_csv('outputs/out_audit_gmp.csv')
for _, w in ag.iterrows():
    if w.feature == 'PackID' or pd.isna(w.get('eta', np.nan)): continue
    rows.append(dict(source='GMP', feature=w.feature, eta=w.eta, r_res=abs(w.r_resid) if pd.notna(w.r_resid) else 0.0,
                     S0=w.S_cond if pd.notna(w.S_cond) else 0.0, cls=w['class']))
an = pd.read_csv('outputs/out_audit_nasa.csv')
for _, w in an.iterrows():
    if pd.isna(w.eta): continue
    rows.append(dict(source='NASA', feature=f'{w.battery}:{w.feature}', eta=w.eta, r_res=abs(w.r_resid), S0=w.S_cond, cls=w['class']))
E = pd.read_csv('outputs/out_ext_audit.csv')
for _, w in E.iterrows():
    if pd.isna(w.eta): continue
    rows.append(dict(source=w.dataset, feature=f'{w.cell}:{w.feature}', eta=w.eta, r_res=abs(w.r_resid), S0=w.S_cond, cls=w['class']))
D = pd.DataFrame(rows); D.to_csv('outputs/out_audit_space.csv', index=False)
print(len(D), 'measured features'); print(D.groupby('cls')[['eta', 'r_res', 'S0']].median().round(3)); print(D.cls.value_counts())
