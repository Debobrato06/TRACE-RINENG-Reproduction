"""E1 generalizability (leave-one-cell-out) and E2 explainability (SHAP) consequences of TRACE classes."""
import numpy as np, pandas as pd, json, sys, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0,'.'); import trace_audit as T
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import shap
f=pd.read_csv('data/nasa_cycle_features.csv')
FS={'Index only':['cycle'],
    'All features':['cycle','t_discharge','t_Tmax','V_mean','V_600s','T_max','T_start','I_mean'],
    'Proxy only (P)':['t_discharge'],
    'Trend-only (T)':['cycle','T_max','T_start','I_mean'],
    'TRACE-admitted (I)':['cycle','V_600s','V_mean']}
rows=[]; shap_rows=[]
for name,cols in FS.items():
    for held in sorted(f.battery.unique()):
        tr=f[f.battery!=held]; te=f[f.battery==held]
        for mname,mk in [('GBR',lambda: GradientBoostingRegressor(n_estimators=300,max_depth=3,learning_rate=0.05,subsample=0.8,random_state=0)),
                         ('Ridge',lambda: make_pipeline(StandardScaler(),RidgeCV(alphas=np.logspace(-3,3,13))))]:
            m=mk().fit(tr[cols],tr.capacity); p=m.predict(te[cols])
            rmse=float(np.sqrt(np.mean((p-te.capacity)**2))); r2=float(1-np.sum((p-te.capacity)**2)/np.sum((te.capacity-te.capacity.mean())**2))
            rows.append(dict(feature_set=name,model=mname,held_out=f'B{held:04d}',RMSE=rmse,R2=r2))
            if mname=='GBR' and name in ('All features','TRACE-admitted (I)'):
                sv=shap.TreeExplainer(m).shap_values(te[cols])
                for j,c in enumerate(cols): shap_rows.append(dict(feature_set=name,held_out=f'B{held:04d}',feature=c,mean_abs_shap=float(np.mean(np.abs(sv[:,j])))))
R=pd.DataFrame(rows); R.to_csv('outputs/out_loco.csv',index=False)
S=pd.DataFrame(shap_rows); S.to_csv('outputs/out_shap_nasa.csv',index=False)
print(R.pivot_table(index=['feature_set'],columns=['model'],values='RMSE',aggfunc='mean').round(4))
print(R.pivot_table(index=['feature_set','held_out'],columns='model',values='R2').round(3))
print(S.groupby(['feature_set','feature']).mean_abs_shap.mean().round(4))
# ---------- GMP: SHAP null calibration with self-surrogates ----------
g=pd.read_csv('data/GMP.csv'); cells=[f'Cell{i}_V' for i in range(1,9)]
ops=['Voltage(V)','Current(A)','Temperature(C)']+cells
X=g[['CycleNumber']+ops].copy(); y=g['Capacity(Ah)'].values
rng=np.random.default_rng(0)
for c in ops: X['NULL_'+c]=rng.permutation(X[c].values)   # self-surrogate null copies
idx=rng.permutation(len(X)); tr,te=idx[:2400],idx[2400:]       # random split, as commonly (mis)used
m=GradientBoostingRegressor(n_estimators=300,max_depth=3,learning_rate=0.05,subsample=0.8,random_state=0).fit(X.iloc[tr],y[tr])
p=m.predict(X.iloc[te]); r2_rand=float(1-np.sum((p-y[te])**2)/np.sum((y[te]-y[te].mean())**2))
sv=shap.TreeExplainer(m).shap_values(X.iloc[te]); ma=np.mean(np.abs(sv),0)
G=pd.DataFrame(dict(feature=X.columns,mean_abs_shap=ma)); G.to_csv('outputs/out_shap_gmp.csv',index=False)
null_band=G[G.feature.str.startswith('NULL_')].mean_abs_shap
real=G[G.feature.isin(ops)].mean_abs_shap
print('GMP random-split R2',r2_rand,'cycle shap',float(G[G.feature=='CycleNumber'].mean_abs_shap.iloc[0]))
print('real ops mean|SHAP| range',real.min(),real.max(),'null copies range',null_band.min(),null_band.max())
from scipy.stats import mannwhitneyu
mw=mannwhitneyu(real,null_band); print('MW p',mw.pvalue)
# chronological split with operating features only (no index)
m2=GradientBoostingRegressor(n_estimators=300,max_depth=3,learning_rate=0.05,random_state=0).fit(g[ops].iloc[:2100],y[:2100])
p2=m2.predict(g[ops].iloc[2100:]); r2_chrono=float(1-np.sum((p2-y[2100:])**2)/np.sum((y[2100:]-y[2100:].mean())**2))
print('GMP chrono ops-only R2',r2_chrono)
json.dump(dict(gmp_r2_random_split=r2_rand,gmp_r2_chrono_ops_only=r2_chrono,
               gmp_shap_cycle=float(G[G.feature=='CycleNumber'].mean_abs_shap.iloc[0]),
               gmp_shap_ops_min=float(real.min()),gmp_shap_ops_max=float(real.max()),
               gmp_shap_null_min=float(null_band.min()),gmp_shap_null_max=float(null_band.max()),
               gmp_shap_mw_p=float(mw.pvalue)),open('outputs/out_consequence.json','w'),indent=1)
