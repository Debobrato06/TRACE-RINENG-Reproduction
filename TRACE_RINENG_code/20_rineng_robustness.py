"""Step 20 - robustness analyses of the Results in Engineering article.
(a) Index-only ceiling: sensitivity of R2_index to the training fraction (60 / 70 / 80 % of cycles).
(b) Transfer: equal-size control (every family reduced to two features: CC Q + CC time, CV Q + CV time,
    voltage mean + std, current mean + std) and a second learner (random forest) for all input sets.
(c) NASA engineering reading: data for Fig. 3 (restatement of capacity by discharge time; beyond-index co-fluctuation of V600).
Outputs: outputs/rineng_robust_ceiling.csv, rineng_robust_transfer.csv, rineng_robust_transfer_summary.csv, rineng_robust.json."""
import json, os, sys, warnings, numpy as np, pandas as pd
warnings.filterwarnings('ignore'); sys.path.insert(0, '.')
import trace_audit as T
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from joblib import Parallel, delayed
O = 'outputs/'; QUICK = os.environ.get('TRACE_QUICK') == '1'
ns = {}; exec(open('11_external.py').read().split("\nif __name__")[0].split("\nD = load()")[0], ns)
E = pd.read_csv(O + 'out_ext_audit.csv'); cells = E.drop_duplicates('cell')[['cell', 'dataset']]
D = ns['load'](); D = D[D.cell.isin(cells.cell)]
R = {}

# (a) ceiling vs training fraction
rows = []
for c, g in D.groupby('cell'):
    k = g.k.values.astype(float); y = g.capacity.values
    for f in (0.6, 0.7, 0.8):
        rows.append(dict(cell=c, dataset=g.dataset.iloc[0], n=len(g), train_frac=f, R2_index=T.index_ceiling(k, y, train_frac=f)['R2_index']))
Cc = pd.DataFrame(rows); Cc.to_csv(O + 'rineng_robust_ceiling.csv', index=False)
summ = Cc.groupby(['train_frac', 'dataset']).R2_index.agg(median='median', ge09=lambda v: (v >= 0.9).mean(), lt0=lambda v: (v < 0).mean()).reset_index()
alls = Cc.groupby('train_frac').R2_index.agg(median='median', ge09=lambda v: (v >= 0.9).mean(), lt0=lambda v: (v < 0).mean()).reset_index(); alls['dataset'] = 'All'
R['ceiling_vs_frac'] = pd.concat([summ, alls]).round(3).to_dict('records')
w = Cc.pivot_table(index='cell', columns='train_frac', values='R2_index')
from scipy import stats
R['ceiling_rank_agreement'] = {'rho_60_70': float(stats.spearmanr(w[0.6], w[0.7])[0]), 'rho_70_80': float(stats.spearmanr(w[0.7], w[0.8])[0])}
print(pd.concat([summ, alls]).round(2).to_string(index=False))

# (b) equal-size control and second learner
SETS = {'index only': [], 'index + CC phase': ['CC Q', 'CC charge time'], 'index + CV phase': ['CV Q', 'CV charge time'],
        'index + voltage mean, std': ['voltage mean', 'voltage std'], 'index + current mean, std': ['current mean', 'current std'],
        'index + voltage statistics': ['voltage mean', 'voltage std', 'voltage kurtosis', 'voltage skewness', 'voltage slope', 'voltage entropy'],
        'index + current statistics': ['current mean', 'current std', 'current kurtosis', 'current skewness', 'current slope', 'current entropy']}
def one(ds, dd, te, cl):
    tr = dd[dd.cell.isin([c for c in cl if c != te])].iloc[::3]; ts = dd[dd.cell == te]; yv = ts.capacity.values; out = []
    for nm, fs in SETS.items():
        X = ['k'] + fs
        for lrn, m in (('GBR', HistGradientBoostingRegressor(max_iter=150, random_state=0)),
                       ('RF', RandomForestRegressor(n_estimators=60, min_samples_leaf=5, max_features=0.6, max_samples=0.5, n_jobs=1, random_state=0))):
            if lrn == 'GBR' and nm in ('index only', 'index + CC phase', 'index + CV phase', 'index + voltage statistics', 'index + current statistics'):
                continue   # already in rineng_family_loco.csv (identical settings)
            p = m.fit(tr[X], tr.capacity).predict(ts[X])
            out.append(dict(dataset=ds, cell=te, set=nm, learner=lrn, rmse_pct=float(100 * np.sqrt(np.mean((p - yv) ** 2)) / yv[:5].mean())))
    return out
jobs = []
for ds, dd in D.groupby('dataset'):
    cl = dd.cell.unique()
    if QUICK: cl = cl[:10]
    jobs += [(ds, dd, te, cl) for te in cl]
res = Parallel(n_jobs=-1, verbose=0)(delayed(one)(*j) for j in jobs)
B = pd.DataFrame([r for rr in res for r in rr])
L = pd.read_csv(O + 'rineng_family_loco.csv').assign(learner='GBR')
B = pd.concat([B, L[['dataset', 'cell', 'set', 'learner', 'rmse_pct']]], ignore_index=True); B.to_csv(O + 'rineng_robust_transfer.csv', index=False)
rng = np.random.default_rng(0); S = []
for (ds, s, l), g in B.groupby(['dataset', 'set', 'learner']):
    v = g.rmse_pct.values; bs = [np.median(rng.choice(v, len(v))) for _ in range(1000)]
    S.append(dict(dataset=ds, set=s, learner=l, cells=len(v), median=float(np.median(v)), lo=float(np.percentile(bs, 2.5)), hi=float(np.percentile(bs, 97.5))))
S = pd.DataFrame(S); S.to_csv(O + 'rineng_robust_transfer_summary.csv', index=False)
print(S.pivot_table(index=['learner', 'set'], columns='dataset', values='median').round(2))
# does the RF reproduce the GBR ranking of families? Spearman over the 4 families x 4 datasets of fold gains
fam = ['index + CC phase', 'index + CV phase', 'index + voltage statistics', 'index + current statistics']
P = S.pivot_table(index=['dataset', 'set'], columns='learner', values='median').reset_index()
base = P[P.set == 'index only'].set_index('dataset')
P = P[P.set.isin(fam)].copy(); P['gain_GBR'] = P.apply(lambda r: base.loc[r.dataset, 'GBR'] / r.GBR, axis=1); P['gain_RF'] = P.apply(lambda r: base.loc[r.dataset, 'RF'] / r.RF, axis=1)
R['learner_agreement'] = dict(rho=float(stats.spearmanr(P.gain_GBR, P.gain_RF)[0]), p=float(stats.spearmanr(P.gain_GBR, P.gain_RF)[1]),
                              best_family_RF={ds: g.loc[g.gain_RF.idxmax(), 'set'] for ds, g in P.groupby('dataset')})
# identifiability vs RF gain at the cell level (same test as step 19)
fo = {'CC Q': 'CC phase', 'CC charge time': 'CC phase', 'CV Q': 'CV phase', 'CV charge time': 'CV phase'}
for f in SETS['index + voltage statistics']: fo[f] = 'voltage statistics'
for f in SETS['index + current statistics']: fo[f] = 'current statistics'
E['fam'] = E.feature.map(fo); E['I'] = (E['class'] == 'I').astype(int); ci = E.groupby(['cell', 'fam']).I.mean().rename('Ish').reset_index()
rf = B[B.learner == 'RF']; b0 = rf[rf.set == 'index only'][['cell', 'rmse_pct']].rename(columns={'rmse_pct': 'b'})
X = rf[rf.set.isin(fam)].copy(); X['fam'] = X.set.str.replace('index + ', '', regex=False); X = X.merge(b0, on='cell').merge(ci, on=['cell', 'fam'])
X['lg'] = np.log(X.b / X.rmse_pct); r = stats.spearmanr(X.Ish, X.lg)
R['rf_identifiability_vs_gain'] = dict(rho=float(r[0]), p=float(r[1]), pairs=len(X))
json.dump(R, open(O + 'rineng_robust.json', 'w'), indent=1, default=float); print(json.dumps({k: R[k] for k in ['ceiling_rank_agreement', 'learner_agreement', 'rf_identifiability_vs_gain']}, indent=1))
