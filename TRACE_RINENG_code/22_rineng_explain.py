"""Step 22 - explainability and explanation reliability (Results in Engineering article).
(A) Single-record explanations: for each of the 374 public cells a model is fitted within the record with the common
    random 80/20 row split (index k + 16 charge features -> capacity), and explained with TreeSHAP.
    Measured: random-split R2 (the score such a study would report), share of mean |SHAP| given to the index and to
    features of each TRACE class, whether the top-ranked feature is one the audit can credit (class I),
    and explanation stability (Spearman agreement of the feature ranking between two random splits).
(B) Cross-cell explanations: within each dataset, leave-one-cell-out models on index + 16 features, TreeSHAP on each
    held-out cell; reliability of the feature ranking across folds (Kendall's W) and its agreement with the
    single-record class-I share of each feature.
Outputs: outputs/rineng_xai_within.csv, rineng_xai_cross.csv, rineng_xai.json; figures/rineng_fig8_explainability.png."""
import json, os, sys, warnings, numpy as np, pandas as pd, matplotlib, shap
warnings.filterwarnings('ignore'); sys.path.insert(0, '.'); matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score
from joblib import Parallel, delayed
O, FG = 'outputs/', 'figures/'; QUICK = os.environ.get('TRACE_QUICK') == '1'
ns = {}; exec(open('11_external.py').read().split("\nif __name__")[0].split("\nD = load()")[0], ns); F = ns['F']
E = pd.read_csv(O + 'out_ext_audit.csv'); D = ns['load'](); D = D[D.cell.isin(E.cell.unique())]
CLS = E.set_index(['cell', 'feature'])['class']
DSC = {'XJTU': '#1f5fbf', 'TJU': '#d9541e', 'HUST': '#138a5e', 'MIT': '#c2477d'}
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 12, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.grid': True, 'grid.color': '#dddddd', 'savefig.dpi': 300, 'savefig.bbox': 'tight', 'legend.frameon': False})
X_COLS = ['k'] + F


def within(cell, g):
    out = {}
    imps = []
    for seed in (0, 1):
        rng = np.random.default_rng(seed); idx = rng.permutation(len(g)); cut = int(0.8 * len(g))
        tr, te = g.iloc[idx[:cut]], g.iloc[idx[cut:]]
        m = HistGradientBoostingRegressor(max_iter=150, random_state=0).fit(tr[X_COLS], tr.capacity)
        sv = shap.TreeExplainer(m).shap_values(te[X_COLS]); imp = np.abs(sv).mean(0); imps.append(imp)
        if seed == 0: out['R2_random_split'] = r2_score(te.capacity, m.predict(te[X_COLS]))
    imp = imps[0]; tot = imp.sum()
    cls = np.array(['index'] + [CLS[(cell, f)] for f in F])
    for c in ['index', 'I', 'T', 'N', 'P']: out[f'share_{c}'] = float(imp[cls == c].sum() / tot)
    fi = imp[1:]; top = F[int(np.argmax(fi))]
    out.update(cell=cell, dataset=g.dataset.iloc[0], n=len(g), top_feature=top, top_class=CLS[(cell, top)],
               n_I=int((cls[1:] == 'I').sum()), stability=float(stats.spearmanr(imps[0][1:], imps[1][1:])[0]),
               index_rank=int((imp > imp[0]).sum()) + 1)
    return out


cells = list(D.groupby('cell'))
if QUICK: cells = cells[:30]
FIG_ONLY = os.environ.get('FIG_ONLY') == '1'
if FIG_ONLY: W = pd.read_csv(O + 'rineng_xai_within.csv')
else: W = pd.DataFrame(Parallel(n_jobs=-1)(delayed(within)(c, g) for c, g in cells)); W.to_csv(O + 'rineng_xai_within.csv', index=False)


def cross_fold(ds, dd, te, cl):
    tr = dd[dd.cell.isin([c for c in cl if c != te])].iloc[::3]; ts = dd[dd.cell == te]
    if len(ts) > 300: ts = ts.iloc[np.linspace(0, len(ts) - 1, 300).astype(int)]
    m = HistGradientBoostingRegressor(max_iter=150, random_state=0).fit(tr[X_COLS], tr.capacity)
    sv = np.abs(shap.TreeExplainer(m).shap_values(ts[X_COLS])).mean(0)
    return dict(dataset=ds, cell=te, **{c: float(v) for c, v in zip(X_COLS, sv)})


jobs = []
for ds, dd in D.groupby('dataset'):
    cl = dd.cell.unique()
    if QUICK: cl = cl[:8]
    jobs += [(ds, dd, te, cl) for te in cl]
if FIG_ONLY: X = pd.read_csv(O + 'rineng_xai_cross.csv')
else: X = pd.DataFrame(Parallel(n_jobs=-1)(delayed(cross_fold)(*j) for j in jobs)); X.to_csv(O + 'rineng_xai_cross.csv', index=False)


def kendall_w(M):  # rows = folds (raters), cols = features
    R = np.apply_along_axis(stats.rankdata, 1, M); m, n = R.shape
    S = ((R.sum(0) - m * (n + 1) / 2) ** 2).sum(); return float(12 * S / (m ** 2 * (n ** 3 - n)))


Ish = E.assign(I=(E['class'] == 'I')).groupby(['dataset', 'feature']).I.mean()
res = {'within': {}, 'cross': {}}
for ds, g in W.groupby('dataset'):
    res['within'][ds] = dict(cells=len(g), R2_random_median=float(g.R2_random_split.median()), share_index=float(g.share_index.median()),
                             share_I=float(g.share_I.mean()), share_T=float(g.share_T.mean()), share_N=float(g.share_N.mean()),
                             top_is_I=float((g.top_class == 'I').mean()), top_is_T=float((g.top_class == 'T').mean()),
                             stability_median=float(g.stability.median()))
res['within']['All'] = dict(cells=len(W), R2_random_median=float(W.R2_random_split.median()), share_index=float(W.share_index.median()),
                            share_I=float(W.share_I.mean()), share_T=float(W.share_T.mean()), share_N=float(W.share_N.mean()),
                            top_is_I=float((W.top_class == 'I').mean()), top_is_T=float((W.top_class == 'T').mean()),
                            stability_median=float(W.stability.median()), R2_ge_0_99=float((W.R2_random_split >= 0.99).mean()),
                            stability_vs_topI=float(stats.mannwhitneyu(W[W.top_class == 'I'].stability, W[W.top_class != 'I'].stability)[1]))
for ds, g in X.groupby('dataset'):
    M = g[F].values; mean_imp = g[F].mean(); rk = mean_imp.sort_values(ascending=False)
    r = stats.spearmanr(mean_imp.values, Ish.loc[ds].reindex(F).values)
    res['cross'][ds] = dict(folds=len(g), kendall_W=kendall_w(M), top3=list(rk.index[:3]), index_share=float((g['k'] / g[X_COLS].sum(1)).median()),
                            rho_importance_vs_Ishare=float(r[0]), p=float(r[1]))
json.dump(res, open(O + 'rineng_xai.json', 'w'), indent=1); print(json.dumps(res, indent=1))

# ---------------- figure
fig, ax = plt.subplots(1, 3, figsize=(17, 5.2), gridspec_kw=dict(width_ratios=[1.15, 1, 1.25], wspace=0.38))
a = ax[0]; order = ['XJTU', 'TJU', 'HUST', 'MIT']; cols = {'index': '#111111', 'I': '#138a5e', 'T': '#8c8c86', 'N': '#c98a00'}
bottom = np.zeros(4)
for c in ['index', 'I', 'T', 'N']:
    v = np.array([W[W.dataset == d][f'share_{c}'].mean() for d in order]); a.bar(order, 100 * v, bottom=100 * bottom, color=cols[c], label=c if c != 'index' else 'cycle index', width=0.6)
    bottom += v
a.set_ylabel('share of mean |SHAP| (%)'); a.set_ylim(0, 118); a.legend(ncol=4, loc='upper center', fontsize=10.5, bbox_to_anchor=(0.5, 1.02), handlelength=1)
a.set_title('(a)', loc='left', fontsize=15, weight='bold')
a = ax[1]
for d in order:
    g = W[W.dataset == d]; a.scatter(g.stability, g.R2_random_split.clip(0.9, 1), s=16, color=DSC[d], alpha=0.75, label=d)
a.set_xlabel('explanation stability between two splits (ρ)'); a.set_ylabel('random-split R² (clipped at 0.9)'); a.legend(fontsize=10, loc='lower left')
a.set_title('(b)', loc='left', fontsize=15, weight='bold')
a = ax[2]
for d in order:
    g = X[X.dataset == d]; mi = g[F].mean(); a.scatter(Ish.loc[d].reindex(F).values, mi.values / mi.max(), s=40, color=DSC[d], label=f"{d}: ρ = {res['cross'][d]['rho_importance_vs_Ishare']:.2f}, W = {res['cross'][d]['kendall_W']:.2f}")
a.set_xlabel('single-record class-I share of feature'); a.set_ylabel('cross-cell mean |SHAP| (relative)'); a.legend(fontsize=9.5, loc='upper right', bbox_to_anchor=(1.0, 0.88))
a.set_title('(c)', loc='left', fontsize=15, weight='bold')
fig.savefig(FG + 'rineng_fig8_explainability.png'); print('saved fig8')
