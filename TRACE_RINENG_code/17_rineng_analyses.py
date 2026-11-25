"""Step 17 - analyses of the Results in Engineering research article (beyond the MethodsX validation).
(a) Index-attainable accuracy: within-record power-law extrapolation (S2 ceiling) for GMP, the 4 NASA cells and
    374 public cells; GMP remaining-useful-life comparison with the companion LSTM study.
(b) Feature atlas: share of class-I verdicts per charge feature and dataset, by record length and target smoothness.
(c) Transfer value of feature families: within-dataset leave-one-cell-out gradient boosting with
    index only / index + CC-phase / index + CV-phase / index + voltage statistics / index + current statistics.
Outputs: outputs/rineng_*.csv/json and figures/rineng_fig*.png."""
import json, os, sys, warnings, numpy as np, pandas as pd, matplotlib
from scipy import stats
from scipy.optimize import curve_fit, brentq
warnings.filterwarnings('ignore'); sys.path.insert(0, '.')
matplotlib.use('Agg'); import matplotlib.pyplot as plt
QUICK = os.environ.get('TRACE_QUICK') == '1'
O, FG = 'outputs/', 'figures/'
BLUE, ORANGE, GREEN, AMBER, PINK, GREY, INK = '#1f5fbf', '#d9541e', '#138a5e', '#c98a00', '#c2477d', '#8c8c86', '#111111'
DSC = {'XJTU': BLUE, 'TJU': ORANGE, 'HUST': GREEN, 'MIT': PINK}
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 12, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.grid': True, 'grid.color': '#dddddd', 'savefig.dpi': 300, 'savefig.bbox': 'tight', 'legend.frameon': False})
def lab(a, t): a.set_title(t, loc='left', fontsize=15, weight='bold')
def legible(fig, print_w=6.7, min_pt=7.0):
    import matplotlib.text as mt
    need = min_pt * fig.get_figwidth() / print_w
    for t in fig.findobj(mt.Text):
        if t.get_text() and t.get_fontsize() < need: t.set_fontsize(need)

E = pd.read_csv(O + 'out_ext_audit.csv'); A = pd.read_csv(O + 'out_audit_nasa.csv')
FAM = {'CC phase': ['CC Q', 'CC charge time'], 'CV phase': ['CV Q', 'CV charge time'],
       'voltage statistics': ['voltage mean', 'voltage std', 'voltage kurtosis', 'voltage skewness', 'voltage slope', 'voltage entropy'],
       'current statistics': ['current mean', 'current std', 'current kurtosis', 'current skewness', 'current slope', 'current entropy']}
fam_of = {f: k for k, v in FAM.items() for f in v}
R = {}

# ---------------- (a) index-attainable accuracy
d = pd.read_csv('data/GMP.csv'); k = d.CycleNumber.values.astype(float); y = d['Capacity(Ah)'].values; rul = d['RUL(cycles)'].values
pw = lambda k, a, b, c: a - b * k ** c
gmp = []
for h in [500, 1000, 1500]:
    p, _ = curve_fit(pw, k[:h], y[:h], p0=[y[0], 1e-3, 1.2], maxfev=20000)
    eol = brentq(lambda t: pw(t, *p) - 0.7 * y[0], 1, 1e5)
    gmp.append(dict(train_cycles=h, eol=eol, rul_rmse=float(np.sqrt(np.mean((rul[h:] - (eol - k[h:])) ** 2)))))
R['gmp_index_rul'] = gmp; R['cmc_best_lstm_rul_rmse'] = 25.59
C = E.drop_duplicates('cell')[['cell', 'dataset', 'n', 'eta_target', 'R2_index']].copy()
N = A.drop_duplicates('battery')[['battery', 'R2_index', 'eta_target']]
R['nasa_R2_index'] = dict(zip(N.battery, N.R2_index.round(2)))
R['public_R2_index'] = {ds: dict(median=float(g.R2_index.median()), q25=float(g.R2_index.quantile(.25)), q75=float(g.R2_index.quantile(.75)),
                                 share_ge_0_9=float((g.R2_index >= 0.9).mean()), share_lt_0=float((g.R2_index < 0).mean()), cells=len(g))
                        for ds, g in C.groupby('dataset')}
R['public_R2_index_all'] = dict(median=float(C.R2_index.median()), share_ge_0_9=float((C.R2_index >= 0.9).mean()), share_lt_0=float((C.R2_index < 0).mean()))

# ---------------- (b) feature atlas
E['I'] = (E['class'] == 'I').astype(int); E['family'] = E.feature.map(fam_of)
atlas = E.pivot_table(index='feature', columns='dataset', values='I', aggfunc='mean')
atlas['all'] = E.groupby('feature').I.mean(); atlas = atlas.sort_values('all', ascending=False); atlas.round(3).to_csv(O + 'rineng_atlas_feature_dataset.csv')
fam = E.pivot_table(index='family', columns='dataset', values='I', aggfunc='mean'); fam['all'] = E.groupby('family').I.mean(); fam.round(3).to_csv(O + 'rineng_atlas_family.csv')
bins = [0, 300, 600, 1200, 3000]; blab = ['<300', '300–600', '600–1200', '>1200']
E['nbin'] = pd.cut(E.n, bins, labels=blab)
byn = E.groupby(['dataset', 'nbin']).agg(I_share=('I', 'mean'), cells=('cell', 'nunique')).reset_index(); byn.to_csv(O + 'rineng_atlas_length.csv', index=False)
percell = E.groupby('cell').agg(nI=('I', 'sum'), n=('n', 'first'), eta=('eta_target', 'first'), dataset=('dataset', 'first'))
rho_n = stats.spearmanr(percell.n, percell.nI); rho_eta = stats.spearmanr(percell.eta, percell.nI)
R['atlas'] = dict(I_share_all=float(E.I.mean()), cells_with_I=float((percell.nI > 0).mean()),
                  spearman_nI_vs_length=[float(rho_n[0]), float(rho_n[1])], spearman_nI_vs_eta=[float(rho_eta[0]), float(rho_eta[1])],
                  top_features=atlas['all'].head(4).round(3).to_dict(), family_share=fam['all'].round(3).to_dict(),
                  share_by_length={str(b): float(E[E.nbin == b].I.mean()) for b in blab})

# ---------------- (c) transfer value of feature families (leave-one-cell-out within dataset)
from sklearn.ensemble import HistGradientBoostingRegressor
sys.path.insert(0, '.')
import importlib.util
spec = importlib.util.spec_from_file_location('ext', '11_external.py'); src = open('11_external.py').read()
ns = {}; exec(src.split("\nif __name__")[0].split("\nD = load()")[0], ns)   # reuse load() and F without running the audit
FIG_ONLY = os.environ.get('FIG_ONLY') == '1'
D = None if FIG_ONLY else ns['load']()
sets = {'index only': [], **{f'index + {k}': v for k, v in FAM.items()}}
rows = []
for ds, dd in ([] if FIG_ONLY else D[D.cell.isin(C.cell)].groupby('dataset')):
    cl = dd.cell.unique()
    if QUICK: cl = cl[:12]
    for te in cl:
        tr = dd[dd.cell.isin([c for c in cl if c != te])].iloc[::3]; ts = dd[dd.cell == te]; yv = ts.capacity.values
        for nm, fs in sets.items():
            X = ['k'] + fs
            p = HistGradientBoostingRegressor(max_iter=150, random_state=0).fit(tr[X], tr.capacity).predict(ts[X])
            rows.append(dict(dataset=ds, cell=te, set=nm, rmse_pct=float(100 * np.sqrt(np.mean((p - yv) ** 2)) / yv[:5].mean())))
    print(ds, 'family LOCO done', flush=True)
L = pd.read_csv(O + 'rineng_family_loco.csv') if FIG_ONLY else pd.DataFrame(rows)
if not FIG_ONLY: L.to_csv(O + 'rineng_family_loco.csv', index=False)
rng = np.random.default_rng(0); summ = []
for (ds, s), g in L.groupby(['dataset', 'set']):
    v = g.rmse_pct.values; bs = [np.median(rng.choice(v, len(v))) for _ in range(2000)]
    summ.append(dict(dataset=ds, set=s, cells=len(v), median=float(np.median(v)), lo=float(np.percentile(bs, 2.5)), hi=float(np.percentile(bs, 97.5))))
S = pd.DataFrame(summ); S.to_csv(O + 'rineng_family_loco_summary.csv', index=False); print(S.round(2).to_string(index=False))
R['family_loco'] = {ds: {r.set: round(r.median, 3) for r in g.itertuples()} for ds, g in S.groupby('dataset')}
json.dump(R, open(O + 'rineng_results.json', 'w'), indent=1, default=float)

# ---------------- figures
# Fig. 2: index-attainable accuracy
fig, ax = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw=dict(width_ratios=[1, 1.25], wspace=0.3))
a = ax[0]; p, _ = curve_fit(pw, k[:500], y[:500], p0=[y[0], 1e-3, 1.2])
a.plot(k, y, color=INK, lw=3, label='GMP capacity (data)'); a.plot(k, pw(k, *p), color=ORANGE, ls='--', lw=2, label='power law, fit on cycles 1–500')
a.axvline(500, color=GREY, ls=':'); a.axhline(0.7 * y[0], color=GREY, ls=':'); a.set_xlabel('cycle'); a.set_ylabel('capacity (Ah)'); a.legend(loc='lower left')
a.text(0.04, 0.47, f"RUL error (cycles)\nindex only: {gmp[0]['rul_rmse']:.2f}\nbest LSTM [CMC]: 25.6", transform=a.transAxes, fontsize=10, va='top', ha='left', bbox=dict(fc='white', ec='0.8', pad=3))
lab(a, '(a)')
a = ax[1]; order = ['XJTU', 'TJU', 'HUST', 'MIT']
for i, ds in enumerate(order):
    v = C[C.dataset == ds].R2_index.clip(-2, 1).values
    a.boxplot(v, positions=[i], widths=0.55, showfliers=False, medianprops=dict(color=INK, lw=2))
    a.scatter(np.full(len(v), i) + np.random.default_rng(i).uniform(-0.18, 0.18, len(v)), v, s=10, color=DSC[ds], alpha=0.6, zorder=3)
a.scatter([4] * 4, N.R2_index.clip(-2, 1), s=40, color=GREY, zorder=3); a.scatter([5], [1.0], s=80, marker='*', color=ORANGE, zorder=3)
a.set_xticks(range(6)); a.set_xticklabels(order + ['NASA', 'GMP']); a.axhline(0, color=INK, lw=0.8)
a.set_ylabel('index-only R² (last 30% of cycles)'); lab(a, '(b)')
legible(fig); fig.savefig(FG + 'rineng_fig2_index_ceiling.png'); plt.close(fig)

# Fig. 3: NASA audit card
feats = ['ambient_T', 't_discharge', 't_Tmax', 'V_600s', 'V_mean', 'T_max', 'T_start', 'I_mean']
names = ['ambient temperature', 'discharge time', 'time to peak temperature', 'voltage after 600 s', 'mean voltage', 'peak temperature', 'start temperature', 'mean current']
CLS = {'D': '#bdbdb5', 'P': '#f5ae8c', 'I': '#a9c6f0', 'T': '#f3d58c', 'N': '#e5e5e0'}
bats = ['B0005', 'B0006', 'B0007', 'B0018']
fig, a = plt.subplots(figsize=(8.5, 5.2)); a.grid(False)
for i, f in enumerate(feats):
    for j, b in enumerate(bats):
        r = A[(A.feature == f) & (A.battery == b)].iloc[0]; c = r['class']
        a.add_patch(plt.Rectangle((j, i), 0.95, 0.9, fc=CLS[c], ec='white'))
        sub = f"|r|={abs(r.r_resid):.3f}" if c == 'P' else (f"S₀={r.S_cond:.2f}" if c in 'IT' else '')
        a.text(j + 0.47, i + 0.33, c, ha='center', va='center', fontsize=14, weight='bold'); a.text(j + 0.47, i + 0.68, sub, ha='center', va='center', fontsize=10)
a.set_xlim(0, 4); a.set_ylim(len(feats), 0); a.set_xticks(np.arange(4) + 0.47); a.set_xticklabels(bats); a.set_yticks(np.arange(len(feats)) + 0.45)
a.set_yticklabels(names); a.tick_params(length=0); [s.set_visible(False) for s in a.spines.values()]
legible(fig); fig.savefig(FG + 'rineng_fig3_nasa_card.png'); plt.close(fig)

# Fig. 4: feature atlas
fig, ax = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw=dict(width_ratios=[1.2, 1], wspace=0.45))
a = ax[0]; a.grid(False); M = atlas[order].values
from matplotlib.colors import LinearSegmentedColormap
cm = LinearSegmentedColormap.from_list('b', ['#ffffff', '#a9c6f0', '#1f5fbf'])
a.imshow(M, cmap=cm, vmin=0, vmax=0.6, aspect='auto')
for i in range(M.shape[0]):
    for j in range(M.shape[1]): a.text(j, i, f'{M[i, j]:.2f}', ha='center', va='center', fontsize=10)
a.set_xticks(range(4)); a.set_xticklabels(order); a.set_yticks(range(len(atlas))); a.set_yticklabels(atlas.index); lab(a, '(a)')
a = ax[1]
for ds in order:
    q = byn[byn.dataset == ds]; q = q[q.cells >= 3]
    a.plot([blab.index(b) for b in q.nbin], q.I_share, marker='o', color=DSC[ds], lw=2.2, label=ds)
a.axhline(0.05, color=GREY, ls='--', lw=1.2); a.set_xticks(range(4)); a.set_xticklabels(blab); a.set_xlabel('record length (cycles)')
a.set_ylabel('share of features classed I'); a.legend(); lab(a, '(b)')
legible(fig); fig.savefig(FG + 'rineng_fig4_atlas.png'); plt.close(fig)

# Fig. 5: transfer value of feature families
fig, a = plt.subplots(figsize=(11, 4.8)); w = 0.16; sl = list(sets)
cols = [GREY, BLUE, GREEN, AMBER, PINK]
for i, s in enumerate(sl):
    q = S[S.set == s].set_index('dataset').reindex(order)
    a.bar(np.arange(4) + (i - 2) * w, q['median'], w, color=cols[i], label=s, yerr=[q['median'] - q.lo, q.hi - q['median']], capsize=2)
a.set_xticks(range(4)); a.set_xticklabels(order); a.set_ylabel('median LOCO RMSE (% of initial capacity)')
a.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=3)
legible(fig); fig.savefig(FG + 'rineng_fig5_family_transfer.png'); plt.close(fig)
print(json.dumps({k: v for k, v in R.items() if k != 'family_loco'}, indent=1, default=float)[:3000])
