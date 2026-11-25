"""Step 19 - RQ4 of the Results in Engineering article: does single-record identifiability (TRACE class-I share of a
feature family in a cell) predict that family's cross-cell transfer value (leave-one-cell-out error reduction)?
Needs outputs of step 11 (out_ext_audit.csv) and step 17 (rineng_family_loco*.csv).
Outputs: outputs/rineng_rq4.json, outputs/rineng_rq4_pairs.csv, figures/rineng_fig6_identifiability_vs_transfer.png."""
import json, numpy as np, pandas as pd, matplotlib, matplotlib.ticker
from scipy import stats
matplotlib.use('Agg'); import matplotlib.pyplot as plt
O, FG = 'outputs/', 'figures/'
DSC = {'XJTU': '#1f5fbf', 'TJU': '#d9541e', 'HUST': '#138a5e', 'MIT': '#c2477d'}
MK = {'CC phase': 'o', 'CV phase': 's', 'voltage statistics': '^', 'current statistics': 'D'}
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 12, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.grid': True, 'grid.color': '#dddddd', 'savefig.dpi': 300, 'savefig.bbox': 'tight', 'legend.frameon': False})
FAM = {'CC phase': ['CC Q', 'CC charge time'], 'CV phase': ['CV Q', 'CV charge time'],
       'voltage statistics': ['voltage mean', 'voltage std', 'voltage kurtosis', 'voltage skewness', 'voltage slope', 'voltage entropy'],
       'current statistics': ['current mean', 'current std', 'current kurtosis', 'current skewness', 'current slope', 'current entropy']}
fo = {f: k for k, v in FAM.items() for f in v}
E = pd.read_csv(O + 'out_ext_audit.csv'); L = pd.read_csv(O + 'rineng_family_loco.csv')
S = pd.read_csv(O + 'rineng_family_loco_summary.csv'); A = pd.read_csv(O + 'rineng_atlas_family.csv')
E['fam'] = E.feature.map(fo); E['I'] = (E['class'] == 'I').astype(int)
ci = E.groupby(['cell', 'fam']).agg(Ish=('I', 'mean')).reset_index()
base = L[L.set == 'index only'][['cell', 'rmse_pct']].rename(columns={'rmse_pct': 'b'})
X = L[L.set != 'index only'].copy(); X['fam'] = X.set.str.replace('index + ', '', regex=False)
X = X.merge(base, on='cell').merge(ci, on=['cell', 'fam']); X['log_gain'] = np.log(X.b / X.rmse_pct)
X.to_csv(O + 'rineng_rq4_pairs.csv', index=False)
rng = np.random.default_rng(0)
def rho_ci(g):
    r = stats.spearmanr(g.Ish, g.log_gain); bs = []
    cells = g.cell.unique()
    for _ in range(1000):
        s = rng.choice(cells, len(cells)); gg = pd.concat([g[g.cell == c] for c in s]) if False else g.set_index('cell').loc[s]
        bs.append(stats.spearmanr(gg.Ish, gg.log_gain)[0])
    return float(r[0]), float(r[1]), float(np.nanpercentile(bs, 2.5)), float(np.nanpercentile(bs, 97.5))
R = {'pairs': len(X), 'cells': int(X.cell.nunique()), 'pooled': rho_ci(X), 'per_dataset': {ds: rho_ci(g) for ds, g in X.groupby('dataset')}}
agg = []
for ds in DSC:
    b = S[(S.dataset == ds) & (S.set == 'index only')]['median'].item()
    for f in FAM:
        m = S[(S.dataset == ds) & (S.set == 'index + ' + f)]['median'].item()
        agg.append(dict(dataset=ds, family=f, I_share=A[A.family == f][ds].item(), fold=b / m,
                        improved=float((X[(X.dataset == ds) & (X.fam == f)].log_gain > 0).mean())))
G = pd.DataFrame(agg); r16 = stats.spearmanr(G.I_share, G.fold)
R['dataset_family'] = dict(rho=float(r16[0]), p=float(r16[1]), rows=G.round(3).to_dict('records'))
json.dump(R, open(O + 'rineng_rq4.json', 'w'), indent=1); print(json.dumps({k: R[k] for k in ['pairs', 'pooled', 'per_dataset']}, indent=1)); print(G.round(2))

fig, (a, b) = plt.subplots(1, 2, figsize=(12.5, 4.8), gridspec_kw=dict(width_ratios=[1.25, 1], wspace=0.3))
for r in G.itertuples():
    a.scatter(r.I_share, r.fold, s=90, color=DSC[r.dataset], marker=MK[r.family], edgecolor='white', lw=0.8, zorder=3)
a.set_yscale('log'); a.set_ylim(0.8, 9); a.set_yticks([1, 2, 4, 8]); a.set_yticklabels(['1×', '2×', '4×', '8×']); a.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter()); a.axhline(1, color='#555555', lw=1)
a.set_xlabel('share of family features classed I (single record)'); a.set_ylabel('cross-cell error reduction vs index only')
a.text(0.03, 0.97, f'Spearman ρ = {r16[0]:.2f} (p = {r16[1]:.2f})\n16 dataset–family pairs', transform=a.transAxes, ha='left', va='top', fontsize=11)
h1 = [plt.Line2D([], [], ls='', marker='o', color=c, ms=9, label=d) for d, c in DSC.items()]
h2 = [plt.Line2D([], [], ls='', marker=m, color='#555555', ms=9, label=f) for f, m in MK.items()]
a.legend(handles=h1 + h2, loc='upper right', bbox_to_anchor=(1.0, 0.64), fontsize=10, ncol=2); a.set_title('(a)', loc='left', fontsize=15, weight='bold')
names = list(DSC) + ['pooled']; vals = [R['per_dataset'][d] for d in DSC] + [R['pooled']]
for i, (n, v) in enumerate(zip(names, vals)):
    b.errorbar(i, v[0], yerr=[[v[0] - v[2]], [v[3] - v[0]]], fmt='o', ms=9, color=DSC.get(n, '#111111'), capsize=4, lw=2)
b.axhline(0, color='#555555', lw=1); b.set_xticks(range(len(names))); b.set_xticklabels(names)
b.set_ylabel('Spearman ρ, cell-level I share vs gain'); b.set_ylim(-0.6, 0.6); b.set_title('(b)', loc='left', fontsize=15, weight='bold')
fig.savefig(FG + 'rineng_fig6_identifiability_vs_transfer.png'); print('saved fig6')
