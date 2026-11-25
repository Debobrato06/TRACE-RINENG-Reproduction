"""All article figures. Readability rules: large bold text, no dark text on blue fills, boxes sized to their content."""
import numpy as np, pandas as pd, json, sys, os, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
sys.path.insert(0, '.'); import trace_audit as T
os.makedirs('figures', exist_ok=True)

BLUE, ORANGE, GREEN, AMBER, PINK, DKGREEN = '#1f5fbf', '#d9541e', '#138a5e', '#c98a00', '#c2477d', '#006400'
C = [BLUE, ORANGE, GREEN, AMBER, PINK, DKGREEN]
INK, MUTED, GRID = '#111111', '#555555', '#dddddd'
NAVY = '#0b2a5b'
CLS = {'D': ('#bdbdb5', INK), 'P': ('#f5ae8c', INK), 'I': ('#a9c6f0', NAVY), 'T': ('#f3d58c', INK),
       'N': ('#ededea', INK), 'N*': ('#ededea', INK)}          # fill, text colour (dark bold text only; no white text)
from matplotlib.colors import LinearSegmentedColormap
LIGHT = LinearSegmentedColormap.from_list('lightYlOrBr', plt.get_cmap('YlOrBr')(np.linspace(0.0, 0.55, 256)))
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 12, 'font.weight': 'bold',
                     'axes.titleweight': 'bold', 'axes.labelweight': 'bold', 'axes.titlesize': 12.5,
                     'axes.labelsize': 12, 'xtick.labelsize': 11, 'ytick.labelsize': 11, 'legend.fontsize': 10.5,
                     'axes.edgecolor': INK, 'axes.labelcolor': INK, 'xtick.color': INK, 'ytick.color': INK,
                     'axes.linewidth': 1.1, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.7, 'lines.linewidth': 2.4,
                     'legend.frameon': False, 'savefig.dpi': 300, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.04})
import textwrap
def title(a, t):
    """Panel label only: the caption carries the figure title and the panel descriptions (Elsevier artwork rules)."""
    lab = t.split(')')[0] + ')' if t.startswith('(') else ''
    a.set_title(lab, loc='left', fontsize=15, weight='bold', pad=6)


def legible(fig, print_w=6.7, min_pt=7.0):
    """Raise every text element to >= min_pt points at the printed width (17 cm full-page width)."""
    import matplotlib.text as mtext
    need = min_pt * fig.get_figwidth() / print_w
    for t in fig.findobj(mtext.Text):
        if t.get_text() and t.get_fontsize() < need:
            t.set_fontsize(need)

g = pd.read_csv('data/GMP.csv'); f = pd.read_csv('data/nasa_cycle_features.csv')
b5 = f[f.battery == 5]; kk = b5.cycle.values.astype(float)
V = json.load(open('outputs/out_validation.json'))

pt = pd.read_csv('outputs/out_power_trend_gmp.csv')
L = json.load(open('outputs/out_shift_landscape_B0005.json'))
G = pd.read_csv('outputs/out_shap_gmp.csv'); ops = G[~G.feature.str.startswith('NULL_') & (G.feature != 'CycleNumber')]; nb = G[G.feature.str.startswith('NULL_')]
# =====================================================================  Fig. 2 provenance
fig, axs = plt.subplots(1, 3, figsize=(13, 4.0), gridspec_kw=dict(wspace=0.38))
axs[0].plot(g.CycleNumber, g['Capacity(Ah)'], color=BLUE, lw=3); axs[0].set_xlabel('Cycle'); axs[0].set_ylabel('Capacity (Ah)')
title(axs[0], '(a) GMP capacity follows a fixed power law')
axs[1].plot(b5.cycle, b5.capacity, color=ORANGE, lw=2.2, label='measured')
trn, _ = T.crossfit_trend(kk, b5.capacity.values); axs[1].plot(b5.cycle, trn, color=INK, lw=1.8, ls='--', label='cross-fitted trend')
axs[1].set_xlabel('Cycle'); axs[1].set_ylabel('Capacity (Ah)'); axs[1].legend(loc='upper right'); title(axs[1], '(b) NASA B0005 capacity scatters around its trend')
k = g.CycleNumber.values.astype(float); cols = ['Capacity(Ah)', 'Voltage(V)', 'Temperature(C)', 'Cell1_V']
eg = [T.determinism_index(k, g[c].values) for c in cols]
en = [T.determinism_index(kk, b5[c].values) for c in ['capacity', 'V_600s', 'T_max', 'T_start']]
xx = np.arange(4); axs[2].bar(xx - 0.2, eg, 0.38, color=BLUE, label='GMP'); axs[2].bar(xx + 0.2, en, 0.38, color=ORANGE, label='NASA B0005')
axs[2].set_yscale('log'); axs[2].axhline(1e-3, color=INK, lw=1.6, ls='--'); axs[2].text(-0.45, 1.5e-3, 'η = 10⁻³', fontsize=11, weight='bold')
axs[2].set_xticks(xx); axs[2].set_xticklabels(['cap.', 'volt.', 'temp.', 'cell/T₀']); axs[2].set_ylabel('determinism index η')
axs[2].legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=2); title(axs[2], '(c) η separates derived from measured columns')
legible(fig); fig.savefig('figures/fig2_provenance.png'); plt.close(fig)

# =====================================================================  Fig. 3 false-positive control
ar = pd.read_csv('outputs/out_typeI_ar.csv'); cn = pd.read_csv('outputs/out_typeI_cond_nasa.csv')
fig, axs = plt.subplots(1, 3, figsize=(13, 4.0), gridspec_kw=dict(wspace=0.32))
ch = ['V', 'I', 'T'] + [f'C{i}' for i in range(1, 9)] + ['Imb']
axs[0].bar(np.arange(12), V['V2a_trend_FPR'], color=BLUE, width=0.72); axs[0].axhline(0.05, color=INK, lw=1.6, ls='--')
axs[0].set_ylim(0, 0.1); axs[0].set_xticks(np.arange(12)); axs[0].set_xticklabels(ch, rotation=90, fontsize=10.5)
axs[0].set_ylabel('false-positive rate'); title(axs[0], '(a) The trend test stays near 5% on GMP nulls')
for a, df, t in [(axs[1], ar, '(b) The shift test stays near 5% as autocorrelation grows (n = 3000)'), (axs[2], cn.groupby('phi')[['FPR_TTS', 'FPR_naive']].mean().reset_index(), '(c) The beyond-index test stays near 5% on NASA residuals')]:
    a.plot(df.phi.astype(str), df.FPR_naive, 's-', color=ORANGE, ms=8, label='naive Spearman')
    a.plot(df.phi.astype(str), df.FPR_TTS, 'o-', color=BLUE, ms=8, label='time-shift (TRACE)')
    a.axhline(0.05, color=INK, lw=1.6, ls='--'); a.set_ylim(0, 1); a.set_xlabel('AR(1) coefficient φ'); title(a, t)
axs[1].legend(loc='center right')
legible(fig); fig.savefig('figures/fig3_typeI.png'); plt.close(fig)

# =====================================================================  Fig. 4 power
pc = pd.read_csv('outputs/out_power_cond_nasa.csv')
fig, axs = plt.subplots(1, 2, figsize=(12, 4.2), gridspec_kw=dict(wspace=0.25))
for i, kd in enumerate(['generator', 'self-surrogate']):
    q = pt[pt.kind == kd]; axs[0].plot(q.rho, q.power_family12, ['o-', 's--'][i], color=C[i], ms=7, label=['true generator', 'self-surrogate'][i])
axs[0].axhline(0.8, color=INK, lw=1.6, ls=':'); axs[0].set_xlabel('injected trend coupling ρ'); axs[0].set_ylabel('power (family of 12)')
axs[0].set_xticks([0, 0.05, 0.1, 0.15, 0.2]); axs[0].legend(loc='upper left'); title(axs[0], '(a) Self-surrogate power matches true-generator power on GMP')
for i, (b, q) in enumerate(pc.groupby('battery')):
    axs[1].plot(q.rho, q.power_family5, 'o-', color=C[i], ms=6, label=f'B{b:04d}')
axs[1].axhline(0.8, color=INK, lw=1.6, ls=':'); axs[1].set_xlabel('injected beyond-index coupling ρ'); axs[1].set_ylabel('power (family of 5)')
axs[1].legend(loc='upper left'); title(axs[1], '(b) NASA power rises with beyond-index coupling in every cell')
legible(fig); fig.savefig('figures/fig4_power.png'); plt.close(fig)

# =====================================================================  Fig. 6 NASA evidence (classes are given in Table 5 only)
fig, ax = plt.subplots(figsize=(8.5, 3.9))
_, ry = T.crossfit_trend(kk, b5.capacity.values); _, rv = T.crossfit_trend(kk, b5.V_600s.values)
ax.plot(kk, ry / ry.std(), color=ORANGE, lw=2.2, label='capacity residual')
ax.plot(kk, rv / rv.std(), color=BLUE, lw=2.0, ls='--', label='V₆₀₀ residual')
ax.set_xlabel('Cycle'); ax.set_ylabel('standardised residual'); ax.legend(loc='upper right', ncol=2)
legible(fig); fig.savefig('figures/fig6_nasa.png'); plt.close(fig)

# =====================================================================  Fig. 5 reliability landscape
fig = plt.figure(figsize=(13, 9.2)); gs = fig.add_gridspec(2, 3, hspace=0.5, wspace=0.42)
Ls = pd.read_csv('outputs/out_mde_surface.csv'); ns = sorted(Ls.n.unique()); ph = sorted(Ls.phi.unique())
Mm = Ls.pivot(index='phi', columns='n', values='MDE80').loc[ph, ns].values
a = fig.add_subplot(gs[0, 0:2]); a.grid(False)
im = a.imshow(np.where(np.isnan(Mm), 1.0, Mm), cmap=LIGHT, vmin=0.1, vmax=1.0, aspect='auto', origin='lower')
for i in range(len(ph)):
    for j in range(len(ns)):
        v = Mm[i, j]; val = 1.0 if np.isnan(v) else v
        a.text(j, i, 'n/r' if np.isnan(v) else f'{v:.2f}', ha='center', va='center', fontsize=12.5, weight='bold', color=INK)
a.set_xticks(range(len(ns))); a.set_xticklabels(ns); a.set_yticks(range(len(ph))); a.set_yticklabels(ph)
a.set_xlabel('record length n (cycles)'); a.set_ylabel('feature autocorrelation φ')
title(a, '(a) MDE₈₀ of the beyond-index test falls as records lengthen (m = 5; n/r = not resolvable)')
cb = fig.colorbar(im, ax=a, fraction=0.035, pad=0.015); cb.ax.tick_params(labelsize=10)
nul = pd.read_csv('outputs/out_null_pvalues.csv'); u = np.linspace(0, 1, 201)
a = fig.add_subplot(gs[0, 2])
a.plot(u, [np.mean(nul.p_naive <= z) for z in u], color=ORANGE, label='naive Spearman')
a.plot(u, [np.mean(nul.p_tts <= z) for z in u], color=BLUE, label='time-shift')
a.plot([0, 1], [0, 1], color=INK, lw=1.4, ls=':'); a.set_xlabel('nominal α'); a.set_ylabel('rejection rate under H₀')
a.legend(loc='lower right'); title(a, '(b) Null p-values follow the diagonal (n = 168, φ = 0.9)')
a = fig.add_subplot(gs[1, 0:2]); d = np.arange(-67, 68)
lab = {'V_600s': 'V₆₀₀ (I)', 'V_mean': 'V̄ (I)', 'T_start': 'T₀ (I)', 'T_max': 'T_max (I)', 'I_mean': 'Ī (T)'}
for i, c in enumerate(['I_mean', 'T_max', 'T_start', 'V_mean', 'V_600s']):
    a.plot(d, L[c], color=[PINK, AMBER, GREEN, ORANGE, BLUE][i], lw=[1.6, 1.6, 1.6, 2.0, 2.6][i], label=lab[c])
a.axvline(0, color=INK, lw=1.4, ls='--'); a.set_xlabel('time shift d (cycles)'); a.set_ylabel('|ρ_S| of residuals')
a.legend(ncol=5, loc='upper center', bbox_to_anchor=(0.5, -0.2)); title(a, '(c) The observed alignment is compared with every truncated shift (NASA B0005, r = 67)')
a = fig.add_subplot(gs[1, 2])
for i, p_ in enumerate(ph):
    q = Ls[Ls.phi == p_]; a.plot(q.n, q.MDE80, 'o-', color=C[i], ms=7, label=f'φ = {p_}')
a.set_xscale('log'); a.set_xlabel('n (cycles)'); a.set_ylabel('MDE₈₀'); a.set_ylim(0, 0.8)
a.axvspan(90, 130, color='#bdbdb5', alpha=0.6, lw=0); a.text(93, 0.04, 'n/r', fontsize=11, weight='bold')
a.legend(loc='upper right'); title(a, '(d) Power increases with record length n')
legible(fig); fig.savefig('figures/fig5_reliability_landscape.png'); plt.close(fig)

# =====================================================================  Fig. 7 consequences: XAI and generalizability
fig = plt.figure(figsize=(13, 9.4)); gs = fig.add_gridspec(2, 3, hspace=0.5, wspace=0.55)
Rall = pd.read_csv('outputs/out_loco.csv'); order = ['Index only', 'Trend-only (T)', 'TRACE-admitted (I)', 'All features', 'Proxy only (P)']
Mg = Rall[Rall.model == 'GBR'].pivot_table(index='feature_set', columns='held_out', values='R2').loc[order]
a = fig.add_subplot(gs[0, 0:2]); a.grid(False)
im = a.imshow(np.clip(Mg.values, 0, 1), cmap=LIGHT, vmin=0, vmax=1, aspect='auto')
for i in range(Mg.shape[0]):
    for j in range(Mg.shape[1]):
        v = Mg.values[i, j]; a.text(j, i, f'{v:.2f}', ha='center', va='center', fontsize=13, weight='bold', color=INK)
a.set_xticks(range(4)); a.set_xticklabels(Mg.columns); a.set_yticks(range(len(order))); a.set_yticklabels(order)
a.add_patch(Rectangle((-0.5, 2.5), 4, 2, fill=False, ec=BLUE, lw=3, ls='--'))
a.text(3.6, 3.5, 'leak-\ninflated', fontsize=11.5, weight='bold', color=BLUE, va='center')
title(a, '(a) Leave-one-cell-out R² shows that admitted features transfer best without leakage')
mR = Rall.groupby(['feature_set', 'model']).RMSE.mean().unstack().loc[order]
a = fig.add_subplot(gs[0, 2]); y = np.arange(len(order))
a.barh(y - 0.2, mR.GBR * 1000, 0.38, color=BLUE, label='GBR'); a.barh(y + 0.2, mR.Ridge * 1000, 0.38, color=AMBER, label='Ridge')
a.set_yticks(y); a.set_yticklabels(['k', 'T', 'I', 'all', 'P']); a.invert_yaxis(); a.set_xlabel('mean RMSE (mAh)')
a.legend(loc='lower right'); title(a, '(b) With gradient boosting, admitted features give the lowest non-leaking RMSE')
S = pd.read_csv('outputs/out_shap_nasa.csv')
for gi, (fs, t) in enumerate([('All features', '(c) With all features, proxies dominate SHAP'), ('TRACE-admitted (I)', '(d) After TRACE, V_600s dominates SHAP')]):
    a = fig.add_subplot(gs[1, gi]); s_ = S[S.feature_set == fs].groupby('feature').mean_abs_shap.mean().sort_values()
    cc = [ORANGE if c in ('t_discharge', 't_Tmax') else BLUE if c in ('V_600s', 'V_mean') else '#8c8c86' for c in s_.index]
    a.barh(range(len(s_)), s_.values * 1000, color=cc, height=0.7); a.set_yticks(range(len(s_))); a.set_yticklabels(s_.index)
    a.set_xlabel('mean |SHAP| (mAh)'); title(a, t)
a = fig.add_subplot(gs[1, 2]); rg = np.random.default_rng(3)
a.scatter(rg.uniform(-.18, .18, len(ops)), ops.mean_abs_shap * 1e3, s=60, color=BLUE, label='real channels')
a.scatter(1 + rg.uniform(-.18, .18, len(nb)), nb.mean_abs_shap * 1e3, s=60, color=ORANGE, marker='s', label='shuffled copies')
a.set_xticks([0, 1]); a.set_xticklabels(['real', 'null']); a.set_xlim(-.55, 1.55); a.set_ylabel('mean |SHAP| (mAh)')
cj = json.load(open('outputs/out_consequence.json')); title(a, f"(e) GMP channel SHAP is indistinguishable from the shuffled null (p = {cj['gmp_shap_mw_p']:.2f})")
legible(fig); fig.savefig('figures/fig9_xai_generalization.png'); plt.close(fig)

# =====================================================================  Fig. 8 fair benchmark
tab = pd.read_csv('outputs/out_bench2_table.csv', index_col=0); sm = json.load(open('outputs/out_bench2_summary.json'))
meth = ['M1_corr', 'M2_shap', 'M3_detrended', 'M5_prewhiten', 'M6_dcca', 'M7_hac', 'M4v1_TRACE', 'M4_TRACE']
mlab = ['Correlation screen', 'SHAP screen', 'Detrended corr. (iid p)', 'Pre-whitened CCF', 'DCCA + surrogates', 'HAC regression', 'TRACE v1', 'TRACE v2']
types = ['info_res', 'info_rate', 'info_smooth', 'info_nonlin', 'info_lag', 'trend_hi', 'null_ar', 'null_trend', 'proxy', 'label']
tlab = ['residual*', 'rate*', 'smoothed', 'nonlinear', 'lag-1', 'trend HI', 'AR null', 'trend null', 'proxy', 'label']
fig = plt.figure(figsize=(14, 10.2)); gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1], hspace=0.52, wspace=0.34)
a = fig.add_subplot(gs[0, :]); a.grid(False); M_ = tab.loc[types, meth].T.values.astype(float)
a.imshow(M_, cmap=LIGHT, vmin=0, vmax=1, aspect='auto')
for i in range(M_.shape[0]):
    for j in range(M_.shape[1]): a.text(j, i, f'{M_[i, j]:.2f}', ha='center', va='center', fontsize=10.5, weight='bold', color=INK)
a.set_xticks(range(len(types))); a.set_xticklabels(tlab, fontsize=11, rotation=30, ha='right'); a.set_yticks(range(len(meth))); a.set_yticklabels(mlab, fontsize=11)
for xv in [4.5, 7.5]: a.axvline(xv, color=INK, lw=2.5)
tr_ = a.get_xaxis_transform()
for xx_, tt_ in [(2.0, 'informative (ideal 1)'), (6.0, 'not informative\n(ideal 0)'), (8.5, 'leakage\n(ideal 0)')]:
    a.text(xx_, -0.33, tt_, ha='center', va='top', fontsize=11.5, weight='bold', transform=tr_)
title(a, '(a) Share of 360 replicates in which each rule credits a planted feature (12 measured targets; * = design types)')
a = fig.add_subplot(gs[1, 0]); keys = [('precision', 'precision'), ('recall_heldout_nolag', 'recall, held-out'), ('false_credit_trend_null', 'false credit, trend/null')]
x = np.arange(len(meth)); w = 0.27
for i, (kk_, lb_) in enumerate(keys):
    a.bar(x + (i - 1) * w, [sm[m][kk_] for m in meth], w, color=[BLUE, GREEN, ORANGE][i], label=lb_)
a.set_xticks(x); a.set_xticklabels(['Corr', 'SHAP', 'Detr', 'PreW', 'DCCA', 'HAC', 'v1', 'v2'], fontsize=10.5, rotation=45, ha='right'); a.set_ylim(0, 1.05)
a.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol=1, fontsize=10)
title(a, '(b) Precision, held-out recall and false credit of each rule')
a = fig.add_subplot(gs[1, 1]); Sw = pd.read_csv('outputs/out_bench2_sweep.csv')
for m, col, lab_ in [('M7_hac', AMBER, 'HAC'), ('M5_prewhiten', PINK, 'Pre-whitened'), ('M4v1_TRACE', '#8c8c86', 'TRACE v1'), ('M4_TRACE', BLUE, 'TRACE v2')]:
    pw = Sw[Sw.feature == 'info_smooth'].groupby('c_lag')[m].mean(); fpr = Sw[Sw.feature == 'null_trend'].groupby('c_lag')[m].mean().mean()
    a.plot(pw.index, pw.values, color=col, marker='o', lw=2.6, label=f'{lab_} (null credit {fpr:.2f})')
a.axhline(0.05, color=INK, ls='--', lw=1.4); a.set_xlabel('coupling strength c (held-out smoothed type)'); a.set_ylabel('share credited'); a.set_ylim(0, 1.02)
a.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=9.8); title(a, '(c) Detection rises with coupling strength (NASA targets, m = 2)')
legible(fig); fig.savefig('figures/fig8_benchmark.png'); plt.close(fig)

# =====================================================================  Fig. 10 external validation on 374 public cells
E = pd.read_csv('outputs/out_ext_audit.csv'); Lx = pd.read_csv('outputs/out_ext_loco_summary.csv')
fig = plt.figure(figsize=(14, 5.6)); gs = fig.add_gridspec(1, 2, width_ratios=[1.25, 1], wspace=0.42)
a = fig.add_subplot(gs[0, 0]); a.grid(False)
sh = E.pivot_table(index='feature', columns='class', values='cell', aggfunc='count', fill_value=0)
for c_ in ['I', 'T', 'N', 'P', 'D']:
    if c_ not in sh: sh[c_] = 0
sh = (sh[['I', 'T', 'N', 'P']].T / sh.sum(1)).T.sort_values('I', ascending=False)
im = a.imshow(sh.values, cmap=LIGHT, vmin=0, vmax=1, aspect='auto')
for i in range(sh.shape[0]):
    for j in range(sh.shape[1]): a.text(j, i, f'{sh.values[i, j]:.2f}', ha='center', va='center', fontsize=10, weight='bold', color=INK)
a.set_xticks(range(4)); a.set_xticklabels(['I', 'T', 'N', 'P']); a.set_yticks(range(len(sh))); a.set_yticklabels(sh.index, fontsize=10.5)
title(a, f'(a) Class shares of 16 charge features audited cell by cell ({E.cell.nunique()} cells)')
a = fig.add_subplot(gs[0, 1]); ds = ['XJTU', 'TJU', 'HUST', 'MIT']; sets = ['index only', 'index + I', 'index + T', 'index + all non-leaking']
cols_ = ['#8c8c86', BLUE, AMBER, GREEN]; x = np.arange(len(ds)); w = 0.2
for i, (s_, c_) in enumerate(zip(sets, cols_)):
    q = Lx[Lx.set == s_].set_index('dataset').loc[ds]
    a.bar(x + (i - 1.5) * w, q.median_rmse_pct, w, color=c_, label=s_, yerr=[q.median_rmse_pct - q.lo, q.hi - q.median_rmse_pct], capsize=2, error_kw=dict(lw=1))
a.set_xticks(x); a.set_xticklabels(ds); a.set_ylabel('median LOCO RMSE (% of initial capacity)')
a.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=2, fontsize=10)
title(a, '(b) Across cells, features TRACE cannot credit on one trajectory (T) carry most transferable information')
legible(fig); fig.savefig('figures/fig7_external.png'); plt.close(fig)
print('figures done')

# =====================================================================  Fig. 9 3-D audit space and 3-D power surface
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from matplotlib import cm
A3 = pd.read_csv('outputs/out_audit_space.csv')
A3['leta'] = np.log10(A3.eta.clip(lower=1e-6))
fig = plt.figure(figsize=(16, 7.2))
a = fig.add_axes([0.0, 0.0, 0.46, 0.92], projection='3d')
style = {'D': ('#6e6e66', 'X', 'D derived'), 'P': (ORANGE, 's', 'P proxy'), 'I': (BLUE, 'o', 'I informative'),
         'T': (AMBER, '^', 'T trend-only'), 'N': ('#9a9a92', 'v', 'N null'), 'N*': ('#2b2b2b', 'D', 'N* untestable')}
for cl, (col, mk, lab) in style.items():
    q = A3[A3.cls == cl]
    if len(q) == 0: continue
    big = q.source.isin(['GMP', 'NASA'])
    a.scatter(q.leta[~big], q.r_res[~big], q.S0[~big], c=col, marker=mk, s=22, alpha=0.55, depthshade=False, edgecolors='none')
    a.scatter(q.leta[big], q.r_res[big], q.S0[big], c=col, marker=mk, s=95, alpha=1.0, depthshade=False, edgecolors='black', linewidths=0.8, label=lab)
a.set_xlabel('log₁₀ η', labelpad=10); a.set_ylabel('|r_res|', labelpad=10); a.set_zlabel('S₀ (beyond-index)', labelpad=10)
a.set_xlim(-6, 0.3); a.set_ylim(0, 1); a.set_zlim(0, 1); a.view_init(elev=22, azim=-58)
a.tick_params(labelsize=10)
a.legend(loc='upper left', bbox_to_anchor=(-0.05, 1.02), fontsize=10.5, ncol=2, markerscale=0.9)
a.set_title('(a)', loc='left', fontsize=15, weight='bold', pad=0)
# (b) power surface
Ls = pd.read_csv('outputs/out_mde_surface.csv'); rh = np.round(np.arange(0, 1.0, 0.1), 1)
q = Ls[Ls.phi == 0.8].sort_values('n'); ns_ = q.n.values
Z = q[[f'p{r_:.1f}' for r_ in rh]].values
XX, YY = np.meshgrid(rh, np.log10(ns_))
b_ = fig.add_axes([0.50, 0.0, 0.42, 0.92], projection='3d')
surf = b_.plot_surface(XX, YY, Z, cmap='YlOrBr', vmin=0, vmax=1, edgecolor='#333333', linewidth=0.4, antialiased=True, alpha=0.95)
_cs = plt.figure().add_subplot().contour(XX, YY, Z, levels=[0.8]); plt.close()
for _seg in _cs.allsegs[0]:
    b_.plot(_seg[:, 0], _seg[:, 1], np.full(len(_seg), 0.82), color=BLUE, lw=5, zorder=10)
b_.set_xlabel('coupling ρ', labelpad=10); b_.set_ylabel('record length n', labelpad=10); b_.set_zlabel('power', labelpad=8)
b_.set_yticks(np.log10(ns_)); b_.set_yticklabels([str(int(v)) for v in ns_], fontsize=9.5); b_.set_zlim(0, 1)
b_.view_init(elev=26, azim=-130); b_.tick_params(labelsize=10)
b_.text2D(0.03, 0.9, 'blue line: 80 % power contour\n(MDE₈₀ as a function of n)', transform=b_.transAxes, fontsize=10.5, weight='bold', color=BLUE)
b_.set_title('(b)', loc='left', fontsize=15, weight='bold', pad=0)
cax = fig.add_axes([0.93, 0.25, 0.012, 0.45]); cb = fig.colorbar(surf, cax=cax); cb.set_label('power', fontsize=11, weight='bold')
legible(fig); fig.savefig('figures/fig10_audit_space_3d.png'); plt.close(fig)
print('fig9 done')
