"""Step 15 - Supplementary Figures S1-S4 (same visual style as the article figures).
  S1 threshold sensitivity (eta x proxy heatmap; zeta sweep)                 <- out_threshold_sweep.csv, out_zeta_sweep.csv
  S2 size of the Ljung-Box gate and of the modified Mann-Kendall test        <- out_lb_gate.csv, out_diagnostics.json
  S3 p-value floor: truncated time-shift vs circular-shift p (NASA class I)  <- out_pfloor_nasa.csv
  S4 class shares by public dataset and proxy-noise boundary of S1           <- out_ext_audit.csv, out_bench2_proxy.csv"""
import json, os, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

os.makedirs('figures', exist_ok=True); O = 'outputs/'
BLUE, ORANGE, GREEN, AMBER, PINK = '#1f5fbf', '#d9541e', '#138a5e', '#c98a00', '#c2477d'
INK, GRID = '#111111', '#dddddd'
CLS = {'D': '#bdbdb5', 'P': '#f5ae8c', 'I': '#a9c6f0', 'T': '#f3d58c', 'N': '#d9d9d4'}
LIGHT = LinearSegmentedColormap.from_list('lightYlOrBr', plt.get_cmap('YlOrBr')(np.linspace(0.0, 0.55, 256)))
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 12, 'font.weight': 'bold', 'axes.titleweight': 'bold',
                     'axes.labelweight': 'bold', 'axes.edgecolor': INK, 'axes.labelcolor': INK, 'xtick.color': INK,
                     'ytick.color': INK, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True,
                     'grid.color': GRID, 'lines.linewidth': 2.4, 'legend.frameon': False, 'savefig.dpi': 300,
                     'savefig.bbox': 'tight'})
def ttl(a, t):
    lab = t.split(')')[0] + ')' if t.startswith('(') else ''
    if lab: a.set_title(lab, loc='left', fontsize=15, weight='bold')


def legible(fig, print_w=6.7, min_pt=7.0):
    import matplotlib.text as mtext
    need = min_pt * fig.get_figwidth() / print_w
    for t in fig.findobj(mtext.Text):
        if t.get_text() and t.get_fontsize() < need: t.set_fontsize(need)

# ---------------- S1 thresholds
S = pd.read_csv(O + 'out_threshold_sweep.csv'); Z = pd.read_csv(O + 'out_zeta_sweep.csv')
fig, ax = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw=dict(width_ratios=[1.2, 1], wspace=0.35))
M = S.pivot(index='eta_det', columns='proxy_thr', values='changed') * 100
a = ax[0]; a.grid(False); a.imshow(M.values, cmap=LIGHT, vmin=0, vmax=max(0.5, M.values.max()), aspect='auto')
for i in range(M.shape[0]):
    for j in range(M.shape[1]): a.text(j, i, f'{M.values[i, j]:.2f}', ha='center', va='center', color=INK, fontsize=11)
a.set_xticks(range(M.shape[1])); a.set_xticklabels([f'{v:g}' for v in M.columns]); a.set_yticks(range(M.shape[0]))
a.set_yticklabels([f'{v:g}' for v in M.index]); a.set_xlabel('proxy threshold |r_res|'); a.set_ylabel('determinism threshold η')
ttl(a, '(a) % of 6016 verdicts changed vs defaults (10⁻³, 0.98)')
a = ax[1]; a.plot(Z.zeta_thr, Z.n_P_public, 'o-', color=ORANGE, label='public cells'); a.plot(Z.zeta_thr, Z.n_P_nasa, 's-', color=BLUE, label='NASA cells')
a.axvline(0.01, color=INK, ls='--', lw=1.3); a.set_xscale('log'); a.set_xticks(Z.zeta_thr); a.set_xticklabels([f'{v:g}' for v in Z.zeta_thr]); a.minorticks_off(); a.set_xlabel('restatement threshold ζ'); a.set_ylabel('features classed P')
a.legend(); ttl(a, '(b) Class P count vs ζ threshold (default 0.01)')
legible(fig); fig.savefig('figures/figS1_thresholds.png'); plt.close(fig)

# ---------------- S2 test size
LB = pd.read_csv(O + 'out_lb_gate.csv'); D = json.load(open(O + 'out_diagnostics.json'))
fig, ax = plt.subplots(figsize=(7.5, 4.6))
for n, c in [(168, ORANGE), (3000, BLUE)]:
    q = LB[LB.n == n]; ax.plot(q.phi, q.gate_rejects, 'o-', color=c, label=f'Ljung–Box gate, n = {n}')
    if 'chi2_reference_rejects' in q: ax.plot(q.phi, q.chi2_reference_rejects, 'x:', color=c, label=f'χ² reference, n = {n}')
ax.plot([0.9, 0.99], [D['MK_HR_size_phi0.9'], D['MK_HR_size_phi0.99']], 's--', color=PINK, label='modified Mann–Kendall, n = 3000')
ax.axhline(0.05, color=INK, ls=':', lw=1.3); ax.set_xlabel('AR(1) coefficient φ'); ax.set_ylabel('rejection rate under H₀'); ax.set_ylim(0, 1.05)
ax.legend(loc='center right'); ttl(ax, 'Size of the exchangeability gate and of the Mann–Kendall test')
legible(fig); fig.savefig('figures/figS2_test_size.png'); plt.close(fig)

# ---------------- S3 p floor
P = pd.read_csv(O + 'out_pfloor_nasa.csv')
fig, ax = plt.subplots(figsize=(7.5, 5))
for c, mk, col in [('I', 'o', BLUE), ('T', '^', AMBER), ('N', 'v', '#8c8c86')]:
    q = P[P.cls == c]
    if len(q): ax.scatter(q.p_tts_holm, q.p_circ_holm, s=70, marker=mk, color=col, edgecolor=INK, label=f'class {c} (default test)', zorder=3)
lim = [0, max(1.0, P[['p_tts_holm', 'p_circ_holm']].max().max())]; ax.plot(lim, lim, color=INK, lw=1)
for f_ in P.p_floor_holm.unique(): ax.axvline(f_, color=ORANGE, ls='--', lw=1.2)
ax.axhline(0.05, color=INK, ls=':', lw=1.2); ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel('truncated time-shift p (Holm)'); ax.set_ylabel('circular-shift p, resolution 1/n (Holm)'); ax.legend(loc='lower right')
ttl(ax, 'p-value floor (dashed) and confirmation of NASA verdicts')
legible(fig); fig.savefig('figures/figS3_pfloor.png'); plt.close(fig)

# ---------------- S4 dataset shares and proxy boundary
E = pd.read_csv(O + 'out_ext_audit.csv'); PX = pd.read_csv(O + 'out_bench2_proxy.csv')
fig, ax = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw=dict(wspace=0.3))
sh = pd.crosstab(E.dataset, E['class'], normalize='index')
for c in ['I', 'T', 'N', 'P']:
    if c not in sh: sh[c] = 0
a = ax[0]; bot = np.zeros(len(sh))
for c in ['I', 'T', 'N', 'P']:
    a.bar(sh.index, sh[c], bottom=bot, color=CLS[c], edgecolor='white', lw=2, label=c); bot += sh[c].values
for i, d in enumerate(sh.index): a.text(i, sh.loc[d, 'I'] / 2, f"{sh.loc[d, 'I']:.2f}", ha='center', va='center', color=INK)
a.set_ylabel('share of feature verdicts'); a.legend(ncol=4, loc='upper center', bbox_to_anchor=(0.5, -0.1)); ttl(a, '(a) Class shares by public dataset')
cx = pd.crosstab(PX.noise, PX.cls, normalize='index')
a = ax[1]
for c, col in [('P', ORANGE), ('I', BLUE), ('T', AMBER)]:
    if c in cx: a.plot(cx.index, cx[c], 'o-', color=col, label=f'classed {c}')
a.set_xscale('log'); a.set_xlabel('proxy noise (fraction of SD(y))'); a.set_ylabel('share of planted proxies'); a.legend()
ttl(a, '(b) Restatement boundary of S1')
legible(fig); fig.savefig('figures/figS4_external_proxy.png'); plt.close(fig)
print('supplementary figures done')
