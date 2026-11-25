"""Step 18 - Fig. 1 of the Results in Engineering article: methodology / study design.
Four zones built from the real data and results: (1) data, (2) audit of each record (TRACE, co-submitted),
(3) analysis steps, (4) verified outcomes and the screening checklist. Run after steps 11, 17 and 19."""
import json, sys, numpy as np, pandas as pd, matplotlib
sys.path.insert(0, '.'); matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import trace_audit as T
O, FG = 'outputs/', 'figures/'
BLUE, ORANGE, GREEN, PINK, GREY, INK, AMBER, PURPLE = '#1f5fbf', '#d9541e', '#138a5e', '#c2477d', '#8c8c86', '#111111', '#c98a00', '#6a3d9a'
DSC = {'XJTU': BLUE, 'TJU': ORANGE, 'HUST': GREEN, 'MIT': PINK}
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.grid': True, 'grid.color': '#e6e6e6', 'savefig.dpi': 300, 'legend.frameon': False})
W, H = 18, 10.2
fig = plt.figure(figsize=(W, H))
bg = fig.add_axes([0, 0, 1, 1]); bg.set_xlim(0, W); bg.set_ylim(0, H); bg.axis('off')


def zone(x, w, col, title):
    bg.add_patch(FancyBboxPatch((x, 0.75), w, 8.25, boxstyle='round,pad=0.02,rounding_size=0.18', fc='white', ec=col, lw=2.5))
    bg.text(x + w / 2, 8.68, title, ha='center', va='center', fontsize=14, weight='bold', color=INK)


def ax_at(x, y, w, h, **kw):
    return fig.add_axes([x / W, y / H, w / W, h / H], **kw)


def tag(a, t, fs=9.5):
    a.set_title(t, loc='left', fontsize=fs, weight='bold')


def arrow(x0, x1, y, txt):
    bg.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle='-|>', mutation_scale=22, lw=2.2, color=GREY))
    bg.text((x0 + x1) / 2, y + 0.18, txt, ha='center', fontsize=9, color='#555555')


# banner
bg.add_patch(FancyBboxPatch((0.2, 9.25), W - 0.4, 0.75, boxstyle='round,pad=0.02,rounding_size=0.15', fc='#fbf1e4', ec=AMBER, lw=1.5))
bg.text(W / 2, 9.62, 'What can one battery ageing record support?  Index ceiling · provenance · identifiable features · cross-cell transfer · explanation reliability',
        ha='center', va='center', fontsize=14, weight='bold')
zx = [(0.2, 3.7, GREEN, '1  Data (Step 1)'), (4.35, 4.15, BLUE, '2  Audit of each record (Step 2)'),
      (8.95, 4.6, PURPLE, '3  Analyses (Steps 3–8)'), (14.0, 3.8, ORANGE, '4  Verified outcomes')]
for z in zx: zone(*z)
arrow(3.92, 4.33, 4.9, ''); arrow(8.52, 8.93, 4.9, ''); arrow(13.57, 13.98, 4.9, '')
bg.text(W / 2, 0.38, 'one row per cycle  →  remove restatements  →  report the index ceiling  →  check record length  →  test transfer on other cells'
        '     |     reproducible Python / Google Colab code', ha='center', fontsize=11, color=PURPLE, weight='bold')

# ---------- zone 1 data
g = pd.read_csv('data/GMP.csv'); a = ax_at(0.75, 6.25, 2.95, 1.75)
a.plot(g.CycleNumber, g['Capacity(Ah)'], color=BLUE, lw=2); a.set_xlabel('cycle', labelpad=1); a.set_ylabel('capacity (Ah)')
tag(a, '(a) GMP LiFePO$_4$ pack, 25.6 V 40 Ah'); a.text(0.04, 0.08, '1 record, 3000 cycles\ngenerator-reproducible', transform=a.transAxes, fontsize=8.5)
N = pd.read_csv('data/nasa_cycle_features.csv'); a = ax_at(0.75, 3.55, 2.95, 1.75)
for b, gg in N.groupby('battery'): a.plot(gg.cycle, gg.capacity, lw=1.3, label=f'B{b:04d}')
a.legend(fontsize=7.5, ncol=2, loc='upper right'); a.set_xlabel('cycle', labelpad=1); a.set_ylabel('capacity (Ah)')
tag(a, '(b) NASA cells: measured, noisy'); a.set_ylim(1.2, 2.25)
ns = {}; exec(open('11_external.py').read().split("\nif __name__")[0].split("\nD = load()")[0], ns)
E = pd.read_csv(O + 'out_ext_audit.csv'); D = ns['load'](); D = D[D.cell.isin(E.cell.unique())]
a = ax_at(0.75, 1.15, 2.95, 1.6); rng = np.random.default_rng(1)
for ds, col in DSC.items():
    cs = D[D.dataset == ds].cell.unique(); pick = rng.choice(cs, min(8, len(cs)), replace=False)
    for c in pick:
        gg = D[D.cell == c]; a.plot(gg.k, gg.capacity / gg.capacity.iloc[:5].mean(), color=col, lw=0.8, alpha=0.8)
a.set_xscale('log'); a.set_xticks([1, 10, 100, 1000]); a.set_xticklabels(['1', '10', '100', '1000']); a.set_xlabel('cycle', labelpad=1); a.set_ylabel('relative capacity'); a.set_ylim(0.6, 1.08)
tag(a, '(c) 374 public cells, 16 charge features')
for i, (ds, col) in enumerate(DSC.items()): a.text(0.03 + 0.24 * i, 0.06, ds, color=col, transform=a.transAxes, fontsize=8.5, weight='bold')

# ---------- zone 2 audit
b5 = N[N.battery == 5]; k = b5.cycle.values.astype(float)
ey = T.crossfit_trend(k, b5.capacity.values)[1]; ev = T.crossfit_trend(k, b5.V_600s.values)[1]
a = ax_at(4.85, 6.25, 3.4, 1.75)
a.plot(k, (ey - ey.mean()) / ey.std(), color=BLUE, lw=1.2, label='capacity residual'); a.plot(k, (ev - ev.mean()) / ev.std(), color=ORANGE, lw=1.0, label='V$_{600}$ residual')
a.legend(fontsize=7.5, loc='upper right'); a.set_xlabel('cycle', labelpad=1); a.set_ylabel('z')
tag(a, '(d) S3: co-fluctuation beyond the index (B0005)')
bg.text(4.6, 5.62, 'S1  provenance: D (index-derived), P (restated target)\nS2  index-only ceiling: power law, R²$_{index}$\n'
        'S3  time-shift test, Holm: I / T / N\nS4  self-surrogate power: MDE$_{80}$', fontsize=9.5, va='top', linespacing=1.5)
sh = E['class'].value_counts(normalize=True).reindex(['T', 'I', 'N', 'P']).fillna(0)
a = ax_at(4.85, 1.05, 3.4, 3.3)
cols = {'T': GREY, 'I': GREEN, 'N': AMBER, 'P': PINK}
a.bar(sh.index, 100 * sh.values, color=[cols[c] for c in sh.index], width=0.6)
for i, v in enumerate(sh.values): a.text(i, 100 * v + 1.5, f'{100 * v:.1f}%', ha='center', fontsize=9)
a.set_ylabel('% of 5984 verdicts'); a.set_ylim(0, 85); tag(a, '(e) audit card classes, public cells')
a.text(0.98, 0.62, 'method and validation:\nco-submitted MethodsX article', transform=a.transAxes, ha='right', fontsize=8.5, style='italic')

# ---------- zone 3 analyses
from scipy.optimize import curve_fit
kk = g.CycleNumber.values.astype(float); yy = g['Capacity(Ah)'].values
p, _ = curve_fit(lambda k, a_, b_, c_: a_ - b_ * k ** c_, kk[:500], yy[:500], p0=[yy[0], 1e-3, 1.2], maxfev=20000)
a = ax_at(9.45, 6.25, 1.75, 1.75); a.plot(kk, yy, color=INK, lw=2.5); a.plot(kk, p[0] - p[1] * kk ** p[2], color=ORANGE, ls='--', lw=1.5)
a.axvline(500, color=GREY, ls=':'); a.set_xticks([0, 1500, 3000]); tag(a, '(f) Step 3: index ceiling'); a.set_xlabel('cycle', labelpad=1)
a.text(0.05, 0.08, 'RUL error\n0.03 cycles', transform=a.transAxes, fontsize=8.5)
fam = pd.read_csv(O + 'rineng_atlas_family.csv').set_index('family').loc[['CC phase', 'CV phase', 'voltage statistics', 'current statistics'], ['XJTU', 'TJU', 'HUST', 'MIT']]
a = ax_at(11.85, 6.25, 1.5, 1.75); a.imshow(fam.values, cmap='Blues', vmin=0, vmax=0.7, aspect='auto'); a.grid(False)
a.set_xticks(range(4)); a.set_xticklabels(['XJ', 'TJ', 'HU', 'MI'], fontsize=8); a.set_yticks(range(4)); a.set_yticklabels(['CC', 'CV', 'V stats', 'I stats'], fontsize=8)
for i in range(4):
    for j in range(4): a.text(j, i, f'{fam.values[i, j]:.2f}', ha='center', va='center', fontsize=7, color='white' if fam.values[i, j] > 0.4 else INK)
tag(a, '(g) Step 5: atlas', 9.5)
C = E.groupby('cell').agg(n=('n', 'first'), nI=('class', lambda s: (s == 'I').sum()), R2=('R2_index', 'first'), ds=('dataset', 'first'))
a = ax_at(9.05, 2.55, 2.75, 3.1, projection='3d')
for ds, col in DSC.items():
    cc = C[C.ds == ds]; a.scatter(np.log10(cc.n), cc.R2.clip(-1, 1), cc.nI, s=5, color=col, depthshade=True)
a.set_xticks(np.log10([100, 1000])); a.set_xticklabels(['100', '1000'], fontsize=7); a.tick_params(labelsize=7, pad=0)
a.set_xlabel('cycles', fontsize=7.5, labelpad=-6); a.set_ylabel('R²$_{index}$', fontsize=7.5, labelpad=-6); a.set_zlabel('# I', fontsize=7.5, labelpad=-6)
a.set_zticks([0, 10]); a.set_yticks([-1, 0, 1]); a.view_init(elev=20, azim=-58); a.set_title('(h) Step 5: 3-D cell landscape', fontsize=9.5, weight='bold', loc='left')
S = pd.read_csv(O + 'rineng_family_loco_summary.csv'); a = ax_at(12.5, 3.0, 0.95, 2.25)
dss = ['XJTU', 'TJU', 'HUST', 'MIT']
for i, ds in enumerate(dss):
    v0 = S[(S.dataset == ds) & (S.set == 'index only')]['median'].item(); v1 = S[(S.dataset == ds) & (S.set == 'index + voltage statistics')]['median'].item()
    a.bar(i - 0.2, v0, 0.4, color=GREY); a.bar(i + 0.2, v1, 0.4, color=AMBER)
a.set_ylim(0, 4.8); a.set_xticks(range(4)); a.set_xticklabels(['XJ', 'TJ', 'HU', 'MI'], fontsize=8); a.set_ylabel('LOCO RMSE (%)', fontsize=8, labelpad=1)
tag(a, '(i) Step 6: transfer', 9.5); a.text(0.98, 0.92, 'index only', color=GREY, transform=a.transAxes, ha='right', fontsize=7.5, weight='bold')
a.text(0.98, 0.80, '+ voltage stats', color=AMBER, transform=a.transAxes, ha='right', fontsize=7.5, weight='bold')
bg.text(9.3, 2.2, 'Step 4: NASA verdicts → measurement mechanism\nStep 6: leave-one-cell-out GBR / RF, 4 families\n'
        'Step 7: identifiability vs transfer (1496 pairs)\nStep 8: SHAP explanations and their reliability', fontsize=9.3, va='top', linespacing=1.5)

# ---------- zone 4 outcomes
rq4 = json.load(open(O + 'rineng_rq4.json')); R = json.load(open(O + 'rineng_results.json')); X = json.load(open(O + 'rineng_xai.json'))['within']['All']
XAI_TXT = f"stable SHAP (ρ = {X['stability_median']:.2f}), yet top feature\nnot creditable in {100 * (1 - X['top_is_I']):.0f}% of records"
items = [(ORANGE, 'Index ceiling', f"GMP RUL error 0.03 cycles (LSTM 25.6)\nR²$_{{index}}$ ≥ 0.9 in {100 * R['public_R2_index_all']['share_ge_0_9']:.1f}% of cells"),
         (PINK, 'Restatements', 't$_{dis}$, t$_{Tmax}$ = class P in 4/4 NASA cells\nV$_{600}$ = class I in 4/4 cells'),
         (GREEN, 'Identifiable features', f"{100 * R['atlas']['I_share_all']:.1f}% of verdicts; CV phase highest\nnone in 79 records < 300 cycles"),
         (BLUE, 'Identifiability ≠ transfer', f"ρ = {rq4['pooled'][0]:.2f} (1496 pairs)\nvoltage stats: up to 6.2-fold gain"),
         (PURPLE, 'Explanations ≠ evidence', XAI_TXT)]
for i, (col, t, b) in enumerate(items):
    y0 = 7.5 - i * 1.13
    bg.add_patch(FancyBboxPatch((14.25, y0), 3.3, 0.98, boxstyle='round,pad=0.02,rounding_size=0.1', fc='white', ec=col, lw=1.8))
    bg.text(14.4, y0 + 0.76, t, fontsize=10.5, weight='bold', color=col, va='center')
    bg.text(14.4, y0 + 0.52, b, fontsize=8.5, va='top', linespacing=1.35)
bg.add_patch(FancyBboxPatch((14.25, 0.95), 3.3, 1.85, boxstyle='round,pad=0.02,rounding_size=0.1', fc='#f3f0f8', ec=PURPLE, lw=1.8))
bg.text(14.4, 2.6, 'Screening checklist (Section 5)', fontsize=10.5, weight='bold', color=PURPLE, va='center')
bg.text(14.4, 2.38, '1  remove restated columns\n2  report R²$_{index}$ beside every score\n3  require a measured target\n'
        '4  ≥ 300 cycles; report MDE$_{80}$\n5  test transfer on other cells', fontsize=9, va='top', linespacing=1.4)
fig.savefig(FG + 'rineng_fig1_study_design.png'); print('saved figures/rineng_fig1_study_design.png')
