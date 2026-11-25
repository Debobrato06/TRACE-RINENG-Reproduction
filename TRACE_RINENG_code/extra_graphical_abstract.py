"""Graphical abstract in the layout and colours of the original four-zone design, redrawn at its display size
(13 cm wide) so that every text element is >= 7 pt when shown at 13 cm (MethodsX Guide for Authors).
Zones: 1 Data (measured capacity with its cross-fitted trend), 2 TRACE audit (stages S1-S4),
3 Audit card (class grid for the NASA cells), 4 Verified outcomes (benchmark: false credit of each rule).
All plotted values are read from data/ and outputs/. Saved as PNG and TIFF at 600 dpi."""
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import trace_audit as T

INK, NAVY = '#111111', '#14306b'
GREEN, BLUE, PINK, ORANGE, AMBER, GREY = '#138a5e', '#1f5fbf', '#c2477d', '#d9541e', '#c98a00', '#8c8c86'
CLS = {'D': '#bdbdb5', 'P': '#f5ae8c', 'I': '#a9c6f0', 'T': '#f3d58c', 'N': '#d9d9d4'}
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 7, 'axes.linewidth': 0.6,
                     'xtick.major.width': 0.5, 'ytick.major.width': 0.5, 'xtick.major.size': 2, 'ytick.major.size': 2})
WMM, HMM = 130, 68                                        # canvas in mm = display size (13 cm wide)
fig = plt.figure(figsize=(WMM / 25.4, HMM / 25.4)); ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, WMM); ax.set_ylim(0, HMM); ax.axis('off')
inset = lambda x, y, w, h: fig.add_axes([x / WMM, y / HMM, w / WMM, h / HMM])

def zone(x, w, num, title, col, y=7, h=49):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0,rounding_size=1.8', fc='white', ec=col, lw=1.3))
    ax.text(x + w / 2, y + h - 3.2, f'{num}  {title}', ha='center', va='center', fontsize=8, weight='bold', color=INK)

def chip(xc, y, t, fs=7):
    ax.text(xc, y, t, ha='center', va='center', fontsize=fs, color=INK,
            bbox=dict(boxstyle='round,pad=0.25', fc='#f1efe6', ec='none'))

def arrow(x0, x1, y=31.5):
    ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle='-|>', mutation_scale=8, lw=1.2, color=INK))

# title and banner (old layout)
ax.text(WMM / 2, 65, 'TRACE: what can one battery ageing record support before modelling?', ha='center', va='center',
        fontsize=8.5, weight='bold', color=INK)
ax.text(WMM / 2, 59.6, 'Pre-modelling audit: restatements flagged, false credit 1–3 %, valid under autocorrelation',
        ha='center', va='center', fontsize=7, color=INK, bbox=dict(boxstyle='round,pad=0.35', fc='#fbe7b5', ec=AMBER, lw=0.9))

# 1 Data -------------------------------------------------------------------------------------------
X1, W1 = 1, 27; zone(X1, W1, '1', 'Data', GREEN)
d = pd.read_csv('data/nasa_cycle_features.csv'); d = d[d.battery == 5]
k = d.cycle.values.astype(float); y = d.capacity.values; ct, _ = T.crossfit_trend(k, y)
a = inset(X1 + 8, 22.5, W1 - 10, 25)
a.plot(k, y, color=ORANGE, lw=0.9, label='measured'); a.plot(k, ct, color=INK, lw=0.8, ls='--', label='trend')
a.set_xticks([0, 80, 160]); a.set_yticks([1.4, 1.6, 1.8]); a.tick_params(labelsize=7, pad=1)
a.set_xlabel('cycle k', fontsize=7, labelpad=0.5); a.set_title('capacity (Ah)', fontsize=7, pad=2)
for s in ['top', 'right']: a.spines[s].set_visible(False)
chip(X1 + W1 / 2, 13.6, 'GMP + 4 NASA'); chip(X1 + W1 / 2, 9.8, '+ 374 public cells')
arrow(X1 + W1 + 0.3, X1 + W1 + 2.6)

# 2 TRACE audit ------------------------------------------------------------------------------------
X2, W2 = 31, 30; zone(X2, W2, '2', 'TRACE audit', BLUE)
for i, (s, t) in enumerate([('S1', 'provenance'), ('S2', 'index ceiling'), ('S3', 'time-shift test'), ('S4', 'power: MDE₈₀')]):
    yc = 45 - i * 7.6
    ax.add_patch(FancyBboxPatch((X2 + 2, yc - 3), W2 - 4, 6, boxstyle='round,pad=0,rounding_size=1', fc='#eaf1fb', ec=BLUE, lw=0.8))
    ax.add_patch(FancyBboxPatch((X2 + 2, yc - 3), 7, 6, boxstyle='round,pad=0,rounding_size=1', fc=BLUE, ec=BLUE, lw=0.8))
    ax.text(X2 + 5.5, yc, s, ha='center', va='center', fontsize=7.5, weight='bold', color='white')
    ax.text(X2 + 10.5, yc, t, ha='left', va='center', fontsize=7, color=INK)
chip(X2 + W2 / 2, 11.5, 'false positives ≤ 6 %')
arrow(X2 + W2 + 0.3, X2 + W2 + 2.6)

# 3 Audit card -------------------------------------------------------------------------------------
X3, W3 = 64.5, 32; zone(X3, W3, '3', 'Audit card', PINK)
A = pd.read_csv('outputs/out_audit_nasa.csv')
feats = [('ambient_T', 'amb. T'), ('t_discharge', 't_dis'), ('V_600s', 'V₆₀₀'), ('I_mean', 'I mean')]
bats = ['B0005', 'B0006', 'B0007', 'B0018']; cw, chh, gx, gy = 4.4, 4.7, X3 + 11, 45
for j, b in enumerate(bats):
    ax.text(gx + j * cw + cw / 2, gy + 1.0, b[1:].lstrip('0'), ha='center', va='bottom', fontsize=7, color=INK)
for i, (f, lab) in enumerate(feats):
    yy = gy - (i + 1) * chh
    ax.text(gx - 0.8, yy + chh / 2, lab, ha='right', va='center', fontsize=7, color=INK)
    for j, b in enumerate(bats):
        c = A[(A.feature == f) & (A.battery == b)]['class'].iloc[0]
        ax.add_patch(Rectangle((gx + j * cw, yy), cw - 0.4, chh - 0.4, fc=CLS[c], ec='white'))
        ax.text(gx + j * cw + cw / 2 - 0.2, yy + chh / 2 - 0.2, c, ha='center', va='center', fontsize=7.5, weight='bold', color=INK)
ax.text(X3 + W3 / 2, 21, 'D derived · P proxy', ha='center', va='center', fontsize=7, color=INK)
ax.text(X3 + W3 / 2, 17.4, 'I beyond k · T trend', ha='center', va='center', fontsize=7, color=INK)
chip(X3 + W3 / 2, 11.5, 'one class per feature')
arrow(X3 + W3 + 0.3, X3 + W3 + 2.6)

# 4 Verified outcomes: one benchmark chart ---------------------------------------------------------
X4, W4 = 99, 30; zone(X4, W4, '4', 'Benchmark', ORANGE)
B = pd.read_csv('outputs/out_bench2_table.csv', index_col=0)
rules = [('M1_corr', 'Corr.'), ('M6_dcca', 'DCCA'), ('M7_hac', 'HAC'), ('M5_prewhiten', 'Pre-wh.'), ('M4_TRACE', 'TRACE')]
fc = [B.loc[['trend_hi', 'null_ar', 'null_trend'], m].mean() * 100 for m, _ in rules]
a = inset(X4 + 10, 23.5, W4 - 13, 22)
cols = [GREY, PINK, AMBER, GREY, BLUE]
a.barh(range(len(rules)), fc, color=cols, height=0.7); a.invert_yaxis()
a.set_yticks(range(len(rules))); a.set_yticklabels([r[1] for r in rules], fontsize=7)
for i, v in enumerate(fc): a.text(v + 2, i, f'{v:.0f}', va='center', fontsize=7, color=INK)
a.set_xlim(0, 90); a.set_xticks([0, 40, 80]); a.tick_params(labelsize=7, pad=1)
a.set_xlabel('false credit (%)', fontsize=7, labelpad=0.5)
for s in ['top', 'right']: a.spines[s].set_visible(False)
chip(X4 + W4 / 2, 13.6, 'proxies flagged:'); chip(X4 + W4 / 2, 9.8, 'only by TRACE')

ax.text(WMM / 2, 3.2, 'what one record can support  ·  bounded negatives  ·  reproducible Python / Colab code',
        ha='center', va='center', fontsize=7, weight='bold', color=NAVY)
fig.savefig('figures/graphical_abstract.png', dpi=600)
fig.savefig('figures/graphical_abstract.tif', dpi=600, pil_kwargs={'compression': 'tiff_lzw'})
print('graphical abstract done')
