"""Fig. 1 methodology figure: twelve numbered step panels with icon-style graphics (no axes) and no connector lines.
Drawn schematic; reads pipeline outputs only for the icon shapes."""
import numpy as np, pandas as pd, json, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle, Ellipse, Polygon
BLUE, ORANGE, GREEN, AMBER, PINK = '#1f5fbf', '#d9541e', '#138a5e', '#c98a00', '#c2477d'
TINT = {'#1f5fbf': '#a9c6f0', '#d9541e': '#f5ae8c', '#138a5e': '#9fd6b8', '#c98a00': '#f3d58c', '#c2477d': '#eeb3cc'}
INK = '#111111'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.weight': 'bold', 'savefig.dpi': 300})
g = pd.read_csv('data/GMP.csv'); f = pd.read_csv('data/nasa_cycle_features.csv')
L = json.load(open('outputs/out_shift_landscape_B0005.json'))
pt = pd.read_csv('outputs/out_power_trend_gmp.csv'); nul = pd.read_csv('outputs/out_null_pvalues.csv')

W, H = 160, 100
fig = plt.figure(figsize=(16, 10.0)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis('off')
PW, PH = 37.5, 30; GX, GY = 2.0, 2.4; X0, Y0 = 1.5, 1.2
def pos(i):  # i = 0..11, row-major from top
    r, c = divmod(i, 4); return X0 + c * (PW + GX), Y0 + (2 - r) * (PH + GY)
def icon_ax(x, y, w, h):
    a = fig.add_axes([x / W, y / H, w / W, h / H]); a.axis('off'); return a
def panel(i, title, tags, fill):
    x, y = pos(i)
    ax.add_patch(FancyBboxPatch((x, y), PW, PH, boxstyle='round,pad=0.2,rounding_size=1.6', fc='white', ec=INK, lw=1.8))
    ax.add_patch(Circle((x + 3.6, y + PH - 3.6), 2.7, fc=TINT.get(fill, fill), ec=fill, lw=2.5))
    ax.text(x + 3.6, y + PH - 3.6, str(i + 1), ha='center', va='center', color=INK, fontsize=15, weight='bold')
    ax.text(x + 7.4, y + PH - 3.6, title, ha='left', va='center', fontsize=13.2, weight='bold', color=INK)
    for j, t in enumerate(tags):  # tag chips at the bottom, sized to text
        ax.text(x + PW / 2, y + 2.4 + (len(tags) - 1 - j) * 4.1, t, ha='center', va='center', fontsize=10.6, weight='bold', color=INK,
                bbox=dict(boxstyle='round,pad=0.25', fc='#eef0e6', ec='none'))
    return x, y
STAGE = {0: BLUE, 1: BLUE, 2: BLUE, 3: GREEN, 4: GREEN, 5: GREEN, 6: GREEN, 7: ORANGE, 8: ORANGE, 9: ORANGE, 10: ORANGE, 11: ORANGE}

# 1 GMP pack
x, y = panel(0, 'Industrial LFP pack', ['GMP pack, 25.6 V, 40 Ah', '3000 cycles × 17 columns'], STAGE[0])
ax.add_patch(FancyBboxPatch((x + 7, y + 10.8), 21, 11, boxstyle='round,pad=0.2,rounding_size=1', fc='white', ec=INK, lw=2.2))
ax.add_patch(Rectangle((x + 28.4, y + 14.0), 2, 4.6, fc=INK))
for k, c in enumerate([GREEN, GREEN, GREEN, AMBER]): ax.add_patch(Rectangle((x + 8.3 + k * 4.9, y + 12.0), 4.1, 8.6, fc=c))
# 2 generator check
x, y = panel(1, 'Provenance check', ['published Eqs. 1–8, seed 42', 'CSV reproduced to 5×10⁻⁹'], STAGE[1])
a = icon_ax(x + 6, y + 10.0, PW - 12, 13); n = g.CycleNumber.values
a.plot(n, g['Capacity(Ah)'], color=BLUE, lw=6); a.plot(n, 40 - 12 * (n / 3000) ** 1.35, color=ORANGE, lw=2.5, ls='--')
# 3 NASA cells
x, y = panel(2, 'Measured control', ['NASA B0005/6/7/18', '+ 374 public cells (4 labs)'], STAGE[2])
for k, c in enumerate([BLUE, ORANGE, GREEN, AMBER]):
    cx = x + 8 + k * 6.4
    ax.add_patch(FancyBboxPatch((cx, y + 10.0), 4.4, 12.6, boxstyle='round,pad=0.1,rounding_size=0.6', fc=TINT[c], ec=c, lw=2.5))
    ax.add_patch(Rectangle((cx + 1.4, y + 22.7), 1.6, 1.0, fc=c))
    ax.text(cx + 2.2, y + 16.3, ['5', '6', '7', '18'][k], ha='center', va='center', color=INK, fontsize=12, weight='bold', rotation=90)
# 4 cycle table
x, y = panel(3, 'Cycle-level table', ['index k, target y, features x', 'one row per cycle'], STAGE[3])
for r_ in range(4):
    for c_ in range(5):
        ax.add_patch(Rectangle((x + 7 + c_ * 4.8, y + 9.8 + r_ * 3.4), 4.6, 3.2, fc=['#d7e4f7', '#fde0cf', '#e0f0e6', '#e0f0e6', '#e0f0e6'][c_] if r_ < 3 else ['#a9c6f0', '#f5ae8c', '#9fd6b8', '#9fd6b8', '#9fd6b8'][c_], ec='white', lw=1.2))
for c_, t in enumerate(['k', 'y', 'x₁', 'x₂', 'xₚ']): ax.text(x + 9.3 + c_ * 4.8, y + 21.4, t, ha='center', va='center', color=INK, fontsize=12, weight='bold')
# 5 S1
x, y = panel(4, 'S1 Provenance', ['determinism index η', 'proxy check: r_res, ζ'], STAGE[4])
a = icon_ax(x + 7, y + 9.8, PW - 14, 13)
a.bar(range(6), np.log10([7.5e-5, 1, 1, 0.067, 0.095, 0.78]) + 5, color=[BLUE] * 3 + [ORANGE] * 3, width=0.78)
# 6 S2
x, y = panel(5, 'S2 Index ceiling', ['y predicted from k alone', 'chronological 70/30 split'], STAGE[5])
a = icon_ax(x + 6, y + 10.0, PW - 12, 13)
a.plot(n[:2100], g['Capacity(Ah)'][:2100], color=BLUE, lw=6); a.plot(n[2100:], g['Capacity(Ah)'][2100:], color=ORANGE, lw=6)
# 7 S3
x, y = panel(6, 'S3 Association', ['gated trend test', 'shift test: level, rate'], STAGE[6])
a = icon_ax(x + 5, y + 9.8, PW - 10, 13); d = np.arange(-67, 68)
a.fill_between(d, 0, L['T_max'], color=AMBER, alpha=0.8, lw=0); a.plot(d, L['V_600s'], color=BLUE, lw=2.4)
# 8 S4
x, y = panel(7, 'S4 Sensitivity', ['self-surrogate injection', 'power curve → MDE₈₀'], STAGE[7])
a = icon_ax(x + 6, y + 10.0, PW - 12, 13)
for k, kd in enumerate(['generator', 'self-surrogate']):
    q = pt[pt.kind == kd]; a.plot(q.rho, q.power_family12, color=[BLUE, ORANGE][k], lw=[6, 3][k], ls=['-', '--'][k])
# 9 audit card
x, y = panel(8, 'Audit card', ['class per feature', 'permitted claims'], STAGE[8])
CLS = [('D', '#bdbdb5', INK), ('P', '#f5ae8c', INK), ('I', '#a9c6f0', '#0b2a5b'), ('T', '#f3d58c', INK), ('N', '#ededea', INK), ('N*', '#ededea', INK)]
for k, (t, fc, tc) in enumerate(CLS):
    cx = x + 7.5 + (k % 3) * 8; cy = y + 16.2 - (k // 3) * 6.6
    ax.add_patch(Rectangle((cx, cy), 7, 6.0, fc=fc, ec='white', lw=1.5)); ax.text(cx + 3.5, cy + 3.0, t, ha='center', va='center', color=tc, fontsize=15, weight='bold')
# 10 reliability
x, y = panel(9, 'Reliability', ['null P–P calibration', 'FPR 3–6 % vs naive ≤ 92 %'], STAGE[9])
a = icon_ax(x + 8, y + 9.8, PW - 16, 13); u = np.linspace(0, 1, 101)
a.plot(u, [np.mean(nul.p_naive <= z) for z in u], color=ORANGE, lw=5); a.plot(u, [np.mean(nul.p_tts <= z) for z in u], color=BLUE, lw=5)
# 11 explainability + generalizability
x, y = panel(10, 'XAI & generalization', ['SHAP vs null band', 'LOCO on 374 cells'], STAGE[10])
a = icon_ax(x + 6, y + 9.8, PW - 12, 13)
a.barh([3, 2, 1, 0], [79.5, 72.0, 8.3, 7.6], color=[ORANGE, ORANGE, '#8c8c86', BLUE], height=0.75)
# 12 benchmark
x, y = panel(11, 'Benchmark', ['10 planted types, 6 tests', 'false credit 1–3 %'], STAGE[11])
a = icon_ax(x + 8, y + 9.8, PW - 16, 13)
a.bar(range(4), [0.69, 0.15, 0.135, 0.017], color=['#8c8c86', AMBER, PINK, BLUE], width=0.75); a.set_ylim(0, 0.75)
import matplotlib.text as _mt
_need = 7.0 * fig.get_figwidth() / 6.7
for _t in fig.findobj(_mt.Text):
    if _t.get_text() and _t.get_fontsize() < _need: _t.set_fontsize(_need)
fig.savefig('figures/fig1_methodology.png', dpi=300); plt.close(fig)
print('methodology figure done')
