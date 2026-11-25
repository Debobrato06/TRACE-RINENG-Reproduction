"""Step 21 - additional figures of the Results in Engineering article.
Fig. 3  NASA engineering reading: (a) capacity vs charge counted from discharge time (restatement, class P);
        (b) cross-fitted residuals of V600 vs residuals of capacity (co-fluctuation beyond the index, class I);
        (c) cross-fitted residuals of mean current vs capacity (instrumentation check).
Fig. 5  3-D landscape of the 374 public cells: record length x index-only ceiling x number of class-I features.
Needs data/nasa_cycle_features.csv and outputs of steps 01, 11."""
import numpy as np, pandas as pd, matplotlib, sys
sys.path.insert(0, '.'); matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from scipy import stats
import trace_audit as T
FG, O = 'figures/', 'outputs/'
DSC = {'XJTU': '#1f5fbf', 'TJU': '#d9541e', 'HUST': '#138a5e', 'MIT': '#c2477d'}
BC = {5: '#1f5fbf', 6: '#d9541e', 7: '#138a5e', 18: '#c2477d'}
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 12, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.grid': True, 'grid.color': '#dddddd', 'savefig.dpi': 300, 'savefig.bbox': 'tight', 'legend.frameon': False})
lab = lambda a, t: a.set_title(t, loc='left', fontsize=15, weight='bold')

# ---------------- Fig. 3 NASA engineering reading
N = pd.read_csv('data/nasa_cycle_features.csv')
fig, ax = plt.subplots(1, 3, figsize=(15, 4.6), gridspec_kw=dict(wspace=0.32))
a = ax[0]
for b, g in N.groupby('battery'):
    q = np.abs(g.I_mean) * g.t_discharge / 3600
    a.scatter(q, g.capacity, s=12, color=BC[b], label=f'B{b:04d}', alpha=0.8)
lim = [1.25, 2.1]; a.plot(lim, lim, color='#111111', lw=1, ls='--'); a.set_xlim(lim); a.set_ylim(lim)
a.set_xlabel('|I| × discharge time (Ah)'); a.set_ylabel('measured capacity (Ah)'); a.legend(loc='upper left', fontsize=10)
r = stats.pearsonr(np.abs(N.I_mean) * N.t_discharge / 3600, N.capacity)[0]
a.text(0.97, 0.05, f'r = {r:.4f}\nclass P in 4/4 cells', transform=a.transAxes, ha='right', fontsize=11); lab(a, '(a)')
rows = []
for b, g in N.groupby('battery'):
    k = g.cycle.values.astype(float)
    ey = T.crossfit_trend(k, g.capacity.values)[1]
    for f in ['V_600s', 'I_mean']:
        ex = T.crossfit_trend(k, g[f].values)[1]
        rows.append(dict(battery=b, feature=f, ey=ey, ex=ex))
for i, (f, t, nm) in enumerate([('V_600s', '(b)', 'V$_{600}$ residual (z)'), ('I_mean', '(c)', 'mean-current residual (z)')]):
    a = ax[i + 1]
    for rr in rows:
        if rr['feature'] != f: continue
        zx = (rr['ex'] - rr['ex'].mean()) / rr['ex'].std(); zy = (rr['ey'] - rr['ey'].mean()) / rr['ey'].std()
        a.scatter(zx, zy, s=12, color=BC[rr['battery']], alpha=0.75)
        rows_rho = stats.spearmanr(zx, zy)[0]; rr['rho'] = rows_rho
    rhos = ', '.join(f"{rr['rho']:.2f}" for rr in rows if rr['feature'] == f)
    a.set_xlabel(nm); a.set_ylabel('capacity residual (z)'); a.set_xlim(-4.5, 4.5); a.set_ylim(-4.5, 4.5)
    a.text(0.03, 0.97, f'Spearman ρ by cell:\n{rhos}', transform=a.transAxes, va='top', fontsize=10.5); lab(a, t)
fig.savefig(FG + 'rineng_fig3_nasa_reading.png'); plt.close(fig)
pd.DataFrame([dict(battery=r['battery'], feature=r['feature'], rho=r['rho']) for r in rows]).to_csv(O + 'rineng_nasa_residual_rho.csv', index=False)
print('saved Fig. 3; residual rho:', [(r['battery'], r['feature'], round(r['rho'], 2)) for r in rows])

# ---------------- Fig. 5 3-D landscape
E = pd.read_csv(O + 'out_ext_audit.csv'); E['I'] = (E['class'] == 'I')
C = E.groupby('cell').agg(n=('n', 'first'), nI=('I', 'sum'), R2=('R2_index', 'first'), ds=('dataset', 'first')).reset_index()
C['R2c'] = C.R2.clip(-1, 1)
fig = plt.figure(figsize=(10.5, 8)); a = fig.add_subplot(111, projection='3d')
for ds in ['XJTU', 'TJU', 'HUST', 'MIT']:
    g = C[C.ds == ds]
    a.scatter(np.log10(g.n), g.R2c, g.nI, s=34, color=DSC[ds], label=f'{ds} ({len(g)} cells)', depthshade=True, edgecolor='white', linewidth=0.3)
# projected shadows on the floor
a.scatter(np.log10(C.n), C.R2c, np.zeros(len(C)) - 0.3, s=6, color='#bbbbbb', alpha=0.5, depthshade=False)
zt = C.nI.max() + 1
xx, yy = np.meshgrid([np.log10(100.0), np.log10(300.0)], [-1, 1]); a.plot_surface(xx, yy, np.full_like(xx, -0.3), color='#d9541e', alpha=0.18, linewidth=0)
a.text2D(0.03, 0.70, 'shaded floor: records < 300 cycles,\nno class-I verdict in any of 79 cells', transform=a.transAxes, fontsize=11, color='#8a2e0b')
a.set_xticks(np.log10([100, 300, 1000, 3000])); a.set_xticklabels(['100', '300', '1000', '3000'])
a.set_xlabel('record length (cycles)', labelpad=10); a.set_ylabel('index-only R² (clipped to ±1)', labelpad=10); a.set_zlabel('class-I features per cell', labelpad=8)
a.set_zlim(-0.3, zt); a.view_init(elev=22, azim=-58); a.legend(loc='upper left', fontsize=11, bbox_to_anchor=(0.02, 0.92))
for p in (a.xaxis.pane, a.yaxis.pane, a.zaxis.pane): p.set_facecolor((0.97, 0.97, 0.97, 1)); p.set_edgecolor('#cccccc')
fig.savefig(FG + 'rineng_fig5_landscape3d.png'); plt.close(fig)
r1 = stats.spearmanr(C.n, C.nI); r2 = stats.spearmanr(C.R2, C.nI); r3 = stats.spearmanr(C.n, C.R2)
print('3D: rho(n,nI)=%.2f rho(R2,nI)=%.2f p=%.3g rho(n,R2)=%.2f p=%.3g' % (r1[0], r2[0], r2[1], r3[0], r3[1]))
pd.Series(dict(rho_n_nI=r1[0], rho_R2_nI=r2[0], p_R2_nI=r2[1], rho_n_R2=r3[0], p_n_R2=r3[1])).to_csv(O + 'rineng_landscape_rho.csv')
