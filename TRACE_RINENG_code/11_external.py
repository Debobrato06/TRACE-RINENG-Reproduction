"""Step 11 - V8: external validation on 387 measured cells from four public datasets (XJTU, TJU, HUST, MIT;
processed release github.com/wang-fujin/PINN4SOH, commit cc3cc50), each audited as its own single trajectory.
(a) Audit every cell (16 charge features, target = measured discharge capacity); store statistics and classes.
(b) Consequence for generalizability: within each dataset, leave-one-cell-out GBR with
    index only / index + features that TRACE classes I in the majority of training cells / index + T-majority
    features / index + all non-leaking features. Shows that T features, which TRACE cannot credit on one
    trajectory, may still transfer across cells (the identifiability boundary), with cell-level bootstrap CIs."""
import os, sys, glob, subprocess, warnings, numpy as np, pandas as pd
from joblib import Parallel, delayed
warnings.filterwarnings('ignore'); sys.path.insert(0, '.'); import trace_audit as T
QUICK = os.environ.get('TRACE_QUICK') == '1'
REPO, COMMIT = 'https://github.com/wang-fujin/PINN4SOH.git', 'cc3cc5053caf38f04e0665f7f88cb109144d035e'
F = ['voltage mean', 'voltage std', 'voltage kurtosis', 'voltage skewness', 'CC Q', 'CC charge time', 'voltage slope',
     'voltage entropy', 'current mean', 'current std', 'current kurtosis', 'current skewness', 'CV Q', 'CV charge time',
     'current slope', 'current entropy']


def load():
    tgt = 'data/PINN4SOH'
    if not os.path.isdir(tgt):
        subprocess.run(['git', 'clone', '-q', REPO, tgt], check=True); subprocess.run(['git', '-C', tgt, 'checkout', '-q', COMMIT], check=True)
    B = tgt + '/data/'; fs = glob.glob(B + 'XJTU data/*.csv') + glob.glob(B + 'HUST data/*.csv') + glob.glob(B + 'MIT data/*/*.csv') + glob.glob(B + 'TJU data/*/*.csv')
    out = []
    for f in sorted(fs):
        d = pd.read_csv(f).replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
        if len(d) < 100: continue
        ds = 'XJTU' if 'XJTU' in f else 'HUST' if 'HUST' in f else 'MIT' if 'MIT' in f else 'TJU'
        d = d.assign(cell=f.split('data/', 2)[-1], dataset=ds, k=np.arange(1, len(d) + 1))
        out.append(d)
    return pd.concat(out, ignore_index=True)


def audit(cell, g):
    card = T.run_audit(g[['k', 'capacity'] + F].reset_index(drop=True), 'k', 'capacity', F, B=199, seed=0)
    card['cell'] = cell; card['dataset'] = g.dataset.iloc[0]; card['n'] = len(g)
    card['eta_target'] = card.attrs['eta_target']; card['r'] = card.attrs['r']; card['p_floor_holm'] = card.attrs['min_holm_p']
    card['R2_index'] = card.attrs['ceiling']['R2_index']
    return card


if __name__ == '__main__':
    D = load(); cells = list(D.groupby('cell'))
    if QUICK: cells = cells[::8]
    print('cells', len(cells), 'datasets', D.dataset.value_counts().to_dict(), flush=True)
    A = pd.concat(Parallel(n_jobs=2)(delayed(audit)(c, g) for c, g in cells), ignore_index=True)
    A.to_csv('outputs/out_ext_audit.csv', index=False)
    C = A.drop_duplicates('cell')
    print('cells audited', len(C), 'median n', C.n.median(), 'eta_target range', C.eta_target.min(), C.eta_target.max(),
          'cells with deterministic target', int((C.eta_target < 1e-3).sum()))
    tab = A.pivot_table(index='feature', columns='class', values='cell', aggfunc='count', fill_value=0)
    tab = (tab.T / tab.sum(1)).T.round(3); tab.to_csv('outputs/out_ext_class_share.csv'); print(tab)
    by_ds = A.groupby(['dataset', 'class']).size().unstack(fill_value=0); by_ds = (by_ds.T / by_ds.sum(1)).T.round(3)
    by_ds.to_csv('outputs/out_ext_class_by_dataset.csv'); print(by_ds)
    # (b) leave-one-cell-out within each dataset
    from sklearn.ensemble import HistGradientBoostingRegressor
    cls = A.set_index(['cell', 'feature'])['class']
    rows = []
    for ds, dd in D[D.cell.isin([c for c, _ in cells])].groupby('dataset'):
        cl = dd.cell.unique()
        for te in cl:
            tr_cells = [c for c in cl if c != te]
            maj = cls.loc[tr_cells].groupby(level='feature').agg(lambda s: s.value_counts().idxmax())
            sets = {'index only': [], 'index + I': [f for f in F if maj.get(f) == 'I'],
                    'index + T': [f for f in F if maj.get(f) == 'T'], 'index + all non-leaking': [f for f in F if maj.get(f) in ('I', 'T', 'N')]}
            tr = dd[dd.cell.isin(tr_cells)].iloc[::3]; ts = dd[dd.cell == te]
            for nm, fs in sets.items():
                X = ['k'] + fs
                p = HistGradientBoostingRegressor(max_iter=150, random_state=0).fit(tr[X], tr.capacity).predict(ts[X])
                y = ts.capacity.values
                rows.append(dict(dataset=ds, cell=te, set=nm, n_feat=len(fs), rmse=float(np.sqrt(np.mean((p - y) ** 2))),
                                 r2=float(1 - np.sum((p - y) ** 2) / np.sum((y - y.mean()) ** 2)), nominal=float(y[:5].mean())))
        print(ds, 'LOCO done', flush=True)
    L = pd.DataFrame(rows); L['rmse_pct'] = 100 * L.rmse / L.nominal; L.to_csv('outputs/out_ext_loco.csv', index=False)
    rng = np.random.default_rng(0); summ = []
    for (ds, s), g in L.groupby(['dataset', 'set']):
        v = g.rmse_pct.values; bs = [np.median(rng.choice(v, len(v))) for _ in range(2000)]
        summ.append(dict(dataset=ds, set=s, cells=len(v), median_rmse_pct=float(np.median(v)), lo=float(np.percentile(bs, 2.5)), hi=float(np.percentile(bs, 97.5)),
                         mean_n_feat=float(g.n_feat.mean())))
    S = pd.DataFrame(summ); S.to_csv('outputs/out_ext_loco_summary.csv', index=False); print(S.round(3).to_string(index=False))
