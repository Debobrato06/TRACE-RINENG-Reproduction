"""Step 12 - V7 (revised): benchmark of TRACE against established association tests for non-stationary,
autocorrelated series, with ground truth that is not defined through TRACE's own residual.
Targets: measured capacity of the four NASA cells and of 8 public cells (2 per dataset: XJTU, TJU, HUST, MIT).
Planted feature types (truth):
  info_res   (I)  smooth random trend + coupling to the spline residual of capacity (original V7 design; reported for continuity)
  info_lag   (I)  smooth random trend + coupling to the one-cycle-lagged Savitzky-Golay high-pass of capacity (different detrender, lagged)
  info_rate  (I)  smooth random trend + coupling to the first difference of capacity (rate-type feature)
  trend_hi   (T)  cubic polynomial fit of capacity on cycle + AR(0.95) noise: a trend-carrying health indicator
  null_ar    (N)  AR(0.9) noise
  null_trend (N)  random smooth trend + AR(0.9) noise (confounded null)
  proxy      (P)  capacity + noise of 0.2% of its SD
  label      (D)  remaining-cycles countdown n - k
Decision rules (credit = feature declared informative):
  M1 Pearson screen |r| >= 0.3 and iid p < 0.05;  M2 SHAP share >= 5% (random split);
  M3 detrended Spearman, iid p, Holm;  M5 pre-whitened cross-correlation (first differences, AR(p) by AIC, Holm) [Box-Jenkins];
  M6 DCCA coefficient (box n/10) with 299 phase-randomised (Fourier) surrogates of the feature, Holm [Podobnik-Stanley; Theiler];
  M7 detrended regression with Newey-West (HAC) standard errors, Holm;  M4 TRACE (credited = class I; leakage = class P or D).
Part (b): strength sweep of info_smooth (coupling c) against null_trend for TRACE, M5, M6, M7 on the NASA cells.
Part (c): proxy-noise sweep: TRACE class of capacity + noise at 0.2-50% of the capacity SD."""
import os, sys, warnings, json, numpy as np, pandas as pd
from scipy import stats
from scipy.signal import savgol_filter, lfilter
import statsmodels.api as sm
from statsmodels.tsa.ar_model import ar_select_order
from joblib import Parallel, delayed
warnings.filterwarnings('ignore'); sys.path.insert(0, '.'); import trace_audit as T
QUICK = os.environ.get('TRACE_QUICK') == '1'
R_NASA, R_EXT, R_SWEEP = (8, 4, 6) if QUICK else (50, 20, 25)
TYPES = ['info_res', 'info_rate', 'info_smooth', 'info_nonlin', 'info_lag', 'trend_hi', 'null_ar', 'null_trend', 'proxy', 'label']
TRUTH = dict(info_res='I', info_rate='I', info_smooth='I', info_nonlin='I', info_lag='I', trend_hi='T', null_ar='N', null_trend='N', proxy='P', label='D')
DESIGN = ['info_res', 'info_rate']          # types seen while revising the statistic (v2 rate representation)
HELDOUT = ['info_smooth', 'info_nonlin', 'info_lag']   # never used to design or tune TRACE
# two audit families of five features (m = 5, as in the NASA audits), each with informative, negative and leakage members
FAM = [['info_res', 'info_rate', 'trend_hi', 'null_ar', 'proxy'], ['info_smooth', 'info_nonlin', 'info_lag', 'null_trend', 'label']]


def ar(n, phi, rg):
    e = rg.normal(size=n + 300); x = np.zeros_like(e)
    for i in range(1, len(e)): x[i] = phi * x[i - 1] + e[i]
    x = x[300:]; return (x - x.mean()) / x.std()


z = lambda a: (a - a.mean()) / (a.std() + 1e-12)


def plant(k, y, rg, c_lag=0.8):
    n = len(k); _, ry = T.crossfit_trend(k, y)
    smooth = lambda: np.sin(np.pi * k / n * rg.uniform(0.5, 2) + rg.uniform(0, 6))
    w = min(21, (n // 8) * 2 + 1); hp = y - savgol_filter(y, w, 2); lag = np.r_[hp[0], hp[:-1]]
    dy = np.r_[0.0, np.diff(y)]
    return pd.DataFrame(dict(
        info_res=2 * smooth() + 0.8 * z(ry) + 0.6 * rg.normal(size=n),
        info_lag=2 * smooth() + 0.8 * z(lag) + 0.6 * rg.normal(size=n),
        info_rate=2 * smooth() + 0.8 * z(dy) + 0.6 * rg.normal(size=n),
        info_smooth=2 * smooth() + c_lag * z(np.convolve(hp, np.ones(5) / 5, mode='same')) + 0.6 * rg.normal(size=n),
        info_nonlin=2 * smooth() + 0.8 * z(np.tanh(2 * z(hp))) + 0.6 * rg.normal(size=n),
        trend_hi=z(np.polyval(np.polyfit(k, y, 3), k)) + 0.6 * ar(n, 0.95, rg),
        null_ar=ar(n, 0.9, rg), null_trend=2 * smooth() + ar(n, 0.9, rg),
        proxy=y + rg.normal(0, 0.002 * y.std(), n), label=(n - k).astype(float)))


def prewhiten_p(x, y):
    dx, dy = np.diff(x), np.diff(y)
    try:
        sel = ar_select_order(dx, maxlag=min(8, len(dx) // 10), ic='aic', trend='c'); lags = sel.ar_lags or []
    except Exception:
        lags = []
    if lags:
        res = sm.tsa.AutoReg(dx, lags=lags, trend='c').fit(); phi = np.zeros(max(lags) + 1); phi[0] = 1
        for l, cf in zip(lags, res.params[1:]): phi[l] = -cf
        dx, dy = lfilter(phi, [1], dx)[max(lags):], lfilter(phi, [1], dy)[max(lags):]
    return stats.pearsonr(dx, dy)[1]


def dcca_rho(x, y, s):
    X, Y = np.cumsum(x - x.mean()), np.cumsum(y - y.mean()); n = len(x); t = np.arange(s + 1)
    fxx = fyy = fxy = 0.0
    for i in range(0, n - s, max(1, s // 2)):
        xs, ys = X[i:i + s + 1], Y[i:i + s + 1]
        ex = xs - np.polyval(np.polyfit(t, xs, 1), t); ey = ys - np.polyval(np.polyfit(t, ys, 1), t)
        fxx += np.sum(ex * ex); fyy += np.sum(ey * ey); fxy += np.sum(ex * ey)
    return fxy / np.sqrt(fxx * fyy + 1e-30)


def ft_surrogate(x, rg):
    F = np.fft.rfft(x - x.mean()); ph = np.exp(1j * rg.uniform(0, 2 * np.pi, len(F))); ph[0] = 1
    if len(x) % 2 == 0: ph[-1] = 1
    return np.fft.irfft(F * ph, n=len(x)) + x.mean()


def dcca_p(x, y, rg, S=299):
    s = max(8, len(x) // 10); r0 = abs(dcca_rho(x, y, s))
    null = [abs(dcca_rho(ft_surrogate(x, rg), y, s)) for _ in range(S)]
    return (1 + sum(v >= r0 for v in null)) / (S + 1)


def hac_p(rx, ry):
    L = int(np.floor(4 * (len(rx) / 100) ** (2 / 9)))
    return float(sm.OLS(ry, sm.add_constant(rx)).fit(cov_type='HAC', cov_kwds=dict(maxlags=L)).pvalues[1])


def run_rep(cell, k, y, seed, with_shap=True, c_lag=0.8, types=TYPES):
    rg = np.random.default_rng(seed); X = plant(k, y, rg, c_lag)[types]; n = len(k); out = []
    _, ry = T.crossfit_trend(k, y)
    m1 = {c: abs(stats.pearsonr(X[c], y)[0]) >= 0.3 and stats.pearsonr(X[c], y)[1] < 0.05 for c in types}
    if with_shap:
        import shap
        from sklearn.ensemble import GradientBoostingRegressor
        idx = rg.permutation(n); tr = idx[:int(0.8 * n)]
        mdl = GradientBoostingRegressor(n_estimators=150, max_depth=3, learning_rate=0.05, random_state=0).fit(X.iloc[tr], y[tr])
        sv = np.mean(np.abs(shap.TreeExplainer(mdl).shap_values(X)), 0); sh = sv / sv.sum(); m2 = {c: sh[i] >= 0.05 for i, c in enumerate(types)}
    else:
        m2 = {c: np.nan for c in types}
    p3, p5, p6, p7 = {}, {}, {}, {}
    for c in types:
        x = X[c].values; _, rx = T.crossfit_trend(k, x)
        ok = np.std(rx) > 1e-9 * (np.std(x) + 1e-12)
        p3[c] = stats.spearmanr(rx, ry)[1] if ok else 1.0
        p5[c] = prewhiten_p(x, y) if np.std(np.diff(x)) > 0 else 1.0
        p6[c] = dcca_p(x, y, rg) if ok else 1.0
        p7[c] = hac_p(rx, ry) if ok else 1.0
    hol = lambda d: dict(zip(types, T.holm(np.array([d[c] for c in types]))))
    h3, h5, h6, h7 = hol(p3), hol(p5), hol(p6), hol(p7)
    card = T.run_audit(pd.concat([pd.DataFrame(dict(k=k, y=y)), X.reset_index(drop=True)], axis=1), 'k', 'y', types, B=199, seed=seed)
    cls = dict(zip(card.feature, card['class']))
    card1 = T.run_audit(pd.concat([pd.DataFrame(dict(k=k, y=y)), X.reset_index(drop=True)], axis=1), 'k', 'y', types, B=199, seed=seed, version=1)
    cls1 = dict(zip(card1.feature, card1['class']))
    for c in types:
        out.append(dict(target=cell, seed=seed, c_lag=c_lag, feature=c, truth=TRUTH[c], M1_corr=m1[c], M2_shap=m2[c], M3_detrended=h3[c] < .05,
                        M5_prewhiten=h5[c] < .05, M6_dcca=h6[c] < .05, M7_hac=h7[c] < .05, M4_TRACE=cls[c] == 'I', M4v1_TRACE=cls1[c] == 'I', TRACE_v1_class=cls1[c],
                        TRACE_class=cls[c], TRACE_leak=cls[c] in ('P', 'D')))
    return out


def targets():
    f = pd.read_csv('data/nasa_cycle_features.csv'); out = [(f'NASA B{int(b):04d}', h.cycle.values.astype(float), h.capacity.values, R_NASA) for b, h in f.groupby('battery')]
    E = pd.read_csv('outputs/out_ext_audit.csv').drop_duplicates('cell')
    import importlib.util
    spec = importlib.util.spec_from_file_location('ext', '11_external.py'); ext = importlib.util.module_from_spec(spec); spec.loader.exec_module(ext)
    D = ext.load()
    for ds in ['XJTU', 'TJU', 'HUST', 'MIT']:
        cs = E[E.dataset == ds].sort_values('n').cell.values; pick = [cs[len(cs) // 3], cs[2 * len(cs) // 3]]
        for c in pick:
            g = D[D.cell == c]; out.append((f'{ds} {c.split("/")[-1][:-4]}', g.k.values.astype(float), g.capacity.values, R_EXT))
    return out


if __name__ == '__main__':
    TG = targets()
    jobs = [(nm, k, y, 1000 * i + s, True, 0.8, fam) for i, (nm, k, y, R) in enumerate(TG) for s in range(R) for fam in FAM]
    rows = [r for rr in Parallel(n_jobs=2)(delayed(run_rep)(*j) for j in jobs) for r in rr]
    Dd = pd.DataFrame(rows); Dd.to_csv('outputs/out_bench2_raw.csv', index=False)
    M = ['M1_corr', 'M2_shap', 'M3_detrended', 'M5_prewhiten', 'M6_dcca', 'M7_hac', 'M4v1_TRACE', 'M4_TRACE']
    tab = Dd.groupby('feature')[M].mean().loc[TYPES]; tab['TRACE_leak_flag'] = Dd.groupby('feature').TRACE_leak.mean().loc[TYPES]
    tab.to_csv('outputs/out_bench2_table.csv'); print(tab.round(3))
    summ = {}
    pos = Dd.truth == 'I'
    for m in M:
        cr = Dd[m].astype(float)
        tp = cr[pos].sum(); fp = cr[~pos].sum()
        summ[m] = dict(recall=float(cr[pos].mean()), precision=float(tp / (tp + fp)) if tp + fp else 0.0,
                       false_credit=float(cr[~pos].mean()), false_credit_trend_null=float(cr[Dd.truth.isin(['T', 'N'])].mean()),
                       credit_leak=float(cr[Dd.truth.isin(['P', 'D'])].mean()),
                       recall_design=float(cr[Dd.feature.isin(DESIGN)].mean()), recall_heldout=float(cr[Dd.feature.isin(HELDOUT)].mean()),
                       recall_heldout_nolag=float(cr[Dd.feature.isin(['info_smooth', 'info_nonlin'])].mean()))
    summ['TRACE_leak_detection'] = float(Dd[Dd.truth.isin(['P', 'D'])].TRACE_leak.mean())
    summ['TRACE_v1_leak_detection'] = float(Dd[Dd.truth.isin(['P', 'D'])].TRACE_v1_class.isin(['P', 'D']).mean())
    summ['targets'] = [t[0] for t in TG]; summ['replicates'] = len(jobs) // 2
    json.dump(summ, open('outputs/out_bench2_summary.json', 'w'), indent=1); print(json.dumps({k: v for k, v in summ.items() if k != 'targets'}, indent=1))
    # (b) strength sweep on NASA targets
    nasa = [t for t in TG if t[0].startswith('NASA')]
    jobs = [(nm, k, y, 50000 + 1000 * i + 37 * j + s, False, c, ['info_smooth', 'null_trend']) for i, (nm, k, y, _) in enumerate(nasa)
            for j, c in enumerate([0.0, 0.2, 0.4, 0.6, 0.8, 1.2]) for s in range(R_SWEEP)]
    Sw = pd.DataFrame([r for rr in Parallel(n_jobs=2)(delayed(run_rep)(*j) for j in jobs) for r in rr])
    Sw.to_csv('outputs/out_bench2_sweep.csv', index=False)
    print(Sw.groupby(['c_lag', 'feature'])[['M5_prewhiten', 'M6_dcca', 'M7_hac', 'M4v1_TRACE', 'M4_TRACE']].mean().round(3))
    # (c) proxy-noise sweep
    pr = []
    for nm, k, y, _ in TG:
        for lev in [0.002, 0.01, 0.05, 0.1, 0.2, 0.5]:
            for s in range(10 if not QUICK else 3):
                rg = np.random.default_rng(s); x = y + rg.normal(0, lev * y.std(), len(y))
                card = T.run_audit(pd.DataFrame(dict(k=k, y=y, x=x)), 'k', 'y', ['x'], B=99, seed=s)
                pr.append(dict(target=nm, noise=lev, cls=card['class'][0], r_res=card.r_resid[0]))
    P = pd.DataFrame(pr); P.to_csv('outputs/out_bench2_proxy.csv', index=False)
    print(P.groupby('noise').cls.value_counts(normalize=True).unstack(fill_value=0).round(3))
