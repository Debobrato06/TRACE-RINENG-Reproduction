"""Step 13 - V5 (extended): resolution and threshold sensitivity.
(a) p-value floor: for each NASA cell, the smallest attainable Holm-adjusted time-shift p-value m/(2r+1), and which
    class-I verdicts sit at that floor. Sensitivity test with finer resolution: circular-shift p-value over all
    n - 1 non-zero circular shifts of the feature residual (full overlap; resolution 1/n), Holm-adjusted.
(b) Threshold sweep: reclassify the stored audit statistics of the 4 NASA cells and the 374 public cells
    under eta_det in {1e-4, 3e-4, 1e-3, 3e-3, 1e-2} and proxy threshold in {0.90, 0.95, 0.98, 0.99, 0.995};
    report the share of class assignments that change from the defaults (1e-3, 0.98).
(c) Ljung-Box exchangeability gate: rejection rate under AR(1) nulls (n = 168 and 3000; phi 0-0.99), i.e. how often
    the trend test is downgraded to descriptive."""
import sys, numpy as np, pandas as pd, warnings, os
from scipy import stats
warnings.filterwarnings('ignore'); sys.path.insert(0, '.'); import trace_audit as T
QUICK = os.environ.get('TRACE_QUICK') == '1'
f = pd.read_csv('data/nasa_cycle_features.csv'); A = pd.read_csv('outputs/out_audit_nasa.csv')
FE = [c for c in A.feature.unique()]

# (a)
rows = []
for b, h in f.groupby('battery'):
    k = h.cycle.values.astype(float); y = h.capacity.values; a = A[A.battery == f'B{int(b):04d}']
    tested = a[a['class'].isin(['I', 'T', 'N'])]; m = len(tested); r = int(a.r.iloc[0]); n = len(k)
    _, ry = T.crossfit_trend(k, y); pc = {}
    for fe in tested.feature:
        _, rx = T.crossfit_trend(k, h[fe].values.astype(float)); xr, yr = stats.rankdata(rx), stats.rankdata(ry)
        s0 = abs(np.corrcoef(xr, yr)[0, 1]); null = np.array([abs(np.corrcoef(np.roll(xr, d), yr)[0, 1]) for d in range(1, n)])
        pc[fe] = (1 + np.sum(null >= s0 - 1e-12)) / n
    hc = dict(zip(pc, T.holm(np.array(list(pc.values())))))
    for _, rw in tested.iterrows():
        rows.append(dict(battery=b, feature=rw.feature, n=n, r=r, m=m, p_floor_holm=m / (2 * r + 1), cls=rw['class'], p_tts_holm=rw.p_cond_holm,
                         at_floor=bool(abs(rw.p_cond_holm - m / (2 * r + 1)) < 1e-9), p_circ_holm=hc[rw.feature], cls_circ='I' if hc[rw.feature] <= .05 else '-'))
P = pd.DataFrame(rows); P.to_csv('outputs/out_pfloor_nasa.csv', index=False)
print(P[P.cls == 'I'][['battery', 'feature', 'p_floor_holm', 'p_tts_holm', 'at_floor', 'p_circ_holm']].round(4).to_string(index=False))
print('class-I verdicts at floor:', int(P[P.cls == 'I'].at_floor.sum()), 'of', int((P.cls == 'I').sum()),
      '| confirmed by circular-shift test (Holm <= 0.05):', int((P[P.cls == 'I'].p_circ_holm <= .05).sum()))

# (b)
E = pd.read_csv('outputs/out_ext_audit.csv'); N_ = A.copy(); N_['cell'] = N_.battery.astype(str); N_['eta_target'] = N_.eta_target
both = pd.concat([E.assign(src='public'), N_.assign(src='NASA')], ignore_index=True)


def reclass(d, eta_det, pthr, zthr=0.01):
    out = []
    for _, rw in d.iterrows():
        reason = str(rw.get('reason', '')) if pd.notna(rw.get('reason', np.nan)) else ''
        if reason in ('constant', 'affine in index or target'): out.append('D'); continue
        if pd.notna(rw.eta) and rw.eta < eta_det: out.append('D'); continue
        if pd.notna(rw.r_resid) and abs(rw.r_resid) >= pthr: out.append('P'); continue
        if pd.notna(rw.get('zeta', np.nan)) and rw.zeta < zthr: out.append('P'); continue
        if rw['class'] in ('D', 'P'):   # was leakage under defaults but not under the new threshold: test result needed
            out.append('I' if (pd.notna(rw.p_cond) and rw.p_cond * max(1, rw.get('m_eff', 5)) <= .05) else 'T'); continue
        out.append(rw['class'])
    return out


base = both['class'].values; sw = []
for eta in [1e-4, 3e-4, 1e-3, 3e-3, 1e-2]:
    for pt in [0.90, 0.95, 0.98, 0.99, 0.995]:
        c = np.array(reclass(both, eta, pt))
        sw.append(dict(eta_det=eta, proxy_thr=pt, changed=float(np.mean(c != base)),
                       changed_public=float(np.mean(c[both.src == 'public'] != base[both.src == 'public'])),
                       changed_nasa=float(np.mean(c[both.src == 'NASA'] != base[both.src == 'NASA'])),
                       n_P=int((c == 'P').sum()), n_D=int((c == 'D').sum())))
S = pd.DataFrame(sw); S.to_csv('outputs/out_threshold_sweep.csv', index=False); print(S.round(4).to_string(index=False))
zs = []
for z in [0.0025, 0.005, 0.01, 0.02, 0.05]:
    c = np.array(reclass(both, 1e-3, 0.98, z))
    zs.append(dict(zeta_thr=z, changed=float(np.mean(c != base)), n_P=int((c == 'P').sum()),
                   n_P_public=int((c[both.src == 'public'] == 'P').sum()), n_P_nasa=int((c[both.src == 'NASA'] == 'P').sum())))
Z = pd.DataFrame(zs); Z.to_csv('outputs/out_zeta_sweep.csv', index=False); print(Z.round(4).to_string(index=False))
te = pd.concat([E.drop_duplicates('cell')[['cell', 'eta_target']].assign(src='public'), N_.drop_duplicates('cell')[['cell', 'eta_target']].assign(src='NASA')])
M_ = pd.DataFrame([dict(eta_det=e, cells_target_deterministic=int((te.eta_target < e).sum()), of=len(te)) for e in [1e-4, 3e-4, 1e-3, 3e-3, 1e-2]])
M_.to_csv('outputs/out_target_eta_margin.csv', index=False); print(M_.to_string(index=False))
print('public cell target eta: min %.4f, 1st pct %.4f' % (E.drop_duplicates('cell').eta_target.min(), E.drop_duplicates('cell').eta_target.quantile(.01)))
print('feature |r_res| max among non-proxy public features: %.3f' % E[E['class'] != 'P'].r_resid.abs().max())

# (c) size of the exchangeability gate: Monte Carlo-calibrated (TRACE) vs the asymptotic chi-square reference
from statsmodels.stats.diagnostic import acorr_ljungbox
rng = np.random.default_rng(0); lb = []
for n in [168, 3000]:
    k = np.arange(1, n + 1, dtype=float); Bf = T._bspline_basis(k, int(max(3, min(12, n // 15))))
    for phi in [0.0, 0.3, 0.6, 0.9, 0.99]:
        rej = rej_asy = 0; R = 100 if QUICK else (1000 if n < 1000 else 300)
        for i_ in range(R):
            e = rng.normal(size=n + 200); x = np.zeros_like(e)
            for i in range(1, len(e)): x[i] = phi * x[i - 1] + e[i]
            x = x[200:]; res = x - Bf @ np.linalg.lstsq(Bf, x, rcond=None)[0]
            rej_asy += acorr_ljungbox(res, lags=[10], return_df=True)['lb_pvalue'].iloc[0] < 0.05
            rej += T.lb_gate(k, x, lags=10, B=499, seed=i_) < 0.05
        lb.append(dict(n=n, phi=phi, gate_rejects=rej / R, chi2_reference_rejects=rej_asy / R))
L = pd.DataFrame(lb); L.to_csv('outputs/out_lb_gate.csv', index=False); print(L.to_string(index=False))
