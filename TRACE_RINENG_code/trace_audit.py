"""
TRACE: Trajectory-Referenced Audit of Cycle-level Evidence
A pre-modelling information audit for cycle-indexed battery prognostics datasets.

Stages
  S1  Provenance / identity screen  -> determinism index eta, exact identities
  S2  Index-surrogate ceiling        -> skill attainable from the cycle index alone
  S3  Association tests (time-shift) -> marginal and index-conditional, Holm-corrected
  S4  Sensitivity calibration        -> injection power curves, minimum detectable effect

Dependencies: numpy, scipy, pandas (scikit-learn optional, not required).
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats

from scipy.optimize import curve_fit

# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------

def _z(a):
    a = np.asarray(a, float)
    s = a.std()
    return (a - a.mean()) / s if s > 0 else a * 0.0


_PROJ_CACHE = {}


def _bspline_basis(k, n_knots):
    from scipy.interpolate import BSpline
    deg = 3
    lo, hi = k.min(), k.max()
    inner = np.quantile(k, np.linspace(0, 1, n_knots + 2)[1:-1])
    t = np.r_[[lo] * (deg + 1), inner, [hi] * (deg + 1)]
    return BSpline.design_matrix(np.clip(k, lo, hi), t, deg).toarray()


def _crossfit_operator(k, n_knots):
    """Linear operator M such that M @ x is the two-fold interleaved cross-fitted
    cubic regression-spline trend of x on k (cached per index vector)."""
    key = (len(k), float(k[0]), float(k[-1]), n_knots, float(np.sum(k)))
    if key in _PROJ_CACHE:
        return _PROJ_CACHE[key]
    B = _bspline_basis(k, n_knots)
    n = len(k); idx = np.arange(n); M = np.zeros((n, n))
    for part in (0, 1):
        tr = idx % 2 != part; te = ~tr
        P = np.linalg.pinv(B[tr])            # least-squares coefficients map
        M[np.ix_(te, tr)] = B[te] @ P
    _PROJ_CACHE[key] = M
    return M


def crossfit_trend(k, x, n_knots=None):
    """Two-fold interleaved cross-fitted cubic regression spline of x on index k.
    Points at even positions are predicted from a spline fitted to odd positions and
    vice versa, so the trend cannot absorb observation noise.
    Default: min(12, n//15) interior quantile knots. Returns (trend, residual)."""
    k = np.asarray(k, float); x = np.asarray(x, float)
    if n_knots is None:
        n_knots = int(max(3, min(12, len(k) // 15)))
    M = _crossfit_operator(k, n_knots)
    fit = M @ x
    return fit, x - fit


# ----------------------------------------------------------------------------
# S1  provenance / identity screen
# ----------------------------------------------------------------------------

def determinism_index(k, x):
    """eta = RMS(cross-fitted residual) / SD(x).
    eta ~ 1: no smooth dependence on the index (stationary column);
    eta << 1 (< 1e-3): column is a deterministic smooth function of the index."""
    x = np.asarray(x, float)
    sd = x.std()
    if sd == 0:
        return np.nan
    _, r = crossfit_trend(k, x)
    return float(np.sqrt(np.mean(r ** 2)) / sd)


def identity_screen(df: pd.DataFrame, index_col: str, tol=1e-9):
    """Flag constant columns, columns affine in another column (|r| >= 1-tol),
    and columns strictly monotone in the index."""
    out = []
    cols = [c for c in df.columns]
    k = df[index_col].values
    for c in cols:
        x = df[c].values.astype(float)
        rec = dict(column=c)
        rec['constant'] = bool(np.nanstd(x) == 0)
        if rec['constant']:
            rec.update(affine_to='', strictly_monotone_in_index=False,
                       spearman_index=np.nan, eta=np.nan)
            out.append(rec); continue
        partners = []
        for c2 in cols:
            if c2 == c or np.std(df[c2].values) == 0:
                continue
            r = np.corrcoef(x, df[c2].values.astype(float))[0, 1]
            if abs(r) >= 1 - tol:
                partners.append(c2)
        rec['affine_to'] = ';'.join(partners)
        dx = np.diff(x[np.argsort(k)])
        rec['strictly_monotone_in_index'] = bool(np.all(dx > 0) or np.all(dx < 0))
        rec['spearman_index'] = float(stats.spearmanr(k, x)[0])
        rec['eta'] = determinism_index(k, x) if c != index_col else 0.0
        out.append(rec)
    return pd.DataFrame(out)


def trend_r2(k, x):
    """Cross-fitted trend R^2 = 1 - eta^2 (can be slightly negative under no trend)."""
    e = determinism_index(k, x)
    return np.nan if np.isnan(e) else 1 - e ** 2


def _lb_q(R, lags):
    """Ljung-Box Q for each column of R."""
    R = R - R.mean(0); n = R.shape[0]; d = np.sum(R ** 2, 0)
    return n * (n + 2) * sum(np.sum(R[h:] * R[:-h], 0) ** 2 / d ** 2 / (n - h) for h in range(1, lags + 1))


def lb_gate(k, x, lags=10, B=999, seed=0):
    """Exchangeability gate: Ljung-Box statistic of the spline-detrended residual, with its null
    distribution obtained by passing i.i.d. N(0,1) series through the same detrending projection.
    The asymptotic chi-square reference ignores the degrees of freedom removed by the spline and
    rejects white noise far too often on short records (34% at n = 168); this calibration restores
    the nominal size under i.i.d. errors. Returns the Monte Carlo p-value."""
    k = np.asarray(k, float); x = np.asarray(x, float)
    Bf = _bspline_basis(k, int(max(3, min(12, len(k) // 15))))
    Q, _ = np.linalg.qr(Bf)
    proj = lambda V: V - Q @ (Q.T @ V)
    q0 = _lb_q(proj(x[:, None]), lags)[0]
    Z = np.random.default_rng(seed + 7919).normal(size=(len(k), B))
    qn = _lb_q(proj(Z), lags)
    return float((1 + np.sum(qn >= q0)) / (B + 1))


def trend_test(k, x, B=999, seed=0, lb_lags=10):
    """S1b: permutation test of H0 'no dependence of x on the cycle index'.
    Statistic: cross-fitted trend R^2. Null: random permutations of x (valid when
    the observations are exchangeable). Exchangeability is checked by a Ljung-Box
    test on the detrended residual; if it fails, the result is descriptive only."""
    k = np.asarray(k, float); x = np.asarray(x, float)
    M = _crossfit_operator(k, int(max(3, min(12, len(k) // 15))))
    def stat(v):
        r = v - M @ v
        return 1 - np.mean(r ** 2) / np.var(v)
    s0 = stat(x)
    rng = np.random.default_rng(seed)
    P = np.stack([rng.permutation(x) for _ in range(B)], 1)
    R = P - M @ P
    null = 1 - np.mean(R ** 2, 0) / np.var(P, 0)
    p = (1 + np.sum(null >= s0)) / (B + 1)
    # exchangeability gate on the full-sample detrended residual (interleaved cross-fitting itself
    # induces lag-1 structure), calibrated by Monte Carlo for the spline projection (lb_gate)
    lb_p = lb_gate(k, x, lags=lb_lags, seed=seed)
    return dict(R2_trend=float(s0), p_trend=float(p), null_q95=float(np.quantile(null, .95)),
                null_q99=float(np.quantile(null, .99)), LB_p=lb_p, exchangeable=lb_p > 0.05)


# ----------------------------------------------------------------------------
# S2  index-surrogate ceiling
# ----------------------------------------------------------------------------

def _powerlaw(k, a, b, c):
    return a - b * np.power(k, c)


def index_ceiling(k, y, train_frac=0.7):
    """Chronological split. Fits y = a - b*k^c on the first train_frac of cycles
    using the index only and extrapolates to the rest. Returns test R2, RMSE.
    High R2 means the target is (near-)fully determined by the index: any
    feature-based model must be judged against this ceiling."""
    k = np.asarray(k, float); y = np.asarray(y, float)
    n = len(k); m = int(round(train_frac * n))
    kt, yt = k[:m], y[:m]
    try:
        p0 = (yt[0], (yt[0] - yt[-1]) / max(kt[-1] ** 1.0, 1), 1.0)
        p, _ = curve_fit(_powerlaw, kt, yt, p0=p0, maxfev=20000,
                         bounds=([-np.inf, -np.inf, 0.2], [np.inf, np.inf, 4.0]))
        yhat = _powerlaw(k[m:], *p)
    except Exception:
        coef = np.polyfit(kt, yt, 2)
        yhat = np.polyval(coef, k[m:])
    ytest = y[m:]
    ss_res = np.sum((ytest - yhat) ** 2)
    ss_tot = np.sum((ytest - ytest.mean()) ** 2)
    return dict(R2_index=float(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan,
                RMSE_index=float(np.sqrt(np.mean((ytest - yhat) ** 2))))


# ----------------------------------------------------------------------------
# S3  association tests: conservative truncated time-shift test
# ----------------------------------------------------------------------------

def _absspearman(a, b):
    return abs(stats.spearmanr(a, b)[0])


def _shift_corr(x, y, r):
    """|Pearson r| between x[r+d : n-r+d] and y[r : n-r] for every d in [-r, r],
    computed in O(n log n) with FFT cross-correlation and cumulative sums."""
    from scipy.signal import correlate
    n = len(x); L = n - 2 * r
    yw = y[r:n - r]
    yc = yw - yw.mean(); syy = np.sum(yc ** 2)
    sxy = correlate(x, yc, mode='valid', method='fft')        # length 2r+1
    cs = np.concatenate([[0.0], np.cumsum(x)])
    cs2 = np.concatenate([[0.0], np.cumsum(x * x)])
    sx = cs[L:] - cs[:-L]; sxx = cs2[L:] - cs2[:-L]
    vx = sxx - sx ** 2 / L
    with np.errstate(invalid='ignore', divide='ignore'):
        S = np.abs(sxy) / np.sqrt(vx * syy)
    return np.nan_to_num(S)


TTS_DIFF_DEFAULT = True   # TRACE v2: level + rate representation of the target residual


def tts_test(x, y, r=None, rank=True, lags=0, diff=None):
    """Truncated time-shift test (Harris 2020; Yuan & Shou 2024), conservative.
    x is the series assumed stationary; y is held fixed. For each shift d in [-r, r] the
    statistic is computed between x[R+d : n-R+d] and the fixed window y[R : n-R], R = r + lags.
    Basic statistic: |rank correlation| at lag 0 (lags=0, diff=False; TRACE v1).
    TRACE v2 statistic: the maximum |rank correlation| over lags l in [-lags, lags] and over two
    representations of the target, its level residual and its first difference (rate). The same
    statistic is computed at every shift, so the test remains valid for any stationary x.
    p = #{d : S_d >= S_0} / (2r+1); minimum attainable p = 1/(2r+1). Returns (S0, p)."""
    if diff is None:
        diff = TTS_DIFF_DEFAULT
    x = np.asarray(x, float); y = np.asarray(y, float)
    n = len(x)
    if r is None:
        r = n // 4
    R = r + lags
    targets = [y] + ([np.r_[0.0, np.diff(y)]] if diff else [])
    if rank:
        x = stats.rankdata(x); targets = [stats.rankdata(t) for t in targets]
    best = None
    for t in targets:
        S = _shift_corr(x, t, R)                     # shifts -R..R
        if lags:
            S = np.max(np.stack([S[lags + l: lags + l + 2 * r + 1] for l in range(-lags, lags + 1)]), 0)
        best = S if best is None else np.maximum(best, S)
    S0 = best[r]
    if lags:
        # exclusion zone: shifts with |d| <= 2*lags share alignments with d = 0 and are left out of the null
        d = np.arange(-r, r + 1); null = best[np.abs(d) > 2 * lags]
        p = (1 + np.sum(null >= S0 - 1e-12)) / (len(null) + 1)
    else:
        p = np.sum(best >= S0 - 1e-12) / (2 * r + 1)
    return float(S0), float(p)


def holm(p):
    p = np.asarray(p, float); m = len(p)
    o = np.argsort(p); adj = np.empty(m); run = 0.0
    for i, j in enumerate(o):
        run = max(run, min(1.0, (m - i) * p[j]))
        adj[j] = run
    return adj


def association_tests(k, y, X: pd.DataFrame, r=None, conditional=True):
    """Marginal (x vs y) and index-conditional (resid_x vs resid_y) TTS tests."""
    rows = []
    _, ry = crossfit_trend(k, y)
    ry_ok = np.std(ry) > 1e-6 * np.std(y)
    for c in X.columns:
        x = X[c].values.astype(float)
        S0, p = tts_test(x, y, r)
        rec = dict(feature=c, S_marg=S0, p_marg=p)
        if conditional and ry_ok:
            _, rx = crossfit_trend(k, x)
            S0c, pc = tts_test(rx, ry, r)
            rec.update(S_cond=S0c, p_cond=pc)
        else:
            rec.update(S_cond=np.nan, p_cond=np.nan)
        rows.append(rec)
    out = pd.DataFrame(rows)
    out['p_marg_holm'] = holm(out.p_marg.values)
    if out.p_cond.notna().all():
        out['p_cond_holm'] = holm(out.p_cond.values)
    else:
        out['p_cond_holm'] = np.nan
    out.attrs['conditional_testable'] = bool(ry_ok)
    return out


# ----------------------------------------------------------------------------
# S4  sensitivity calibration by signal injection (self-surrogate)
# ----------------------------------------------------------------------------

def inject(xnull, target, rho):
    """Return x_inj whose population correlation with target is rho when xnull
    is independent of target; preserves the mean/SD of xnull."""
    zx, zt = _z(xnull), _z(target)
    zi = np.sqrt(1 - rho ** 2) * zx + rho * zt
    return xnull.mean() + xnull.std() * zi


def self_surrogate_power(x, y, rhos, n_rep=200, alpha=0.05, r=None, seed=0,
                         target=None, n_tests=1):
    """Power of the (Holm-adjusted, n_tests-family) TTS test to detect an injected
    coupling of strength rho. Null realisations are random circular shifts of the
    observed x (self-surrogates), which preserve its autocorrelation."""
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float); y = np.asarray(y, float)
    t = y if target is None else target
    n = len(x)
    a_eff = alpha / n_tests  # Bonferroni bound on the Holm step for the worst rank
    res = []
    for rho in rhos:
        hits = 0
        for _ in range(n_rep):
            xs = np.roll(x, rng.integers(n // 10, n - n // 10))
            xi = inject(xs, t, rho)
            _, p = tts_test(xi, y, r)
            hits += p <= a_eff
        res.append(hits / n_rep)
    return np.array(res)


def mde(rhos, power, target=0.8):
    """Interpolated smallest coupling with power >= target (MDE80)."""
    rhos = np.asarray(rhos); power = np.asarray(power)
    above = np.where(power >= target)[0]
    if len(above) == 0:
        return np.nan
    i = above[0]
    if i == 0:
        return float(rhos[0])
    # linear interpolation between grid points
    r0, r1, p0, p1 = rhos[i - 1], rhos[i], power[i - 1], power[i]
    return float(r0 + (target - p0) * (r1 - r0) / (p1 - p0)) if p1 > p0 else float(r1)


# ----------------------------------------------------------------------------
# verdicts
# ----------------------------------------------------------------------------

def verdict(row, mde_val, rho_rel=0.1, alpha=0.05, derived=False):
    if derived:
        return 'D'  # derived / deterministic: exclude as feature
    if row['p_marg_holm'] <= alpha or (not np.isnan(row.get('p_cond_holm', np.nan))
                                       and row['p_cond_holm'] <= alpha):
        return 'I'  # informative
    if not np.isnan(mde_val) and mde_val <= rho_rel:
        return 'N'  # null with adequate power
    return 'U'      # underpowered / inconclusive


# ----------------------------------------------------------------------------
# full audit card
# ----------------------------------------------------------------------------

def proxy_check(k, y, x):
    """Residual correlation between feature and target after removing the index
    trend. |r_res| >= 0.98 means the feature reproduces the target's own
    cycle-to-cycle fluctuations: a likely co-computed proxy (verify provenance)."""
    _, ry = crossfit_trend(k, y); _, rx = crossfit_trend(k, x)
    if np.std(ry) < 1e-6 * np.std(y):
        return np.nan
    return float(np.corrcoef(rx, ry)[0, 1])


def restatement_index(x, y):
    """zeta = RMS(y - a - b x) / SD(y): how closely a linear map of the feature restates the target on
    the raw scale. zeta < 0.01 means the feature reproduces the target to within 1% of its spread."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    if np.std(x) == 0 or np.std(y) == 0:
        return np.nan
    A = np.column_stack([np.ones_like(x), x]); res = y - A @ np.linalg.lstsq(A, y, rcond=None)[0]
    return float(np.sqrt(np.mean(res ** 2)) / np.std(y))


def run_audit(df, index_col, target_col, feature_cols, r=None, B=999,
              eta_det=1e-3, proxy_thr=0.98, zeta_thr=0.01, alpha=0.05, seed=0, version=2,
              mde=False, mde_rhos=(0, .1, .2, .3, .4, .5, .6, .7, .8, .9), mde_reps=200):
    """TRACE audit card (stages S1-S4).
    S1 identity / determinism (eta) / proxy (|r_res|, zeta) screen; S2 index-only ceiling (attrs['ceiling']);
    S3 gated permutation trend test and truncated time-shift beyond-index test, Holm within the tested family;
    S4 (mde=True) self-surrogate MDE80 of the beyond-index test for every T and N feature (column 'MDE80').
    The truncation radius r is fixed before classification, so the resolution rule uses m = number of candidate
    features (an upper bound on the tested family); Holm then uses the tested family.
    version=2 (default): level + rate beyond-index statistic and restatement index zeta. version=1: original TRACE."""
    lags, use_diff = (0, True) if version >= 2 else (0, False)
    k = df[index_col].values.astype(float); y = df[target_col].values.astype(float)
    n = len(k)
    m_tests = len(feature_cols)
    if r is None:
        # resolution rule: the smallest Holm-adjusted p, m/(2r+1), must not exceed alpha
        r_floor = int(np.ceil((m_tests / alpha - 1) / 2))
        r = int(min(max(n // 4, r_floor), np.floor(0.4 * n) - lags))
    eta_y = determinism_index(k, y)
    ceil = index_ceiling(k, y)
    rows = []
    for c in feature_cols:
        x = df[c].values.astype(float)
        rec = dict(feature=c)
        if np.std(x) == 0:
            rec.update(cls='D', reason='constant'); rows.append(rec); continue
        rx_aff = abs(np.corrcoef(x, k)[0, 1]) >= 1 - 1e-9 or abs(np.corrcoef(x, y)[0, 1]) >= 1 - 1e-9
        eta = determinism_index(k, x)
        tt = trend_test(k, x, B=B, seed=seed)
        rp = proxy_check(k, y, x)
        zt = restatement_index(x, y) if version >= 2 else np.nan
        rec.update(eta=eta, R2_trend=tt['R2_trend'], p_trend=tt['p_trend'],
                   exch=tt['exchangeable'], r_resid=rp, zeta=zt)
        if eta_y < eta_det:
            rec.update(S_cond=np.nan, p_cond=np.nan)
        else:
            _, ry = crossfit_trend(k, y); _, rx = crossfit_trend(k, x)
            S, p = tts_test(rx, ry, r, lags=lags, diff=use_diff)
            rec.update(S_cond=S, p_cond=p)
        if rx_aff:
            rec['reason'] = 'affine in index or target'
        elif eta < eta_det:
            rec['reason'] = 'deterministic in index'
        elif not np.isnan(rp) and abs(rp) >= proxy_thr:
            rec['reason'] = 'target proxy (residual |r|>=%.2f)' % proxy_thr
        elif not np.isnan(zt) and zt < zeta_thr:
            rec['reason'] = 'target proxy (restatement zeta<%.3f)' % zeta_thr
        else:
            rec['reason'] = ''
        rows.append(rec)
    out = pd.DataFrame(rows)
    tested = out['reason'].fillna('') == ''
    out['p_cond_holm'] = np.nan
    if tested.any() and out.loc[tested, 'p_cond'].notna().all():
        out.loc[tested, 'p_cond_holm'] = holm(out.loc[tested, 'p_cond'].values)
    out['p_trend_holm'] = np.nan
    ex = tested & out['exch'].fillna(False).astype(bool)
    if ex.any():
        out.loc[ex, 'p_trend_holm'] = holm(out.loc[ex, 'p_trend'].values)
    cls = []
    for _, rw in out.iterrows():
        if rw.get('reason', ''):
            cls.append('D' if 'proxy' not in rw['reason'] else 'P'); continue
        if not np.isnan(rw['p_cond_holm']) and rw['p_cond_holm'] <= alpha:
            cls.append('I'); continue
        trending = (rw['exch'] and rw['p_trend_holm'] <= alpha) or (not rw['exch'] and rw['R2_trend'] > 0.1)
        if trending:
            cls.append('T'); continue
        cls.append('N' if eta_y >= eta_det else 'N*')
    out['class'] = cls
    m_eff = int(tested.sum())
    out['MDE80'] = np.nan
    if mde and eta_y >= eta_det and m_eff > 0:
        _, ry = crossfit_trend(k, y)
        for i in out.index[out['class'].isin(['T', 'N'])]:
            _, rx = crossfit_trend(k, df[out.at[i, 'feature']].values.astype(float))
            pw = self_surrogate_power(rx, ry, mde_rhos, n_rep=mde_reps, alpha=alpha, r=r, seed=seed, n_tests=m_eff)
            out.at[i, 'MDE80'] = mde_fn(mde_rhos, pw)
    out.attrs.update(eta_target=eta_y, ceiling=ceil, r=r, conditional_testable=eta_y >= eta_det,
                     m_tested=m_eff, min_holm_p=m_eff / (2 * r + 1 - (4 * lags if lags else 0)))
    return out


mde_fn = mde   # alias used inside run_audit, whose argument 'mde' is a switch
