"""Step 3 - V2 (false positives of the trend test and the beyond-index test) and V3 (power / MDE80)."""
import numpy as np, pandas as pd, sys, time, json, os
sys.path.insert(0, '.'); import trace_audit as T
from scipy import stats
QUICK = os.environ.get('TRACE_QUICK') == '1'
t0 = time.time(); res = {}
N = 3000; k = np.arange(1, N + 1.); cap = 40 - 12 * (k / N) ** 1.35
# fast factorised two-fold cross-fitted spline operator (identical to T._crossfit_operator(k, 12))
Bb = T._bspline_basis(k, 12); idx = np.arange(N); PARTS = []
for part in (0, 1):
    tr = idx % 2 != part; te = ~tr; PARTS.append((tr, te, np.linalg.pinv(Bb[tr]), Bb[te]))
def M(P):
    out = np.empty_like(P)
    for tr, te, L, Bt in PARTS: out[te] = Bt @ (L @ P[tr])
    return out
def r2(v): r = v - M(v); return 1 - np.mean(r ** 2) / np.var(v)

# ---- V2a trend-test FPR on regenerated GMP null datasets (permutation, B = 399)
R = 100 if QUICK else 1000; fp = np.zeros(12); fw = 0
for s in range(R):
    rng = np.random.RandomState(1000 + s)
    V = rng.normal(25.6, .12, N); I = rng.uniform(-40, 40, N); Tt = rng.normal(28, 1.8, N); C = rng.normal(3.2, .008, (N, 8))
    X = np.column_stack([V, I, Tt, C, C.max(1) - C.min(1)]); ps = []
    for j in range(12):
        x = X[:, j]; s0 = r2(x)
        P = np.stack([rng.permutation(x) for _ in range(399)], 1); Rr = P - M(P)
        nul = 1 - np.mean(Rr ** 2, 0) / np.var(P, 0); ps.append((1 + np.sum(nul >= s0)) / 400)
    ps = np.array(ps); fp += ps <= 0.05; fw += np.any(T.holm(ps) <= 0.05)
res['V2a_trend_FPR'] = list(fp / R); res['V2a_trend_FWER'] = fw / R
print('V2a', np.round(fp / R, 3), fw / R, round(time.time() - t0), flush=True)

# ---- V3a trend-injection power on GMP: true generator vs self-surrogate
g = pd.read_csv('data/GMP.csv'); v_obs = g['Voltage(V)'].values; rng = np.random.default_rng(3)
P = np.stack([rng.permutation(v_obs) for _ in range(4000)], 1); nul = 1 - np.mean((P - M(P)) ** 2, 0) / np.var(P, 0)
ct = np.quantile(nul, 0.95); cf = np.quantile(nul, 1 - 0.05 / 12)
rows = []; RR = 100 if QUICK else 500
for kind in ['generator', 'self-surrogate']:
    for rho in [0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20]:
        h1 = h2 = 0
        for s in range(RR):
            x = np.random.RandomState(20000 + s).normal(25.6, .12, N) if kind == 'generator' else rng.permutation(v_obs)
            st = r2(T.inject(x, cap, rho)); h1 += st > ct; h2 += st > cf
        rows.append(dict(kind=kind, rho=rho, power_per_test=h1 / RR, power_family12=h2 / RR))
pw = pd.DataFrame(rows); pw.to_csv('outputs/out_power_trend_gmp.csv', index=False)
for kind in ['generator', 'self-surrogate']:
    q = pw[pw.kind == kind]; res[f'MDE80_trend_{kind}_test'] = T.mde(q.rho, q.power_per_test); res[f'MDE80_trend_{kind}_fam'] = T.mde(q.rho, q.power_family12)
print({k_: v for k_, v in res.items() if 'MDE' in k_}, round(time.time() - t0), flush=True)

# ---- V2b beyond-index FPR on NASA targets (AR(1) + random smooth trend nulls) vs naive test
f = pd.read_csv('data/nasa_cycle_features.csv'); rows = []; RN = 200 if QUICK else 1000
def r_rule(n, m=8): return int(min(max(n // 4, int(np.ceil((m / 0.05 - 1) / 2))), np.floor(0.4 * n)))
for b, h in f.groupby('battery'):
    kk = h.cycle.values.astype(float); y = h.capacity.values; n = len(kk); r = r_rule(n)
    _, ry = T.crossfit_trend(kk, y)
    for phi in [0.0, 0.5, 0.9, 0.97]:
        a = bn = 0
        for s in range(RN):
            rg = np.random.default_rng(40000 + s); e = rg.normal(size=n + 500); x = np.zeros_like(e)
            for i in range(1, len(e)): x[i] = phi * x[i - 1] + e[i]
            x = x[500:] + rg.normal(0, 2) * np.sin(np.pi * kk / n * rg.uniform(0.5, 2) + rg.uniform(0, 6))
            _, rx = T.crossfit_trend(kk, x)
            a += T.tts_test(rx, ry, r)[1] <= 0.05; bn += stats.spearmanr(rx, ry)[1] <= 0.05
        rows.append(dict(battery=b, phi=phi, FPR_TTS=a / RN, FPR_naive=bn / RN))
pd.DataFrame(rows).to_csv('outputs/out_typeI_cond_nasa.csv', index=False); print('V2b done', round(time.time() - t0), flush=True)

# ---- V3b beyond-index power on NASA (self-surrogate: circular shifts of the start-temperature residual)
rows = []; rhos = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]; RP = 100 if QUICK else 400
for b, h in f.groupby('battery'):
    kk = h.cycle.values.astype(float); y = h.capacity.values; n = len(kk); r = r_rule(n)
    _, ry = T.crossfit_trend(kk, y); _, rx = T.crossfit_trend(kk, h.T_start.values); rg = np.random.default_rng(7)
    for rho in rhos:
        h1 = h2 = 0
        for s in range(RP):
            xi = T.inject(np.roll(rx, rg.integers(n // 10, n - n // 10)), ry, rho)
            p = T.tts_test(xi, ry, r)[1]; h1 += p <= 0.05; h2 += p <= 0.05 / 5
        rows.append(dict(battery=b, rho=rho, power_per_test=h1 / RP, power_family5=h2 / RP))
pc = pd.DataFrame(rows); pc.to_csv('outputs/out_power_cond_nasa.csv', index=False)
for b, q in pc.groupby('battery'):
    res[f'MDE80_cond_B{b}_test'] = T.mde(q.rho, q.power_per_test); res[f'MDE80_cond_B{b}_fam'] = T.mde(q.rho, q.power_family5)
json.dump(res, open('outputs/out_validation.json', 'w'), indent=1, default=float)
print({k_: (round(v, 3) if isinstance(v, float) else v) for k_, v in res.items() if 'MDE' in k_}, round(time.time() - t0))
