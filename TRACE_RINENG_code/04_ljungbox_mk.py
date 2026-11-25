"""Step 4 - diagnostics quoted in the text: size of the Monte Carlo-calibrated Ljung-Box gate (and of the chi-square reference) on detrended iid noise; Mann-Kendall (Hamed-Rao) size under AR(1)."""
import numpy as np, sys, json, os
sys.path.insert(0, '.'); import trace_audit as T
from statsmodels.stats.diagnostic import acorr_ljungbox
import pymannkendall as mk
QUICK = os.environ.get('TRACE_QUICK') == '1'
k = np.arange(1, 3001.); Bf = T._bspline_basis(k, 12); out = {}
R = 100 if QUICK else 400; f = []
for i in range(R):
    x = np.random.default_rng(i).normal(size=3000); r = x - Bf @ np.linalg.lstsq(Bf, x, rcond=None)[0]
    f.append(acorr_ljungbox(r, lags=[10], return_df=True)['lb_pvalue'].iloc[0] <= 0.05)
out['LB_size_lag10_chi2_reference'] = float(np.mean(f))
out['LB_size_lag10'] = float(np.mean([T.lb_gate(k, np.random.default_rng(i).normal(size=3000), B=299, seed=i) <= 0.05 for i in range(R)]))
R = 60 if QUICK else 300
for phi in [0.9, 0.99]:
    c = 0
    for s in range(R):
        rng = np.random.default_rng(7000 + s); e = rng.normal(size=5000); x = np.zeros_like(e)
        for i in range(1, len(e)): x[i] = phi * x[i - 1] + e[i]
        c += mk.hamed_rao_modification_test(x[2000:]).p <= 0.05
    out[f'MK_HR_size_phi{phi}'] = c / R
json.dump(out, open('outputs/out_diagnostics.json', 'w'), indent=1); print(out)
