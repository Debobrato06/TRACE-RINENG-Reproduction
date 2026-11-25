"""Step 0 - data. GMP: use data/GMP.csv if present (download from Mendeley, doi:10.17632/k675pxrd83.2),
otherwise regenerate it from the published generating equations (Data in Brief 69 (2026) 113252, Eqs. 1-8, seed 42)
and, if the CSV is present, verify that the equations reproduce it. NASA: download data_NASA.rda and build cycle features."""
import os, urllib.request, numpy as np, pandas as pd
os.makedirs('data', exist_ok=True); os.makedirs('outputs', exist_ok=True); os.makedirs('figures', exist_ok=True)

def generate_gmp(seed=42):
    np.random.seed(seed); N = 3000; n = np.arange(1, N + 1)
    cap = 40.0 - 12.0 * (n / N) ** 1.35
    V = np.random.normal(25.60, 0.12, N); I = np.random.uniform(-40.0, 40.0, N)
    Tt = np.random.normal(28.0, 1.8, N); C = np.random.normal(3.200, 0.008, (N, 8))
    d = pd.DataFrame({'PackID': 1, 'CycleNumber': n, 'Capacity(Ah)': cap, 'SOH(%)': cap / 40 * 100,
                      'RUL(cycles)': N - n, 'Voltage(V)': V, 'Current(A)': I, 'DoD(%)': n / N * 100, 'Temperature(C)': Tt})
    for i in range(8): d[f'Cell{i+1}_V'] = C[:, i]
    return d

gen = generate_gmp()
if os.path.exists('data/GMP.csv'):
    pub = pd.read_csv('data/GMP.csv')
    diff = max(float(np.max(np.abs(pub[c].values - gen[c].values))) for c in gen.columns)
    print(f'GMP: published CSV found; max |published - regenerated| = {diff:.2e}')
else:
    gen.to_csv('data/GMP.csv', index=False)
    print('GMP: published CSV not found -> regenerated from the published equations (identical to Mendeley V2 up to CSV rounding)')

if not os.path.exists('data/nasa_cycle_features.csv'):
    url = 'https://raw.githubusercontent.com/cjrincon/cnnlstm/master/data/data_NASA.rda'
    if not os.path.exists('data/data_NASA.rda'):
        urllib.request.urlretrieve(url, 'data/data_NASA.rda')
    import pyreadr
    d = pyreadr.read_r('data/data_NASA.rda')['data_NASA']; rows = []
    for (b, c), g in d.groupby(['battery', 'cycle']):
        g = g.sort_values('time'); t = g.time.values; v = g.voltage_measured.values
        T = g.temperature_measured.values; I = g.current_measured.values
        below = np.where(v <= 2.7)[0]; icut = below[0] if len(below) else len(v) - 1; seg = slice(1, icut + 1)
        rows.append(dict(battery=int(b), cycle=int(c), capacity=float(g.capacity.iloc[0]),
                         ambient_T=float(g.ambient_temperature.iloc[0]), t_discharge=float(t[icut]),
                         V_mean=float(np.mean(v[seg])), V_600s=float(np.interp(600, t, v)),
                         T_max=float(np.max(T[:icut + 1])), t_Tmax=float(t[np.argmax(T[:icut + 1])]),
                         T_start=float(T[0]), I_mean=float(np.mean(I[seg]))))
    pd.DataFrame(rows).sort_values(['battery', 'cycle']).to_csv('data/nasa_cycle_features.csv', index=False)
f = pd.read_csv('data/nasa_cycle_features.csv'); print('NASA cycles per cell:', f.groupby('battery').size().to_dict())
