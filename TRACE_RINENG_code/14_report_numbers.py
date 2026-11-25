"""Step 14 - assemble the Method details and Method validation chapters from the outputs.

Writes
  outputs/tables/Table2_classes.csv, Table3_parameters.csv, Table4_experiments.csv,
  outputs/tables/Table5_nasa_audit_card.csv, TableS1..S4 (*.csv)
  outputs/results_numbers.md   every number quoted in the Method validation chapter, by section (V1-V8),
                               each with the output file it comes from.
Run after scripts 00-13. Uses only files in outputs/."""
import json, os, numpy as np, pandas as pd

O = 'outputs/'; TD = O + 'tables/'; os.makedirs(TD, exist_ok=True)
rd = lambda f, **k: pd.read_csv(O + f, **k)
js = lambda f: json.load(open(O + f))
L = []                                    # markdown lines
def H(t): L.append(f'\n## {t}\n')
def item(txt, src): L.append(f'- {txt}  _(source: {src})_')
def rng(a, d=3): a = np.asarray(a, float); return f'{a.min():.{d}f}–{a.max():.{d}f}'

# ------------------------------------------------------------------ Method details tables (static definitions)
pd.DataFrame([
    ['D', 'Constant; affine in k or y; or deterministic in k (eta < 1e-3)', 'Not a predictor; may be a label.'],
    ['P', 'Restates the target: |r_res| >= 0.98 or restatement index zeta < 0.01', 'Co-computed with y; exclude and check provenance.'],
    ['I', 'Beyond-index time-shift test significant after Holm correction', 'Carries same-cycle health information that k does not.'],
    ['T', 'Co-trends with k; beyond-index test not significant', 'Not identifiable on one record; may still transfer across cells.'],
    ['N', 'No trend coupling; beyond-index test not significant', 'Coupling below the reported MDE80.'],
    ['N*', 'Target deterministic in k (eta(y) < 1e-3): beyond-index test undefined', 'No claim beyond k is possible.'],
], columns=['Class', 'Decision rule', 'Permitted claim']).to_csv(TD + 'Table2_classes.csv', index=False)
pd.DataFrame([
    ['Trend smoother', 'Cubic regression spline, min(12, n//15) quantile knots', 'knots 6-12 in V5'],
    ['Cross-fitting', 'Two folds, interleaved odd/even cycles', 'trend at a cycle never uses that cycle'],
    ['Determinism threshold', 'eta < 1e-3', 'swept 1e-4 to 1e-2 (V5)'],
    ['Proxy thresholds', '|r_res| >= 0.98; zeta < 0.01', 'swept 0.90-0.995 and 0.0025-0.05 (V5)'],
    ['Trend test', 'Permutation, B = 999; Ljung-Box (lag 10) gate', 'descriptive if the gate rejects'],
    ['Beyond-index statistic', 'max(|Spearman| with e_y, with diff(y)), lag 0', 'multi-lag maximum rejected (inflated size)'],
    ['Truncation radius r', 'min(max(n//4, ceil((p/alpha-1)/2)), floor(0.4 n)), p = candidate features', 'fixed before classification; m/(2r+1) <= alpha for tested m <= p'],
    ['Multiplicity', 'Holm within the feature family', 'family-wise error control'],
    ['Power', '80 %, coupling grid rho in [0, 0.9]', 'MDE80 per family'],
], columns=['Parameter', 'Default', 'Rationale; sensitivity']).to_csv(TD + 'Table3_parameters.csv', index=False)
pd.DataFrame([
    ['V1', 'Are known column classes recovered?', 'GMP published file'],
    ['V2', 'Are false positives controlled?', '1000 GMP nulls; AR(1) nulls n=3000; 1000 per NASA cell and phi'],
    ['V3', 'Does self-surrogate power match true power?', '500 replicates per rho (GMP); 400 per rho (NASA)'],
    ['V4', 'Are real signals and restatements found?', '4 NASA cells'],
    ['V5', 'Robust to settings, thresholds, p-resolution?', 'knots x radius; threshold sweeps on 378 cells; circular-shift re-test'],
    ['V6', 'What do the classes mean across many cells?', '374 public cells; within-dataset LOCO'],
    ['V7', 'Comparison with established tests', '10 planted types x 360 replicates on 12 targets; 6 competitors'],
    ['V8', 'Do the classes change modelling and explanation?', 'NASA LOCO, SHAP; GMP null band'],
], columns=['Exp.', 'Question', 'Data and replication']).to_csv(TD + 'Table4_experiments.csv', index=False)

# ------------------------------------------------------------------ V1 GMP
G = rd('out_audit_gmp.csv'); GA = js('out_audit_gmp_attrs.json')
H('V1. Recovery of the known structure of the GMP dataset')
item('Class D: ' + ', '.join(G[G['class'] == 'D'].feature), 'out_audit_gmp.csv')
item(f"Capacity determinism index eta = {GA['eta_target']:.2e}", 'out_audit_gmp_attrs.json')
item(f"Index-only ceiling R2 = {GA['ceiling']['R2_index']:.3f}, RMSE = {GA['ceiling']['RMSE_index']:.1e} Ah", 'out_audit_gmp_attrs.json')
item(f"Operating channels classed N*: {(G['class'] == 'N*').sum()} of {len(G) - (G['class'] == 'D').sum()}", 'out_audit_gmp.csv')
# index-only RUL: power law fitted to the first H cycles, end of life at 70% SOH (compared with the companion CMC study)
from scipy.optimize import curve_fit, brentq
_d = pd.read_csv('data/GMP.csv'); _k = _d.CycleNumber.values.astype(float); _y = _d['Capacity(Ah)'].values; _R = _d['RUL(cycles)'].values
_pw = lambda k, a, b, c: a - b * k ** c
for _h in [500, 1000, 1500]:
    _p, _ = curve_fit(_pw, _k[:_h], _y[:_h], p0=[_y[0], 1e-3, 1.2], maxfev=20000)
    _eol = brentq(lambda t: _pw(t, *_p) - 0.7 * _y[0], 1, 1e5)
    _rm = float(np.sqrt(np.mean((_R[_h:] - (_eol - _k[_h:])) ** 2)))
    item(f"Index-only RUL from the first {_h} cycles: predicted end of life {_eol:.1f}, RUL RMSE {_rm:.3f} cycles (CMC best LSTM: 25.59 cycles)", 'data/GMP.csv')

# ------------------------------------------------------------------ V2 false positives
V = js('out_validation.json'); TG = rd('out_typeI_gmp.csv'); AR = rd('out_typeI_ar.csv'); CN = rd('out_typeI_cond_nasa.csv')
fw = open(O + 'out_typeI_fwer.txt').read().split(','); LB = rd('out_lb_gate.csv'); D = js('out_diagnostics.json')
H('V2. False-positive control')
item(f"Trend test on 1000 GMP nulls: per-channel {rng(V['V2a_trend_FPR'])}; Holm FWER {V['V2a_trend_FWER']:.3f}", 'out_validation.json')
item(f"Time-shift test on GMP nulls: per-channel {rng(TG.FPR_TTS)}; FWER {float(fw[0]):.3f}", 'out_typeI_gmp.csv, out_typeI_fwer.txt')
item('AR(1), n = 3000: ' + '; '.join(f'phi {r.phi}: shift {r.FPR_TTS:.3f} vs naive {r.FPR_naive:.3f}' for _, r in AR.iterrows()), 'out_typeI_ar.csv')
item(f"NASA residual + AR(1) features: shift {rng(CN.FPR_TTS)}; naive at phi >= 0.9 {rng(CN[CN.phi >= 0.9].FPR_naive)}", 'out_typeI_cond_nasa.csv')
item('Ljung-Box gate rejection: ' + '; '.join(f'n={int(r.n)}, phi={r.phi}: {r.gate_rejects:.3f}' for _, r in LB.iterrows()), 'out_lb_gate.csv')
item(f"Modified Mann-Kendall size: {D['MK_HR_size_phi0.9']:.2f} (phi 0.9), {D['MK_HR_size_phi0.99']:.2f} (phi 0.99)", 'out_diagnostics.json')

# ------------------------------------------------------------------ V3 power
MS = rd('out_mde_surface.csv')
H('V3. Self-surrogate sensitivity')
item(f"Trend MDE80 family of 12: generator {V['MDE80_trend_generator_fam']:.3f}, self-surrogate {V['MDE80_trend_self-surrogate_fam']:.3f}; "
     f"per test {V['MDE80_trend_generator_test']:.3f} / {V['MDE80_trend_self-surrogate_test']:.3f}", 'out_validation.json')
item('NASA beyond-index MDE80 per test: ' + ', '.join(f"B{b}: {V[f'MDE80_cond_B{b}_test']:.2f}" for b in [5, 6, 7, 18]), 'out_validation.json')
for n in [150, 2000]:
    item(f"Simulated MDE80 at n = {n}: {rng(MS[MS.n == n].MDE80, 2)} (phi 0-0.95)", 'out_mde_surface.csv')
item(f"n = 100: MDE80 resolvable = {MS[MS.n == 100].MDE80.notna().any()}", 'out_mde_surface.csv')

# ------------------------------------------------------------------ V4 NASA
A = rd('out_audit_nasa.csv')
def cell(r):
    if r['class'] == 'P': return f"P ({abs(r.r_resid):.3f})"
    if r['class'] in ('I', 'T', 'N'): return f"{r['class']} ({r.S_cond:.3f}; {r.p_cond_holm:.3f})"
    return r['class']
card = A.assign(v=A.apply(cell, axis=1)).pivot(index='feature', columns='battery', values='v')
meta = A.groupby('battery').agg(n=('r', 'size'), r=('r', 'first'), eta=('eta_target', 'first'), R2=('R2_index', 'first'))
card.to_csv(TD + 'Table5_nasa_audit_card.csv')
H('V4. Measured cells: NASA (Table 5)')
L.append('\n' + card.to_markdown() + '\n')
item(f"Target eta {rng(meta.eta)}; index-only R2 {rng(meta.R2, 2)}; r per cell {', '.join(map(str, meta.r))}", 'out_audit_nasa.csv')
for c in ['P', 'I']:
    item(f"Class {c} counts per feature: " + ', '.join(f'{k} {v}/4' for k, v in A[A['class'] == c].feature.value_counts().items()), 'out_audit_nasa.csv')

# ------------------------------------------------------------------ V5 robustness
RB = rd('out_robust_nasa.csv'); TS = rd('out_threshold_sweep.csv'); ZS = rd('out_zeta_sweep.csv')
EM = rd('out_target_eta_margin.csv'); PF = rd('out_pfloor_nasa.csv'); E = rd('out_ext_audit.csv')
H('V5. Robustness')
res = RB[RB.r.astype(str) != '0.25']
item(f"Agreement with the default audit over knots x radius (excluding r = n/4): {res.agree.mean() * 100:.1f}% of {len(res)} verdicts; "
     f"D and P agreement {res[res.base.isin(['D', 'P'])].agree.mean() * 100:.0f}%; with r = n/4: {RB[RB.r.astype(str) == '0.25'].agree.mean() * 100:.1f}%", 'out_robust_nasa.csv')
item(f"Threshold sweep (eta x proxy) max share changed {TS.changed.max() * 100:.2f}%; within proxy 0.95-0.995 {TS[TS.proxy_thr >= 0.95].changed.max() * 100:.2f}%", 'out_threshold_sweep.csv')
item('zeta sweep: ' + '; '.join(f'{r.zeta_thr:g}: changed {r.changed * 100:.2f}%, P public {int(r.n_P_public)}' for _, r in ZS.iterrows()), 'out_zeta_sweep.csv')
item('Cells whose target would be deterministic: ' + '; '.join(f'{r.eta_det:g}: {int(r.cells_target_deterministic)}/{int(r.of)}' for _, r in EM.iterrows()), 'out_target_eta_margin.csv')
item(f"Smallest public target eta {E.drop_duplicates('cell').eta_target.min():.4f}; max |r_res| of a non-P public feature {E[E['class'] != 'P'].r_resid.abs().max():.3f}", 'out_ext_audit.csv')
I_ = PF[PF.cls == 'I']
item(f"p-floor: {int(I_.at_floor.sum())} of {len(I_)} class-I verdicts at the floor ({', '.join(f'{v:.3f}' for v in sorted(I_.p_floor_holm.unique()))}); "
     f"circular-shift Holm p {rng(I_.p_circ_holm)}; confirmed {int((I_.p_circ_holm <= 0.05).sum())}/{len(I_)}", 'out_pfloor_nasa.csv')
T_ = PF[PF.cls == 'T']
item(f"Class-T verdicts significant under the circular-shift test: {int((T_.p_circ_holm <= 0.05).sum())} of {len(T_)} ("
     + ', '.join(f'B{int(r.battery):04d} {r.feature}' for _, r in T_[T_.p_circ_holm <= 0.05].iterrows()) + ')', 'out_pfloor_nasa.csv')

# ------------------------------------------------------------------ V6 external
LX = rd('out_ext_loco_summary.csv')
H('V6. External validation on public cells (Fig. 7)')
item(f"{E.cell.nunique()} cells: " + ', '.join(f'{k} {v}' for k, v in E.groupby('dataset').cell.nunique().items()) +
     f"; cycles {int(E.n.min())}-{int(E.n.max())}; {len(E)} verdicts", 'out_ext_audit.csv')
item('Class shares: ' + ', '.join(f'{k} {v * 100:.1f}%' for k, v in E['class'].value_counts(normalize=True).items()), 'out_ext_audit.csv')
item(f"Cells with >= 1 class-I feature: {E.groupby('cell')['class'].apply(lambda c: (c == 'I').any()).mean() * 100:.0f}%", 'out_ext_audit.csv')
sh = E.pivot_table(index='feature', columns='class', values='cell', aggfunc='count', fill_value=0); sh = (sh.T / sh.sum(axis=1)).T
sh.round(3).to_csv(TD + 'Fig7a_class_share.csv')
item('Highest I shares: ' + ', '.join(f'{k} {v * 100:.0f}%' for k, v in sh['I'].sort_values(ascending=False).head(4).items()), 'out_ext_audit.csv')
piv = LX.pivot(index='dataset', columns='set', values='median_rmse_pct').round(2); piv.to_csv(TD + 'Fig7b_loco_rmse_pct.csv')
L.append('\n' + piv.to_markdown() + '\n')
fold = piv['index only'] / piv['index + T']
item(f"Error reduction index-only -> index + T: {fold.min():.1f}- to {fold.max():.0f}-fold", 'out_ext_loco_summary.csv')

# ------------------------------------------------------------------ V7 benchmark
BT = rd('out_bench2_table.csv', index_col=0); BS = js('out_bench2_summary.json'); SW = rd('out_bench2_sweep.csv'); PX = rd('out_bench2_proxy.csv')
H('V7. Benchmark (Fig. 8, Table S3)')
L.append('\n' + BT.round(3).to_markdown() + '\n')
NI = ['trend_hi', 'null_ar', 'null_trend']
for m in ['M1_corr', 'M3_detrended', 'M5_prewhiten', 'M6_dcca', 'M7_hac', 'M4v1_TRACE', 'M4_TRACE']:
    s = BS[m]; item(f"{m}: false credit on trend/null types {rng(BT.loc[NI, m], 3)}; precision {s['precision']:.3f}; recall design {s['recall_design']:.3f}, "
                    f"held-out {s['recall_heldout']:.3f}; proxy credited {BT.loc['proxy', m]:.2f}", 'out_bench2_table.csv, out_bench2_summary.json')
item(f"TRACE v2 leak detection {BS['TRACE_leak_detection']:.2f}; v1 {BS['TRACE_v1_leak_detection']:.2f}; replicates {BS['replicates']}", 'out_bench2_summary.json')
sw = SW.groupby(['feature', 'c_lag'])[['M5_prewhiten', 'M7_hac', 'M4v1_TRACE', 'M4_TRACE']].mean().round(2)
sw.to_csv(TD + 'Fig8c_strength_sweep.csv'); L.append('\n' + sw.to_markdown() + '\n')
px = pd.crosstab(PX.noise, PX.cls, normalize='index').round(2); px.to_csv(TD + 'proxy_noise_sweep.csv')
L.append('\nProxy-noise sweep (share of classes by noise level, fraction of SD(y)):\n\n' + px.to_markdown() + '\n')

# ------------------------------------------------------------------ V8 consequences
LO = rd('out_loco.csv'); C8 = js('out_consequence.json'); SN = rd('out_shap_nasa.csv')
H('V8. Consequences for modelling and explanation (Fig. 9, Table S1)')
m = LO[LO.model == 'GBR'].groupby('feature_set').RMSE.mean() * 1000
item('Mean GBR LOCO RMSE (mAh, % of 2 Ah): ' + '; '.join(f'{k} {v:.1f} ({v / 20:.1f}%)' for k, v in m.items()), 'out_loco.csv')
b7 = LO[(LO.model == 'GBR') & (LO.held_out == 'B0007')].set_index('feature_set').R2
item(f"B0007 R2: index only {b7['Index only']:.2f} -> TRACE-admitted {b7['TRACE-admitted (I)']:.2f}", 'out_loco.csv')
s = SN.groupby(['feature_set', 'feature']).mean_abs_shap.mean() * 1000
item('Mean |SHAP| (mAh), all features: ' + ', '.join(f'{k} {v:.1f}' for k, v in s['All features'].sort_values(ascending=False).head(3).items()), 'out_shap_nasa.csv')
item(f"Mean |SHAP| of V_600s after removing D and P: {s['TRACE-admitted (I)']['V_600s']:.1f} mAh", 'out_shap_nasa.csv')
item(f"GMP random split R2 {C8['gmp_r2_random_split']:.3f}; ops |SHAP| {C8['gmp_shap_ops_min']:.1e}-{C8['gmp_shap_ops_max']:.1e}; "
     f"Mann-Whitney p {C8['gmp_shap_mw_p']:.2f}; chronological ops-only R2 {C8['gmp_r2_chrono_ops_only']:.1f}", 'out_consequence.json')
LO.to_csv(TD + 'TableS1_nasa_loco.csv', index=False)
BT.to_csv(TD + 'TableS3_benchmark.csv'); TS.to_csv(TD + 'TableS4a_threshold_sweep.csv', index=False)
ZS.to_csv(TD + 'TableS4b_zeta_sweep.csv', index=False); EM.to_csv(TD + 'TableS4c_target_eta_margin.csv', index=False); PF.to_csv(TD + 'TableS4d_pfloor.csv', index=False)
pd.concat([LB.assign(test='Ljung-Box gate'), pd.DataFrame([dict(test='modified Mann-Kendall', n=3000, phi=0.9, gate_rejects=D['MK_HR_size_phi0.9']),
                                                            dict(test='modified Mann-Kendall', n=3000, phi=0.99, gate_rejects=D['MK_HR_size_phi0.99'])])]
          ).rename(columns={'gate_rejects': 'rejection_rate'}).to_csv(TD + 'TableS2_test_size.csv', index=False)

open(O + 'results_numbers.md', 'w').write('# TRACE - numbers quoted in the Method validation chapter\n' + '\n'.join(L) + '\n')
print('\n'.join(L)); print('\nwrote outputs/results_numbers.md and outputs/tables/')
