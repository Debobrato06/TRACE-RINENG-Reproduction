# TRACE — Trajectory-Referenced Audit of Cycle-level Evidence
Code that reproduces, from the raw data, every number, table and figure (Figs. 1–10, graphical abstract) of the MethodsX article and its supplementary material (Tables S1–S4, Figs. S1–S4).

**TRACE v2** (trace_audit.py) is the version reported in the article: beyond-index statistic = max of |Spearman| with the target residual level and with its first difference (rate), lag 0; restatement index zeta < 0.01 added to the proxy check. The original version is kept as trace_audit_v1.py and is compared in the benchmark (V7).

## Option A: Google Colab (recommended)
Open `TRACE_reproduce_Colab.ipynb` in Colab → Runtime → Run all.
* `QUICK = True`  → ~3-5 min smoke test with fewer Monte-Carlo replicates.
* `QUICK = False` → exact paper numbers (~60-90 min on a free CPU runtime).
In Step 1 you may upload `GMP_LFP_40Ah_3000Cycle.csv` (Mendeley, doi:10.17632/k675pxrd83.2). If you skip it, the dataset is regenerated
from the published generating equations (Data in Brief 69 (2026) 113252, seed 42); both routes give identical audit results.
The NASA PCoE data (B0005/6/7/18) are downloaded automatically, and the 374 public cells (XJTU, TJU, HUST, MIT) are cloned from github.com/wang-fujin/PINN4SOH at commit cc3cc5053caf38f04e0665f7f88cb109144d035e (requires git).

## Option B: local Python
    pip install -r requirements.txt
    bash run_all.sh                 # exact paper numbers
    TRACE_QUICK=1 bash run_all.sh   # quick smoke test
Optional: put the Mendeley CSV at data/GMP.csv first.

## Pipeline
| Script | Produces | Paper |
|---|---|---|
| trace_audit.py | the TRACE method (S1-S4, run_audit) | Method details |
| 00_data.py | data/GMP.csv (verified/regenerated), data/nasa_cycle_features.csv | Validation design |
| 01_audits.py | outputs/out_audit_gmp.csv, out_audit_nasa.csv | V1, V4, Table 5 |
| 02_typeI.py | outputs/out_typeI_ar.csv | V2, Fig. 3b |
| 03_validation.py | out_validation.json, power and FPR tables | V2, V3, Figs. 3-4 |
| 04_ljungbox_mk.py | out_diagnostics.json | text (Ljung-Box, Mann-Kendall) |
| 05_robustness.py | out_robust_nasa.csv | V5 |
| 06_consequences.py | out_loco.csv, out_shap_*.csv, out_consequence.json | V8, Fig. 9, Table S1 |
| 07_landscape.py | out_mde_surface.csv, out_null_pvalues.csv, out_shift_landscape_B0005.json | Fig. 5 |
| 11_external.py | out_ext_audit.csv, out_ext_class_*.csv, out_ext_loco*.csv | V6, Fig. 7 |
| 12_benchmark_fair.py | out_bench2_raw/table/summary, out_bench2_sweep.csv, out_bench2_proxy.csv | V7, Fig. 8, Table S3 |
| 13_sensitivity.py | out_pfloor_nasa.csv, out_threshold_sweep.csv, out_zeta_sweep.csv, out_target_eta_margin.csv, out_lb_gate.csv | V5, V2, Tables S2, S4 |
| 08_audit_space.py (after 11) | out_audit_space.csv | Fig. 10a |
| 10_figures.py | figures/fig2 ... fig10 | Figs. 2-10 |
| extra_methodology_figure.py | figures/fig1_methodology.png (schematic with data icons) | Fig. 1 |
| extra_graphical_abstract.py | figures/graphical_abstract.png (built from the result figures) | Graphical abstract |
| 14_report_numbers.py | outputs/tables/*.csv (Tables 2-5, S1-S4, figure data), outputs/results_numbers.md (every quoted number with its source) | Method details, Method validation |
| 15_supp_figures.py | figures/figS1 ... figS4 | Supplementary Figs. S1-S4 |
| 16_tables_docx.py | outputs/TRACE_main_tables.docx, outputs/TRACE_Supplementary_Material.docx | Tables 2-5; Supplementary Tables S1-S4 and Figs. S1-S4 |

All random seeds are fixed; the pipeline is deterministic.

## Audit your own data
    import pandas as pd, trace_audit as T
    df = pd.read_csv("my_cell.csv")          # one row per cycle
    card = T.run_audit(df, index_col="cycle", target_col="capacity", mde=True,
                       feature_cols=[c for c in df if c not in ("cycle", "capacity")])
    print(card[["feature", "class", "eta", "r_resid", "zeta", "S_cond", "p_cond_holm", "MDE80"]])
Classes: D derived, P target restatement, I informative beyond the index (same cycle, level or rate),
T trend-coupled only (not identifiable on one record; may still transfer across cells), N null (report MDE80), N* untestable.

Versions used for the published outputs are pinned in requirements.txt (Python 3.13).

## Results in Engineering research article (co-submitted with the MethodsX method article)

Run after steps 00, 01 and 11 (all in `run_all.sh`), in this order. `TRACE_RINENG_Colab.ipynb` runs exactly these steps in Google Colab.

| Script | Produces | Article item |
|---|---|---|
| `17_rineng_analyses.py` | `rineng_results.json`, `rineng_atlas_*.csv`, `rineng_family_loco*.csv` | Steps 3, 5, 6; Figs. 2, 4, 6; Tables 5, 8, 10 |
| `19_rineng_rq4.py` | `rineng_rq4.json`, `rineng_rq4_pairs.csv` | Step 7; Fig. 7; Table 12 |
| `20_rineng_robustness.py` | `rineng_robust*.csv/json` | Steps 3, 6 controls; Tables 6, 11 |
| `21_rineng_extra_figures.py` | `rineng_nasa_residual_rho.csv`, `rineng_landscape_rho.csv` | Steps 4, 5; Figs. 3, 5 (3-D) |
| `22_rineng_explain.py` | `rineng_xai*.csv/json` | Step 8; Fig. 8; Table 13 |
| `18_rineng_fig1.py` | `rineng_fig1_study_design.png` | Fig. 1 |

`FIG_ONLY=1` redraws the figures of steps 17 and 22 from saved outputs.
