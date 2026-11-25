#!/usr/bin/env bash
# Reproduce every number, table and figure of the TRACE MethodsX article, its supplementary material,
# and the companion Results in Engineering research article (steps 17-22, figures/rineng_*).
# Full run ~60-90 min on 2 CPU cores; TRACE_QUICK=1 gives a short smoke test with fewer replicates.
# 11_external clones the public PINN4SOH repository (XJTU, TJU, HUST, MIT cells) at a pinned commit (needs git).
set -e
export PYTHONPATH=.
for s in 00_data 01_audits 02_typeI 03_validation 04_ljungbox_mk 05_robustness 06_consequences 07_landscape \
         11_external 12_benchmark_fair 13_sensitivity 08_audit_space 10_figures \
         extra_methodology_figure extra_graphical_abstract 14_report_numbers 15_supp_figures 16_tables_docx \
         17_rineng_analyses 19_rineng_rq4 20_rineng_robustness 21_rineng_extra_figures 22_rineng_explain 18_rineng_fig1; do
  echo "=== $s ==="; python $s.py
done
