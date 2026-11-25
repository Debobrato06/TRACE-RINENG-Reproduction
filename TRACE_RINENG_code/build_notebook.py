import nbformat as nbf
nb = nbf.v4.new_notebook(); cells = []
md = nbf.v4.new_markdown_cell; code = nbf.v4.new_code_cell
cells.append(md("""# TRACE — Trajectory-Referenced Audit of Cycle-level Evidence
**Reproducibility notebook for the MethodsX article.** Running all cells regenerates every number, table and figure of the Method details and Method validation chapters (Figs. 1–10, Tables 2–5, graphical abstract) and the Supplementary material (Tables S1–S4, Figs. S1–S4), plus `outputs/results_numbers.md`, which lists every quoted number with its source file.

* **Data 1 – GMP LiFePO₄ dataset** (Mendeley Data, doi:10.17632/k675pxrd83.2). Upload `GMP_LFP_40Ah_3000Cycle.csv` in Step 1 to audit the published file. If you skip the upload, the file is regenerated exactly from the published generating equations (Data in Brief 69 (2026) 113252, seed 42).
* **Data 2 – NASA PCoE cells B0005/6/7/18**, downloaded automatically.\n* **Data 3 – 374 public cells (XJTU, TJU, HUST, MIT)**, cloned automatically from github.com/wang-fujin/PINN4SOH at commit cc3cc50.
* Set `QUICK = True` for a ~5-minute smoke test with fewer Monte-Carlo replicates. Leave it `False` to reproduce the paper exactly (~60–90 min on a free Colab CPU)."""))
cells.append(code("""QUICK = False   # True = fast smoke test; False = exact paper numbers
import os; os.environ['TRACE_QUICK'] = '1' if QUICK else '0'
!pip -q install pyreadr==0.5.9 pymannkendall==1.4.3 python-docx tabulate shap statsmodels scikit-learn
!mkdir -p data outputs figures"""))
cells.append(md("## Step 1 – (optional) upload the published GMP CSV"))
cells.append(code("""try:
    from google.colab import files
    up = files.upload()            # choose GMP_LFP_40Ah_3000Cycle.csv, or press Cancel to regenerate it
    for name in up:
        os.replace(name, 'data/GMP.csv'); print('saved data/GMP.csv')
except Exception as e:
    print('No upload (', e, ') - the dataset will be regenerated from the published equations.')"""))
cells.append(md("## Step 2 – the TRACE method (single module)"))
cells.append(code("%%writefile trace_audit.py\n" + open('trace_audit.py').read()))
steps = [('00_data', 'Step 3 – data loading, generator verification and NASA feature extraction'),
         ('01_audits', 'Step 4 – TRACE audit of GMP and NASA (V1, V4; Table 5)'),
         ('02_typeI', 'Step 5 – shift test vs naive test under autocorrelation (V2, Fig. 3b)'),
         ('03_validation', 'Step 6 – false-positive control and power / MDE₈₀ (V2, V3; Figs. 3–4)'),
         ('04_ljungbox_mk', 'Step 7 – diagnostics quoted in the text (Ljung–Box size, Mann–Kendall size)'),
         ('05_robustness', 'Step 8 – robustness to knots and truncation radius (V5)'),
         ('06_consequences', 'Step 9 – NASA leave-one-cell-out and SHAP consequences (V8, Fig. 9)'),
         ('07_landscape', 'Step 10 – reliability landscape: MDE₈₀ surface, null P–P, shift landscape (Fig. 5)'),
         ('11_external', 'Step 11 – external validation: 374 public cells (XJTU, TJU, HUST, MIT; cloned from PINN4SOH at a pinned commit) and cross-cell LOCO (V6, Fig. 7)'),
         ('12_benchmark_fair', 'Step 12 – benchmark of 10 planted feature types against 6 established tests on 12 measured targets (V7, Fig. 8)'),
         ('13_sensitivity', 'Step 13 – p-value floor, threshold and ζ sweeps, Ljung–Box gate size (V5, Supplementary Tables S2, S4)'),
         ('08_audit_space', 'Step 14 – coordinates of every measured feature in the 3-D audit space (Fig. 10a)'),
         ('10_figures', 'Step 15 – all data figures (Figs. 2–10)'),
         ('extra_methodology_figure', 'Step 16 – methodology figure (Fig. 1)'),
         ('extra_graphical_abstract', 'Step 17 – graphical abstract (built from the result figures)'),
         ('14_report_numbers', 'Step 18 – Method details and Method validation tables (Tables 2–5, S1–S4) and every quoted number with its source (results_numbers.md)'),
         ('15_supp_figures', 'Step 19 – Supplementary Figures S1–S4'),
         ('16_tables_docx', 'Step 20 – Word files: main tables and Supplementary material'),
         ('17_rineng_analyses', 'Step 21 – Results in Engineering article: index-only ceiling, feature atlas, family transfer (Figs. 2, 4, 6)'),
         ('19_rineng_rq4', 'Step 22 – Results in Engineering article: identifiability vs transfer (Fig. 7)'),
         ('20_rineng_robustness', 'Step 23 – Results in Engineering article: robustness (training horizon, equal-size families, random forest)'),
         ('21_rineng_extra_figures', 'Step 24 – Results in Engineering article: NASA engineering reading (Fig. 3) and 3-D landscape (Fig. 5)'),
         ('22_rineng_explain', 'Step 25 – Results in Engineering article: SHAP explainability and explanation reliability (Fig. 8)'),
         ('18_rineng_fig1', 'Step 26 – Results in Engineering article: methodology figure (Fig. 1)')]
for s, title in steps:
    cells.append(md('## ' + title))
    cells.append(code(f"%%writefile {s}.py\n" + open(f'{s}.py').read()))
    cells.append(code(f"!python {s}.py"))
cells.append(md("## Step 27 – view the figures and tables and key outputs"))
cells.append(code("""from IPython.display import Image, display
import glob, json, pandas as pd
for f in sorted(glob.glob('figures/*.png')): print(f); display(Image(f, width=900))
print(open('outputs/results_numbers.md').read())
display(pd.read_csv('outputs/out_audit_nasa.csv')[['battery','feature','r_resid','S_cond','p_cond_holm','class']])"""))
cells.append(code("!zip -qr TRACE_outputs.zip outputs figures && echo 'download TRACE_outputs.zip from the Files pane'"))
nb['cells'] = cells
nb['metadata'] = {'kernelspec': {'name': 'python3', 'display_name': 'Python 3'}, 'language_info': {'name': 'python'}, 'colab': {'provenance': []}}
nbf.write(nb, 'TRACE_reproduce_Colab.ipynb'); print('notebook written')
