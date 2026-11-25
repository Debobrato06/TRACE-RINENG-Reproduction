"""Builds TRACE_RINENG_Colab.ipynb: self-contained Google Colab notebook for the Results in Engineering article."""
import nbformat as nbf
nb = nbf.v4.new_notebook(); cells = []
md = nbf.v4.new_markdown_cell; code = nbf.v4.new_code_cell
cells.append(md("""# What can one battery ageing record support? — Google Colab reproduction
**Results in Engineering research article** (co-submitted with the TRACE method article in *MethodsX*).
Running all cells regenerates every number, table and figure of the article (Figs. 1–8, Tables 4–13):
index-only ceiling, NASA engineering reading, feature atlas and 3-D landscape, cross-cell transfer with robustness
controls, identifiability vs transfer, and SHAP explainability / explanation reliability.

* **GMP LiFePO₄ dataset** (Mendeley Data, doi:10.17632/k675pxrd83.2): upload `GMP_LFP_40Ah_3000Cycle.csv` in Step 1, or skip to regenerate it from the published equations.
* **NASA PCoE cells B0005/6/7/18**: downloaded automatically.
* **374 public cells (XJTU, TJU, HUST, MIT)**: cloned automatically from github.com/wang-fujin/PINN4SOH (commit cc3cc50).
* `QUICK = True` gives a short smoke test; `False` reproduces the article (about 1.5–2 h on a free Colab CPU)."""))
cells.append(code("""QUICK = False   # True = fast smoke test; False = exact article numbers
import os; os.environ['TRACE_QUICK'] = '1' if QUICK else '0'; os.environ['PYTHONPATH'] = '.'
!pip -q install pyreadr==0.5.9 pymannkendall==1.4.3 shap statsmodels scikit-learn joblib
!mkdir -p data outputs figures"""))
cells.append(md("## Step 1 – (optional) upload the published GMP CSV"))
cells.append(code("""try:
    from google.colab import files
    up = files.upload()
    for name in up:
        os.replace(name, 'data/GMP.csv'); print('saved data/GMP.csv')
except Exception as e:
    print('No upload (', e, ') - the dataset will be regenerated from the published equations.')"""))
cells.append(md("## Step 2 – the TRACE audit module"))
cells.append(code("%%writefile trace_audit.py\n" + open('trace_audit.py').read()))
steps = [('00_data', 'Step 3 – data preparation: GMP record, NASA discharge features (methods Step 1)'),
         ('01_audits', 'Step 4 – audit of the GMP record and the NASA cells (methods Step 2)'),
         ('11_external', 'Step 5 – audit of the 374 public cells (methods Step 2)'),
         ('17_rineng_analyses', 'Step 6 – index-only ceiling, feature atlas, family transfer (methods Steps 3, 5, 6; Figs. 2, 4, 6)'),
         ('19_rineng_rq4', 'Step 7 – identifiability vs transfer (methods Step 7; Fig. 7)'),
         ('20_rineng_robustness', 'Step 8 – robustness: training horizon, equal-size families, random forest (Tables 6, 11)'),
         ('21_rineng_extra_figures', 'Step 9 – NASA engineering reading (Fig. 3) and 3-D landscape (Fig. 5) (methods Steps 4, 5)'),
         ('22_rineng_explain', 'Step 10 – SHAP explainability and explanation reliability (methods Step 8; Fig. 8)'),
         ('18_rineng_fig1', 'Step 11 – methodology figure (Fig. 1)')]
for s, title in steps:
    cells.append(md('## ' + title))
    cells.append(code(f"%%writefile {s}.py\n" + open(f'{s}.py').read()))
    cells.append(code(f"!PYTHONPATH=. python {s}.py"))
cells.append(md("## Step 12 – view figures and key numbers"))
cells.append(code("""from IPython.display import Image, display
import glob, json
for f in sorted(glob.glob('figures/rineng_*.png')): print(f); display(Image(f, width=900))
for j in ['rineng_results.json', 'rineng_rq4.json', 'rineng_robust.json', 'rineng_xai.json']:
    print(j); print(json.dumps(json.load(open('outputs/' + j)), indent=1)[:3000])"""))
cells.append(code("!zip -qr TRACE_RINENG_outputs.zip outputs figures && echo 'download TRACE_RINENG_outputs.zip from the Files pane'"))
nb['cells'] = cells
nb['metadata'] = {'kernelspec': {'name': 'python3', 'display_name': 'Python 3'}, 'language_info': {'name': 'python'}, 'colab': {'provenance': []}}
nbf.write(nb, 'TRACE_RINENG_Colab.ipynb'); print('notebook written')
