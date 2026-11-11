# Training on Kaggle (about 2 hours) - exact steps

1. Kaggle account -> open https://www.kaggle.com/c/ieee-fraud-detection -> **Join Competition** -> accept the rules (needed to use the data).
2. **Datasets -> New Dataset** -> upload `Fraud_Detection_Management.zip` (title e.g. `egt-code`, keep it Private) -> Create.
3. **Code -> New Notebook**. Right panel: *Session options -> Accelerator = None* (the code runs on CPU), *Internet = Off* is fine.
4. **Add Input** -> *Competitions* tab -> IEEE-CIS Fraud Detection -> Add. Then **Add Input** -> *Your datasets* -> `egt-code` -> Add.
5. Paste each cell below into its own notebook cell and run them in order (cell A, B first - they take about 2 minutes).
6. When A and B succeed: **Save Version -> Save & Run All (Commit)**. It runs in the background (you can close the browser) and saves `/kaggle/working`.
7. When it finishes: notebook page -> **Output** tab -> download `EGT_results.zip` (and optionally `EGT_cache.zip`). Unzip `models/` and `outputs/` into your local project.

## Cell A - set up
```python
import glob, shutil, os, subprocess, sys
mains = glob.glob('/kaggle/input/**/main.py', recursive=True)
if not mains:                                    # dataset was stored as a zip
    z = glob.glob('/kaggle/input/**/*.zip', recursive=True)[0]
    shutil.unpack_archive(z, '/kaggle/working/_unz'); mains = glob.glob('/kaggle/working/_unz/**/main.py', recursive=True)
src = os.path.dirname(mains[0])
shutil.copytree(src, '/kaggle/working/EGT', dirs_exist_ok=True)
os.chdir('/kaggle/working/EGT')
TX = glob.glob('/kaggle/input/**/train_transaction.csv', recursive=True)[0]
ID = glob.glob('/kaggle/input/**/train_identity.csv', recursive=True)[0]
print(TX); print(ID); print(os.listdir('.'))
```
## Cell B - self test (must print ALL TESTS PASSED)
```python
!python -u main.py test
```
## Cell C - real training (report + prepare + train, about 80-100 min)
```python
!python -u main.py run-all --transaction "{TX}" --identity "{ID}" --seeds 42 43 --epochs 40 --patience 6 --max-minutes 25 --no-ablation
```
## Cell D - collect results
```python
import shutil, os
os.makedirs('/kaggle/working/results', exist_ok=True)
for d in ('models', 'outputs'):
    shutil.copytree(d, f'/kaggle/working/results/{d}', dirs_exist_ok=True)
shutil.make_archive('/kaggle/working/EGT_results', 'zip', '/kaggle/working/results')
shutil.make_archive('/kaggle/working/EGT_cache', 'zip', 'data/cache')      # large; only needed for the local GUI "prepared dataset"
shutil.rmtree('/kaggle/working/EGT/data/cache', ignore_errors=True)
print(os.listdir('/kaggle/working'))
print(open('outputs/training_report.txt').read()[:3000])
```
