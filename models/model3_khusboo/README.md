# Model 3 — Khusboo building load profiles

This folder is an independent experiment. It adds no changes to Model 1 or Model 2 files. Branch: `codex/model-3-khusboo-buildings`, based on Model 1 commit `d2b3da873a3c350ad11ff19a750469d37ab10bd9`.

**Task:** recognize Academic, Facilities or Mess from a completed day's power and power-factor measurements. Each class is one specific building/meter. This is not a classifier of physical R/I/RC/RLC load types, and it does not establish performance on new buildings.

The selected classifier is an **RBF support vector machine**, using 69 daily features, training-only median imputation and standard scaling. Three expanding chronological validation folds selected it from five fixed candidates. The final test starts on **10 February 2017**, with a seven-day gap before it. All buildings on a date stay in the same partition.

Final test: **85.07% accuracy, 0.8517 macro F1, 792 correct out of 931 days**. Mean validation macro F1 was 0.9889. Performance deteriorates later in the test year, reaching **53.76% in December 2017**. This limitation is included in the report and monthly chart. The test was not used to tune or replace the model.

Read [the professor report](outputs/model3/report.md) for the simple explanation, per-class results, figures and limitations. [The Excel workbook](outputs/model3/chart_data.xlsx) contains editable charts and their underlying values. [Verification results](results/verification.json) record checks on splits, transformations, model serialization and metrics.

## Contents

- `artifacts/model.joblib`: saved pipeline plus ordered feature names and environment metadata.
- `artifacts/preprocessing.joblib`: separately saved imputation and scaling.
- `results/`: archive audit/hash, daily features, quality decisions, split and CV assignments, all candidate scores, predictions, metrics, uncertainty intervals, feature definitions and chart source CSVs.
- `outputs/model3/`: report, five PNG figures and Excel workbook.
- `inspect_dataset.py`, `pipeline.py`, `predict.py`: audit, feature extraction/training and inference.
- `test_pipeline.py`, `verify_results.py`, `verify_inference.py`, `verify_workbook.py`: feature/split tests, frozen-result checks, a raw-CSV inference check and exported-workbook checks.
- `summarize_results.py`, `build_workbook.mjs`: reproducible report/chart and Excel builders.

The original 148 MB ZIP remains on the user's computer and is not uploaded to Git. Its SHA-256 is recorded in `results/dataset_audit.json`. The original Readme describes a larger dataset than the subset actually supplied. Only the supplied three meter files are used. The status table is used to mask unavailable measurements. Calendar files are audited but are not predictors. Attached document instructions are not executed.

## Reproduce in an isolated environment

Use Python 3.13 and the versions in this folder's `requirements.txt`, independently of Model 1's environment. Run commands from the repository root. Keep the ZIP at an accessible local path.

```powershell
python -m venv models/model3_khusboo/.venv
models/model3_khusboo/.venv/Scripts/python -m pip install -r models/model3_khusboo/requirements.txt
models/model3_khusboo/.venv/Scripts/python -m unittest discover -s models/model3_khusboo -p test_pipeline.py -v
```

The committed final experiment is frozen. `pipeline.py` refuses to train again while `results/test_metrics.json` exists. For a deliberate reproduction, copy this folder to a new experiment directory, omit its generated `results/`, `artifacts/`, `outputs/`, `node_modules/` and `.venv/`, then audit and train there. Keep this committed experiment unchanged. For example, inside that fresh copy:

```powershell
python inspect_dataset.py 'C:\path\khusboo dataset .zip'
python pipeline.py 'C:\path\khusboo dataset .zip'
python verify_results.py
python summarize_results.py
```

`--use-features` reuses the existing `daily_features.csv` instead of re-reading the ZIP. It does not bypass the final-test guard. Extraction and temporal rules are deterministic. Fit runtimes can vary. Raw data is not required for inference from an already extracted daily feature table.

The workbook builder uses `@oai/artifact-tool` from the bundled document runtime. Link its `node_modules` to an ignored `node_modules` folder here and run the bundled Node executable with `build_workbook.mjs`. Its checks reconcile Excel formulas with Python metrics and test that an edited prediction updates accuracy before restoring the original value.

## Classify new daily profiles

From the repository root, using an environment with the pinned dependencies:

```powershell
python models/model3_khusboo/predict.py 'C:\path\measurements.csv' --output 'C:\path\daily_predictions.csv'
```

The CSV must contain Unix-second `timestamp`, `power` in watts, and signed `power_factor`. It should contain measurements from a single meter, with one row per minute. Filename and meter name do not enter the model. If an external availability mask is needed, remove/mask those readings before prediction, as in training. The command returns daily predictions or explicit insufficient-coverage rejections. Only load trusted local joblib files. Unseen meters are outside the measured scope of this experiment.

## Checks

```powershell
python -m unittest discover -s models/model3_khusboo -p test_pipeline.py -v
python models/model3_khusboo/verify_results.py
python models/model3_khusboo/verify_workbook.py
python models/model3_khusboo/verify_inference.py 'C:\path\khusboo dataset .zip'
```

The checks verify known-signal feature calculations, local dates, the 80% quality boundary, gaps, duplicate timestamps, invalid values, no duplicate profiles across partitions, development-only preprocessing, metric reconciliation and saved-model predictions. They cannot guarantee performance under future changes in equipment or use.
