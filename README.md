# Electrical load classification

Predict the supplied **R, I, RC or RLC** label from a complete timestamped
electrical recording. The new data replaces the previous simulated workbook
and its CC/CP/CZ prediction task. That work remains in Git history.

The dataset contains **209,554 windows from 1,074 recordings in 55 houses**.
Each window covers 500 samples at 30,000 Hz, or about 16.67 ms. The supplied
JSON provides labels and house identifiers. The Drive CSV provides already
computed electrical measurements; raw waveforms and the feature extractor
were not supplied with these files.

## Model and evaluation

The selected model is **Extra Trees with 400 trees**, balanced class weights
and all available features considered at each split. Selection uses mean
macro-F1 across four validation folds containing separate houses. Candidates
include a majority baseline, logistic regression, two SVM settings, a random
forest, gradient boosting, a neural network and Extra Trees settings.
The initial comparison was followed by three additional Extra Trees variants,
using development results only. No test score was used for this choice.

The selected setting achieved **92.87% mean validation accuracy** and **0.9131
mean validation macro-F1**. These are validation results. Final held-out test
results are written to `runs/timestamped/report.md` by `evaluate`.

- Development: 859 recordings from 43 houses, with four grouped validation folds.
- Test: 215 recordings from 12 other houses, reserved before model comparison.
- One prediction per recording. All windows from a recording stay together.
- No house appears in both development and test, or in multiple validation folds.
- Imputation and scaling are fitted inside each training fold.
- The selected pipeline is finally fitted on all 859 development recordings.

## How time is used

For each of 23 electrical measurements, the code calculates eight summaries:
10th percentile, median, 90th percentile, interquartile range, early median,
late median, slope per second and typical adjacent change per second. Early and
late refer to the first and last 20% of elapsed time. This produces **184 inputs**
per recording. The full sequence is required at inference; this is an offline
classifier, not a real-time or future-load forecasting model.

Inputs include current, voltage, real/apparent/nonactive power, power factor,
phase sine/cosine, waveform shape and harmonics. ID, label, appliance name,
house, collection date, window count and recording duration are excluded.
Raw timestamps and sample/window indices are used for ordering and rate
calculations, rather than directly as predictors. Equivalent-component
estimates and algebraically redundant derived columns are omitted.

The median-only ablation for the selected setting scored **0.8993 validation
macro-F1**, compared with **0.9131** for the full summaries. This comparison
changes both distribution and time summaries; it does not isolate a causal
benefit from timestamps alone. The default Extra Trees setting had a stronger
median-only result, so time summaries do not help every classifier setting.

## Run

Use Python 3.12:

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest -q
```

Prepared summaries are included, so training does not require the large CSV.
Use a new run directory for each experiment:

```sh
python -m load_classifier compare --run runs/reproduction
python -m load_classifier train --run runs/reproduction
python -m load_classifier evaluate --run runs/reproduction
python -m load_classifier verify --run runs/reproduction
```

`compare` refuses a populated directory, `train` refuses an existing trained
model, and `evaluate` refuses an existing test result. `verify` replays inference
and recounts saved answers without training or changing model selection. For
future improvements, use development folds and keep test results out of model
selection. After substantial development, obtain a new external test set rather
than repeatedly optimizing against the existing test scores.

To rebuild summaries from the exact Drive source:

```sh
python scripts/download_data.py
python -m load_classifier prepare --output /tmp/ctee-prepared
```

The downloader checks the complete 214,166,054-byte CSV against its SHA-256.
The raw CSV stays in ignored `data/raw/` because it exceeds GitHub's regular
file-size limit. The JSON and compact prepared summaries are tracked. Drive
access is needed only when downloading the original windows.

## Predict

Provide complete recordings with `id`, the five time/sample columns and the
23 measurement columns listed in `data/processed/audit.json`. Labels, appliance
names and house columns are optional and ignored during prediction.

```sh
python -m load_classifier predict --input new_recordings.csv --output predictions.csv
```

The output has one row per recording, its predicted class and four probabilities.
These probabilities have not been independently calibrated. Load only trusted
model files: joblib uses pickle and must not open arbitrary model uploads.

## Layout

```text
data/meta_labeled.json       Supplied recording descriptions and labels
data/source.json             Drive source and raw-file fingerprint
data/processed/              Recording summaries, fixed splits and audit
load_classifier/data.py     Input validation and house splits
load_classifier/features.py Timestamp-aware recording summaries
load_classifier/models.py   Declared classifier settings
load_classifier/experiment.py Comparison and final training
load_classifier/evaluate.py Test evaluation, plots, report and verification
load_classifier/predict.py  Saved-model inference on new feature CSVs
runs/timestamped/            Comparison, selected model and results
scripts/download_data.py     Verified download of the source CSV
tests/                      Time arithmetic, leakage guards and pipeline checks
```

## Scope

**I is kept as written**, not silently renamed to L. There are no CC/CP/CZ
labels, so those outputs cannot be trained or evaluated from this dataset.
Each appliance type maps to one supplied load label. Agreement with those
annotations does not independently establish a physical equivalent circuit.

All collection dates are `July, 2013`, so calendar date cannot distinguish the
classes. The metadata warns that some recordings are not well calibrated.
Testing checks new houses within this collection; it does not establish
performance for unseen appliance categories, countries, simultaneous loads
or arbitrary electrical conditions. Old workbook scores refer to a different
dataset and task and should not be compared directly.

Algorithm and evaluation background: [Extra Trees documentation](https://scikit-learn.org/stable/modules/ensemble.html#extremely-randomized-trees)
and [grouped cross-validation](https://scikit-learn.org/stable/modules/cross_validation.html#stratifiedgroupkfold).
See `data/README.md` for provenance and source checks.
