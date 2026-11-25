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
and half of the available features considered at each split. Selection uses mean
macro-F1 across four validation folds containing separate houses. Candidates
include a majority baseline, logistic regression, two SVM settings, a random
forest, gradient boosting, a neural network and Extra Trees settings.
The initial comparison was followed by three additional Extra Trees variants,
using development results only. No test score was used for this choice.

The selected setting achieved **92.39% mean validation accuracy** and **0.9056
mean validation macro-F1**. On the reserved test houses it achieved **93.95%
accuracy (202/215)**, **0.9275 macro-F1** and **92.50% balanced accuracy**.
The majority-class test baseline is 31.63%. See the [full test report](runs/timestamped/report.md)
and [professor explanation](docs/professor-guide.md).

| Test class | Precision | Recall | F1 | Recordings |
| --- | ---: | ---: | ---: | ---: |
| R | 1.0000 | 0.9500 | 0.9744 | 60 |
| I | 1.0000 | 0.7500 | 0.8571 | 40 |
| RC | 1.0000 | 1.0000 | 1.0000 | 68 |
| RLC | 0.7833 | 1.0000 | 0.8785 | 47 |

All 13 mistakes were false RLC predictions: 10 actual I recordings and three
actual R recordings. I recall is the main weakness at 75%. Training accuracy is 100%,
which is an apparent fit score, not evidence of generalization. Validation
and test scores provide the useful estimate. Whole-house bootstrap gives an
approximate 95% accuracy interval of 89.30%–97.77% on this collection.

- Development: 859 recordings from 43 houses, with four grouped validation folds.
- Test: 215 recordings from 12 other houses, reserved before model comparison.
- One prediction per recording. All windows from a recording stay together.
- No house appears in both development and test, or in multiple validation folds.
- Imputation and scaling are fitted inside each training fold.
- The selected pipeline is finally fitted on all 859 development recordings.

Five pairs of recordings have identical electrical/time inputs despite different
IDs and house names. They link house11 and house19, both in development. The
split keeps these houses in the same validation fold. There are 1,069 distinct
summary rows and 54 duplicate-connected house groups overall. All recording
IDs are retained, and no duplicate summary crosses the test boundary.

This duplicate connection was found after an initial test evaluation. The
development comparison was corrected with the same 12 candidate settings and
unchanged test IDs. Final selection used corrected development scores, not
test performance. The reported result is the re-evaluation after that fix.

## How time is used

For each of 23 electrical measurements, the code calculates eight summaries:
10th percentile, median, 90th percentile, interquartile range, early median,
late median, slope per second and typical adjacent change per second. Early and
late refer to the first and last 20% of elapsed time. This produces **184 inputs**
per recording. The full sequence is required at inference; this is an offline
classifier, not a real-time or future-load forecasting model.

The fitted imputer adds 64 missing-value indicators, so the selected forest
actually receives 248 numeric columns: 184 summaries plus 64 indicators.

Inputs include current, voltage, real/apparent/nonactive power, power factor,
phase sine/cosine, waveform shape and harmonics. ID, label, appliance name,
house, collection date, window count and recording duration are excluded.
Raw timestamps and sample/window indices are used for ordering and rate
calculations, rather than directly as predictors. Equivalent-component
estimates and algebraically redundant derived columns are omitted.

The median-only ablation for the selected setting scored **0.8899 validation
macro-F1**, compared with **0.9056** for the full summaries. This comparison
changes both distribution and time summaries; it does not isolate a causal
benefit from timestamps alone. Adding features does not automatically improve every classifier setting.

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

For a runnable example, use `--input examples/recordings.csv`. This file contains
four recordings from development houses with their labels and names removed.
It is a usage example, not an additional test set.

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

## Verification

All 20 tests passed. Saved-model inference replays all 215 test predictions,
and an independent recount matches the reported confusion matrix and scores.
Rebuilding the prepared data reproduced all three data files byte for byte.
A second unchanged-seed comparison selected the same model and reproduced
all cross-validation class predictions, all 400 fitted tree structures and
all 1,074 recording class predictions. Probabilities matched within 1e-14.
This establishes local reproducibility, not robustness across random seeds.
Evidence is recorded in `runs/timestamped/verification.json`.

GitHub Actions runs the test suite and saved-result verification on pushes
and pull requests. The four-recording prediction example also runs without
labels or appliance names.
