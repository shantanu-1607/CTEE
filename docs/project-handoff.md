# Electrical load classification: project handoff

Updated: 27 November 2026

This file provides context for continuing the project on another laptop or account. It describes the completed experiment and the work now required. Paths below are relative to the repository so they also work after cloning elsewhere.

## 1. Current assignment and next task

The professor now requires **three models using three different datasets**. One dataset experiment is complete. We plan to build the second on the original laptop and the third on another laptop/account.

The aim remains electrical load classification: use electrical measurements to predict the supplied load category. This is supervised classification because the training examples have known labels.

**The other two datasets have not been supplied in this handoff.** Inspect the new dataset before deciding its labels, features, classifier or split. Three classifier settings on the existing dataset do not fulfil the requirement for three datasets. We have not established whether the professor also requires three different classifier families; clarify that requirement if it affects the new experiment.

## 2. Repository and people

- Repository: <https://github.com/shantanu-1607/CTEE>
- Existing completed-work branch: `timestamped-load-classifier`.
- Existing pull request: <https://github.com/shantanu-1607/CTEE/pull/2>.
- Completed model/report checkpoint: `183062fb57290ecb13066c68166a035a6c476c5a`.
- Student: Shantanu Singh — **BTECH/10232/24**.
- Student: Aanyaa Sinha — **BTECH/10789/24**.

Check the remote state when starting; branches may have advanced since this handoff. A fresh clone's default branch may not yet contain the completed experiment, so explicitly check out the existing-work branch.

Earlier work used a simulated Excel dataset and predicted behaviour labels CC/CP/CZ as well as circuit classes. That task was replaced when the dataset changed. It remains in Git history. **Do not mix its neural-network architecture, scores or graphs with the current timestamped experiment.**

## 3. What the completed model predicts

The current model produces **one load label for a complete electrical recording**, choosing among:

| Supplied label | Number of recordings |
| --- | ---: |
| R | 305 |
| I | 191 |
| RC | 347 |
| RLC | 231 |
| Total | 1,074 |

The label `I` is preserved exactly as supplied. Its meaning has not been confirmed; do not silently rename it to `L`. There are no CC/CP/CZ labels in this dataset, so this model cannot learn or evaluate those outputs.

This is an offline recording classifier. It uses a complete sequence, rather than predicting future electricity demand or providing a validated live-streaming result.

## 4. Data used for the completed experiment

The user provided `meta_labeled.json`, containing recording IDs, house identifiers, appliance types and load labels. Electrical measurements came from CSV files in this Drive folder:

<https://drive.google.com/drive/folders/1_aO3gFYwigITWH0YmCPAoB48wRg2B5mF>

The combined source file, `all_features.csv`, contains **209,554 rows, 64 columns and 1,074 recording IDs from 55 houses**. Its size is 214,166,054 bytes. It contains precomputed feature windows, not raw voltage/current waveforms. Each window represents 500 samples at 30,000 Hz, approximately 16.67 milliseconds.

All combined-file IDs, house annotations, appliance types and labels were checked against the JSON. Time/sample consistency, ordering, overlap, missing values and duplicate inputs were also checked. Source fingerprints are saved in `data/source.json` and `data/processed/audit.json`.

The schema is consistent with original PLAID, but that is a provenance inference. The supplied feature extractor, physical label-assignment process and raw waveforms were not provided. Describe the inputs as the user's supplied feature CSV and labelled JSON, rather than claiming independent verification of their origin.

The large raw CSV is ignored by Git. The metadata and prepared recording summaries are tracked, so the completed model can be reproduced without downloading the raw file. The downloader can retrieve and verify the exact source when Drive access is available.

## 5. Features and use of timestamps

We selected **23 existing electrical measurements**:

```text
v_rms, i_rms, P_real_W, S_apparent_VA, Q_nonactive_var, pf,
i_crest_factor, v_crest_factor, i_form_factor, i_skew, i_kurtosis,
i_zero_cross, f0_Hz, cos_phi, sin_phi, i_h3_ratio, i_h5_ratio,
i_h7_ratio, i_thd, v_thd, i_spec_centroid_Hz, i_spec_entropy,
i_hf_energy_ratio
```

These describe voltage/current magnitude, power, power factor, phase, waveform shape and harmonic content. For each measurement, its time-ordered windows become eight recording-level summaries:

| Summary | Plain-language meaning |
| --- | --- |
| p10 | A representative low value: the 10th percentile |
| median | The middle value |
| p90 | A representative high value: the 90th percentile |
| iqr | Spread between the 25th and 75th percentiles |
| early | Median during the first 20% of elapsed recording time |
| late | Median during the last 20% of elapsed recording time |
| slope | Overall trend per second, fitted against actual elapsed time |
| step | Median absolute change per second between adjacent valid windows |

Thus, **23 × 8 = 184 engineered summary features** represent each recording. Compared with a baseline of 23 medians, this adds **161 summary columns**, rather than 184 new sensor measurements.

Missing summaries are filled using medians learned only from the training partition. The final fitted imputer also adds **64 missing-value indicator columns**, giving **248 numeric inputs to the final forest**. Indicator counts can differ when fitting on other data.

Timestamps are used for ordering, early/late intervals and rates of change. Raw timestamps, sample/window indices, recording duration and window count are not direct predictors. Recording ID, house, appliance name, collection date and target label are also excluded from model inputs. Equivalent-component estimates and redundant derived columns are omitted.

## 6. Splitting and leakage prevention

Windows from one recording are related, and recordings from one house may be related. Randomly splitting individual windows could therefore produce an overly optimistic score.

The experiment instead uses:

- **Development:** 859 recordings from 43 houses.
- **Test:** 215 recordings from 12 other houses.
- **Model selection:** four development cross-validation folds, grouped by house and duplicate connections.
- **Preprocessing:** imputation and, where needed, scaling fitted within each training fold.
- **Final fit:** selected pipeline trained on all 859 development recordings.

Five identical electrical/time-input pairs were found: `234/389`, `235/390`, `236/391`, `237/392` and `238/393`. They link house11 and house19, both in development. These houses are kept together in validation. All original IDs are retained for traceability: 1,074 IDs correspond to 1,069 distinct summary rows. No duplicate summary crosses the development/test boundary.

**Evaluation history matters:** this deeper duplicate check happened after an initial test evaluation. Development cross-validation was corrected and the same 12 candidate settings were compared again, keeping test IDs unchanged. The final model was chosen using corrected development scores, not test scores. However, the test had previously been inspected; do not describe it as an entirely untouched second experiment. See `runs/timestamped/protocol_notes.md`.

## 7. Classifiers compared and selected model

Twelve candidate settings were evaluated, including a majority-class baseline, logistic regression, two RBF SVM settings, random forest, histogram gradient boosting, a neural network and several Extra Trees settings.

Selected comparison results are below. These are development cross-validation means, not test scores.

| Candidate | Mean validation accuracy | Mean macro F1 |
| --- | ---: | ---: |
| Selected Extra Trees, half the features per split | 92.39% | 0.9056 |
| Extra Trees, all features per split | 91.68% | 0.8954 |
| Extra Trees, square-root feature setting | 91.45% | 0.8939 |
| Random forest | 91.13% | 0.8911 |
| Histogram gradient boosting | 89.22% | 0.8669 |
| Logistic regression | 82.90% | 0.8050 |
| Neural network, hidden layers 64 and 32 | 82.87% | 0.8007 |
| Majority-class baseline | 32.44% | 0.1224 |

All 12 settings and fold results are saved in `comparison.csv` and `cv_folds.csv` under `runs/timestamped/`.

The final classifier is **ExtraTreesClassifier**, internally named `extra_trees_half`, with:

- 400 trees; Gini split criterion.
- `max_features=0.5`: half the available input columns considered at each split.
- Balanced class weights.
- Minimum leaf size 1; minimum split size 2; no fixed maximum depth.
- No bootstrap sampling; random seed 42; four worker jobs.
- Training-median imputation and missing-value indicators before the forest.

Extra Trees combines many randomized decision trees and averages their class probabilities. It was selected because it achieved the highest mean grouped-validation macro F1 among the tested settings. **The current final model is not a neural network**, although one was compared. Macro F1 gives each class equal importance, which is useful when class counts differ.

## 8. Did the extra features help?

For the selected classifier setting, the development-only feature comparison gave:

| Representation | Mean validation macro F1 |
| --- | ---: |
| 23 median summaries | 0.8899 |
| 184 full summaries | 0.9056 |

The mean gain is **0.0157 F1**, or approximately 1.57 points on a 0–100 scale. It is not a 1.57 percentage-point test-accuracy gain.

The full representation improved one fold and was slightly worse in the other three. This does not establish a consistent or statistically significant improvement. Distribution summaries, time summaries and missingness indicators change together, so we cannot attribute the gain solely to timestamps. No separate median-only test result was measured. Detailed values are in `runs/timestamped/selection.json` under `ablation`.

## 9. Final test results

| Metric | Result |
| --- | ---: |
| Correct predictions | 202 of 215 |
| Accuracy | **93.95%** |
| Balanced accuracy | **92.50%** |
| Macro F1 | **0.9275** |
| Weighted F1 | **0.9397** |
| Majority-class baseline accuracy | 31.63% |

| Class | Precision | Recall | F1 | Test recordings |
| --- | ---: | ---: | ---: | ---: |
| R | 1.0000 | 0.9500 | 0.9744 | 60 |
| I | 1.0000 | 0.7500 | 0.8571 | 40 |
| RC | 1.0000 | 1.0000 | 1.0000 | 68 |
| RLC | 0.7833 | 1.0000 | 0.8785 | 47 |

Confusion matrix: rows are actual labels; columns are predictions.

| Actual / predicted | R | I | RC | RLC |
| --- | ---: | ---: | ---: | ---: |
| R | 57 | 0 | 0 | 3 |
| I | 0 | 30 | 0 | 10 |
| RC | 0 | 0 | 68 | 0 |
| RLC | 0 | 0 | 0 | 47 |

All 13 errors were predictions of RLC: ten actual I recordings and three actual R recordings. I recall, at 75%, is the main weakness. Training accuracy was 100%, which is an apparent fit score, not a claim of perfect generalization.

Resampling whole test houses gives an approximate 95% accuracy interval of **89.30%–97.77%**. The test contains only 12 houses, so uncertainty remains. These scores measure agreement with supplied labels within this collection, not proven physical circuit identification or performance on every new appliance.

## 10. Code flow and files to read

```text
JSON labels + feature-window CSV
    -> validate IDs, measurements and timestamps
    -> order windows within each recording
    -> compute 184 recording summaries
    -> apply fixed house/duplicate-group partitions
    -> compare candidate pipelines on development folds
    -> select by mean validation macro F1
    -> fit chosen pipeline on all development recordings
    -> evaluate on reserved test houses
    -> save model, probabilities, metrics, plots and reports
```

| File or folder | Purpose |
| --- | --- |
| `README.md` | Main project overview and commands |
| `data/README.md` | Provenance, duplicate checks and data limitations |
| `data/meta_labeled.json` | Supplied labels and recording metadata |
| `data/source.json` | Raw source location and fingerprint |
| `data/processed/recordings.csv` | Recording-level features and audit metadata |
| `data/processed/splits.csv` | Fixed development/test and validation-fold assignments |
| `data/processed/audit.json` | Counts, feature names, validation and fingerprints |
| `load_classifier/data.py` | Schema validation and grouped splits |
| `load_classifier/features.py` | Timestamp-aware feature summaries |
| `load_classifier/models.py` | Candidate classifier pipelines |
| `load_classifier/experiment.py` | Preparation, comparison and final training |
| `load_classifier/evaluate.py` | Evaluation, figures and saved-result verification |
| `load_classifier/metrics.py` | Metric calculations |
| `load_classifier/predict.py` | Inference on complete new recordings |
| `load_classifier/__main__.py` | Command-line interface |
| `runs/timestamped/model.joblib` | Saved preprocessing and selected classifier |
| `runs/timestamped/test_metrics.json` | Exact final metrics and uncertainty estimates |
| `runs/timestamped/test_predictions.csv` | All test answers and probabilities |
| `runs/timestamped/selection.json` | Selection settings, feature study and environment |
| `runs/timestamped/verification.json` | Reproducibility evidence |
| `docs/professor-guide.md` | Easy explanation and likely questions |
| `tests/` | Data, time arithmetic, leakage and pipeline checks |
| `.github/workflows/tests.yml` | Automated tests and saved-result verification |

Twenty tests passed at the completed checkpoint. Saved-model inference reproduces all 215 test predictions, and an independent recount matches the metrics. Rebuilding prepared data reproduced the data files byte for byte. A fixed-seed reproduction matched selected model, tree structures and class predictions, with probabilities agreeing within 1e-14. This demonstrates local reproducibility, not robustness across many random seeds.

## 11. Report and Excel deliverables already produced

The directory `outputs/timestamped-report-20261127/` contains:

- **`electrical_load_classification_report.docx`:** 20-page original report following the supplied reference's formatting and nine-section structure. Covers context, literature, features, methodology, architecture, results, limitations and further work. Includes both confirmed student names and roll numbers; no unconfirmed supervisor or institution details were copied.
- **`chart_data.xlsx`:** nine worksheets and seven native editable charts. Contains numerical data behind report figures, classifier comparisons, feature experiments, time-pattern examples, 248 feature definitions/importances, all 215 test predictions and formula-based metrics/confusion counts.

Builders are `scripts/build_submission.py` and `scripts/build_chart_workbook.mjs`. These report-building scripts depend on the document/spreadsheet runtime used for authoring and contain environment-specific setup; they are not part of the ordinary Python training CLI. Adapt their runtime paths if regenerating on another laptop. The finished Word/Excel files can be opened independently.

## 12. Start on another laptop

With Git and Python 3.12 available, run:

```sh
git clone https://github.com/shantanu-1607/CTEE.git
cd CTEE
git switch timestamped-load-classifier
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest -q
python -m load_classifier verify --run runs/timestamped
```

The activation command above is for macOS/Linux. Use the equivalent virtual-environment activation on Windows. Access to a different account does not automatically provide repository write permissions or Drive access; establish those connections when needed.

To reproduce the existing experiment, use a **new, unused** output directory:

```sh
python -m load_classifier compare --run runs/reproduction
python -m load_classifier train --run runs/reproduction
python -m load_classifier evaluate --run runs/reproduction
python -m load_classifier verify --run runs/reproduction
```

The commands intentionally refuse to overwrite populated experiment/model/test outputs. To try inference without labels:

```sh
python -m load_classifier predict --input examples/recordings.csv --output predictions.csv
```

Those example recordings come from development houses and are a usage demonstration, not another independent test. The existing CLI expects the existing dataset's schema; it is not automatically compatible with the next dataset.

## 13. Approach for the new dataset

1. **Inspect the actual files.** Record source, sample counts, units, target labels, missing values, timestamps and available house/device/session identifiers. Determine whether rows are raw samples, windows or entire recordings.
2. **Define the prediction task.** Confirm what one prediction represents and whether supplied labels match the professor's intended output. Do not invent unavailable circuit or behaviour labels.
3. **Audit leakage before splitting.** Look for duplicates and related windows, recordings, devices or sessions. Keep related examples together. Use chronological evaluation where future prediction is the task; house grouping is appropriate only when houses exist and match the intended generalization question.
4. **Reserve the test partition.** Fit preprocessing only on training folds. Select classifiers and features using development validation, keeping final test performance out of selection.
5. **Establish a baseline and compare sensible candidates.** A different dataset may favour a different model. A neural network is an option, not a guaranteed improvement. Use macro F1, accuracy, class recall and a majority baseline to assess results.
6. **Test feature changes explicitly.** Compare original features with added/removed features under identical development splits. Separate timestamp benefits from other representation changes where possible. Report actual gains and regressions.
7. **Save a complete experiment.** Include source fingerprints, split assignments, settings/seeds, preprocessing, trained model, fold results, test predictions/probabilities, per-class metrics, confusion matrix, error analysis and reproducible commands.
8. **Prepare presentation material.** Explain the method simply, provide charts and export their numerical data to Excel. Use the existing report structure where relevant, with original dataset-specific details and measured results.

Do not promise a target accuracy before examining the data. Scores on different datasets have different meanings, so do not call one model universally better just because its raw accuracy is higher.

## 14. Coordination and working preferences

- Preserve the completed dataset, model, splits, results and report. The older broad permission to remove unnecessary files does not mean these now-required experiment records should be deleted.
- Use separate dataset-specific folders and experiment outputs for the new work. Create a separate task-based branch on each laptop, and coordinate shared-file edits before merging. Do not force-push over the other laptop's work.
- The user authorized working autonomously and committing/pushing project changes. Proceed with routine reversible work; ask when essential information is unavailable. Respect the permissions actually available to the new account.
- Keep code organized, readable and reproducible. Use plain task-based filenames, branch names and commit messages. Do not add unsolicited assistant attribution or signatures.
- Explain what changed, why, how it was evaluated and what its limitations are. The user needs to answer the professor's cross-questions, not just show a high number.

Known scope limits of Model 1: no raw-waveform deep learning, validated real-time deployment, forecasting, seasonality experiment, power-factor correction, SHAP analysis or independently calibrated probabilities were implemented. All collection dates are July 2013. Supplied metadata warns of calibration limitations. Do not present planned work as completed work.

**First action for the receiving account:** read the existing README and this handoff, obtain the new dataset, audit its contents, and then propose an appropriate independent experiment while preserving Model 1.
