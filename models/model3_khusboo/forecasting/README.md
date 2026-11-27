# Model 3 — seasonal demand and peak / non-peak forecasting

This is the current Model 3 objective, following the user's clarification: forecast electrical demand and identify high-demand periods. All work stays inside `models/model3_khusboo/`. Model 1 and Model 2 files are untouched. The older building-recognition experiment and circuit reassessment remain as historical records; their scores do not describe these forecasts.

## What to read

- [Simple report, results and charts](outputs/report.md)
- [Excel results and editable charts](outputs/forecast_results.xlsx)
- [Forecast protocol](results/protocol.json)
- [Verification](results/verification.json)

## Outputs

The day-ahead model forecasts each of tomorrow's 24 hourly mean power levels for each of the three known buildings. A peak flag is derived from the forecast: predicted hourly power strictly greater than the building's training-data 90th percentile. This rule is an explicit data-based assumption because no tariff or prescribed peak definition was supplied. It is not an instantaneous surge/capacity classifier.

The seasonal model forecasts average power for each of the next three calendar months. It uses completed monthly history, the same month last year and the target month's annual cycle. Seasonal groups (Dec–Feb, Mar–May, Jun–Sep and Oct–Nov) are defined explicitly for reporting. There are no measured season/weather labels in the archive.

The selected hourly regressor is gradient boosting; the monthly regressor is ridge regression. The 2017 evaluation uses models fitted before 2017. Input history updates before each forecast origin, while fitted evaluation parameters stay frozen. Monthly training targets that would finish after a split are purged. The first 14 days/hourly and 12 months/monthly are warm-up history. Tests perturb future measurements to prove that earlier feature values do not change.

2017 hourly results: MAE **2.749 kW**, WAPE **12.19%**, versus baseline MAE **3.546 kW**. Monthly results: MAE **2.127 kW**, WAPE **9.10%**, versus baseline MAE **2.134 kW**. The monthly result is effectively tied with the seasonal baseline, and its RMSE is worse. Peak precision is **80.52%**, recall **71.60%**, and peak F1 **0.7580**. Do not convert WAPE to “forecast accuracy.”

The monthly test has 93 forecast records with overlapping target months/horizons; it does not have 93 independent months. The 2017 archive had previously been inspected for the abandoned experiment. This is a retrospective holdout for forecasting, with selection frozen before computing its forecast scores. New unseen data would be a stronger external check.

## Saved models and predictions

- `artifacts/hourly_evaluation.joblib` and `monthly_evaluation.joblib`: frozen pre-2017 models underlying reported scores.
- `artifacts/*_operational.joblib`: separate refits using observations through December 2017, fitted only after final evaluation was saved.
- `artifacts/*_preprocessing.joblib`: separate preprocessing, also contained in each full model bundle.
- `results/*_test_predictions.csv.gz`: all scored forecasts and baseline predictions; hourly rows also include peak thresholds/labels.
- `results/*_future_forecasts.csv`: 72 hourly forecasts for 1 January 2018 and nine average-load forecasts for January–March 2018.
- `results/*_split_manifest.csv.gz`, `*_folds.csv`, `*_cv.csv`: time boundaries, target eligibility, fold summaries and all selection scores.
- `results/hourly_measurements.csv.gz`, `monthly_measurements.csv`: cleaned source aggregates; the original archive remains outside Git.
- `outputs/`: report, six charts and Excel data. All chart sources are also in `results/chart_*.csv`.

The issued future forecasts start from the dataset endpoint, **1 January 2018**. They are not current 2026 predictions. Future actuals were not supplied, and predictive intervals were not estimated. Unknown weather, outages, missing months, new buildings and equipment changes limit generalization. Monthly mean power is measured in kW, not kWh of energy. The original ZIP SHA-256 is retained in the protocol and original audit.

## Run checks and inference

Use Python 3.13 and `../requirements.txt` in an isolated environment. Run from the repository root:

```powershell
python -m unittest discover -s models/model3_khusboo/forecasting -p test_forecasts.py -v
python models/model3_khusboo/forecasting/verify_forecasts.py
python models/model3_khusboo/forecasting/verify_workbook.py
```

To reproduce training, copy this experiment into a fresh directory without its generated `results`, `artifacts`, `outputs` and `previews`. Keep the frozen experiment unchanged. Run `train_forecasts.py 'C:\path\khusboo dataset .zip'` there. Its archive hash is checked against `../results/dataset_audit.json`; preserve that original audit alongside the fresh copy. `--reuse-hourly` can reuse the cleaned hourly file. Training refuses to overwrite existing final test metrics. Do not tune using the test scores.

For a new forecast, supply already-cleaned hourly history with columns `timestamp`, `building`, `power_kw`. Timestamps are timezone-naive **Asia/Kolkata local hours**. The history must cover all three known buildings, obey the same aggregation/coverage rules, and precede the origin. Forecasts before the saved fitting cutoff are rejected.

```powershell
python models/model3_khusboo/forecasting/predict_forecasts.py --history models/model3_khusboo/forecasting/results/hourly_measurements.csv.gz --origin 2018-01-01 --task hourly --output models/model3_khusboo/forecasting/results/hourly_inference.csv
python models/model3_khusboo/forecasting/predict_forecasts.py --history models/model3_khusboo/forecasting/results/hourly_measurements.csv.gz --origin 2018-01-01 --task monthly --output models/model3_khusboo/forecasting/results/monthly_inference.csv
```

Monthly origins must be the first day of a month and need at least 12 months of history through the preceding month. Hourly forecasts need recent history; more complete history improves lag availability. The operational models do not automatically refit in inference. Only use trusted joblib files.

Report builder: `build_report.py`. Excel builder: `build_workbook.mjs` with the bundled Node runtime and `@oai/artifact-tool` linked through the ignored parent `node_modules`. Scientific charts use matplotlib. The workbook cross-checks Python metrics and source-edit recalculation, and the exported OOXML is checked separately.
