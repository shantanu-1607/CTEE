# Timestamped load classification results

Selected classifier: **extra_trees_half**. One prediction per complete recording.

Test accuracy: **93.95%** (202/215). Macro-F1: **0.9275**. Balanced accuracy: **92.50%**.

These test recordings come from 12 houses absent from training and model selection.

The majority-class test baseline is 31.63%. Apparent training accuracy is 100.00%; it is not a generalization score. All test mistakes are preserved in the prediction export.

## Test classes

| Load | Precision | Recall | F1 | Recordings |
| --- | ---: | ---: | ---: | ---: |
| R | 1.0000 | 0.9500 | 0.9744 | 60 |
| I | 1.0000 | 0.7500 | 0.8571 | 40 |
| RC | 1.0000 | 1.0000 | 1.0000 | 68 |
| RLC | 0.7833 | 1.0000 | 0.8785 | 47 |

## Model selection

Test data was not used to rank or tune models. The table reports four-fold cross-validation on development houses. Preprocessing was fitted separately inside each training fold.

| Classifier | Mean macro-F1 | Fold SD | Mean accuracy |
| --- | ---: | ---: | ---: |
| extra_trees_half | 0.9056 | 0.0326 | 92.39% |
| extra_trees_all | 0.8954 | 0.0348 | 91.68% |
| extra_trees | 0.8939 | 0.0450 | 91.45% |
| extra_trees_compact | 0.8932 | 0.0212 | 91.36% |
| random_forest | 0.8911 | 0.0397 | 91.13% |
| extra_trees_leaf3 | 0.8893 | 0.0371 | 90.93% |
| gradient_boosting | 0.8669 | 0.0288 | 89.22% |
| logistic | 0.8050 | 0.0571 | 82.90% |
| mlp | 0.8007 | 0.0364 | 82.87% |
| svm_c1 | 0.7954 | 0.0484 | 81.07% |
| svm_c10 | 0.7809 | 0.0465 | 80.78% |
| majority | 0.1224 | 0.0058 | 32.44% |

## Using time

The model receives 23 electrical measurements summarized eight ways: 10th percentile, median, 90th percentile, interquartile range, early-window median, late-window median, slope per second, and median absolute adjacent change per second. Early and late mean the first and last 20% of elapsed time. Windows are sorted by their timestamps. This gives 184 inputs per recording.

The fitted preprocessing produces 248 numeric columns after adding training-observed missing-value indicators. These columns are available to the fitted classifier. `feature_importance.csv` records training-based impurity importances. These are descriptive, can be affected by correlated inputs, and do not establish causation.

For the selected classifier, median-only development CV macro-F1 was 0.8899; the temporal summaries scored 0.9056. This ablation compares complete feature sets; it does not isolate timestamps from distribution summaries.

## Uncertainty and limitations

A whole-house bootstrap (2,000 resamples) gives an approximate 95% accuracy interval of 89.30%–97.77%, and macro-F1 interval 0.8502–0.9730. This is conditional on the selected model and these held-out houses; it does not include model-selection or label uncertainty.

- The supplied labels are R, I, RC and RLC. I is preserved verbatim. No CC/CP/CZ target exists in this dataset.
- Each appliance type maps to a single supplied load label. Scores measure agreement with these annotations, not independent verification of a physical equivalent circuit. Appliance type is never a predictor.
- Source JSON warns that some recordings are not well calibrated. The feature-extraction code and raw waveforms for the Drive CSVs were not supplied; electrical feature formulas cannot be independently certified here.
- Recording ID, appliance name, load, house, collection date, window count and recording duration are excluded as predictors. Time is used for changes within each recording, not calendar forecasting.
- Missing harmonic summaries are imputed from training data only. Equivalent L/C columns and redundant derived columns are omitted.
- This is an offline recording classifier. It does not identify multiple simultaneous appliances, unseen load classes, or prove generalization outside this collection. Reported probabilities are uncalibrated model outputs.
- House grouping prevents known within-house overlap. The metadata has no physical appliance ID to verify independence across houses.
- There are 5 excess exact-duplicate summary rows in 5 duplicate groups. Houses connected by these summaries stay together in development validation; no duplicate group crosses the test boundary. Original recording IDs are retained for traceability.
- The test set contains 10 of 11 appliance types. Types with zero test support: Washing Machine. Their type-specific test performance is not established. All four load labels are present in test.
- The previous workbook results describe a different dataset and must not be compared as an improvement or regression.

## Files

`test_predictions.csv` contains every test answer and class probability. `test_metrics.json` stores exact counts, per-house results, confidence intervals and the majority baseline. `cv_predictions.csv` contains development-only out-of-fold predictions for all candidates. `model.joblib` contains the fitted preprocessing pipeline and classifier.

## Test errors and appliance coverage

| Appliance type | Test recordings | Correct |
| --- | ---: | ---: |
| Air Conditioner | 17 | 17 |
| Compact Fluorescent Lamp | 23 | 23 |
| Fan | 25 | 18 |
| Fridge | 10 | 7 |
| Hairdryer | 29 | 29 |
| Heater | 6 | 3 |
| Incandescent Light Bulb | 25 | 25 |
| Laptop | 45 | 45 |
| Microwave | 30 | 30 |
| Vacuum | 5 | 5 |
| Washing Machine | 0 | 0 |

There were 13 mistakes.

| Appliance type | Actual load | Predicted load | Mistakes |
| --- | --- | --- | ---: |
| Fan | I | RLC | 7 |
| Fridge | I | RLC | 3 |
| Heater | R | RLC | 3 |

These are observations about saved test errors. They were not used to retune the model.

## Evaluation correction

After an initial evaluation, a deeper check found five exact duplicate input
pairs linking house11 and house19: 234/389, 235/390, 236/391, 237/392 and 238/393.
The required electrical measurements and time/sample columns match across each
pair. Both houses are in development; none of the 215 test recordings is part
of these duplicate groups.

Development cross-validation was corrected to keep the connected houses in
the same fold. The same 12 candidate settings were compared again, and the
corrected validation scores selected `extra_trees_half`. Test houses and IDs
were unchanged. The final result is a re-evaluation after this data-integrity
fix. Test scores were not used to select the corrected model, but the test set
had previously been inspected. The first result must not be treated as an
independent second test set or combined with the final result.

