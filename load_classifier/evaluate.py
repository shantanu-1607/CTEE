"""Evaluate once on the reserved houses, and export a readable result report."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from .data import LABELS, file_hash, write_json
from .experiment import input_fingerprint, load_prepared
from .metrics import classification_scores, house_bootstrap
from .predict import load_bundle, predict_records


def plot_results(run, metrics, comparison):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
    matrix = np.asarray(metrics["confusion_matrix"])
    axes[0].imshow(matrix, cmap="Blues", vmin=0)
    axes[0].set(xticks=range(4), yticks=range(4), xticklabels=LABELS, yticklabels=LABELS,
                xlabel="Predicted load", ylabel="Actual load", title="Test recordings from unseen houses")
    for (i, j), count in np.ndenumerate(matrix):
        axes[0].text(j, i, str(count), ha="center", va="center",
                     color="white" if count > matrix.max() / 2 else "black")
    for column, offset in (("precision", -.23), ("recall", 0), ("f1-score", .23)):
        axes[1].bar(np.arange(4) + offset, [metrics["per_class"][k][column] for k in LABELS],
                    width=.23, label=column.replace("-score", ""))
    axes[1].set(xticks=range(4), xticklabels=LABELS, ylim=(0, 1.08),
                title="Per-class test results", ylabel="Score")
    axes[1].legend(loc="lower right")
    fig.savefig(run / "test_results.png", dpi=180)
    plt.close(fig)
    table = comparison.sort_values("macro_f1_mean")
    fig, ax = plt.subplots(figsize=(9, 5), layout="constrained")
    ax.barh(table.model, table.macro_f1_mean, xerr=table.macro_f1_std, color="#3879a8", capsize=3)
    ax.set(xlim=(0, 1.05), xlabel="Mean macro-F1 (error bars: standard deviation across four folds)",
           title="Classifier comparison on development houses")
    fig.savefig(run / "model_comparison.png", dpi=180)
    plt.close(fig)


def write_report(run, audit, selection, metrics, comparison):
    test = metrics["test"]
    confidence = metrics["house_bootstrap"]
    lines = ["# Timestamped load classification results", "",
             f"Selected classifier: **{selection['selected_model']}**. One prediction per complete recording.", "",
             f"Test accuracy: **{test['accuracy']:.2%}** ({test['correct']}/{test['recordings']}). "
             f"Macro-F1: **{test['macro_f1']:.4f}**. Balanced accuracy: **{test['balanced_accuracy']:.2%}**.", "",
             f"These test recordings come from {confidence['houses']} houses absent from training and model selection.", "",
             "## Test classes", "", "| Load | Precision | Recall | F1 | Recordings |",
             "| --- | ---: | ---: | ---: | ---: |"]
    for label in LABELS:
        row = test["per_class"][label]
        lines.append(f"| {label} | {row['precision']:.4f} | {row['recall']:.4f} | {row['f1-score']:.4f} | {int(row['support'])} |")
    lines += ["", "## Model selection", "", "Test data was not used to rank or tune models. "
              "The table reports four-fold cross-validation on development houses. Preprocessing was fitted separately inside each training fold.", "",
              "| Classifier | Mean macro-F1 | Fold SD | Mean accuracy |", "| --- | ---: | ---: | ---: |"]
    for row in comparison.itertuples():
        lines.append(f"| {row.model} | {row.macro_f1_mean:.4f} | {row.macro_f1_std:.4f} | {row.accuracy_mean:.2%} |")
    ablation = selection["ablation"]
    lines += ["", "## Using time", "",
              "The model receives 23 electrical measurements summarized eight ways: 10th percentile, median, "
              "90th percentile, interquartile range, early-window median, late-window median, slope per second, "
              "and median absolute adjacent change per second. Early and late mean the first and last 20% of elapsed time. "
              "Windows are sorted by their timestamps. This gives 184 inputs per recording.", "",
              f"For the selected classifier, median-only development CV macro-F1 was {ablation['median_only_macro_f1_mean']:.4f}; "
              f"the temporal summaries scored {ablation['temporal_macro_f1_mean']:.4f}. "
              "This ablation compares complete feature sets; it does not isolate timestamps from distribution summaries.", "",
              "## Uncertainty and limitations", "",
              f"A whole-house bootstrap (2,000 resamples) gives an approximate 95% accuracy interval of "
              f"{confidence['accuracy_95_percent'][0]:.2%}–{confidence['accuracy_95_percent'][1]:.2%}, and "
              f"macro-F1 interval {confidence['macro_f1_95_percent'][0]:.4f}–{confidence['macro_f1_95_percent'][1]:.4f}. "
              "This is conditional on the selected model and these held-out houses; it does not include model-selection or label uncertainty.", "",
              "- The supplied labels are R, I, RC and RLC. I is preserved verbatim. No CC/CP/CZ target exists in this dataset.",
              "- Each appliance type maps to a single supplied load label. Scores measure agreement with these annotations, "
              "not independent verification of a physical equivalent circuit. Appliance type is never a predictor.",
              "- Source JSON warns that some recordings are not well calibrated. The feature-extraction code and raw waveforms "
              "for the Drive CSVs were not supplied; electrical feature formulas cannot be independently certified here.",
              "- Recording ID, appliance name, load, house, collection date, window count and recording duration are excluded as predictors. "
              "Time is used for changes within each recording, not calendar forecasting.",
              "- Missing harmonic summaries are imputed from training data only. Equivalent L/C columns and redundant derived columns are omitted.",
              "- This is an offline recording classifier. It does not identify multiple simultaneous appliances, unseen load classes, "
              "or prove generalization outside this collection. Reported probabilities are uncalibrated model outputs.",
              "- House grouping prevents known within-house overlap. The metadata has no physical appliance ID to verify independence across houses.",
              "- The previous workbook results describe a different dataset and must not be compared as an improvement or regression.",
              "", "## Files", "", "`test_predictions.csv` contains every test answer and class probability. "
              "`test_metrics.json` stores exact counts, per-house results, confidence intervals and the majority baseline. "
              "`cv_predictions.csv` contains development-only out-of-fold predictions for all candidates. "
              "`model.joblib` contains the fitted preprocessing pipeline and classifier.", ""]
    (run / "report.md").write_text("\n".join(lines))


def evaluate(directory, run):
    run = Path(run)
    if (run / "test_metrics.json").exists():
        raise FileExistsError("This run has already been evaluated; use verify to check saved results")
    records, audit = load_prepared(directory)
    selection = json.loads((run / "selection.json").read_text())
    training = json.loads((run / "training.json").read_text())
    if file_hash(run / "model.joblib") != training["model_sha256"]:
        raise ValueError("Model changed after training")
    bundle = load_bundle(run / "model.joblib")
    if bundle["input_fingerprint"] != input_fingerprint(audit):
        raise ValueError("Inputs changed after training")
    if bundle["selection_sha256"] != file_hash(run / "selection.json"):
        raise ValueError("Selection changed after training")
    test = records[records.partition.eq("test")].copy()
    with threadpool_limits(limits=1):
        predictions = predict_records(bundle, test)
    output = test[["id", "location", "load", "appliance_type"]].merge(predictions, on="id", validate="one_to_one")
    output = output.rename(columns={"load": "actual_load"})
    output["correct"] = output.actual_load.eq(output.predicted_load)
    output.to_csv(run / "test_predictions.csv", index=False)
    majority = records.loc[records.partition.eq("development"), "load"].mode().iloc[0]
    metrics = {
        "selected_model": bundle["selected_model"], "test": classification_scores(output.actual_load, output.predicted_load),
        "majority_baseline": classification_scores(output.actual_load, np.repeat(majority, len(output))),
        "house_bootstrap": house_bootstrap(output.actual_load, output.predicted_load, output.location),
        "per_house": {house: classification_scores(part.actual_load, part.predicted_load)
                      for house, part in output.groupby("location")},
        "model_sha256": file_hash(run / "model.joblib"),
        "test_predictions_sha256": file_hash(run / "test_predictions.csv"),
        "input_fingerprint": input_fingerprint(audit),
    }
    write_json(run / "test_metrics.json", metrics)
    comparison = pd.read_csv(run / "comparison.csv")
    plot_results(run, metrics["test"], comparison)
    write_report(run, audit, selection, metrics, comparison)
    return metrics


def verify(directory, run):
    """Read-only replay and independent recount; never retrain or select on test data."""
    run = Path(run)
    records, audit = load_prepared(directory)
    metrics = json.loads((run / "test_metrics.json").read_text())
    if file_hash(run / "model.joblib") != metrics["model_sha256"]:
        raise ValueError("Model hash differs from the evaluated model")
    if file_hash(run / "test_predictions.csv") != metrics["test_predictions_sha256"]:
        raise ValueError("Prediction export changed after evaluation")
    if metrics["input_fingerprint"] != input_fingerprint(audit):
        raise ValueError("Input data changed after evaluation")
    saved = pd.read_csv(run / "test_predictions.csv")
    test = records[records.partition.eq("test")]
    expected = test[["id", "load", "location"]].sort_values("id").reset_index(drop=True)
    actual = saved[["id", "actual_load", "location"]].rename(columns={"actual_load": "load"})
    pd.testing.assert_frame_equal(expected, actual.sort_values("id").reset_index(drop=True))
    recount = classification_scores(saved.actual_load, saved.predicted_load)
    if recount != metrics["test"]:
        raise ValueError("Exported predictions do not reproduce the test metrics")
    replay = predict_records(load_bundle(run / "model.joblib"), test)
    pd.testing.assert_frame_equal(saved[replay.columns].reset_index(drop=True), replay.reset_index(drop=True), rtol=1e-12, atol=1e-12)
    return {"passed": True, "checks": ["input hashes", "house separation", "test coverage and targets",
            "model hash", "prediction hash", "exact metric recount", "saved model inference replay"],
            "test_recordings": len(saved)}
