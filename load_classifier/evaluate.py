"""Evaluate a frozen checkpoint and export the predictions behind every score."""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from .data import BEHAVIOURS, CIRCUITS, arrays, file_hash
from .metrics import score
from .model import TwoHeadMLP


def prediction_table(data, probabilities):
    result = data[[name for name in ("sample_id", "condition", "behaviour", "circuit")
                   if name in data]].copy()
    for head, labels in (("behaviour", BEHAVIOURS), ("circuit", CIRCUITS)):
        values = probabilities[head]
        result[f"predicted_{head}"] = np.array(labels)[values.argmax(axis=1)]
        for i, label in enumerate(labels):
            result[f"p_{head}_{label}"] = values[:, i]
        if head in data:
            result[f"{head}_correct"] = result[head] == result[f"predicted_{head}"]
    if {"behaviour", "circuit"} <= set(data):
        result["joint_correct"] = result["behaviour_correct"] & result["circuit_correct"]
    return result


def make_plots(output, metrics, history, best_epoch):
    os.environ.setdefault("MPLCONFIGDIR", str(Path(".cache/matplotlib").resolve()))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    axes[0].plot(history["epoch"], history["training_loss"], label="Training")
    axes[0].plot(history["epoch"], history["validation_loss"], label="Validation")
    axes[0].axvline(best_epoch, color="0.5", linestyle="--", label="Saved epoch")
    axes[0].set(xlabel="Epoch", ylabel="Sum of cross-entropy losses", title="Learning curves")
    axes[0].legend()
    for column, label in (("validation_behaviour_accuracy", "Behaviour"),
                          ("validation_circuit_accuracy", "Circuit"),
                          ("validation_joint_accuracy", "Both correct")):
        axes[1].plot(history["epoch"], history[column] * 100, label=label)
    axes[1].set(xlabel="Epoch", ylabel="Accuracy (%)", ylim=(0, 101), title="Validation accuracy")
    axes[1].legend()
    fig.savefig(output / "learning_curves.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), constrained_layout=True)
    for ax, head in zip(axes, ("behaviour", "circuit")):
        labels = metrics["test"][head]["labels"]
        matrix = np.array(metrics["test"][head]["confusion_matrix"])
        ax.imshow(matrix, cmap="Blues")
        ax.set(xticks=range(len(labels)), yticks=range(len(labels)), xticklabels=labels,
               yticklabels=labels, xlabel="Predicted", ylabel="Actual", title=f"Test: {head}")
        for row, col in np.ndindex(matrix.shape):
            ax.text(col, row, str(matrix[row, col]), ha="center", va="center",
                    color="white" if matrix[row, col] > matrix.max() / 2 else "black", fontsize=9)
    fig.savefig(output / "confusion_matrices.png", dpi=160)
    plt.close(fig)


def write_report(output, results, training):
    lines = ["# Baseline test results", "",
        "One shared MLP predicts behaviour and circuit type from 19 distinct measurements.",
        "The test partition was evaluated after training and validation checkpoint selection.", "",
        "| Partition | Rows | Behaviour accuracy | Circuit accuracy | Both correct |",
        "|---|---:|---:|---:|---:|"]
    for name in ("train", "validation", "test"):
        row = results[name]
        lines.append(f"| {name.title()} | {row['rows']:,} | {row['behaviour']['accuracy']:.2%} | "
                     f"{row['circuit']['accuracy']:.2%} | {row['joint_accuracy']:.2%} |")
    lines += ["", "## Test conditions", "",
        "| Condition | Rows | Behaviour accuracy | Circuit accuracy | Both correct |",
        "|---|---:|---:|---:|---:|"]
    for name, row in results["test_conditions"].items():
        lines.append(f"| {name.title()} | {row['rows']:,} | {row['behaviour']['accuracy']:.2%} | "
                     f"{row['circuit']['accuracy']:.2%} | {row['joint_accuracy']:.2%} |")
    test = results["test"]
    lines += ["", f"Both labels match on **{test['joint_correct']:,}/{test['rows']:,}** test examples.",
        f"Macro-F1: behaviour **{test['behaviour']['macro_f1']:.4f}**, "
        f"circuit **{test['circuit']['macro_f1']:.4f}**."]
    for head in ("behaviour", "circuit"):
        lines += ["", f"## {head.title()} classes on test", "",
            "| Class | Precision | Recall | F1 | Support |", "|---|---:|---:|---:|---:|"]
        for label, values in test[head]["per_class"].items():
            lines.append(f"| {label} | {values['precision']:.4f} | {values['recall']:.4f} | "
                         f"{values['f1']:.4f} | {values['support']} |")
    widths = " → ".join(str(n) for n in (len(training["features"]), *training["config"]["hidden_sizes"]))
    lines += ["", "## Training", "",
        f"- Architecture: {widths}, followed by separate 3-class and 7-class softmax heads.",
        f"- Trainable parameters: {training['parameter_count']:,}.",
        f"- Training/split seed: {training['config']['seed']} / 42.",
        f"- Saved epoch: {training['best_epoch']} of {training['epochs_completed']} completed epochs.",
        f"- Selection: {training['selection']}.",
        "- ReLU hidden layers; Adam; equally weighted behaviour and circuit cross-entropy.",
        f"- Learning rate {training['config']['learning_rate']}; batch size {training['config']['batch_size']}; "
        f"L2 coefficient {training['config']['l2']}.",
        "- Standardization is fitted on training rows only. No resampling, feature engineering or ensemble.",
        "- History losses exclude the training-only L2 term, so training/validation curves are comparable.",
        "", "## Interpretation and next experiments", "",
        "This is a starting measurement, not an optimized classifier. Keep the split fixed and use",
        "validation results for future feature and model choices. Do not tune against this test report.",
        "Compare voltage/current/power sweep ratios or slopes, removal of redundant measurement groups,",
        "and small network-size or training-setting changes. Compare other classifiers only after the",
        "neural baseline is fixed. Repeat promising experiments with several training seeds.", "",
        "The circuit head may confuse circuits with similar available measurements. The workbook has",
        "aggregate distortion features, not full harmonic impedance or transient measurements. Higher",
        "scores are not guaranteed from model size alone.", "",
        "These are simulated labelled loads informed by real load-profile values. The workbook defines",
        "CC/CP/CZ as dominant ZIP behaviour. Source keys are unique, but physical circuit IDs and the",
        "generator are unavailable, so physical independence and real-appliance performance are not established.",
        "Normal and tough test subsets come from the same two condition families used during training;",
        "this is not a test on a completely unseen condition family.", "",
        "![Training and validation](learning_curves.png)", "",
        "![Test confusion matrices](confusion_matrices.png)", "",
        "Exact scores and confusion matrices: [metrics.json](metrics.json).",
        "Every test prediction and class probability: [test_predictions.csv](test_predictions.csv).", "",
        "## Provenance", "",
        f"- Workbook SHA-256: `{training['workbook_sha256']}`",
        f"- Split SHA-256: `{training['splits_sha256']}`",
        f"- Checkpoint SHA-256: `{training['checkpoint_sha256']}`", ""]
    (output / "report.md").write_text("\n".join(lines))


def evaluate_run(data, output, workbook, split_path):
    output = Path(output)
    model, embedded = TwoHeadMLP.load(output / "model.npz")
    training = json.loads((output / "training.json").read_text())
    checks = (("workbook_sha256", workbook), ("splits_sha256", split_path),
              ("checkpoint_sha256", output / "model.npz"))
    if any(training[key] != file_hash(path) for key, path in checks):
        raise ValueError("Dataset, split or checkpoint changed since training")
    if any(embedded[key] != training[key] for key in embedded):
        raise ValueError("Training record differs from checkpoint metadata")
    results = {}
    test_probabilities = None
    for split in ("train", "validation", "test"):
        subset = data[data["split"] == split]
        x, behaviour, circuit = arrays(subset, model.features)
        probabilities = model.predict_proba(x)
        results[split] = score(behaviour, circuit, probabilities)
        if split == "test":
            test_probabilities = probabilities
    test = data[data["split"] == "test"]
    results["test_conditions"] = {}
    for condition in ("normal", "tough"):
        mask = (test["condition"] == condition).to_numpy()
        _, behaviour, circuit = arrays(test[mask], model.features)
        results["test_conditions"][condition] = score(behaviour, circuit,
            {head: values[mask] for head, values in test_probabilities.items()})
    results["provenance"] = {key: training[key] for key, _ in checks}
    metrics_path = output / "metrics.json"
    if metrics_path.exists() and json.loads(metrics_path.read_text()) != results:
        raise ValueError("Existing results differ; preserve this run and investigate")
    metrics_path.write_text(json.dumps(results, indent=2) + "\n")
    prediction_table(test, test_probabilities).to_csv(output / "test_predictions.csv", index=False)
    make_plots(output, results, pd.read_csv(output / "history.csv"), training["best_epoch"])
    write_report(output, results, training)
    print(f"Test: behaviour {results['test']['behaviour']['accuracy']:.2%}, circuit "
          f"{results['test']['circuit']['accuracy']:.2%}, both {results['test']['joint_accuracy']:.2%}")
    return results
