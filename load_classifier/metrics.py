"""Recording-level metrics and uncertainty across held-out houses."""

import numpy as np
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             classification_report, confusion_matrix, f1_score)

from .data import LABELS


def classification_scores(actual, predicted):
    actual, predicted = np.asarray(actual), np.asarray(predicted)
    if len(actual) == 0 or actual.shape != predicted.shape:
        raise ValueError("Predictions and targets must have the same nonempty shape")
    if not set(actual) <= set(LABELS) or not set(predicted) <= set(LABELS):
        raise ValueError("Unknown load label")
    report = classification_report(actual, predicted, labels=list(LABELS),
                                   output_dict=True, zero_division=0)
    return {
        "recordings": len(actual), "correct": int(np.sum(actual == predicted)),
        "accuracy": float(accuracy_score(actual, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(actual, predicted)),
        "macro_f1": float(f1_score(actual, predicted, labels=list(LABELS), average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(actual, predicted, labels=list(LABELS), average="weighted", zero_division=0)),
        "per_class": {name: report[name] for name in LABELS},
        "confusion_matrix": confusion_matrix(actual, predicted, labels=list(LABELS)).tolist(),
        "labels": list(LABELS),
    }


def house_bootstrap(actual, predicted, houses, repetitions=2000, seed=2026):
    actual, predicted, houses = map(np.asarray, (actual, predicted, houses))
    groups = np.unique(houses)
    indices = [np.flatnonzero(houses == group) for group in groups]
    rng = np.random.default_rng(seed)
    accuracies, f1s = [], []
    for _ in range(repetitions):
        rows = np.concatenate([indices[i] for i in rng.integers(len(groups), size=len(groups))])
        accuracies.append(float(np.mean(actual[rows] == predicted[rows])))
        matrix = confusion_matrix(actual[rows], predicted[rows], labels=list(LABELS))
        denominator = matrix.sum(axis=0) + matrix.sum(axis=1)
        f1 = np.divide(2 * np.diag(matrix), denominator, out=np.zeros(4), where=denominator > 0)
        f1s.append(float(f1.mean()))
    return {"method": "resample whole test houses with replacement",
            "repetitions": repetitions, "seed": seed, "houses": len(groups),
            "accuracy_95_percent": np.quantile(accuracies, [.025, .975]).tolist(),
            "macro_f1_95_percent": np.quantile(f1s, [.025, .975]).tolist()}
