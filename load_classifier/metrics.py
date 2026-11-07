"""Accuracy and per-class precision/recall/F1 with fixed label orders."""
import numpy as np

from .data import BEHAVIOURS, CIRCUITS


def classification_scores(targets, probabilities, labels):
    targets = np.asarray(targets)
    probabilities = np.asarray(probabilities)
    if (targets.ndim != 1 or len(targets) == 0 or not np.issubdtype(targets.dtype, np.integer)
            or probabilities.shape != (len(targets), len(labels))
            or not np.isfinite(probabilities).all() or (probabilities < 0).any()
            or not np.allclose(probabilities.sum(axis=1), 1.)
            or (targets < 0).any() or (targets >= len(labels)).any()):
        raise ValueError("Invalid targets or class probabilities")
    predicted = probabilities.argmax(axis=1)
    matrix = np.zeros((len(labels), len(labels)), dtype=int)
    np.add.at(matrix, (targets, predicted), 1)
    per_class = {}
    for i, label in enumerate(labels):
        actual, guessed, correct = int(matrix[i].sum()), int(matrix[:, i].sum()), int(matrix[i, i])
        precision = correct / guessed if guessed else 0.
        recall = correct / actual if actual else 0.
        f1 = 2 * correct / (actual + guessed) if actual + guessed else 0.
        per_class[label] = {"precision": precision, "recall": recall, "f1": f1, "support": actual}
    return {
        "accuracy": float(np.mean(targets == predicted)),
        "correct": int(np.sum(targets == predicted)),
        "macro_f1": float(np.mean([row["f1"] for row in per_class.values()])),
        "per_class": per_class,
        "confusion_matrix": matrix.tolist(),
        "labels": list(labels),
    }


def score(behaviour, circuit, probabilities):
    if len(behaviour) != len(circuit):
        raise ValueError("Both targets must describe the same examples")
    b = classification_scores(behaviour, probabilities["behaviour"], BEHAVIOURS)
    c = classification_scores(circuit, probabilities["circuit"], CIRCUITS)
    joint = ((probabilities["behaviour"].argmax(axis=1) == behaviour)
             & (probabilities["circuit"].argmax(axis=1) == circuit))
    return {"rows": len(behaviour), "behaviour": b, "circuit": c,
            "joint_accuracy": float(joint.mean()), "joint_correct": int(joint.sum())}
