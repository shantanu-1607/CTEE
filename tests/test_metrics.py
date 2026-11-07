import numpy as np
import pytest

from load_classifier.metrics import classification_scores, score


def test_confusion_and_macro_f1_by_hand():
    targets = np.array([0, 0, 1, 1, 2, 2])
    probabilities = np.eye(3)[[0, 1, 1, 1, 0, 2]]
    metrics = classification_scores(targets, probabilities, ("a", "b", "c"))
    assert metrics["confusion_matrix"] == [[1, 1, 0], [0, 2, 0], [1, 0, 1]]
    assert metrics["accuracy"] == 4 / 6
    assert metrics["macro_f1"] == pytest.approx((.5 + .8 + 2 / 3) / 3)


def test_joint_accuracy_requires_both_answers_on_same_row():
    behaviour = np.array([0, 1, 2, 0])
    circuit = np.array([0, 1, 2, 3])
    probabilities = {"behaviour": np.eye(3)[[0, 1, 1, 0]], "circuit": np.eye(7)[[0, 0, 2, 3]]}
    metrics = score(behaviour, circuit, probabilities)
    assert metrics["behaviour"]["accuracy"] == .75
    assert metrics["circuit"]["accuracy"] == .75
    assert metrics["joint_accuracy"] == .5


def test_invalid_probabilities_are_rejected():
    with pytest.raises(ValueError):
        classification_scores(np.array([0]), np.array([[.8, .8]]), ("a", "b"))
