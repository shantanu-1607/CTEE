import numpy as np
import pytest

from load_classifier.metrics import classification_scores, house_bootstrap


def test_metrics_recount_a_known_confusion_matrix():
    actual = ["R", "R", "I", "I", "RC", "RC", "RLC", "RLC"]
    predicted = ["R", "I", "I", "I", "RC", "RLC", "RC", "RLC"]
    result = classification_scores(actual, predicted)
    assert result["correct"] == 5
    assert result["accuracy"] == 5 / 8
    assert result["confusion_matrix"] == [[1, 1, 0, 0], [0, 2, 0, 0], [0, 0, 1, 1], [0, 0, 1, 1]]
    assert result["macro_f1"] == pytest.approx((2 / 3 + 4 / 5 + .5 + .5) / 4)


def test_bootstrap_resamples_entire_houses():
    result = house_bootstrap(["R", "I", "RC", "RLC"], ["R", "I", "R", "R"],
                             ["a", "a", "b", "b"], repetitions=500)
    assert result["houses"] == 2
    assert result["accuracy_95_percent"] == [0, 1]


def test_unknown_labels_are_rejected():
    with pytest.raises(ValueError, match="Unknown"):
        classification_scores(["L"], ["I"])
