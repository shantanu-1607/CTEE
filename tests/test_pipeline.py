import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from load_classifier.__main__ import predict
from load_classifier.data import FEATURES, apply_splits, arrays, read_dataset
from load_classifier.model import TwoHeadMLP
from load_classifier.train import TrainingConfig, train_run


def test_prediction_ignores_supplied_labels_and_column_order(tmp_path):
    model = TwoHeadMLP(FEATURES, (4, 3))
    x = np.random.default_rng(3).normal(size=(4, len(FEATURES)))
    frame = pd.DataFrame(x, columns=FEATURES)
    frame["sample_id"] = ["a", "b", "c", "d"]
    frame["behaviour"] = ["CP"] * 4
    frame["circuit"] = ["RLC"] * 4
    model.save(tmp_path / "model.npz", {})
    input_path = tmp_path / "measurements.csv"
    output_path = tmp_path / "predictions.csv"
    frame[list(reversed(frame.columns))].to_csv(input_path, index=False)
    predict(tmp_path / "model.npz", input_path, output_path)
    result = pd.read_csv(output_path)
    expected = model.predict_proba(x)
    for head in ("behaviour", "circuit"):
        probability_columns = [name for name in result if name.startswith(f"p_{head}_")]
        np.testing.assert_allclose(result[probability_columns].to_numpy(), expected[head], atol=1e-15)
    with pytest.raises(FileExistsError):
        predict(tmp_path / "model.npz", input_path, output_path)


def test_prediction_rejects_missing_feature(tmp_path):
    TwoHeadMLP(FEATURES, (4,)).save(tmp_path / "model.npz", {})
    pd.DataFrame({"Vrms": [220.]}).to_csv(tmp_path / "input.csv", index=False)
    with pytest.raises(ValueError, match="Missing measurements"):
        predict(tmp_path / "model.npz", tmp_path / "input.csv", tmp_path / "output.csv")


def test_training_cannot_overwrite_an_existing_run(tmp_path):
    (tmp_path / "model.npz").write_bytes(b"keep this checkpoint")
    with pytest.raises(FileExistsError):
        train_run(None, TrainingConfig(), tmp_path, None, None)
    assert (tmp_path / "model.npz").read_bytes() == b"keep this checkpoint"


def test_saved_baseline_predictions_and_training_scaler():
    data = apply_splits(read_dataset(), pd.read_csv("data/splits.csv"))
    model, _ = TwoHeadMLP.load("runs/baseline/model.npz")
    x_train, _, _ = arrays(data[data["split"] == "train"])
    np.testing.assert_array_equal(model.mean, x_train.mean(axis=0))
    np.testing.assert_array_equal(model.scale, x_train.std(axis=0))
    test = data[data["split"] == "test"]
    exported = pd.read_csv("runs/baseline/test_predictions.csv")
    assert test["sample_id"].tolist() == exported["sample_id"].tolist()
    probabilities = model.predict_proba(arrays(test)[0])
    for head in ("behaviour", "circuit"):
        columns = [name for name in exported if name.startswith(f"p_{head}_")]
        np.testing.assert_allclose(exported[columns].to_numpy(), probabilities[head], atol=1e-15)
    metrics = json.loads(Path("runs/baseline/metrics.json").read_text())
    joint = exported.behaviour.eq(exported.predicted_behaviour) & exported.circuit.eq(exported.predicted_circuit)
    assert int(joint.sum()) == metrics["test"]["joint_correct"] == 2392
