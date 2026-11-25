import json

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import ExtraTreesClassifier

from load_classifier import experiment
from load_classifier.data import LABELS, MEASUREMENTS
from load_classifier.features import FEATURE_NAMES, aggregate_windows
from load_classifier.models import pipeline
from load_classifier.predict import load_bundle, predict_csv, predict_records


def test_imputer_and_scaler_fit_only_the_training_values():
    from sklearn.linear_model import LogisticRegression
    train = pd.DataFrame({"x": [1., 3., 1., 3.], "missing": [np.nan, 2., np.nan, 4.]})
    model = pipeline(LogisticRegression(), scale=True).fit(train, ["R", "I", "R", "I"])
    imputer, scaler = model.named_steps["impute"], model.named_steps["scale"]
    np.testing.assert_allclose(imputer.statistics_, [2, 3])
    np.testing.assert_allclose(scaler.mean_[:2], [2, 3])
    before = imputer.statistics_.copy()
    model.predict(pd.DataFrame({"x": [100000.], "missing": [np.nan]}))
    np.testing.assert_array_equal(imputer.statistics_, before)


def test_training_saving_and_unlabelled_inference(tmp_path, monkeypatch):
    from load_classifier.evaluate import evaluate, verify
    from load_classifier.metrics import house_bootstrap
    entries, rows = [], []
    for house in range(25):
        for label, load in enumerate(LABELS):
            recording = house * 4 + label + 1
            entries.append({"id": str(recording), "meta": {
                "appliance": {"load": load, "type": "sample"}, "location": f"house{house}",
                "header": {"sampling_frequency": "30000Hz", "collection_time": "July, 2013"}}})
            for i in range(6):
                row = {name: float(label + 1) + .01 * i for name in MEASUREMENTS}
                row.update(id=recording, window_idx=i, start_sample=i * 500, end_sample=(i + 1) * 500 - 1,
                           t_start_s=i / 60, t_end_s=(i + 1) / 60, load=load,
                           location=f"house{house}", appliance_type="sample")
                rows.append(row)
    source, metadata, data, run = (tmp_path / name for name in ("windows.csv", "meta.json", "data", "run"))
    pd.DataFrame(rows).to_csv(source, index=False)
    metadata.write_text(json.dumps(entries))
    monkeypatch.setattr(experiment, "candidates", lambda: {
        "small_tree": pipeline(ExtraTreesClassifier(n_estimators=10, random_state=42))})
    monkeypatch.setattr("load_classifier.evaluate.house_bootstrap", lambda a, p, h: house_bootstrap(a, p, h, repetitions=30))
    experiment.prepare(source, metadata, data)
    selection = experiment.compare(data, run)
    assert selection["test_used_in_selection"] is False
    experiment.train(data, run)
    evaluate(data, run)
    assert verify(data, run)["passed"]
    model = load_bundle(run / "model.joblib")
    unlabelled = pd.DataFrame(rows).drop(columns=["load", "appliance_type", "location"])
    input_path, output_path = tmp_path / "unlabelled.csv", tmp_path / "answers.csv"
    unlabelled.to_csv(input_path, index=False)
    actual = predict_csv(run / "model.joblib", input_path, output_path)
    expected = predict_records(model, aggregate_windows(unlabelled))
    pd.testing.assert_frame_equal(expected, actual)
    assert set(actual.columns) == {"id", "predicted_load", *[f"probability_{name}" for name in LABELS]}
    with pytest.raises(FileExistsError):
        predict_csv(run / "model.joblib", input_path, input_path)
    with pytest.raises(FileExistsError):
        experiment.train(data, run)
