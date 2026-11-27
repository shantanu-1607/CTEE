"""Verify frozen results without fitting, selecting or tuning another model."""
import hashlib
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from pipeline import ROOT, LABELS, scores

def main():
    r = ROOT / "results"
    data = pd.read_csv(r / "daily_features.csv")
    splits = pd.read_csv(r / "split_manifest.csv")
    pred = pd.read_csv(r / "test_predictions.csv")
    metrics = json.loads((r / "test_metrics.json").read_text())
    cv = pd.read_csv(r / "cv_assignments.csv")
    bundle = joblib.load(ROOT / "artifacts" / "model.joblib")
    features = bundle["metadata"]["features"]
    assert len(features) == 69 and len(set(features)) == 69
    assert data.sample_id.is_unique and splits.sample_id.is_unique
    assert splits.groupby("date").split.nunique().max() == 1
    assert data.profile_hash.is_unique and data.feature_hash.is_unique
    dev = data[data.sample_id.isin(splits.loc[splits.split.eq("development"), "sample_id"])]
    test = data[data.sample_id.isin(splits.loc[splits.split.eq("test"), "sample_id"])]
    assert set(dev.sample_id).isdisjoint(test.sample_id)
    assert set(dev.profile_hash).isdisjoint(test.profile_hash)
    assert set(dev.feature_hash).isdisjoint(test.feature_hash)
    assert (pd.Timestamp(test.date.min()) - pd.Timestamp(dev.date.max())).days > 7
    assert set(pred.sample_id) == set(test.sample_id)
    assert set(test.label) == set(dev.label) == set(LABELS)
    for fold, rows in cv.groupby("fold"):
        tr, va = [rows[rows.role.eq(k)] for k in ["train", "validation"]]
        assert set(tr.sample_id).isdisjoint(va.sample_id)
        assert set(rows.sample_id).issubset(set(dev.sample_id))
        assert (pd.Timestamp(va.date.min())-pd.Timestamp(tr.date.max())).days > 7
    # Compare learned imputer with development medians only.
    saved_imputer = bundle["pipeline"].steps[0][1]
    np.testing.assert_allclose(saved_imputer.statistics_, np.nanmedian(dev[features], axis=0), rtol=1e-10)
    preprocessing = joblib.load(ROOT / "artifacts" / "preprocessing.joblib")
    np.testing.assert_allclose(preprocessing.transform(test[features]), bundle["pipeline"][:-1].transform(test[features]))
    # This is a serialization round-trip check, not another model experiment.
    restored = bundle["pipeline"].predict(test[features])
    recorded = pred.set_index("sample_id").loc[test.sample_id]
    np.testing.assert_array_equal(restored, recorded.predicted)
    np.testing.assert_array_equal(recorded.label, test.label)
    for k,v in scores(recorded.label, recorded.predicted).items():
        assert abs(metrics[k]-v) < 1e-12
    np.testing.assert_array_equal(metrics["confusion_matrix"], confusion_matrix(recorded.label, recorded.predicted, labels=LABELS))
    assert metrics["correct"] == recorded.correct.sum()
    verification = {"passed": True, "unique_examples": len(data), "development_examples": len(dev), "test_examples": len(test), "features": len(features), "checks": ["dates stay in one partition", "no IDs or exact profile/feature duplicates cross splits", "seven-day gaps in holdout and all CV folds", "all CV samples belong to development", "all classes represented", "imputer matches development-only medians", "separate preprocessing equals pipeline preprocessing", "serialized model reproduces frozen test predictions", "metrics and confusion matrix recomputed from predictions"], "test_metrics_sha256": hashlib.sha256((r / "test_metrics.json").read_bytes()).hexdigest()}
    (r / "verification.json").write_text(json.dumps(verification, indent=2), encoding="utf-8")
    print(json.dumps(verification, indent=2))

if __name__ == "__main__": main()
