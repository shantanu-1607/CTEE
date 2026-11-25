"""Run a locally trained model on new, complete recording feature windows."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .data import LABELS
from .features import FEATURE_NAMES, aggregate_windows


def load_bundle(path):
    # joblib/pickle files must come from a trusted source; never load arbitrary uploads.
    bundle = joblib.load(path)
    if (bundle.get("version") != 1 or bundle.get("features") != list(FEATURE_NAMES)
            or bundle.get("labels") != list(LABELS)):
        raise ValueError("Unsupported saved model schema")
    return bundle


def predict_records(bundle, features):
    x = features[bundle["features"]]
    model = bundle["pipeline"]
    predictions = model.predict(x)
    probabilities = model.predict_proba(x)
    if not np.isfinite(probabilities).all() or not np.allclose(probabilities.sum(axis=1), 1):
        raise ValueError("Invalid model probabilities")
    output = pd.DataFrame({"id": features.id.to_numpy(), "predicted_load": predictions})
    for label, column in zip(model.classes_, probabilities.T):
        output[f"probability_{label}"] = column
    return output[["id", "predicted_load", *[f"probability_{label}" for label in LABELS]]]


def predict_csv(model_path, input_path, output_path):
    input_path, output_path = Path(input_path), Path(output_path)
    if input_path.resolve() == output_path.resolve() or output_path.exists():
        raise FileExistsError("Prediction output must be a new file")
    bundle = load_bundle(model_path)
    features = aggregate_windows(pd.read_csv(input_path))
    result = predict_records(bundle, features)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    return result
