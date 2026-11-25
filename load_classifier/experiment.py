"""Prepare data, compare models on development houses, and fit the selected model."""

import json
import platform
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits

from .data import (LABELS, MEASUREMENTS, duplicate_groups, file_hash, make_splits, read_metadata,
                   validate_splits, validate_windows, write_json)
from .features import FEATURE_NAMES, STATIC_FEATURES, aggregate_windows
from .metrics import classification_scores
from .models import candidates


def prepare(csv_path, metadata_path, output):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Prepared-data directory is not empty: {output}")
    metadata = read_metadata(metadata_path)
    windows = validate_windows(pd.read_csv(csv_path), metadata)
    summaries = aggregate_windows(windows, validate=False)
    records = metadata.merge(summaries, on="id", validate="one_to_one")
    splits = make_splits(metadata, summaries)
    _, duplicate_summary_groups = duplicate_groups(metadata, summaries)
    output.mkdir(parents=True, exist_ok=True)
    records.to_csv(output / "recordings.csv", index=False)
    splits.to_csv(output / "splits.csv", index=False)
    audit = {
        "raw_csv_sha256": file_hash(csv_path), "metadata_sha256": file_hash(metadata_path),
        "raw_csv_bytes": Path(csv_path).stat().st_size,
        "windows": len(windows), "recordings": len(records), "houses": int(metadata.location.nunique()),
        "sampling_hz": sorted(metadata.sampling_hz.unique().tolist()),
        "window_seconds_min": float((windows.t_end_s - windows.t_start_s).min()),
        "window_seconds_max": float((windows.t_end_s - windows.t_start_s).max()),
        "windows_per_recording": {"min": int(windows.groupby("id").size().min()),
                                  "max": int(windows.groupby("id").size().max())},
        "duplicate_recording_window_keys": int(windows.duplicated(["id", "window_idx"]).sum()),
        "exact_duplicate_summary_groups": duplicate_summary_groups,
        "duplicate_summary_excess_recordings": int(records.duplicated(list(FEATURE_NAMES)).sum()),
        "metadata_matches_all_recordings": True,
        "load_counts": {k: int(v) for k, v in metadata.load.value_counts().items()},
        "appliance_to_load": {str(k): sorted(v.load.unique().tolist())
                              for k, v in metadata.groupby("appliance_type")},
        "raw_missing_values": {k: int(v) for k, v in windows.isna().sum().items() if v},
        "summary_missing_values": {k: int(v) for k, v in records.isna().sum().items() if v},
        "measurements": list(MEASUREMENTS), "features": list(FEATURE_NAMES),
        "excluded_inputs": [c for c in windows.columns if c not in MEASUREMENTS],
        "timestamp_use": "Order windows; use first/last 20% of elapsed time, slope per second and adjacent change per second. Raw time, ID and duration are not predictors.",
        "collection_time_values": sorted(metadata.collection_time.unique().tolist()),
        "partition_counts": {},
        "recordings_sha256": file_hash(output / "recordings.csv"),
        "splits_sha256": file_hash(output / "splits.csv"),
        "split_protocol": {"outer_folds": 5, "test_fold": 0, "seed": 42,
                           "development_cv_folds": 4, "development_cv_seed": 43,
                           "group": "house; development houses sharing exact summaries stay in one CV fold", "stratify": "load"},
    }
    for name, part in splits.groupby("partition"):
        audit["partition_counts"][name] = {
            "recordings": len(part), "houses": int(part.location.nunique()),
            "classes": {k: int(v) for k, v in part.load.value_counts().items()},
            "house_ids": sorted(part.location.unique().tolist()),
        }
    write_json(output / "audit.json", audit)
    return audit


def load_prepared(directory):
    directory = Path(directory)
    audit = json.loads((directory / "audit.json").read_text())
    for filename, key in (("recordings.csv", "recordings_sha256"), ("splits.csv", "splits_sha256")):
        if file_hash(directory / filename) != audit[key]:
            raise ValueError(f"Prepared {filename} has changed since the audit")
    records, splits = pd.read_csv(directory / "recordings.csv"), pd.read_csv(directory / "splits.csv")
    validate_splits(records, splits)
    if list(audit["features"]) != list(FEATURE_NAMES):
        raise ValueError("Prepared feature schema differs from this code")
    if records[list(FEATURE_NAMES)].isin([np.inf, -np.inf]).any().any():
        raise ValueError("Prepared features contain infinite values")
    grouping, _ = duplicate_groups(records, records[["id", *FEATURE_NAMES]])
    expected_groups = pd.Series(grouping.to_numpy(), index=records.id)
    if "group_id" not in splits or not splits.group_id.eq(splits.id.map(expected_groups)).all():
        raise ValueError("Split groups do not match duplicate-connected houses")
    return records.merge(splits[["id", "partition", "cv_fold"]], on="id", validate="one_to_one"), audit


def input_fingerprint(audit):
    return {key: audit[key] for key in ("raw_csv_sha256", "metadata_sha256", "recordings_sha256", "splits_sha256")}


def compare(directory, run):
    run = Path(run)
    if run.exists() and any(run.iterdir()):
        raise FileExistsError(f"Run directory is not empty: {run}")
    records, audit = load_prepared(directory)
    development = records[records.partition.eq("development")].copy()
    run.mkdir(parents=True, exist_ok=True)
    fold_rows, oof_rows, warning_rows = [], [], []
    specs = candidates()
    # No test rows are passed into this loop, preprocessing, model fitting or selection.
    with threadpool_limits(limits=1):
        for name, spec in specs.items():
            for fold in range(4):
                train = development[development.cv_fold.ne(fold)]
                validation = development[development.cv_fold.eq(fold)]
                model = clone(spec)
                started = time.perf_counter()
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always", ConvergenceWarning)
                    model.fit(train[list(FEATURE_NAMES)], train.load)
                prediction = model.predict(validation[list(FEATURE_NAMES)])
                metrics = classification_scores(validation.load, prediction)
                fold_rows.append({"model": name, "fold": fold, "features": "temporal",
                                  "accuracy": metrics["accuracy"], "macro_f1": metrics["macro_f1"],
                                  "fit_seconds": time.perf_counter() - started})
                oof_rows.extend({"id": int(row.id), "location": row.location, "actual": row.load,
                                 "predicted": str(guess), "model": name, "fold": fold}
                                for row, guess in zip(validation.itertuples(), prediction))
                warning_rows.extend({"model": name, "fold": fold, "message": str(w.message)} for w in caught)
            recent = [r for r in fold_rows if r["model"] == name]
            print(f"{name}: CV macro-F1 {np.mean([r['macro_f1'] for r in recent]):.4f}", flush=True)
    folds = pd.DataFrame(fold_rows)
    summary = folds.groupby("model", sort=False).agg(
        macro_f1_mean=("macro_f1", "mean"), macro_f1_std=("macro_f1", "std"),
        accuracy_mean=("accuracy", "mean"), accuracy_std=("accuracy", "std"),
        fit_seconds_total=("fit_seconds", "sum"),
    ).reset_index().sort_values(["macro_f1_mean", "accuracy_mean", "model"], ascending=[False, False, True])
    selected = str(summary.iloc[0].model)
    # A development-only ablation measures whether time summaries help the selected family.
    ablation = []
    # For a compact candidate, remove its column selection before fitting median-only inputs.
    ablation_spec = clone(specs[selected])
    if "select" in ablation_spec.named_steps:
        ablation_spec.steps = [(key, value) for key, value in ablation_spec.steps if key != "select"]
    with threadpool_limits(limits=1):
        for fold in range(4):
            train = development[development.cv_fold.ne(fold)]
            validation = development[development.cv_fold.eq(fold)]
            model = clone(ablation_spec)
            model.fit(train[list(STATIC_FEATURES)], train.load)
            score = classification_scores(validation.load, model.predict(validation[list(STATIC_FEATURES)]))
            ablation.append({"fold": fold, "accuracy": score["accuracy"], "macro_f1": score["macro_f1"]})
    folds.to_csv(run / "cv_folds.csv", index=False)
    summary.to_csv(run / "comparison.csv", index=False)
    pd.DataFrame(oof_rows).to_csv(run / "cv_predictions.csv", index=False)
    selection = {
        "selected_model": selected, "selection_metric": "mean validation macro-F1 across four house-grouped folds",
        "training_seed": 42, "test_used_in_selection": False,
        "input_fingerprint": input_fingerprint(audit), "features": list(FEATURE_NAMES),
        "candidate_parameters": {name: {key: str(value) for key, value in spec.get_params().items()
                                        if "__" in key} for name, spec in specs.items()},
        "warnings": warning_rows,
        "ablation": {"model": selected, "median_only_folds": ablation,
                     "median_only_macro_f1_mean": float(np.mean([r["macro_f1"] for r in ablation])),
                     "temporal_macro_f1_mean": float(summary.iloc[0].macro_f1_mean),
                     "purpose": "Development-only comparison; the requested model uses temporal summaries."},
        "comparison_sha256": file_hash(run / "comparison.csv"),
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "pandas": pd.__version__, "scikit_learn": sklearn.__version__},
    }
    write_json(run / "selection.json", selection)
    return selection


def train(directory, run):
    run = Path(run)
    if (run / "model.joblib").exists() or (run / "test_metrics.json").exists():
        raise FileExistsError("This run already has a trained or evaluated model")
    records, audit = load_prepared(directory)
    selection = json.loads((run / "selection.json").read_text())
    if selection["input_fingerprint"] != input_fingerprint(audit):
        raise ValueError("Input data changed after model comparison")
    if file_hash(run / "comparison.csv") != selection["comparison_sha256"]:
        raise ValueError("Model comparison changed after selection")
    development = records[records.partition.eq("development")]
    model = candidates()[selection["selected_model"]]
    current_parameters = {key: str(value) for key, value in model.get_params().items() if "__" in key}
    if current_parameters != selection["candidate_parameters"][selection["selected_model"]]:
        raise ValueError("Selected model parameters changed after comparison")
    started = time.perf_counter()
    with threadpool_limits(limits=1):
        model.fit(development[list(FEATURE_NAMES)], development.load)
    bundle = {"pipeline": model, "features": list(FEATURE_NAMES), "labels": list(LABELS),
              "selected_model": selection["selected_model"], "input_fingerprint": input_fingerprint(audit),
              "selection_sha256": file_hash(run / "selection.json"),
              "version": 1}
    joblib.dump(bundle, run / "model.joblib", compress=3)
    imputer = model.named_steps["impute"]
    transformed_names = imputer.get_feature_names_out()
    classifier = model.named_steps["classifier"]
    if hasattr(classifier, "feature_importances_"):
        pd.DataFrame({"feature": transformed_names, "importance": classifier.feature_importances_}) \
            .sort_values("importance", ascending=False).to_csv(run / "feature_importance.csv", index=False)
    apparent = classification_scores(development.load, model.predict(development[list(FEATURE_NAMES)]))
    write_json(run / "training.json", {
        "model": selection["selected_model"], "recordings": len(development),
        "houses": int(development.location.nunique()), "features": len(FEATURE_NAMES),
        "fit_seconds": time.perf_counter() - started, "test_used_for_training": False,
        "fitted_input_count": len(transformed_names), "apparent_training_metrics": apparent,
        "model_sha256": file_hash(run / "model.joblib"),
        "selection_sha256": bundle["selection_sha256"],
        "input_fingerprint": input_fingerprint(audit),
    })
    return bundle
