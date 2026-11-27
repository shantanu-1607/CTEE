"""Model 3: classify a completed daily profile from one of three known meters.

No calendar, building ID, absolute timestamp or availability flag is a predictor.
All learned transformations are fitted inside each training fold.
"""
import argparse
import hashlib
import json
import math
import platform
import time
import zipfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
SOURCES = {"acad_build_mains.csv": "Academic", "facilities_build_mains.csv": "Facilities", "mess_build_mains.csv": "Mess"}
LABELS = list(SOURCES.values())
SEED = 20261127
MIN_MINUTES = 1152  # 80% of a 1440-minute local day
META = ["sample_id", "date", "label", "valid_minutes", "valid_hours", "profile_hash", "feature_hash"]

def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False, default=str), encoding="utf-8")

def prepare_readings(df):
    required = {"timestamp", "power", "power_factor"}
    if not required.issubset(df):
        raise ValueError(f"Required CSV columns: {sorted(required)}")
    df = df[list(sorted(required))].copy()
    for k in required:
        df[k] = pd.to_numeric(df[k], errors="coerce")
    if df.timestamp.isna().any() or (df.timestamp % 60 != 0).any():
        raise ValueError("Invalid timestamp or samples not on a one-minute grid")
    # Identical duplicate rows collapse; conflicting timestamps are rejected.
    df = df.drop_duplicates()
    if df.timestamp.duplicated().any():
        raise ValueError("Conflicting duplicate timestamps")
    df = df.sort_values("timestamp")
    df.index = pd.to_datetime(df.timestamp, unit="s", utc=True).dt.tz_convert("Asia/Kolkata")
    df.loc[~np.isfinite(df.power) | df.power.lt(0), "power"] = np.nan
    df.loc[~np.isfinite(df.power_factor) | df.power_factor.abs().gt(1), "power_factor"] = np.nan
    return df

def summarize_day(day):
    """Return fixed electrical features and quality; needs a full local day."""
    grid = pd.date_range(day.index[0].normalize(), periods=1440, freq="min")
    frame = day[["power", "power_factor"]].reindex(grid)
    valid = frame.notna().all(axis=1)
    hours = valid.groupby(frame.index.hour).sum().reindex(range(24), fill_value=0)
    quality = {"valid_minutes": int(valid.sum()), "valid_hours": int(hours.ge(30).sum())}
    if quality["valid_minutes"] < MIN_MINUTES or quality["valid_hours"] < 20:
        return None, quality
    # Use the same paired observations for both signals. Never fill from another day.
    frame = frame.where(valid, np.nan)
    p, pf = frame.power, frame.power_factor
    f = {}
    for col, values in [("power", p), ("pf", pf)]:
        f.update({f"{col}_{k}": float(v) for k, v in {
            "mean": values.mean(), "std": values.std(ddof=0), "min": values.min(),
            "q10": values.quantile(.1), "median": values.median(),
            "q90": values.quantile(.9), "max": values.max(),
        }.items()})
    f["power_iqr"] = float(p.quantile(.75) - p.quantile(.25))
    f["power_cv"] = float(p.std(ddof=0) / p.mean()) if p.mean() > 0 else np.nan
    f["pf_negative_fraction"] = float(pf.dropna().lt(0).mean())
    ramps = p.diff().abs()  # reindexed minutes: no jump across a missing interval
    f["power_ramp_mean"] = float(ramps.mean())
    f["power_ramp_q90"] = float(ramps.quantile(.9))
    night = p[(p.index.hour < 6)].mean()
    daytime = p[(p.index.hour >= 9) & (p.index.hour < 18)].mean()
    f["day_night_power_ratio"] = float(daytime / night) if night > 0 else np.nan
    f["peak_mean_power_ratio"] = float(p.max() / p.mean()) if p.mean() > 0 else np.nan
    for h in range(24):
        enough = hours.loc[h] >= 30
        for col, prefix in [("power", "power"), ("power_factor", "pf")]:
            f[f"{prefix}_hour_{h:02d}"] = float(frame.loc[frame.index.hour == h, col].mean()) if enough else np.nan
    # Canonical bytes include minute position and missingness, but no date or label.
    canonical = np.nan_to_num(frame.to_numpy(dtype="<f8"), nan=-1e30)
    quality["profile_hash"] = hashlib.sha256(canonical.tobytes()).hexdigest()
    return f, quality

def extract(archive):
    RESULTS.mkdir(parents=True, exist_ok=True)
    expected = json.loads((RESULTS / "dataset_audit.json").read_text())["archive_sha256"]
    with archive.open("rb") as stream:
        actual = hashlib.file_digest(stream, "sha256").hexdigest()
    if actual != expected:
        raise ValueError("Archive changed since inspection")
    rows, quality_rows = [], []
    with zipfile.ZipFile(archive) as z:
        # Source documentation describes 0/1 presence flags; use only to reject
        # unavailable measurements. These flags and other buildings never enter X.
        status = pd.read_csv(z.open("data_present_status_buildings.csv"), usecols=["timestamp", *LABELS])
        status.index = pd.DatetimeIndex(pd.to_datetime(status.pop("timestamp"))).tz_localize("Asia/Kolkata")
        if not status.index.is_unique:
            raise ValueError("Duplicate availability keys")
        for filename, label in SOURCES.items():
            df = prepare_readings(pd.read_csv(z.open(filename), usecols=["timestamp", "power", "power_factor"]))
            present = status[label].reindex(df.index).eq(1)
            df.loc[~present, ["power", "power_factor"]] = np.nan
            for date, day in df.groupby(df.index.normalize(), sort=True):
                features, quality = summarize_day(day)
                key = f"{label}_{date.date()}"
                quality_rows.append({"sample_id": key, "date": str(date.date()), "label": label, **quality, "eligible": features is not None})
                if features is not None:
                    vals = np.nan_to_num(np.array(list(features.values()), dtype="<f8"), nan=-1e30)
                    fh = hashlib.sha256(np.round(vals, 8).tobytes()).hexdigest()
                    rows.append({"sample_id": key, "date": str(date.date()), "label": label, **quality, "feature_hash": fh, **features})
            print(f"Extracted {label}", flush=True)
    data = pd.DataFrame(rows).sort_values(["date", "label"]).reset_index(drop=True)
    # Keep earliest duplicate. Any identical representation with contradictory
    # class labels is unusable and removed entirely rather than arbitrarily relabeled.
    conflict = data.groupby("feature_hash").label.transform("nunique").gt(1)
    repeated = data.duplicated("profile_hash") | data.duplicated("feature_hash")
    excluded = data.loc[conflict | repeated, META].copy()
    excluded["reason"] = np.where(conflict.loc[excluded.index], "conflicting_labels", "duplicate_profile_or_features")
    data = data.loc[~conflict & ~repeated].reset_index(drop=True)
    data.to_csv(RESULTS / "daily_features.csv", index=False, float_format="%.12g")
    pd.DataFrame(quality_rows).to_csv(RESULTS / "day_quality.csv", index=False)
    excluded.to_csv(RESULTS / "duplicate_exclusions.csv", index=False)
    dictionary = []
    for col in data.columns.difference(META, sort=False):
        meaning = "Daily electrical summary"
        if "_hour_" in col:
            meaning = "Mean of paired valid samples during this local hour (at least 30 minutes)"
        elif "ramp" in col:
            meaning = "Absolute change between adjacent valid one-minute power measurements"
        elif col == "day_night_power_ratio":
            meaning = "Mean power 09:00-17:59 / mean power 00:00-05:59"
        elif col == "pf_negative_fraction":
            meaning = "Fraction of measured power factors below zero; sign retained without interpreting reactive load type"
        unit = "W" if col.startswith("power_") and col not in ["power_cv"] else "ratio"
        if col == "power_cv": meaning = "Power standard deviation / mean power"
        if col == "peak_mean_power_ratio": meaning = "Maximum power / mean power"
        dictionary.append({"feature": col, "unit": unit, "definition": meaning})
    pd.DataFrame(dictionary).to_csv(RESULTS / "feature_dictionary.csv", index=False)
    return data

def partition(data):
    dates = np.array(sorted(data.date.unique()))
    boundary = pd.Timestamp(str(dates[math.floor(.8 * len(dates))]))
    gap_start = boundary - pd.Timedelta(days=7)
    d = pd.to_datetime(data.date)
    split = np.where(d >= boundary, "test", np.where(d >= gap_start, "embargo", "development"))
    manifest = data[META].assign(split=split)
    manifest.to_csv(RESULTS / "split_manifest.csv", index=False)
    dev = data.loc[split == "development"].copy()
    test = data.loc[split == "test"].copy()
    for subset in (dev, test):
        if set(subset.label) != set(LABELS): raise ValueError("A split lacks a class")
    # Expanding-time folds, always all meters on a given date in the same fold.
    days = np.array(sorted(dev.date.unique()))
    cuts = [len(days)//2, 2*len(days)//3, 5*len(days)//6, len(days)]
    folds, fold_rows = [], []
    for i in range(3):
        start = pd.Timestamp(str(days[cuts[i]]))
        end = pd.Timestamp(str(days[cuts[i+1]-1]))
        tr = np.flatnonzero(pd.to_datetime(dev.date).lt(start - pd.Timedelta(days=7)))
        va = np.flatnonzero(pd.to_datetime(dev.date).between(start, end))
        assert set(dev.iloc[tr].label) == set(LABELS) == set(dev.iloc[va].label)
        folds.append((tr, va))
        for role, indices in [("train", tr), ("validation", va)]:
            fold_rows.extend({"fold": i+1, "sample_id": dev.iloc[j].sample_id, "role": role, "date": dev.iloc[j].date} for j in indices)
    pd.DataFrame(fold_rows).to_csv(RESULTS / "cv_assignments.csv", index=False)
    protocol = {"seed": SEED, "target": "source building of a completed daily electrical profile", "labels": LABELS, "timezone": "Asia/Kolkata", "test_start": str(boundary.date()), "embargo_start": str(gap_start.date()), "test_rule": "last 20% of eligible distinct dates; all buildings share each date's partition", "cv": "3 expanding chronological folds, each with seven calendar days purged before validation", "selection": "highest mean validation macro F1; ties follow fixed candidate order", "minimum_paired_minutes": MIN_MINUTES, "minimum_hours_with_30_paired_minutes": 20, "predictors": "power and signed power factor summaries only", "excluded_predictors": ["source filename", "building", "calendar activity", "working_day", "date", "year", "current", "voltage", "frequency", "availability flags", "coverage counts"], "test_policy": "select and serialize the development-fitted model before one final test evaluation; no refit on test"}
    save_json(RESULTS / "protocol.json", protocol)
    return dev, test, folds

def candidates():
    imputer = lambda: SimpleImputer(strategy="median", keep_empty_features=True)
    return {
        "majority_baseline": make_pipeline(imputer(), DummyClassifier(strategy="most_frequent")),
        "logistic_regression": make_pipeline(imputer(), StandardScaler(), LogisticRegression(C=1, class_weight="balanced", max_iter=3000, random_state=SEED)),
        "rbf_svm": make_pipeline(imputer(), StandardScaler(), SVC(C=10, gamma="scale", class_weight="balanced", random_state=SEED)),
        "random_forest": make_pipeline(imputer(), RandomForestClassifier(n_estimators=400, min_samples_leaf=2, max_features="sqrt", class_weight="balanced_subsample", n_jobs=2, random_state=SEED)),
        "hist_gradient_boosting": make_pipeline(imputer(), HistGradientBoostingClassifier(max_iter=200, max_leaf_nodes=15, l2_regularization=1, learning_rate=.08, early_stopping=False, random_state=SEED)),
    }

def scores(y, pred):
    return {"accuracy": float(accuracy_score(y, pred)), "balanced_accuracy": float(balanced_accuracy_score(y, pred)), "macro_f1": float(f1_score(y, pred, labels=LABELS, average="macro", zero_division=0))}

def bootstrap_by_week(test, pred, repetitions=1000):
    # Sample weeks, not individual correlated days/buildings. Approximate intervals.
    groups = pd.to_datetime(test.date).dt.to_period("W-SUN").astype(str).to_numpy()
    weeks = np.unique(groups)
    indices = {week: np.flatnonzero(groups == week) for week in weeks}
    rng = np.random.default_rng(SEED)
    draws = []
    for _ in range(repetitions):
        idx = np.concatenate([indices[w] for w in rng.choice(weeks, len(weeks), replace=True)])
        draws.append(scores(test.label.to_numpy()[idx], pred[idx]))
    return {k: {"low": float(np.quantile([d[k] for d in draws], .025)), "high": float(np.quantile([d[k] for d in draws], .975))} for k in draws[0]}

def train(data):
    if (RESULTS / "test_metrics.json").exists():
        raise RuntimeError("Final test already evaluated in this directory. Use a fresh copy for a documented reproduction.")
    dev, test, folds = partition(data)
    features = [c for c in data if c not in META]
    X, y = dev[features], dev.label
    models = candidates()
    rows, predictions = [], []
    for name, model in models.items():
        for i, (tr, va) in enumerate(folds):
            estimator = clone(model)
            start = time.perf_counter()
            estimator.fit(X.iloc[tr], y.iloc[tr])
            pred = estimator.predict(X.iloc[va])
            row = {"model": name, "fold": i+1, "train_n": len(tr), "validation_n": len(va), "train_end": dev.iloc[tr].date.max(), "validation_start": dev.iloc[va].date.min(), "validation_end": dev.iloc[va].date.max(), **scores(y.iloc[va], pred), "fit_predict_seconds": time.perf_counter()-start}
            rows.append(row)
            predictions.extend({"model": name, "fold": i+1, "sample_id": dev.iloc[j].sample_id, "actual": y.iloc[j], "predicted": p} for j, p in zip(va, pred))
            print(name, i+1, row["macro_f1"], flush=True)
    cv = pd.DataFrame(rows)
    cv.to_csv(RESULTS / "cv_scores.csv", index=False)
    pd.DataFrame(predictions).to_csv(RESULTS / "cv_predictions.csv", index=False)
    comparison = cv.groupby("model", sort=False).agg(mean_macro_f1=("macro_f1", "mean"), sd_macro_f1=("macro_f1", "std"), mean_accuracy=("accuracy", "mean")).reset_index().sort_values("mean_macro_f1", ascending=False, kind="stable")
    comparison.to_csv(RESULTS / "model_comparison.csv", index=False)
    selected = comparison.iloc[0].model
    # Feature explanation from the last DEVELOPMENT validation fold only.
    tr, va = folds[-1]
    explainer = clone(models[selected]).fit(X.iloc[tr], y.iloc[tr])
    imp = permutation_importance(explainer, X.iloc[va], y.iloc[va], scoring="f1_macro", n_repeats=5, random_state=SEED, n_jobs=1)
    pd.DataFrame({"feature": features, "mean_f1_drop": imp.importances_mean, "sd_f1_drop": imp.importances_std}).sort_values("mean_f1_drop", ascending=False).to_csv(RESULTS / "validation_feature_importance.csv", index=False)
    fitted = clone(models[selected]).fit(X, y)
    model_path = ROOT / "artifacts"
    model_path.mkdir(exist_ok=True)
    metadata = {"selected_model": selected, "features": features, "classes": fitted.classes_.tolist(), "sklearn": sklearn.__version__, "numpy": np.__version__, "pandas": pd.__version__, "python": platform.python_version(), "development_n": len(dev), "test_n": len(test), "selection_macro_f1": float(comparison.iloc[0].mean_macro_f1)}
    joblib.dump({"pipeline": fitted, "metadata": metadata}, model_path / "model.joblib", compress=3)
    joblib.dump(fitted[:-1], model_path / "preprocessing.joblib", compress=3)
    save_json(model_path / "metadata.json", metadata)
    save_json(RESULTS / "selection.json", metadata)
    # Selection and saved model now frozen. No other candidate touches the test.
    pred = fitted.predict(test[features])
    predictions = test[META].copy()
    predictions["predicted"] = pred
    predictions["correct"] = (test.label.to_numpy() == pred).astype(int)
    predictions.to_csv(RESULTS / "test_predictions.csv", index=False)
    metrics = {**scores(test.label, pred), "n": len(test), "correct": int(predictions.correct.sum()), "labels": LABELS, "confusion_matrix": confusion_matrix(test.label, pred, labels=LABELS).tolist(), "per_class": classification_report(test.label, pred, labels=LABELS, output_dict=True, zero_division=0), "approximate_week_block_bootstrap_95_ci": bootstrap_by_week(test, pred), "majority_baseline": scores(test.label, models["majority_baseline"].fit(X,y).predict(test[features]))}
    save_json(RESULTS / "test_metrics.json", metrics)
    print("FINAL TEST", json.dumps(metrics), flush=True)
    return metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("zip", type=Path)
    parser.add_argument("--use-features", action="store_true")
    args = parser.parse_args()
    data = pd.read_csv(RESULTS / "daily_features.csv") if args.use_features else extract(args.zip)
    with threadpool_limits(limits=2):
        train(data)

if __name__ == "__main__":
    main()
