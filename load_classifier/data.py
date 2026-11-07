"""Read the final workbook and keep one reproducible split for all experiments."""
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd


BEHAVIOURS = ("CC", "CP", "CZ")
CIRCUITS = ("R", "L", "C", "RL", "RC", "LC", "RLC")
MEASUREMENTS = (
    "Vrms", "Irms", "P", "Q1", "S", "PF", "phi_deg", "Zmag", "THD_V", "THD_I",
    "V_L0", "I_L0", "P_L0", "V_L1", "I_L1", "P_L1",
    "V_L2", "I_L2", "P_L2", "V_L3", "I_L3", "P_L3",
)
REPEATED_COLUMNS = {"V_L2": "Vrms", "I_L2": "Irms", "P_L2": "P"}
FEATURES = tuple(name for name in MEASUREMENTS if name not in REPEATED_COLUMNS)
WORKBOOK = Path("data/load_dataset_baseline_v3.xlsx")


def file_hash(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def read_dataset(path=WORKBOOK):
    data = pd.read_excel(path, sheet_name="Dataset", engine="openpyxl")
    expected = {"sample_id", "behaviour", "circuit", *MEASUREMENTS}
    if set(data.columns) != expected or data.empty:
        raise ValueError("The Dataset sheet does not match the final workbook schema")
    if data.isna().any().any() or data["sample_id"].duplicated().any():
        raise ValueError("Missing cells or duplicate sample IDs in Dataset")
    if not data["behaviour"].isin(BEHAVIOURS).all() or not data["circuit"].isin(CIRCUITS).all():
        raise ValueError("Unknown behaviour or circuit label")
    values = data[list(MEASUREMENTS)].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Measurements must be finite numbers")
    if data.duplicated(subset=list(MEASUREMENTS)).any():
        raise ValueError("Duplicate measurement rows need reviewing before splitting")
    for repeated, original in REPEATED_COLUMNS.items():
        if not np.array_equal(data[repeated], data[original]):
            raise ValueError(f"{repeated} no longer duplicates {original}; review feature selection")

    source = pd.read_excel(path, sheet_name="Source link", engine="openpyxl")
    if source["sample_id"].duplicated().any() or set(source["sample_id"]) != set(data["sample_id"]):
        raise ValueError("Source link must contain exactly one entry per sample")
    if source.duplicated(subset=["IITK_timestamp", "IITK_column"]).any():
        raise ValueError("Repeated source readings require a grouped split")
    source = source.set_index("sample_id").loc[data["sample_id"]]
    data["condition"] = source["condition"].to_numpy()
    if not data["condition"].isin(("normal", "tough")).all():
        raise ValueError("Unknown measurement condition")
    return data


def make_splits(data, seed=42):
    """Allocate 70/15/15 within each behaviour/circuit/condition group."""
    rng = np.random.default_rng(seed)
    assignments = pd.Series(index=data.index, dtype="object")
    for _, group in data.groupby(["behaviour", "circuit", "condition"], sort=True):
        indices = rng.permutation(group.index.to_numpy())
        train_end = int(len(indices) * .70)
        val_end = train_end + int(len(indices) * .15)
        if min(train_end, val_end - train_end, len(indices) - val_end) < 1:
            raise ValueError("Every label/condition group needs examples in all three splits")
        assignments.loc[indices[:train_end]] = "train"
        assignments.loc[indices[train_end:val_end]] = "validation"
        assignments.loc[indices[val_end:]] = "test"
    return pd.DataFrame({"sample_id": data["sample_id"], "split": assignments})


def apply_splits(data, manifest):
    if set(manifest.columns) != {"sample_id", "split"} or manifest["sample_id"].duplicated().any():
        raise ValueError("Invalid split manifest")
    if set(manifest["sample_id"]) != set(data["sample_id"]):
        raise ValueError("Split manifest does not cover this dataset exactly")
    if set(manifest["split"]) != {"train", "validation", "test"}:
        raise ValueError("Expected train, validation and test partitions")
    data = data.copy()
    data["split"] = manifest.set_index("sample_id").loc[data["sample_id"], "split"].to_numpy()
    return data


def arrays(data, features=FEATURES):
    if not features or len(set(features)) != len(features) or not set(features) <= set(FEATURES):
        raise ValueError("Choose distinct measurement features; labels and metadata are not inputs")
    x = data[list(features)].to_numpy(dtype=np.float64)
    behaviour = data["behaviour"].map({label: i for i, label in enumerate(BEHAVIOURS)})
    circuit = data["circuit"].map({label: i for i, label in enumerate(CIRCUITS)})
    return x, behaviour.to_numpy(dtype=int), circuit.to_numpy(dtype=int)


def dataset_audit(data, workbook):
    counts = data.groupby(["behaviour", "circuit", "condition", "split"]).size()
    return {
        "workbook_sha256": file_hash(workbook),
        "rows": len(data),
        "measurement_columns": len(MEASUREMENTS),
        "features": list(FEATURES),
        "removed_duplicates": REPEATED_COLUMNS,
        "missing_cells": int(data.isna().sum().sum()),
        "duplicate_measurement_rows": int(data.duplicated(subset=list(MEASUREMENTS)).sum()),
        "split_counts": {name: int(n) for name, n in data["split"].value_counts().items()},
        "strata": [dict(zip(("behaviour", "circuit", "condition", "split"), key), rows=int(n))
                   for key, n in counts.items()],
        "source": "Simulated labelled loads using load-profile values, as documented in Read me",
        "independence_check": "Unique sample IDs and unique IITK timestamp/column pairs; no physical circuit IDs supplied",
    }
