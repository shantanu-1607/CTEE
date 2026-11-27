"""Validate the supplied window features and attach the recording-level labels."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

LABELS = ("R", "I", "RC", "RLC")
TIME_COLUMNS = ("window_idx", "start_sample", "end_sample", "t_start_s", "t_end_s")
# Omit algebraic copies, equivalent-component estimates and identifying metadata.
MEASUREMENTS = (
    "v_rms", "i_rms", "P_real_W", "S_apparent_VA", "Q_nonactive_var", "pf",
    "i_crest_factor", "v_crest_factor", "i_form_factor", "i_skew", "i_kurtosis",
    "i_zero_cross", "f0_Hz", "cos_phi", "sin_phi", "i_h3_ratio", "i_h5_ratio",
    "i_h7_ratio", "i_thd", "v_thd", "i_spec_centroid_Hz", "i_spec_entropy",
    "i_hf_energy_ratio",
)


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def read_metadata(path):
    entries = json.loads(Path(path).read_text())
    if not isinstance(entries, list) or not entries:
        raise ValueError("Metadata must be a nonempty list of recordings")
    rows = []
    for item in entries:
        meta = item["meta"]
        rows.append({
            "id": int(item["id"]), "load": meta["appliance"]["load"],
            "location": meta["location"], "appliance_type": meta["appliance"]["type"],
            "sampling_hz": float(meta["header"]["sampling_frequency"].removesuffix("Hz")),
            "collection_time": meta["header"]["collection_time"],
        })
    frame = pd.DataFrame(rows).sort_values("id").reset_index(drop=True)
    if frame.id.duplicated().any() or (frame.id <= 0).any():
        raise ValueError("Recording IDs must be unique positive integers")
    if not set(frame.load) <= set(LABELS):
        raise ValueError("Metadata contains unsupported load labels")
    if frame.location.isna().any() or frame.location.eq("").any():
        raise ValueError("House identifiers are required for grouped evaluation")
    if not np.isfinite(frame.sampling_hz).all() or (frame.sampling_hz <= 0).any():
        raise ValueError("Invalid sampling frequency")
    return frame


def validate_windows(frame, metadata=None):
    required = {"id", *TIME_COLUMNS, *MEASUREMENTS}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError("Missing window columns: " + ", ".join(sorted(missing)))
    if frame.empty:
        raise ValueError("No feature windows were supplied")
    frame = frame.copy()
    for name in ("id", *TIME_COLUMNS, *MEASUREMENTS):
        frame[name] = pd.to_numeric(frame[name], errors="raise")
    integers = ("id", "window_idx", "start_sample", "end_sample")
    for name in integers:
        values = frame[name].to_numpy()
        if not np.isfinite(values).all() or not np.equal(values, np.floor(values)).all():
            raise ValueError(f"{name} must contain finite integers")
        frame[name] = frame[name].astype("int64")
    if (frame.id <= 0).any() or (frame[list(integers[1:])] < 0).any().any():
        raise ValueError("IDs must be positive; sample and window indices must be nonnegative")
    if frame.duplicated(["id", "window_idx"]).any():
        raise ValueError("Duplicate recording/window keys")
    times = frame[["t_start_s", "t_end_s"]].to_numpy()
    if (not np.isfinite(times).all() or (times < 0).any()
            or (frame.t_end_s <= frame.t_start_s).any()
            or (frame.end_sample < frame.start_sample).any()):
        raise ValueError("Invalid time or sample interval")
    values = frame[list(MEASUREMENTS)].to_numpy(float)
    if np.isinf(values).any():
        raise ValueError("Infinite measurements are not supported")
    if frame.i_rms.isna().any() or frame.v_rms.isna().any():
        raise ValueError("RMS measurements cannot be missing")
    if (frame.i_rms < 0).any() or (frame.v_rms <= 0).any():
        raise ValueError("Invalid RMS measurements")
    frame = frame.sort_values(["id", "t_start_s"]).reset_index(drop=True)
    for _, part in frame.groupby("id", sort=False):
        if len(part) < 3:
            raise ValueError("Each recording needs at least three time windows")
        if (np.diff(part.t_start_s) <= 0).any() or (np.diff(part.window_idx) <= 0).any():
            raise ValueError("Window order disagrees with elapsed time")
        if (part.t_start_s.to_numpy()[1:] < part.t_end_s.to_numpy()[:-1] - 1e-9).any():
            raise ValueError("Overlapping time windows")
        if (part.start_sample.to_numpy()[1:] <= part.end_sample.to_numpy()[:-1]).any():
            raise ValueError("Overlapping sample intervals")
        # t_end_s is exclusive, whereas end_sample is inclusive.
        hz = (part.end_sample - part.start_sample + 1) / (part.t_end_s - part.t_start_s)
        if not np.allclose(hz, hz.iloc[0], rtol=1e-6, atol=1e-5):
            raise ValueError("Inconsistent sample rate within a recording")
        if not np.allclose(part.t_start_s, part.start_sample / hz.iloc[0], atol=1e-8):
            raise ValueError("Timestamps disagree with sample indices")
    if metadata is not None:
        if set(frame.id) != set(metadata.id):
            raise ValueError("CSV recording IDs do not exactly match the JSON")
        lookup = metadata.set_index("id")
        expected_hz = frame.id.map(lookup.sampling_hz)
        observed_hz = ((frame.end_sample - frame.start_sample + 1)
                       / (frame.t_end_s - frame.t_start_s))
        if not np.allclose(expected_hz, observed_hz, rtol=1e-6):
            raise ValueError("CSV sample rate disagrees with the JSON")
        for name in ("load", "location", "appliance_type"):
            if name not in frame or not frame[name].eq(frame.id.map(lookup[name])).all():
                raise ValueError(f"CSV {name} disagrees with the JSON")
    return frame


def duplicate_groups(metadata, summaries):
    """Link houses sharing exact recording summaries, including transitive links."""
    from .features import FEATURE_NAMES
    joined = metadata[["id", "location", "load"]].merge(summaries, on="id", validate="one_to_one")
    parent = {house: house for house in metadata.location.unique()}

    def root(house):
        while parent[house] != house:
            parent[house] = parent[parent[house]]
            house = parent[house]
        return house

    duplicated = joined[joined.duplicated(list(FEATURE_NAMES), keep=False)]
    groups = []
    for _, part in duplicated.groupby(list(FEATURE_NAMES), dropna=False, sort=False):
        if part.load.nunique() != 1:
            raise ValueError("Identical recording summaries have conflicting load labels")
        houses = sorted(part.location.unique())
        for house in houses[1:]:
            left, right = sorted((root(houses[0]), root(house)))
            parent[right] = left
        groups.append({"ids": part.id.astype(int).tolist(), "houses": houses})
    return metadata.location.map(lambda house: root(house)), groups


def make_splits(metadata, summaries=None):
    """Reserve fold zero for testing; assign four CV folds to the remaining houses."""
    metadata = metadata.sort_values("id").reset_index(drop=True)
    if summaries is None:
        grouping = metadata.location.copy()
    else:
        grouping, _ = duplicate_groups(metadata, summaries)
    if set(metadata.load) != set(LABELS):
        raise ValueError("The dataset must include all four load labels")
    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    development, test = next(outer.split(metadata, metadata.load, metadata.location))
    manifest = metadata[["id", "location", "load"]].copy()
    manifest["group_id"] = grouping
    manifest["partition"] = "development"
    manifest["cv_fold"] = -1
    manifest.loc[test, "partition"] = "test"
    if manifest.groupby("group_id").partition.nunique().max() != 1:
        raise ValueError("Exact duplicates cross the reserved test-house boundary; a new evaluation protocol is required")
    inner = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=43)
    part = metadata.iloc[development]
    for fold, (train, validation) in enumerate(inner.split(part, part.load, grouping.iloc[development])):
        if set(part.iloc[train].load) != set(LABELS) or set(part.iloc[validation].load) != set(LABELS):
            raise ValueError("A CV fold is missing a load class")
        manifest.loc[development[validation], "cv_fold"] = fold
    validate_splits(metadata, manifest)
    return manifest


def validate_splits(metadata, manifest):
    if manifest.id.duplicated().any() or set(manifest.id) != set(metadata.id):
        raise ValueError("Split manifest must cover every recording exactly once")
    joined = metadata.merge(manifest, on="id", suffixes=("", "_split"), validate="one_to_one")
    for name in ("location", "load"):
        if not joined[name].eq(joined[name + "_split"]).all():
            raise ValueError("Split manifest metadata has changed")
    if set(joined.partition) != {"development", "test"}:
        raise ValueError("Both development and test partitions are required")
    if joined.groupby("location").partition.nunique().max() != 1:
        raise ValueError("A house occurs in both development and test")
    if "group_id" in joined and joined.groupby("group_id").partition.nunique().max() != 1:
        raise ValueError("A duplicate-connected group crosses the test boundary")
    for _, part in joined.groupby("partition"):
        if set(part.load) != set(LABELS):
            raise ValueError("A partition is missing a load class")
    development = joined[joined.partition.eq("development")]
    if set(development.cv_fold) != {0, 1, 2, 3}:
        raise ValueError("Development rows must have one of four CV folds")
    if development.groupby("location").cv_fold.nunique().max() != 1:
        raise ValueError("A house occurs in multiple validation folds")
    if "group_id" in development and development.groupby("group_id").cv_fold.nunique().max() != 1:
        raise ValueError("A duplicate-connected group occurs in multiple validation folds")
    if not joined.loc[joined.partition.eq("test"), "cv_fold"].eq(-1).all():
        raise ValueError("Test recordings cannot be used in cross-validation")
