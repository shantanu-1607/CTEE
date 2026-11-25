import numpy as np
import pandas as pd
import pytest

from load_classifier.data import (LABELS, duplicate_groups, make_splits, read_metadata,
                                 validate_splits, validate_windows)
from load_classifier.features import FEATURE_NAMES
from load_classifier.experiment import load_prepared


def test_duplicate_windows_are_rejected(windows):
    with pytest.raises(ValueError, match="Duplicate"):
        validate_windows(pd.concat([windows, windows.iloc[:1]]))


@pytest.mark.parametrize("column,value", [("t_start_s", np.nan), ("start_sample", .5), ("i_rms", -1)])
def test_invalid_inputs_are_rejected(windows, column, value):
    windows[column] = windows[column].astype(float)
    windows.loc[0, column] = value
    with pytest.raises(ValueError):
        validate_windows(windows)


def test_sample_rate_and_timestamps_must_agree(windows):
    windows.loc[2, "t_start_s"] += .001
    with pytest.raises(ValueError, match="sample rate|Timestamps"):
        validate_windows(windows)


def test_overlapping_windows_are_rejected(windows):
    windows.loc[1, "t_start_s"] = .001
    with pytest.raises(ValueError, match="Overlapping"):
        validate_windows(windows)


def test_json_labels_are_checked_against_csv(windows):
    metadata = pd.DataFrame([dict(id=1, location="house1", load="R", appliance_type="Heater", sampling_hz=30000)])
    windows["load"], windows["location"], windows["appliance_type"] = "RC", "house1", "Heater"
    with pytest.raises(ValueError, match="load disagrees"):
        validate_windows(windows, metadata)


def test_split_keeps_entire_houses_together():
    metadata = pd.DataFrame([dict(id=4 * house + label + 1, location=f"house{house}", load=load)
                             for house in range(25) for label, load in enumerate(LABELS)])
    manifest = make_splits(metadata)
    assert manifest.groupby("location").partition.nunique().eq(1).all()
    development = manifest[manifest.partition.eq("development")]
    assert development.groupby("location").cv_fold.nunique().eq(1).all()
    pd.testing.assert_frame_equal(manifest, make_splits(metadata.sample(frac=1, random_state=99)))
    changed = manifest.copy()
    test_house = changed.loc[changed.partition.eq("test"), "location"].iloc[0]
    row = changed.index[changed.location.eq(test_house)][0]
    changed.loc[row, ["partition", "cv_fold"]] = ["development", 0]
    with pytest.raises(ValueError, match="house occurs"):
        validate_splits(metadata, changed)


def test_real_prepared_data_covers_the_supplied_json():
    records, audit = load_prepared("data/processed")
    metadata = read_metadata("data/meta_labeled.json")
    assert len(records) == len(metadata) == 1074
    assert audit["windows"] == 209554
    assert set(records.id) == set(metadata.id)
    assert audit["partition_counts"]["test"]["recordings"] == 215
    assert not set(records.loc[records.partition.eq("test"), "location"]) & set(
        records.loc[records.partition.eq("development"), "location"])
    duplicates = records[records.duplicated(list(FEATURE_NAMES), keep=False)]
    assert set(duplicates.id) == {234, 235, 236, 237, 238, 389, 390, 391, 392, 393}
    assert duplicates.partition.eq("development").all()
    assert duplicates.cv_fold.nunique() == 1


def test_duplicate_groups_link_houses_transitively_and_reject_conflicting_labels():
    metadata = pd.DataFrame({"id": [1, 2, 3, 4], "location": ["a", "b", "b", "c"], "load": ["R"] * 4})
    summaries = pd.DataFrame([{**{name: value for name in FEATURE_NAMES}, "id": i}
                              for i, value in [(1, 10.), (2, 10.), (3, 20.), (4, 20.)]])
    groups, pairs = duplicate_groups(metadata, summaries)
    assert groups.nunique() == 1
    assert len(pairs) == 2
    metadata.loc[1, "load"] = "RC"
    with pytest.raises(ValueError, match="conflicting"):
        duplicate_groups(metadata, summaries)


def test_duplicate_link_cannot_cross_the_reserved_test_boundary():
    metadata = pd.DataFrame([dict(id=4 * house + label + 1, location=f"house{house}", load=load)
                             for house in range(25) for label, load in enumerate(LABELS)])
    original = make_splits(metadata)
    test_id = original[original.partition.eq("test") & original.load.eq("R")].id.iloc[0]
    train_id = original[original.partition.eq("development") & original.load.eq("R")].id.iloc[0]
    summaries = pd.DataFrame([{**{name: float(i) for name in FEATURE_NAMES}, "id": i} for i in metadata.id])
    summaries.loc[summaries.id.eq(test_id), list(FEATURE_NAMES)] = float(train_id)
    with pytest.raises(ValueError, match="cross the reserved"):
        make_splits(metadata, summaries)
