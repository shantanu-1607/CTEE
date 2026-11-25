import json

import numpy as np
import pandas as pd
import pytest

from load_classifier.data import LABELS, make_splits, read_metadata, validate_splits, validate_windows
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
