from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from load_classifier.data import FEATURES, apply_splits, arrays, make_splits, read_dataset


@pytest.fixture(scope="module")
def dataset():
    return read_dataset()


def test_final_dataset_and_stratified_split(dataset):
    assert len(dataset) == 16800
    assert len(FEATURES) == 19
    assert dataset.groupby(["behaviour", "circuit", "condition"]).size().eq(400).all()
    manifest = make_splits(dataset)
    pd.testing.assert_frame_equal(manifest, pd.read_csv("data/splits.csv"))
    prepared = apply_splits(dataset, manifest)
    assert prepared["split"].value_counts().to_dict() == {"train": 11760, "validation": 2520, "test": 2520}
    for name, expected in (("train", 280), ("validation", 60), ("test", 60)):
        subset = prepared[prepared["split"] == name]
        assert subset.groupby(["behaviour", "circuit", "condition"]).size().eq(expected).all()
    assert prepared["sample_id"].is_unique


def test_row_order_does_not_change_saved_split_membership(dataset):
    manifest = make_splits(dataset)
    shuffled = dataset.sample(frac=1, random_state=13).reset_index(drop=True)
    actual = apply_splits(shuffled, manifest).set_index("sample_id")["split"].sort_index()
    expected = manifest.set_index("sample_id")["split"].sort_index()
    pd.testing.assert_series_equal(actual, expected)


def test_metadata_and_labels_cannot_be_selected_as_inputs(dataset):
    x, behaviour, circuit = arrays(dataset.iloc[:10])
    assert x.shape == (10, 19)
    assert behaviour.shape == circuit.shape == (10,)
    for forbidden in ("sample_id", "behaviour", "circuit", "condition"):
        with pytest.raises(ValueError):
            arrays(dataset, (forbidden,))


def test_missing_and_duplicate_split_assignments_are_rejected(dataset):
    manifest = make_splits(dataset)
    with pytest.raises(ValueError):
        apply_splits(dataset, manifest.iloc[:-1])
    with pytest.raises(ValueError):
        apply_splits(dataset, pd.concat([manifest, manifest.iloc[:1]]))
