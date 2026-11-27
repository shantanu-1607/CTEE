import numpy as np
import pandas as pd
import pytest

from load_classifier.features import FEATURE_NAMES, aggregate_windows


def test_time_slope_uses_seconds_and_irregular_intervals(windows):
    result = aggregate_windows(windows).iloc[0]
    assert result["i_rms__slope"] == pytest.approx(2)
    assert result["i_rms__step"] == pytest.approx(2)
    assert result["i_rms__median"] == 7
    assert result["i_rms__early"] == 4
    assert result["i_rms__late"] == 19
    assert len(FEATURE_NAMES) == 184


def test_row_order_and_identifying_columns_do_not_change_features(windows):
    expected = aggregate_windows(windows)
    changed = windows.sample(frac=1, random_state=1).copy()
    for name in ("load", "appliance_type", "location", "status"):
        changed[name] = "deliberately wrong label"
    pd.testing.assert_frame_equal(expected, aggregate_windows(changed))
    changed["id"] = 9999
    pd.testing.assert_frame_equal(expected.drop(columns="id"), aggregate_windows(changed).drop(columns="id"))


def test_rescaling_time_changes_rates_but_not_amplitudes(windows):
    expected = aggregate_windows(windows).iloc[0]
    windows[["t_start_s", "t_end_s"]] *= 2
    actual = aggregate_windows(windows).iloc[0]
    assert actual["i_rms__slope"] == pytest.approx(expected["i_rms__slope"] / 2)
    assert actual["i_rms__step"] == pytest.approx(expected["i_rms__step"] / 2)
    assert actual["i_rms__median"] == expected["i_rms__median"]


def test_missing_summaries_are_not_filled_from_other_recordings(windows):
    windows["i_thd"] = np.nan
    result = aggregate_windows(windows)
    assert result.filter(like="i_thd__").isna().all().all()
