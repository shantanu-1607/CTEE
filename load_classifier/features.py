"""Summarize an ordered electrical recording without using its labels or IDs."""

import numpy as np
import pandas as pd

from .data import MEASUREMENTS, validate_windows

STATISTICS = ("p10", "median", "p90", "iqr", "early", "late", "slope", "step")
FEATURE_NAMES = tuple(f"{name}__{stat}" for name in MEASUREMENTS for stat in STATISTICS)
STATIC_FEATURES = tuple(f"{name}__median" for name in MEASUREMENTS)


def summarize_recording(part):
    times = ((part.t_start_s + part.t_end_s) / 2).to_numpy(float)
    times -= times[0]
    span = times[-1]
    features = {}
    for name in MEASUREMENTS:
        values = part[name].to_numpy(float)
        valid = np.isfinite(values)
        if not valid.any():
            summary = [np.nan] * len(STATISTICS)
        else:
            t, v = times[valid], values[valid]
            p10, p25, median, p75, p90 = np.quantile(v, [.1, .25, .5, .75, .9])
            early = values[valid & (times <= .2 * span)]
            late = values[valid & (times >= .8 * span)]
            # Missing values stay missing until a training-only imputer is fitted.
            early = float(np.median(early)) if len(early) else np.nan
            late = float(np.median(late)) if len(late) else np.nan
            tc = t - t.mean()
            slope = float(tc @ (v - v.mean()) / (tc @ tc)) if len(v) >= 2 else np.nan
            # Preserve real adjacent intervals rather than joining across missing windows.
            pairs = valid[1:] & valid[:-1]
            rates = np.abs(np.diff(values)[pairs] / np.diff(times)[pairs])
            step = float(np.median(rates)) if len(rates) else np.nan
            summary = [p10, median, p90, p75 - p25, early, late, slope, step]
        features.update({f"{name}__{stat}": float(value)
                         for stat, value in zip(STATISTICS, summary)})
    return features


def aggregate_windows(windows, validate=True):
    windows = validate_windows(windows) if validate else windows
    rows = [{"id": int(recording), **summarize_recording(part)}
            for recording, part in windows.groupby("id", sort=True)]
    return pd.DataFrame(rows, columns=["id", *FEATURE_NAMES])
