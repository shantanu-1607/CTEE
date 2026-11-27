"""Independent checks for feature quality, timestamp handling and temporal isolation."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
import pipeline as p

def full_day():
    idx = pd.date_range("2016-02-29", periods=1440, freq="min", tz="Asia/Kolkata")
    return pd.DataFrame({"timestamp": idx.asi8 // 10**9, "power": 1000., "power_factor": .9})

class Checks(unittest.TestCase):
    def test_timezone_and_known_signal(self):
        df = p.prepare_readings(full_day())
        self.assertEqual(str(df.index[0]), "2016-02-29 00:00:00+05:30")
        f, q = p.summarize_day(df)
        self.assertEqual(q["valid_minutes"], 1440)
        self.assertEqual(len(f), 69)
        self.assertAlmostEqual(f["power_mean"], 1000)
        self.assertAlmostEqual(f["power_std"], 0)
        self.assertAlmostEqual(f["power_ramp_mean"], 0)
        self.assertAlmostEqual(f["pf_mean"], .9)
        self.assertAlmostEqual(f["day_night_power_ratio"], 1)
        self.assertAlmostEqual(f["power_hour_23"], 1000)

    def test_coverage_boundary(self):
        df = p.prepare_readings(full_day())
        df.iloc[::5, df.columns.get_loc("power")] = np.nan
        f, q = p.summarize_day(df)
        self.assertIsNotNone(f)
        self.assertEqual(q["valid_minutes"], 1152)
        df.iloc[1, df.columns.get_loc("power")] = np.nan
        self.assertIsNone(p.summarize_day(df)[0])

    def test_missing_intervals_do_not_create_ramps(self):
        df = p.prepare_readings(full_day())
        df.iloc[720, df.columns.get_loc("power")] = np.nan
        df.iloc[721:, df.columns.get_loc("power")] = 2000
        f, _ = p.summarize_day(df)
        self.assertEqual(f["power_ramp_mean"], 0)

    def test_duplicates_and_invalid_values(self):
        source = full_day()
        identical = pd.concat([source, source.iloc[:1]])
        self.assertEqual(len(p.prepare_readings(identical)), 1440)
        other = source.iloc[:1].assign(power=123)
        with self.assertRaisesRegex(ValueError, "Conflicting"):
            p.prepare_readings(pd.concat([source, other]))
        source.loc[0, "power_factor"] = 1.1
        source.loc[1, "power"] = -5
        clean = p.prepare_readings(source)
        self.assertTrue(pd.isna(clean.power_factor.iloc[0]))
        self.assertTrue(pd.isna(clean.power.iloc[1]))

    def test_temporal_partitions(self):
        rows = []
        for date in pd.date_range("2014-01-01", periods=100):
            for label in p.LABELS:
                key = f"{date.date()}_{label}"
                rows.append(dict(sample_id=key, date=str(date.date()), label=label, valid_minutes=1440, valid_hours=24, profile_hash=key, feature_hash=key))
        with tempfile.TemporaryDirectory() as tmp, patch.object(p, "RESULTS", Path(tmp)):
            dev, test, folds = p.partition(pd.DataFrame(rows))
            self.assertGreater((pd.Timestamp(test.date.min()) - pd.Timestamp(dev.date.max())).days, 7)
            self.assertFalse(set(dev.date) & set(test.date))
            for tr, va in folds:
                self.assertGreater((pd.Timestamp(dev.iloc[va].date.min()) - pd.Timestamp(dev.iloc[tr].date.max())).days, 7)
                self.assertFalse(set(dev.iloc[tr].date) & set(dev.iloc[va].date))

if __name__ == "__main__":
    unittest.main()
