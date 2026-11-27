"""Check the raw-CSV inference command on one frozen test day, with no training."""
import argparse
import json
import subprocess
import sys
import zipfile
import numpy as np
import pandas as pd
from pipeline import ROOT, prepare_readings, summarize_day

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("zip")
    args = parser.parse_args()
    results = ROOT / "results"
    expected = pd.read_csv(results / "test_predictions.csv").query("label == 'Academic'").iloc[0]
    start = pd.Timestamp(expected.date, tz="Asia/Kolkata").timestamp()
    end = start+86400
    chunks = []
    with zipfile.ZipFile(args.zip) as archive:
        for chunk in pd.read_csv(archive.open("acad_build_mains.csv"), chunksize=200000):
            selected = chunk[chunk.timestamp.ge(start) & chunk.timestamp.lt(end)]
            if len(selected): chunks.append(selected)
            if chunk.timestamp.max() >= end: break
    raw = pd.concat(chunks)
    raw_dir = ROOT / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    source = raw_dir / "anonymous_daily_profile.csv"
    output = raw_dir / "inference_smoke_predictions.csv"
    raw.to_csv(source, index=False)
    subprocess.run([sys.executable,str(ROOT/"predict.py"),str(source),"--output",str(output)],check=True)
    actual = pd.read_csv(output).iloc[0]
    assert actual.status == "classified"
    assert actual.date == expected.date
    assert actual.predicted == expected.predicted
    f, _ = summarize_day(prepare_readings(raw))
    saved = pd.read_csv(results / "daily_features.csv").set_index("sample_id").loc[expected.sample_id]
    np.testing.assert_allclose(list(f.values()), saved[list(f)].to_numpy(dtype=float), rtol=1e-9, atol=1e-9)
    record = {"passed": True, "sample_id": expected.sample_id, "date": expected.date, "predicted": actual.predicted, "raw_minutes": len(raw), "checks": ["anonymous filename inference command", "all raw-derived features match saved daily features", "prediction matches frozen test record"]}
    (results/"inference_smoke.json").write_text(json.dumps(record,indent=2),encoding="utf-8")
    print(json.dumps(record,indent=2))

if __name__ == "__main__": main()
