"""Audit the supplied ZIP without executing or extracting its contents."""
import argparse
import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
SOURCES = {"acad_build_mains.csv": "Academic", "facilities_build_mains.csv": "Facilities", "mess_build_mains.csv": "Mess"}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("zip", type=Path)
    args = parser.parse_args()
    out = ROOT / "results"
    out.mkdir(parents=True, exist_ok=True)
    with args.zip.open("rb") as stream:
        archive_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    audit = {"archive_sha256": archive_hash, "files": [], "measurements": {}, "calendar": {}}
    with zipfile.ZipFile(args.zip) as z:
        audit["files"] = [{"name": i.filename, "bytes": i.file_size} for i in z.infolist()]
        # Preserve source text, normalizing only trailing whitespace for Git.
        source_text = z.read("Readme.txt").decode("utf-8")
        (out / "source_readme.txt").write_text("\n".join(line.rstrip() for line in source_text.splitlines()).rstrip() + "\n", encoding="utf-8")
        calendars = []
        for name in z.namelist():
            if name.startswith("calender"):
                c = pd.read_csv(z.open(name))
                calendars.append(c)
        c = pd.concat(calendars)
        audit["calendar"] = {"rows": len(c), "start": c.Date.min(), "end": c.Date.max(), "duplicates": int(c.Date.duplicated().sum()), "activity_counts": c.activity.value_counts().to_dict(), "working_day_counts": c.working_day.value_counts().to_dict()}
        for name, label in SOURCES.items():
            df = pd.read_csv(z.open(name))
            ts = pd.to_datetime(df.timestamp, unit="s", utc=True).dt.tz_convert("Asia/Kolkata")
            numeric = df.drop(columns="timestamp")
            info = {"label": label, "rows": len(df), "start": str(ts.min()), "end": str(ts.max()), "timestamps_monotonic": bool(ts.is_monotonic_increasing), "duplicate_timestamps": int(ts.duplicated().sum()), "cadence_seconds_counts": df.timestamp.diff().value_counts().head(10).to_dict(), "missing": numeric.isna().sum().to_dict(), "nonfinite": {k: int((~np.isfinite(v) & v.notna()).sum()) for k, v in numeric.items()}, "quantiles": numeric.quantile([0,.01,.5,.99,1]).to_dict(), "negative_power": int(df.power.lt(0).sum()), "pf_out_of_range": int(df.power_factor.abs().gt(1.0001).sum()), "off_minute_timestamps": int(df.timestamp.mod(60).ne(0).sum())}
            audit["measurements"][name] = info
            print(name, json.dumps(info, default=str), flush=True)
        status = pd.read_csv(z.open("data_present_status_buildings.csv"))
        audit["status"] = {"rows": len(status), "start": str(status.timestamp.min()), "end": str(status.timestamp.max()), "duplicate_timestamps": int(status.timestamp.duplicated().sum()), "presence_counts": {k: v.value_counts().to_dict() for k, v in status.drop(columns="timestamp").items()}}
    (out / "dataset_audit.json").write_text(json.dumps(audit, indent=2, default=str), encoding="utf-8")
    print("Audit saved", flush=True)

if __name__ == "__main__":
    main()
