"""Predict completed local days from a CSV; its filename is never a feature."""
import argparse
from pathlib import Path
import joblib
import pandas as pd
from pipeline import ROOT, prepare_readings, summarize_day

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=ROOT / "artifacts" / "model.joblib")
    args = parser.parse_args()
    bundle = joblib.load(args.model)
    df = prepare_readings(pd.read_csv(args.csv))
    rows = []
    for date, day in df.groupby(df.index.normalize(), sort=True):
        features, quality = summarize_day(day)
        row = {"date": str(date.date()), "valid_minutes": quality["valid_minutes"], "valid_hours": quality["valid_hours"]}
        if features is None:
            row.update(status="rejected_insufficient_coverage", predicted=None)
        else:
            X = pd.DataFrame([features])[bundle["metadata"]["features"]]
            row.update(status="classified", predicted=bundle["pipeline"].predict(X)[0])
        rows.append(row)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)
    print(f"Saved {len(rows)} daily outcomes to {args.output}")

if __name__ == "__main__":
    main()
