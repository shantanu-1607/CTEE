"""Run with python -m load_classifier from the repository root."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data import WORKBOOK, apply_splits, dataset_audit, file_hash, make_splits, read_dataset
from .evaluate import evaluate_run, prediction_table
from .model import TwoHeadMLP
from .train import TrainingConfig, train_run


def prepare(workbook, split_path):
    data = read_dataset(workbook)
    manifest = make_splits(data)
    audit = dataset_audit(apply_splits(data, manifest), workbook)
    split_path.parent.mkdir(parents=True, exist_ok=True)
    if split_path.exists():
        pd.testing.assert_frame_equal(pd.read_csv(split_path), manifest)
    else:
        manifest.to_csv(split_path, index=False)
    audit.update({"split_seed": 42, "splits_sha256": file_hash(split_path)})
    (split_path.parent / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(f"Prepared {len(data):,} rows; split counts: {audit['split_counts']}")


def load_prepared(workbook, split_path):
    audit = json.loads((split_path.parent / "audit.json").read_text())
    if audit["workbook_sha256"] != file_hash(workbook) or audit["splits_sha256"] != file_hash(split_path):
        raise ValueError("Workbook or split changed after preparation")
    return apply_splits(read_dataset(workbook), pd.read_csv(split_path))


def predict(model_path, input_path, output_path):
    if input_path.resolve() == output_path.resolve():
        raise ValueError("Prediction output must not overwrite the input")
    if output_path.exists():
        raise FileExistsError("Prediction output already exists")
    model, _ = TwoHeadMLP.load(model_path)
    frame = pd.read_csv(input_path)
    missing = set(model.features) - set(frame.columns)
    if missing:
        raise ValueError("Missing measurements: " + ", ".join(sorted(missing)))
    x = frame[list(model.features)].to_numpy(dtype=np.float64)
    probabilities = model.predict_proba(x)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    prediction_table(frame, probabilities).to_csv(output_path, index=False)
    print(f"Wrote {len(frame)} predictions to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Electrical load classification baseline")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "train", "evaluate"):
        command = commands.add_parser(name)
        command.add_argument("--workbook", type=Path, default=WORKBOOK)
        command.add_argument("--splits", type=Path, default=Path("data/splits.csv"))
        if name != "prepare":
            command.add_argument("--run", type=Path, default=Path("runs/baseline"))
        if name == "train":
            command.add_argument("--config", type=Path, default=Path("configs/baseline.json"))
    command = commands.add_parser("predict")
    command.add_argument("--model", type=Path, default=Path("runs/baseline/model.npz"))
    command.add_argument("--input", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "prepare":
            prepare(args.workbook, args.splits)
        elif args.command == "predict":
            predict(args.model, args.input, args.output)
        else:
            data = load_prepared(args.workbook, args.splits)
            if args.command == "train":
                train_run(data, TrainingConfig.read(args.config), args.run, args.workbook, args.splits)
            else:
                evaluate_run(data, args.run, args.workbook, args.splits)
    except (ValueError, FileNotFoundError, FileExistsError, AssertionError) as error:
        parser.exit(2, f"{error}\n")


if __name__ == "__main__":
    main()
