"""Command line interface for the supplied timestamped recording features."""

import argparse
import json
from pathlib import Path

from .experiment import compare, prepare, train
from .predict import predict_csv


def main():
    parser = argparse.ArgumentParser(description="Timestamped electrical load classification")
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser("prepare")
    command.add_argument("--input", type=Path, default=Path("data/raw/all_features.csv"))
    command.add_argument("--metadata", type=Path, default=Path("data/meta_labeled.json"))
    command.add_argument("--output", type=Path, default=Path("data/processed"))
    for name in ("compare", "train", "evaluate", "verify"):
        command = commands.add_parser(name)
        command.add_argument("--data", type=Path, default=Path("data/processed"))
        command.add_argument("--run", type=Path, default=Path("runs/timestamped"))
    command = commands.add_parser("predict")
    command.add_argument("--input", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    command.add_argument("--model", type=Path, default=Path("runs/timestamped/model.joblib"))
    args = parser.parse_args()
    try:
        if args.command == "prepare":
            result = prepare(args.input, args.metadata, args.output)
            print(f"Prepared {result['recordings']} recordings from {result['windows']} windows")
        elif args.command == "compare":
            result = compare(args.data, args.run)
            print("Selected:", result["selected_model"])
        elif args.command == "train":
            result = train(args.data, args.run)
            print("Trained:", result["selected_model"])
        elif args.command == "evaluate":
            from .evaluate import evaluate
            result = evaluate(args.data, args.run)
            print(json.dumps(result["test"], indent=2))
        elif args.command == "verify":
            from .evaluate import verify
            print(json.dumps(verify(args.data, args.run), indent=2))
        else:
            result = predict_csv(args.model, args.input, args.output)
            print(f"Wrote {len(result)} recording predictions to {args.output}")
    except (ValueError, FileNotFoundError, FileExistsError, AssertionError) as error:
        parser.exit(2, f"{error}\n")


if __name__ == "__main__":
    main()
