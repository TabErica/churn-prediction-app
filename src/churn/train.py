"""Command-line training entry point.

Examples:
    python -m churn.train                                          # LR + KNN voting
    python -m churn.train --data path/to/train.parquet --model "Extra Trees"
"""

from __future__ import annotations

import argparse
import json

from churn.data import load_logs
from churn.model import MODEL_NAMES, VOTING, train_from_logs


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train a churn model from event logs.")
    parser.add_argument(
        "--data", default="data/sample/train.parquet", help="Training logs (.parquet or .csv)"
    )
    parser.add_argument("--model", default=VOTING, choices=MODEL_NAMES)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--no-oversample", action="store_true")
    parser.add_argument("--out", default="models/model.joblib")
    args = parser.parse_args(argv)

    raw = load_logs(args.data)
    model = train_from_logs(
        raw, model_name=args.model, n_splits=args.folds, oversample=not args.no_oversample
    )
    model.save(args.out)
    print(json.dumps({"model": args.model, "saved_to": args.out, **model.metrics}, indent=2))


if __name__ == "__main__":
    main()
