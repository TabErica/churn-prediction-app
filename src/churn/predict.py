"""Command-line scoring: event logs -> submission CSV.

Example:
    python -m churn.predict --model models/model.joblib \
        --data data/sample/test.parquet --template data/sample/example_submission.csv \
        --out submission.csv
"""

from __future__ import annotations

import argparse

import pandas as pd

from churn.data import load_logs
from churn.model import ChurnModel, make_submission


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Score event logs with a trained churn model.")
    parser.add_argument("--model", default="models/model.joblib")
    parser.add_argument("--data", required=True, help="Logs to score (.parquet or .csv)")
    parser.add_argument("--template", help="example_submission.csv (its 'id' column is used)")
    parser.add_argument("--pos-rate", type=float, default=0.5)
    parser.add_argument("--out", default="submission.csv")
    args = parser.parse_args(argv)

    model = ChurnModel.load(args.model)
    proba = model.predict_proba_from_logs(load_logs(args.data))
    ids = pd.read_csv(args.template)["id"] if args.template else None
    submission = make_submission(proba, ids, pos_rate=args.pos_rate)
    submission.to_csv(args.out, index=False)
    print(f"Saved {args.out}: {len(submission)} rows")
    print(submission["target"].value_counts().to_string())


if __name__ == "__main__":
    main()
