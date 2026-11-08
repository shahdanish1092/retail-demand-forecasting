"""CLI for batch inference: generate-forecast console script."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> None:  # noqa: D401
    """Entry point for the ``generate-forecast`` console script.

    Runs batch inference against test.csv (or any CSV supplied via
    ``--target-csv``) and either prints a JSON summary or writes a Kaggle
    submission CSV.

    Examples::

        generate-forecast --raw-dir data/raw --model-path model/latest.joblib
        generate-forecast --raw-dir data/raw --model-path model/latest.joblib --kaggle
    """
    parser = argparse.ArgumentParser(
        prog="generate-forecast",
        description="Run batch inference with the saved model.",
    )
    parser.add_argument(
        "--raw-dir",
        default="data/raw",
        help="Directory that contains train.csv, store.csv, test.csv (default: data/raw).",
    )
    parser.add_argument(
        "--model-path",
        default="model/latest.joblib",
        help="Path to the saved *.joblib model file (default: model/latest.joblib).",
    )
    parser.add_argument(
        "--kaggle",
        action="store_true",
        help="Generate a Kaggle-formatted submission CSV instead of JSON output.",
    )
    parser.add_argument(
        "--output",
        default="submissions/submission.csv",
        help="Destination for the Kaggle submission CSV (default: submissions/submission.csv).",
    )

    args = parser.parse_args()

    try:
        if args.kaggle:
            from retail_demand.modeling.inference import generate_kaggle_submission

            out_path = generate_kaggle_submission(
                raw_dir=args.raw_dir,
                model_path=args.model_path,
                output_path=args.output,
            )
            report = {"status": "ok", "submission_path": str(out_path)}
        else:
            from retail_demand.modeling.inference import score_from_files

            predictions = score_from_files(
                raw_dir=args.raw_dir,
                model_path=args.model_path,
            )
            report = {
                "status": "ok",
                "rows_scored": len(predictions),
                "sample_predictions": predictions[["Store", "Date", "Sales_Predicted"]]
                .head(10)
                .to_dict(orient="records"),
            }
    except Exception as exc:  # noqa: BLE001
        report = {"status": "error", "message": str(exc)}
        json.dump(report, sys.stdout, indent=2, default=str)
        sys.stdout.write("\n")
        sys.exit(1)

    json.dump(report, sys.stdout, indent=2, default=str)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
