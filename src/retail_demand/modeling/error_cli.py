from __future__ import annotations

import argparse
import json
from pathlib import Path

from retail_demand.data.ingestion import RawDataPaths, load_store, load_train
from retail_demand.evaluation.error_analysis import model_error_analysis_report
from retail_demand.modeling.training import fit_first_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the first model and inspect holdout error segments."
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/raw"),
        help="Directory containing raw Rossmann CSV files.",
    )
    parser.add_argument(
        "--holdout-days",
        type=int,
        default=42,
        help="Number of final calendar days to use as local holdout.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = RawDataPaths(args.raw_dir)
    model_run = fit_first_model(
        load_train(paths),
        load_store(paths),
        holdout_days=args.holdout_days,
    )
    report = model_error_analysis_report(model_run.holdout_predictions)
    print(json.dumps(report, indent=2, default=str))

