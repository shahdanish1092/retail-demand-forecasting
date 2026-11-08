from __future__ import annotations

import argparse
import json
from pathlib import Path

from retail_demand.data.ingestion import RawDataPaths, load_store, load_train
from retail_demand.modeling.persistence import save_model
from retail_demand.modeling.training import train_and_log_to_mlflow


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate the first sklearn forecasting model."
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
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("model/latest.joblib"),
        help="Destination path for the saved model (default: model/latest.joblib).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = RawDataPaths(args.raw_dir)
    report, model_run = train_and_log_to_mlflow(
        load_train(paths),
        load_store(paths),
        holdout_days=args.holdout_days,
        experiment_name="Phase 4: First Model",
    )

    # Persist the fitted pipeline so inference and the demo app can load it
    save_model(
        model_run.pipeline,
        path=args.model_path,
        metadata=report,
    )
    report["model_saved_to"] = str(args.model_path)

    print(json.dumps(report, indent=2, default=str))

