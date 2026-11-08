from __future__ import annotations

import argparse
import json
from pathlib import Path

from retail_demand.data.ingestion import RawDataPaths, load_train
from retail_demand.modeling.baselines import (
    run_baseline_experiment,
    run_walk_forward_baseline_experiment,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run mandatory baseline forecasts on a temporal holdout."
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
        "--walk-forward",
        action="store_true",
        help="Evaluate baselines on expanding walk-forward folds.",
    )
    parser.add_argument("--initial-train-days", type=int, default=365)
    parser.add_argument("--validation-days", type=int, default=42)
    parser.add_argument("--step-days", type=int, default=42)
    parser.add_argument("--n-splits", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = RawDataPaths(args.raw_dir)
    train_df = load_train(paths)
    if args.walk_forward:
        report = run_walk_forward_baseline_experiment(
            train_df,
            initial_train_days=args.initial_train_days,
            validation_days=args.validation_days,
            step_days=args.step_days,
            n_splits=args.n_splits,
        )
    else:
        report = run_baseline_experiment(train_df, holdout_days=args.holdout_days)
    print(json.dumps(report, indent=2, default=str))
