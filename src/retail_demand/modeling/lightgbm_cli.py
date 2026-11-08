from __future__ import annotations

import argparse
import json
from pathlib import Path

from retail_demand.data.ingestion import RawDataPaths, load_store, load_train
from retail_demand.modeling.training import evaluate_walk_forward_lightgbm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate LightGBM on leakage-safe walk-forward folds."
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/raw"),
        help="Directory containing raw Rossmann CSV files.",
    )
    parser.add_argument("--initial-train-days", type=int, default=365)
    parser.add_argument("--validation-days", type=int, default=42)
    parser.add_argument("--step-days", type=int, default=42)
    parser.add_argument("--n-splits", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = RawDataPaths(args.raw_dir)
    report = evaluate_walk_forward_lightgbm(
        load_train(paths),
        load_store(paths),
        initial_train_days=args.initial_train_days,
        validation_days=args.validation_days,
        step_days=args.step_days,
        n_splits=args.n_splits,
    )
    print(json.dumps(report, indent=2, default=str))
