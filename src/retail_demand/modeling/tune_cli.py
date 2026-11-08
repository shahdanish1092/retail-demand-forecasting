from __future__ import annotations

import argparse
import json
from pathlib import Path

from retail_demand.data.ingestion import RawDataPaths, load_store, load_train
from retail_demand.modeling.tuning import tune_hist_gradient_boosting


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tune HistGradientBoostingRegressor on a temporal holdout."
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
    report = tune_hist_gradient_boosting(
        load_train(paths),
        load_store(paths),
        holdout_days=args.holdout_days,
    )
    print(json.dumps(report, indent=2, default=str))

