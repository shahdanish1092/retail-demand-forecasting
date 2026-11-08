from __future__ import annotations

import argparse
import json
from pathlib import Path

from retail_demand.data.ingestion import RawDataPaths, inspect_raw_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect the raw Rossmann train.csv and store.csv files."
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/raw"),
        help="Directory containing raw Rossmann CSV files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = RawDataPaths(args.raw_dir)
    report = inspect_raw_dataset(paths)
    print(json.dumps(report, indent=2, default=str))

