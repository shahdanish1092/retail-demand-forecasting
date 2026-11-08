from __future__ import annotations

import argparse
import json
from pathlib import Path

from retail_demand.data.ingestion import RawDataPaths, load_store, load_train
from retail_demand.features.build import build_features, summarize_feature_table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build and summarize the initial feature table."
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
    feature_df = build_features(load_train(paths), load_store(paths))
    print(json.dumps(summarize_feature_table(feature_df), indent=2, default=str))

