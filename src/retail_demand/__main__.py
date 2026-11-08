"""Unified CLI entry point for the retail demand forecasting pipeline.

Run with:
    python -m retail_demand [command]
or via the console script:
    retail-demand [command]

Commands
--------
train       Run the full training pipeline and save the model.
forecast    Generate predictions on test.csv using the saved model.
baselines   Run baseline benchmark.
features    Build and summarise the feature table.
inspect     Inspect raw data files.

Examples
--------
    retail-demand train --raw-dir data/raw --model-path model/latest.joblib
    retail-demand forecast --raw-dir data/raw --model-path model/latest.joblib --kaggle
    retail-demand baselines --raw-dir data/raw
    retail-demand --help
"""

from __future__ import annotations

import argparse
import sys


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="retail-demand",
        description="Retail Demand Forecasting — unified pipeline CLI.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", metavar="command")

    # ── train ──────────────────────────────────────────────────────────────
    train_p = sub.add_parser("train", help="Train model and save artifact.")
    train_p.add_argument("--raw-dir", default="data/raw", help="Raw CSV directory.")
    train_p.add_argument("--holdout-days", type=int, default=42)
    train_p.add_argument(
        "--model-path", default="model/latest.joblib", help="Where to save the model."
    )

    # ── forecast ───────────────────────────────────────────────────────────
    fc_p = sub.add_parser("forecast", help="Generate predictions on test.csv.")
    fc_p.add_argument("--raw-dir", default="data/raw")
    fc_p.add_argument("--model-path", default="model/latest.joblib")
    fc_p.add_argument(
        "--kaggle",
        action="store_true",
        help="Write a Kaggle-formatted submission CSV.",
    )
    fc_p.add_argument("--output", default="submissions/submission.csv")

    # ── baselines ──────────────────────────────────────────────────────────
    bl_p = sub.add_parser("baselines", help="Run baseline benchmark.")
    bl_p.add_argument("--raw-dir", default="data/raw")
    bl_p.add_argument("--holdout-days", type=int, default=42)

    # ── features ───────────────────────────────────────────────────────────
    ft_p = sub.add_parser("features", help="Build and summarise feature table.")
    ft_p.add_argument("--raw-dir", default="data/raw")

    # ── inspect ────────────────────────────────────────────────────────────
    ins_p = sub.add_parser("inspect", help="Inspect raw data files.")
    ins_p.add_argument("--raw-dir", default="data/raw")

    return parser


def main() -> None:  # noqa: C901
    """Entry point for the ``retail-demand`` console script."""
    import json
    from pathlib import Path

    parser = _build_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    # ── train ──────────────────────────────────────────────────────────────
    if args.command == "train":
        from retail_demand.data.ingestion import RawDataPaths, load_store, load_train
        from retail_demand.modeling.persistence import save_model
        from retail_demand.modeling.training import train_and_log_to_mlflow

        paths = RawDataPaths(Path(args.raw_dir))
        report, model_run = train_and_log_to_mlflow(
            load_train(paths),
            load_store(paths),
            holdout_days=args.holdout_days,
        )
        save_model(model_run.pipeline, path=args.model_path, metadata=report)
        report["model_saved_to"] = args.model_path
        print(json.dumps(report, indent=2, default=str))

    # ── forecast ───────────────────────────────────────────────────────────
    elif args.command == "forecast":
        if args.kaggle:
            from retail_demand.modeling.inference import generate_kaggle_submission

            out = generate_kaggle_submission(
                raw_dir=args.raw_dir,
                model_path=args.model_path,
                output_path=args.output,
            )
            print(json.dumps({"status": "ok", "submission_path": str(out)}, indent=2))
        else:
            from retail_demand.modeling.inference import score_from_files

            preds = score_from_files(raw_dir=args.raw_dir, model_path=args.model_path)
            summary = {
                "status": "ok",
                "rows_scored": len(preds),
                "sample": preds[["Store", "Date", "Sales_Predicted"]]
                .head(10)
                .to_dict(orient="records"),
            }
            print(json.dumps(summary, indent=2, default=str))

    # ── baselines ──────────────────────────────────────────────────────────
    elif args.command == "baselines":
        from retail_demand.data.ingestion import RawDataPaths, load_train
        from retail_demand.modeling.baselines import run_baseline_experiment

        paths = RawDataPaths(Path(args.raw_dir))
        report = run_baseline_experiment(load_train(paths), holdout_days=args.holdout_days)
        print(json.dumps(report, indent=2, default=str))

    # ── features ───────────────────────────────────────────────────────────
    elif args.command == "features":
        from retail_demand.data.ingestion import RawDataPaths, load_store, load_train
        from retail_demand.features.build import build_features, summarize_feature_table

        paths = RawDataPaths(Path(args.raw_dir))
        feature_df = build_features(load_train(paths), load_store(paths))
        print(json.dumps(summarize_feature_table(feature_df), indent=2, default=str))

    # ── inspect ────────────────────────────────────────────────────────────
    elif args.command == "inspect":
        from retail_demand.data.ingestion import RawDataPaths, inspect_raw_dataset

        paths = RawDataPaths(Path(args.raw_dir))
        report = inspect_raw_dataset(paths)
        print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
