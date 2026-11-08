"""Batch inference pipeline for the retail demand forecasting model.

This module provides two entry points:

1. **score_from_files** – Load data files and produce a scored DataFrame.
2. **generate_kaggle_submission** – Run inference on ``test.csv`` and write
   a Kaggle-formatted submission CSV.

Usage (programmatic)::

    from retail_demand.modeling.inference import score_from_files
    predictions = score_from_files(
        raw_dir="data/raw",
        model_path="model/latest.joblib",
    )
    print(predictions[["Store", "Date", "Sales_Predicted"]].head())

Usage (CLI)::

    generate-forecast --raw-dir data/raw --model-path model/latest.joblib
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from retail_demand.data.ingestion import RawDataPaths, load_store, load_train
from retail_demand.features.build import build_prediction_features
from retail_demand.modeling.persistence import load_model
from retail_demand.modeling.training import (
    NON_FEATURE_COLUMNS,
    TARGET_COLUMN,
)

logger = logging.getLogger(__name__)

_SUBMISSION_COLUMNS = ["Id", "Sales"]


def _prepare_test_df(raw_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load train, store, and test DataFrames from *raw_dir*.

    Returns:
        (train_df, store_df, test_df)
    """
    paths = RawDataPaths(Path(raw_dir))
    missing = paths.missing_required_files()
    if missing:
        raise FileNotFoundError(f"Missing required raw files: {missing}")

    train_df = load_train(paths)
    store_df = load_store(paths)

    test_path = Path(raw_dir) / "test.csv"
    if not test_path.exists():
        raise FileNotFoundError(f"test.csv not found at '{test_path}'")

    test_df = pd.read_csv(test_path, parse_dates=["Date"])

    # Kaggle test.csv: Open has 11 NaN rows — treat them as open (1) per competition notes
    test_df["Open"] = test_df["Open"].fillna(1).astype(int)

    return train_df, store_df, test_df


def score_dataframe(
    target_df: pd.DataFrame,
    history_df: pd.DataFrame,
    store_df: pd.DataFrame,
    pipeline: Any,
) -> pd.DataFrame:
    """Run inference on *target_df* using history to build lag/rolling features.

    Args:
        target_df: DataFrame of rows to score (must contain at least Store, Date, Open).
        history_df: Historical Sales data used to compute lag/rolling features.
        store_df: Store metadata table (store.csv).
        pipeline: Fitted sklearn Pipeline returned by :func:`load_model`.

    Returns:
        A copy of *target_df* with an added ``Sales_Predicted`` column.
        Rows with ``Open == 0`` receive a prediction of 0.
    """
    result = target_df.copy()
    result["Sales_Predicted"] = 0.0

    open_mask = result["Open"] == 1
    open_rows = result[open_mask].copy()

    if open_rows.empty:
        logger.warning("No open store rows in target_df — all predictions set to 0.")
        return result

    # Build leakage-safe features using only historical data
    feature_df = build_prediction_features(
        target_df=open_rows,
        history_df=history_df,
        store_df=store_df,
    )

    # Select feature columns (same logic as training)
    drop_cols = NON_FEATURE_COLUMNS | {TARGET_COLUMN}
    feature_cols = [c for c in feature_df.columns if c not in drop_cols]

    X = feature_df[feature_cols]
    preds: np.ndarray = pipeline.predict(X)
    # Sales cannot be negative
    preds = np.clip(preds, 0, None)

    result.loc[open_mask, "Sales_Predicted"] = preds
    logger.info(
        "Scored %d open rows out of %d total rows",
        open_mask.sum(),
        len(result),
    )
    return result


def score_from_files(
    raw_dir: str | Path = "data/raw",
    model_path: str | Path = "model/latest.joblib",
) -> pd.DataFrame:
    """Full end-to-end batch inference from raw data files.

    Loads ``train.csv`` (history) and ``test.csv`` (target), builds
    features, and produces predictions using the saved model.

    Args:
        raw_dir: Directory containing ``train.csv``, ``store.csv``, ``test.csv``.
        model_path: Path to the ``*.joblib`` model file created by :func:`save_model`.

    Returns:
        Scored DataFrame with an extra ``Sales_Predicted`` column.
    """
    logger.info("Loading data from '%s'", raw_dir)
    train_df, store_df, test_df = _prepare_test_df(raw_dir)

    logger.info("Loading model from '%s'", model_path)
    pipeline = load_model(model_path)

    logger.info(
        "Scoring %d test rows against %d history rows",
        len(test_df),
        len(train_df),
    )
    return score_dataframe(test_df, train_df, store_df, pipeline)


def generate_kaggle_submission(
    raw_dir: str | Path = "data/raw",
    model_path: str | Path = "model/latest.joblib",
    output_path: str | Path = "submissions/submission.csv",
) -> Path:
    """Generate a Kaggle submission CSV from ``test.csv``.

    Args:
        raw_dir: Directory containing raw CSVs.
        model_path: Path to the saved model file.
        output_path: Destination CSV path.

    Returns:
        Resolved path to the written submission file.
    """
    predictions = score_from_files(raw_dir=raw_dir, model_path=model_path)

    # Kaggle format requires Id + Sales columns
    if "Id" not in predictions.columns:
        raise KeyError(
            "'Id' column missing from test.csv — cannot generate Kaggle submission."
        )

    submission = predictions[["Id"]].copy()
    submission["Sales"] = predictions["Sales_Predicted"].round(0).astype(int)

    out_path = Path(output_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(out_path, index=False)
    logger.info("Submission written to '%s' (%d rows)", out_path, len(submission))

    return out_path
