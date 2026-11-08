from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from retail_demand.data.validation import (
    issues_to_dicts,
    validate_store_data,
    validate_train_data,
    validate_train_store_relationship,
)


@dataclass(frozen=True)
class RawDataPaths:
    """Filesystem locations for the raw Rossmann CSV files."""

    raw_dir: Path

    @property
    def train(self) -> Path:
        return self.raw_dir / "train.csv"

    @property
    def store(self) -> Path:
        return self.raw_dir / "store.csv"

    @property
    def test(self) -> Path:
        return self.raw_dir / "test.csv"

    def missing_required_files(self) -> list[str]:
        required = [self.train, self.store]
        return [str(path) for path in required if not path.exists()]


def load_train(paths: RawDataPaths) -> pd.DataFrame:
    """Load train.csv with Date parsed as a pandas datetime column."""

    if not paths.train.exists():
        raise FileNotFoundError(f"Missing required file: {paths.train}")

    return pd.read_csv(paths.train, parse_dates=["Date"], low_memory=False)


def load_store(paths: RawDataPaths) -> pd.DataFrame:
    """Load store.csv."""

    if not paths.store.exists():
        raise FileNotFoundError(f"Missing required file: {paths.store}")

    return pd.read_csv(paths.store, low_memory=False)


def summarize_dataframe(df: pd.DataFrame) -> dict[str, Any]:
    """Create a compact schema and quality summary for one dataframe."""

    return {
        "shape": {
            "rows": int(df.shape[0]),
            "columns": int(df.shape[1]),
        },
        "columns": list(df.columns),
        "dtypes": {column: str(dtype) for column, dtype in df.dtypes.items()},
        "missing_values": {
            column: int(count)
            for column, count in df.isna().sum().items()
            if int(count) > 0
        },
        "duplicate_rows": int(df.duplicated().sum()),
    }


def summarize_train_dates(train_df: pd.DataFrame) -> dict[str, Any]:
    """Summarize the date range in train.csv."""

    if "Date" not in train_df.columns:
        return {"error": "Date column not found"}

    return {
        "min_date": train_df["Date"].min(),
        "max_date": train_df["Date"].max(),
        "unique_dates": int(train_df["Date"].nunique()),
    }


def summarize_forecasting_context(train_df: pd.DataFrame) -> dict[str, Any]:
    """Summarize target and operational behavior relevant to forecasting."""

    summary: dict[str, Any] = {}

    if "Open" in train_df.columns:
        summary["open_counts"] = {
            str(open_value): int(count)
            for open_value, count in train_df["Open"].value_counts(dropna=False).items()
        }

    if {"Open", "Sales"}.issubset(train_df.columns):
        open_sales = train_df.loc[train_df["Open"].eq(1), "Sales"]
        summary["open_day_sales"] = {
            "count": int(open_sales.shape[0]),
            "mean": float(open_sales.mean()),
            "median": float(open_sales.median()),
            "min": int(open_sales.min()),
            "max": int(open_sales.max()),
            "zero_sales_open_days": int(
                train_df.loc[train_df["Open"].eq(1), "Sales"].eq(0).sum()
            ),
        }

    if "StateHoliday" in train_df.columns:
        summary["state_holiday_values"] = {
            str(value): int(count)
            for value, count in train_df["StateHoliday"]
            .value_counts(dropna=False)
            .items()
        }

    return summary


def summarize_test_dates(test_df: pd.DataFrame) -> dict[str, Any]:
    """Summarize the date range in test.csv."""

    if "Date" not in test_df.columns:
        return {"error": "Date column not found"}

    return {
        "min_date": test_df["Date"].min(),
        "max_date": test_df["Date"].max(),
        "unique_dates": int(test_df["Date"].nunique()),
    }


def inspect_raw_dataset(paths: RawDataPaths) -> dict[str, Any]:
    """Inspect required raw files without modifying them."""

    missing_files = paths.missing_required_files()
    if missing_files:
        return {
            "status": "missing_files",
            "missing_required_files": missing_files,
            "next_step": "Place Rossmann train.csv and store.csv in data/raw/.",
        }

    train_df = load_train(paths)
    store_df = load_store(paths)

    report: dict[str, Any] = {
        "status": "ok",
        "train": summarize_dataframe(train_df),
        "store": summarize_dataframe(store_df),
        "validation_issues": issues_to_dicts(
            validate_train_data(train_df)
            + validate_store_data(store_df)
            + validate_train_store_relationship(train_df, store_df)
        ),
        "train_dates": summarize_train_dates(train_df),
        "forecasting_context": summarize_forecasting_context(train_df),
        "unique_stores_in_train": int(train_df["Store"].nunique())
        if "Store" in train_df.columns
        else None,
        "unique_stores_in_store": int(store_df["Store"].nunique())
        if "Store" in store_df.columns
        else None,
    }

    if paths.test.exists():
        test_df = pd.read_csv(paths.test, parse_dates=["Date"], low_memory=False)
        report["test"] = summarize_dataframe(test_df)
        report["test_dates"] = summarize_test_dates(test_df)

    return report
