from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import pandas as pd


TARGET_COLUMN = "Sales"
KEY_COLUMNS = ["Store", "Date"]


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add date-derived features known before the sales day happens."""

    if "Date" not in df.columns:
        raise ValueError("Missing required column: Date")

    result = df.copy()
    date = result["Date"]
    iso_calendar = date.dt.isocalendar()

    result["year"] = date.dt.year
    result["month"] = date.dt.month
    result["day"] = date.dt.day
    result["week_of_year"] = iso_calendar.week.astype("int64")
    result["is_month_start"] = date.dt.is_month_start.astype("int64")
    result["is_month_end"] = date.dt.is_month_end.astype("int64")

    return result


def add_store_features(df: pd.DataFrame, store_df: pd.DataFrame) -> pd.DataFrame:
    """Join store attributes and add date-aware store/business features."""

    if "Store" not in df.columns:
        raise ValueError("Input data is missing required column: Store")
    if "Store" not in store_df.columns:
        raise ValueError("store_df is missing required column: Store")
    if store_df["Store"].duplicated().any():
        raise ValueError("store_df must contain one row per Store")

    result = df.merge(store_df, on="Store", how="left", validate="many_to_one")
    result["competition_open_months"] = _competition_open_months(result)
    result["promo2_active"] = _promo2_active(result)

    return result


def add_lag_features(
    df: pd.DataFrame,
    lags: Iterable[int] = (7, 14, 28),
    target_column: str = TARGET_COLUMN,
) -> pd.DataFrame:
    """Add same-store target lags using only earlier rows."""

    _require_columns(df, {"Store", "Date", target_column})

    result = _sort_for_time_features(df)
    grouped_sales = result.groupby("Store", sort=False)[target_column]

    for lag in lags:
        if lag <= 0:
            raise ValueError("Lag values must be positive.")
        result[f"sales_lag_{lag}"] = grouped_sales.shift(lag)

    return result


def add_rolling_features(
    df: pd.DataFrame,
    windows: Iterable[int] = (7, 28),
    target_column: str = TARGET_COLUMN,
) -> pd.DataFrame:
    """Add same-store rolling sales statistics using rows before the current date."""

    _require_columns(df, {"Store", "Date", target_column})

    result = _sort_for_time_features(df)
    shifted_sales = result.groupby("Store", sort=False)[target_column].shift(1)

    for window in windows:
        if window <= 0:
            raise ValueError("Rolling window values must be positive.")

        rolling = shifted_sales.groupby(result["Store"], sort=False).rolling(
            window=window,
            min_periods=1,
        )
        result[f"sales_rolling_mean_{window}"] = rolling.mean().reset_index(
            level=0,
            drop=True,
        )
        result[f"sales_rolling_std_{window}"] = rolling.std().reset_index(
            level=0,
            drop=True,
        )

    return result


def build_features(train_df: pd.DataFrame, store_df: pd.DataFrame) -> pd.DataFrame:
    """Build the initial leakage-safe feature table from raw Rossmann data."""

    result = train_df.copy()
    result = add_calendar_features(result)
    result = add_store_features(result, store_df)
    result = add_lag_features(result)
    result = add_rolling_features(result)
    return result


def build_prediction_features(
    target_df: pd.DataFrame,
    history_df: pd.DataFrame,
    store_df: pd.DataFrame,
    lags: Iterable[int] = (7, 14, 28),
    windows: Iterable[int] = (7, 28),
) -> pd.DataFrame:
    """Build features for future rows using labels from history only.

    ``target_df`` represents rows to predict and may not contain ``Sales``.
    ``history_df`` must contain only observations available before the target
    period. The explicit separation prevents validation or inference labels
    from being used as lag or rolling features.
    """

    _require_columns(target_df, {"Store", "Date"})
    _require_columns(history_df, {"Store", "Date", TARGET_COLUMN})
    lags = tuple(lags)
    windows = tuple(windows)

    if target_df.empty:
        return add_store_features(add_calendar_features(target_df), store_df)
    if history_df.empty:
        raise ValueError("history_df must contain observations before prediction.")
    if history_df["Date"].max() >= target_df["Date"].min():
        raise ValueError("history_df contains labels on or after the target period.")

    result = add_store_features(add_calendar_features(target_df), store_df)
    history = history_df[["Store", "Date", TARGET_COLUMN]].copy()
    history = history.sort_values(["Date", "Store"]).reset_index(drop=True)

    for lag in lags:
        if lag <= 0:
            raise ValueError("Lag values must be positive.")
        lookup = history.rename(columns={TARGET_COLUMN: f"sales_lag_{lag}"}).copy()
        lookup["Date"] = lookup["Date"] + pd.Timedelta(days=lag)
        result = result.merge(
            lookup[["Store", "Date", f"sales_lag_{lag}"]],
            on=["Store", "Date"],
            how="left",
            validate="one_to_one",
        )

    rolling_history = history.copy()
    grouped_sales = rolling_history.groupby("Store", sort=False)[TARGET_COLUMN]
    for window in windows:
        if window <= 0:
            raise ValueError("Rolling window values must be positive.")
        rolling_history[f"sales_rolling_mean_{window}"] = (
            grouped_sales.rolling(window=window, min_periods=1).mean()
            .reset_index(level=0, drop=True)
        )
        rolling_history[f"sales_rolling_std_{window}"] = (
            grouped_sales.rolling(window=window, min_periods=1).std()
            .reset_index(level=0, drop=True)
        )

    rolling_columns = [
        "Store",
        "Date",
        *[
            name
            for window in windows
            for name in (
                f"sales_rolling_mean_{window}",
                f"sales_rolling_std_{window}",
            )
        ],
    ]
    right = rolling_history[rolling_columns].sort_values(["Date", "Store"])
    left = result.sort_values(["Date", "Store"])
    result = pd.merge_asof(
        left,
        right,
        on="Date",
        by="Store",
        direction="backward",
        allow_exact_matches=False,
    )
    return result.sort_values(["Store", "Date"]).reset_index(drop=True)


def summarize_feature_table(df: pd.DataFrame) -> dict[str, Any]:
    """Create a compact report for the engineered feature table."""

    engineered_columns = [
        column
        for column in df.columns
        if column.startswith(("sales_lag_", "sales_rolling_"))
        or column
        in {
            "year",
            "month",
            "day",
            "week_of_year",
            "is_month_start",
            "is_month_end",
            "competition_open_months",
            "promo2_active",
        }
    ]

    return {
        "shape": {"rows": int(df.shape[0]), "columns": int(df.shape[1])},
        "engineered_columns": engineered_columns,
        "missing_values_in_engineered_columns": {
            column: int(count)
            for column, count in df[engineered_columns].isna().sum().items()
            if int(count) > 0
        },
    }


def _require_columns(df: pd.DataFrame, columns: set[str]) -> None:
    missing = columns.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")


def _sort_for_time_features(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(["Store", "Date"]).reset_index(drop=True)


def _competition_open_months(df: pd.DataFrame) -> pd.Series:
    required_columns = {
        "Date",
        "CompetitionOpenSinceMonth",
        "CompetitionOpenSinceYear",
    }
    if not required_columns.issubset(df.columns):
        return pd.Series(pd.NA, index=df.index, dtype="Float64")

    open_year = df["CompetitionOpenSinceYear"]
    open_month = df["CompetitionOpenSinceMonth"]
    has_open_date = open_year.notna() & open_month.notna()

    months = (df["Date"].dt.year - open_year) * 12 + (df["Date"].dt.month - open_month)
    months = months.where(has_open_date)
    months = months.where(months >= 0, 0)

    return months.astype("Float64")


def _promo2_active(df: pd.DataFrame) -> pd.Series:
    required_columns = {
        "Date",
        "Promo2",
        "Promo2SinceWeek",
        "Promo2SinceYear",
        "PromoInterval",
    }
    if not required_columns.issubset(df.columns):
        return pd.Series(0, index=df.index, dtype="int64")

    date_month_name = df["Date"].dt.month_name().str[:3]
    promo_interval = df["PromoInterval"].fillna("")
    month_is_active = [
        month in {part.strip() for part in interval.split(",") if part.strip()}
        for month, interval in zip(date_month_name, promo_interval, strict=False)
    ]

    promo2_start_date = _promo2_start_date(df)
    started = promo2_start_date.notna() & (df["Date"] >= promo2_start_date)

    active = df["Promo2"].eq(1) & started & pd.Series(month_is_active, index=df.index)
    return active.astype("int64")


def _promo2_start_date(df: pd.DataFrame) -> pd.Series:
    has_start = df["Promo2SinceYear"].notna() & df["Promo2SinceWeek"].notna()
    start_dates = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")

    year_week = (
        df.loc[has_start, "Promo2SinceYear"].astype("int64").astype(str)
        + "-"
        + df.loc[has_start, "Promo2SinceWeek"].astype("int64").astype(str)
        + "-1"
    )
    start_dates.loc[has_start] = pd.to_datetime(
        year_week,
        format="%G-%V-%u",
        errors="coerce",
    )

    return start_dates
