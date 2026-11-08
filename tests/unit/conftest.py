"""Shared pytest fixtures for retail_demand test suite.

All fixtures build minimal but realistic synthetic DataFrames so individual
test modules do not have to repeat boilerplate construction code.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Primitive building blocks
# ---------------------------------------------------------------------------


def _make_date_range(start: str, periods: int) -> pd.DatetimeIndex:
    return pd.date_range(start=start, periods=periods, freq="D")


# ---------------------------------------------------------------------------
# Store metadata
# ---------------------------------------------------------------------------


@pytest.fixture()
def minimal_store_df() -> pd.DataFrame:
    """Single-store store.csv with all required columns present."""
    return pd.DataFrame(
        {
            "Store": [1],
            "StoreType": ["a"],
            "Assortment": ["a"],
            "CompetitionDistance": [500.0],
            "CompetitionOpenSinceMonth": [1],
            "CompetitionOpenSinceYear": [2010],
            "Promo2": [0],
            "Promo2SinceWeek": [np.nan],
            "Promo2SinceYear": [np.nan],
            "PromoInterval": [np.nan],
        }
    )


@pytest.fixture()
def two_store_df() -> pd.DataFrame:
    """Two-store store.csv (store IDs 1 and 2)."""
    return pd.DataFrame(
        {
            "Store": [1, 2],
            "StoreType": ["a", "b"],
            "Assortment": ["a", "c"],
            "CompetitionDistance": [500.0, 1200.0],
            "CompetitionOpenSinceMonth": [1, 6],
            "CompetitionOpenSinceYear": [2010, 2012],
            "Promo2": [0, 1],
            "Promo2SinceWeek": [np.nan, 10.0],
            "Promo2SinceYear": [np.nan, 2013.0],
            "PromoInterval": [np.nan, "Mar,Jun,Sept,Dec"],
        }
    )


# ---------------------------------------------------------------------------
# Training DataFrames
# ---------------------------------------------------------------------------


@pytest.fixture()
def minimal_train_df() -> pd.DataFrame:
    """50 days of sales for a single store (all days open)."""
    dates = _make_date_range("2015-01-01", 50)
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "Store": 1,
            "DayOfWeek": [(d.dayofweek % 7) + 1 for d in dates],
            "Date": dates,
            "Sales": rng.integers(3000, 8000, size=50).astype(float),
            "Customers": rng.integers(200, 800, size=50),
            "Open": 1,
            "Promo": rng.integers(0, 2, size=50),
            "StateHoliday": "0",
            "SchoolHoliday": 0,
        }
    )


@pytest.fixture()
def two_store_train_df() -> pd.DataFrame:
    """50 days of sales for stores 1 and 2 (all days open)."""
    dates = _make_date_range("2015-01-01", 50)
    rng = np.random.default_rng(42)
    rows = []
    for store_id in [1, 2]:
        for date in dates:
            rows.append(
                {
                    "Store": store_id,
                    "DayOfWeek": (date.dayofweek % 7) + 1,
                    "Date": date,
                    "Sales": float(rng.integers(3000, 9000)),
                    "Customers": int(rng.integers(200, 900)),
                    "Open": 1,
                    "Promo": int(rng.integers(0, 2)),
                    "StateHoliday": "0",
                    "SchoolHoliday": 0,
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Full feature DataFrames (after build_features)
# ---------------------------------------------------------------------------


@pytest.fixture()
def feature_df(minimal_train_df: pd.DataFrame, minimal_store_df: pd.DataFrame) -> pd.DataFrame:
    """Feature table built from minimal_train_df + minimal_store_df."""
    from retail_demand.features.build import build_features

    return build_features(minimal_train_df, minimal_store_df)


@pytest.fixture()
def two_store_feature_df(
    two_store_train_df: pd.DataFrame, two_store_df: pd.DataFrame
) -> pd.DataFrame:
    """Feature table built from two_store_train_df + two_store_df."""
    from retail_demand.features.build import build_features

    return build_features(two_store_train_df, two_store_df)


# ---------------------------------------------------------------------------
# Prediction DataFrames
# ---------------------------------------------------------------------------


@pytest.fixture()
def prediction_df(feature_df: pd.DataFrame) -> pd.DataFrame:
    """A minimal prediction DataFrame with actual and predicted Sales columns."""
    df = feature_df[feature_df["Sales"].notna()].copy()
    rng = np.random.default_rng(7)
    df = df.head(30).copy()
    df["pred"] = df["Sales"] * rng.uniform(0.8, 1.2, size=len(df))
    return df
