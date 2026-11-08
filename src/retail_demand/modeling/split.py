from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class TemporalSplit:
    """Train and holdout data separated by time."""

    train: pd.DataFrame
    holdout: pd.DataFrame
    cutoff_date: pd.Timestamp


@dataclass(frozen=True)
class WalkForwardSplit:
    """One expanding-training-window validation fold."""

    train: pd.DataFrame
    validation: pd.DataFrame
    train_end: pd.Timestamp
    validation_start: pd.Timestamp
    validation_end: pd.Timestamp


def make_walk_forward_splits(
    df: pd.DataFrame,
    date_column: str = "Date",
    initial_train_days: int = 180,
    validation_days: int = 42,
    step_days: int = 42,
    n_splits: int | None = None,
) -> list[WalkForwardSplit]:
    """Create expanding-window folds without shuffling time.

    Each validation period occurs strictly after its training period. The
    training window expands as the origin moves forward.
    """

    if date_column not in df.columns:
        raise ValueError(f"Missing date column: {date_column}")
    if min(initial_train_days, validation_days, step_days) <= 0:
        raise ValueError("Fold durations must be positive.")
    if n_splits is not None and n_splits <= 0:
        raise ValueError("n_splits must be positive when provided.")

    dates = pd.Series(df[date_column].dropna().unique()).sort_values()
    if dates.empty:
        raise ValueError("Cannot create folds from an empty date column.")

    first_train_end = dates.iloc[0] + pd.Timedelta(days=initial_train_days - 1)
    folds: list[WalkForwardSplit] = []
    train_end = first_train_end

    while True:
        validation_start = train_end + pd.Timedelta(days=1)
        validation_end = validation_start + pd.Timedelta(days=validation_days - 1)
        train = df.loc[df[date_column] <= train_end].copy()
        validation = df.loc[
            df[date_column].between(validation_start, validation_end)
        ].copy()

        if train.empty or validation.empty:
            break
        folds.append(
            WalkForwardSplit(
                train=train,
                validation=validation,
                train_end=train_end,
                validation_start=validation_start,
                validation_end=validation_end,
            )
        )
        if n_splits is not None and len(folds) >= n_splits:
            break
        train_end += pd.Timedelta(days=step_days)

    if not folds:
        raise ValueError("Unable to create a complete walk-forward fold.")
    return folds


def make_final_days_holdout(
    df: pd.DataFrame,
    date_column: str = "Date",
    holdout_days: int = 42,
) -> TemporalSplit:
    """Use the final N calendar days as a temporal holdout."""

    if holdout_days <= 0:
        raise ValueError("holdout_days must be positive.")
    if date_column not in df.columns:
        raise ValueError(f"Missing date column: {date_column}")

    max_date = df[date_column].max()
    cutoff_date = max_date - pd.Timedelta(days=holdout_days - 1)

    train = df.loc[df[date_column] < cutoff_date].copy()
    holdout = df.loc[df[date_column] >= cutoff_date].copy()

    if train.empty:
        raise ValueError("Temporal split produced an empty training set.")
    if holdout.empty:
        raise ValueError("Temporal split produced an empty holdout set.")

    return TemporalSplit(train=train, holdout=holdout, cutoff_date=cutoff_date)
