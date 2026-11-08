from __future__ import annotations

import pandas as pd

from retail_demand.evaluation.metrics import regression_metrics
from retail_demand.modeling.split import (
    make_final_days_holdout,
    make_walk_forward_splits,
)


BASELINE_TARGET_COLUMN = "Sales"


def add_calendar_lag_baseline(
    target_rows: pd.DataFrame,
    history: pd.DataFrame,
    lag_days: int,
    prediction_column: str,
) -> pd.DataFrame:
    """Attach a sales baseline using the same store's value N calendar days ago."""

    required_columns = {"Store", "Date", BASELINE_TARGET_COLUMN}
    missing_history = required_columns.difference(history.columns)
    missing_target = {"Store", "Date"}.difference(target_rows.columns)
    if missing_history:
        raise ValueError(f"History is missing columns: {sorted(missing_history)}")
    if missing_target:
        raise ValueError(f"Target rows are missing columns: {sorted(missing_target)}")

    lag_lookup = history[["Store", "Date", BASELINE_TARGET_COLUMN]].copy()
    lag_lookup["Date"] = lag_lookup["Date"] + pd.Timedelta(days=lag_days)
    lag_lookup = lag_lookup.rename(columns={BASELINE_TARGET_COLUMN: prediction_column})

    return target_rows.merge(lag_lookup, on=["Store", "Date"], how="left")


def add_store_mean_baseline(
    target_rows: pd.DataFrame,
    training_rows: pd.DataFrame,
    prediction_column: str = "pred_store_mean",
) -> pd.DataFrame:
    """Attach each store's historical mean sales from the training period."""

    required_columns = {"Store", BASELINE_TARGET_COLUMN}
    missing_training = required_columns.difference(training_rows.columns)
    missing_target = {"Store"}.difference(target_rows.columns)
    if missing_training:
        raise ValueError(f"Training rows are missing columns: {sorted(missing_training)}")
    if missing_target:
        raise ValueError(f"Target rows are missing columns: {sorted(missing_target)}")

    store_means = (
        training_rows.groupby("Store", as_index=False)[BASELINE_TARGET_COLUMN]
        .mean()
        .rename(columns={BASELINE_TARGET_COLUMN: prediction_column})
    )
    return target_rows.merge(store_means, on="Store", how="left")


def evaluate_baseline_column(
    df: pd.DataFrame,
    prediction_column: str,
    target_column: str = BASELINE_TARGET_COLUMN,
) -> dict[str, float | int]:
    """Evaluate one prediction column after dropping rows with missing predictions."""

    evaluation_rows = df.loc[df[prediction_column].notna()].copy()
    metrics = regression_metrics(
        evaluation_rows[target_column],
        evaluation_rows[prediction_column],
    )
    return {
        "rows_evaluated": int(evaluation_rows.shape[0]),
        **metrics,
    }


def run_baseline_experiment(
    train_df: pd.DataFrame,
    holdout_days: int = 42,
) -> dict[str, object]:
    """Run mandatory Phase 2 baselines on an open-day temporal holdout."""

    split = make_final_days_holdout(train_df, holdout_days=holdout_days)

    holdout_open = split.holdout.loc[split.holdout["Open"].eq(1)].copy()
    train_open = split.train.loc[split.train["Open"].eq(1)].copy()

    predictions = holdout_open.copy()
    predictions = add_calendar_lag_baseline(
        predictions,
        history=train_df,
        lag_days=1,
        prediction_column="pred_naive_yesterday",
    )
    predictions = add_calendar_lag_baseline(
        predictions,
        history=train_df,
        lag_days=7,
        prediction_column="pred_seasonal_naive_7",
    )
    predictions = add_store_mean_baseline(
        predictions,
        training_rows=train_open,
        prediction_column="pred_store_mean",
    )

    metric_columns = [
        "pred_naive_yesterday",
        "pred_seasonal_naive_7",
        "pred_store_mean",
    ]

    return {
        "cutoff_date": split.cutoff_date,
        "train_rows": int(split.train.shape[0]),
        "holdout_rows": int(split.holdout.shape[0]),
        "holdout_open_rows": int(holdout_open.shape[0]),
        "baselines": {
            column: evaluate_baseline_column(predictions, column)
            for column in metric_columns
        },
    }


def run_walk_forward_baseline_experiment(
    train_df: pd.DataFrame,
    initial_train_days: int = 365,
    validation_days: int = 42,
    step_days: int = 42,
    n_splits: int = 3,
) -> dict[str, object]:
    """Evaluate baselines on expanding folds without future-label leakage."""

    folds = make_walk_forward_splits(
        train_df,
        initial_train_days=initial_train_days,
        validation_days=validation_days,
        step_days=step_days,
        n_splits=n_splits,
    )
    metric_columns = [
        "pred_naive_yesterday",
        "pred_seasonal_naive_7",
        "pred_store_mean",
    ]
    fold_reports: list[dict[str, object]] = []
    all_predictions = {column: [] for column in metric_columns}

    for fold_number, fold in enumerate(folds, start=1):
        validation_open = fold.validation.loc[fold.validation["Open"].eq(1)].copy()
        train_open = fold.train.loc[fold.train["Open"].eq(1)].copy()
        predictions = add_calendar_lag_baseline(
            validation_open,
            history=fold.train,
            lag_days=1,
            prediction_column="pred_naive_yesterday",
        )
        predictions = add_calendar_lag_baseline(
            predictions,
            history=fold.train,
            lag_days=7,
            prediction_column="pred_seasonal_naive_7",
        )
        predictions = add_store_mean_baseline(
            predictions,
            training_rows=train_open,
            prediction_column="pred_store_mean",
        )

        fold_reports.append(
            {
                "fold": fold_number,
                "train_end": fold.train_end,
                "validation_start": fold.validation_start,
                "validation_end": fold.validation_end,
                "validation_open_rows": int(validation_open.shape[0]),
                "baselines": {
                    column: evaluate_baseline_column(predictions, column)
                    for column in metric_columns
                },
            }
        )
        for column in metric_columns:
            evaluated = predictions.loc[predictions[column].notna()]
            all_predictions[column].append(evaluated[["Sales", column]])

    combined = {}
    for column in metric_columns:
        evaluated = pd.concat(all_predictions[column], ignore_index=True)
        combined[column] = {
            "rows_evaluated": int(evaluated.shape[0]),
            **regression_metrics(evaluated["Sales"], evaluated[column]),
        }

    return {
        "validation_strategy": "walk_forward_expanding_window",
        "initial_train_days": initial_train_days,
        "validation_days": validation_days,
        "step_days": step_days,
        "fold_count": len(folds),
        "folds": fold_reports,
        "combined_baselines": combined,
    }
