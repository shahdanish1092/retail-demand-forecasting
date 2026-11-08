from __future__ import annotations

from typing import Any

import pandas as pd

from retail_demand.evaluation.metrics import regression_metrics


def segment_error_report(
    prediction_df: pd.DataFrame,
    segment_column: str,
    target_column: str = "Sales",
    prediction_column: str = "prediction",
) -> list[dict[str, Any]]:
    """Calculate MAE and WAPE for each value of one segment column."""

    required_columns = {segment_column, target_column, prediction_column}
    missing = required_columns.difference(prediction_df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    rows: list[dict[str, Any]] = []
    for segment_value, segment_df in prediction_df.groupby(segment_column, dropna=False):
        metrics = regression_metrics(
            segment_df[target_column],
            segment_df[prediction_column],
        )
        rows.append(
            {
                "segment": str(segment_value),
                "rows": int(segment_df.shape[0]),
                "actual_sales_sum": float(segment_df[target_column].sum()),
                **metrics,
            }
        )

    return sorted(rows, key=lambda row: row["wape"], reverse=True)


def top_absolute_errors(
    prediction_df: pd.DataFrame,
    n: int = 10,
    target_column: str = "Sales",
    prediction_column: str = "prediction",
) -> list[dict[str, Any]]:
    """Return the largest individual absolute forecast errors."""

    required_columns = {"Store", "Date", target_column, prediction_column}
    missing = required_columns.difference(prediction_df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if n <= 0:
        raise ValueError("n must be positive.")

    result = prediction_df.copy()
    result["absolute_error"] = (
        result[target_column] - result[prediction_column]
    ).abs()

    columns = ["Store", "Date", target_column, prediction_column, "absolute_error"]
    optional_columns = [
        "Promo",
        "StateHoliday",
        "SchoolHoliday",
        "DayOfWeek",
        "StoreType",
        "Assortment",
    ]
    columns.extend(column for column in optional_columns if column in result.columns)

    return result.nlargest(n, "absolute_error")[columns].to_dict(orient="records")


def model_error_analysis_report(
    prediction_df: pd.DataFrame,
    segment_columns: list[str] | None = None,
) -> dict[str, Any]:
    """Create the initial error analysis report for one model run."""

    if segment_columns is None:
        segment_columns = [
            "Promo",
            "DayOfWeek",
            "StateHoliday",
            "SchoolHoliday",
            "StoreType",
            "Assortment",
        ]

    available_segments = [
        column for column in segment_columns if column in prediction_df.columns
    ]

    return {
        "overall_metrics": regression_metrics(
            prediction_df["Sales"],
            prediction_df["prediction"],
        ),
        "segment_metrics": {
            column: segment_error_report(prediction_df, column)
            for column in available_segments
        },
        "top_absolute_errors": top_absolute_errors(prediction_df, n=10),
    }

