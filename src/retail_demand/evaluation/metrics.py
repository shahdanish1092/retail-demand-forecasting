from __future__ import annotations

import numpy as np
import pandas as pd


def mae(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Mean absolute error in the same unit as the target."""

    return float(np.mean(np.abs(y_true.to_numpy() - y_pred.to_numpy())))


def wape(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Weighted absolute percentage error.

    WAPE = sum(abs(actual - prediction)) / sum(abs(actual))

    For this project, lower is better. A value of 0.15 means the total absolute
    error is 15 percent of total actual sales over the evaluated rows.
    """

    denominator = float(np.sum(np.abs(y_true.to_numpy())))
    if denominator == 0.0:
        raise ValueError("WAPE is undefined when the sum of actual values is zero.")
    numerator = float(np.sum(np.abs(y_true.to_numpy() - y_pred.to_numpy())))
    return numerator / denominator


def regression_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    """Return the initial metrics used for baseline comparison."""

    return {
        "mae": mae(y_true, y_pred),
        "wape": wape(y_true, y_pred),
    }


def pinball_loss(
    y_true: pd.Series,
    y_pred: pd.Series,
    quantile: float,
) -> float:
    """Measure asymmetric error for a forecast quantile."""

    if not 0.0 < quantile < 1.0:
        raise ValueError("quantile must be strictly between 0 and 1.")
    error = y_true.to_numpy() - y_pred.to_numpy()
    return float(np.mean(np.maximum(quantile * error, (quantile - 1) * error)))

