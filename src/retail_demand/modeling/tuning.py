from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import pandas as pd

from retail_demand.evaluation.metrics import regression_metrics
from retail_demand.modeling.training import (
    TARGET_COLUMN,
    make_hist_gradient_boosting_pipeline,
    make_training_frame,
    select_feature_columns,
)


DEFAULT_PARAM_GRID: list[dict[str, Any]] = [
    {
        "learning_rate": 0.05,
        "max_iter": 200,
        "max_leaf_nodes": 31,
        "l2_regularization": 0.0,
    },
    {
        "learning_rate": 0.08,
        "max_iter": 150,
        "max_leaf_nodes": 31,
        "l2_regularization": 0.01,
    },
    {
        "learning_rate": 0.05,
        "max_iter": 250,
        "max_leaf_nodes": 63,
        "l2_regularization": 0.01,
    },
    {
        "learning_rate": 0.03,
        "max_iter": 300,
        "max_leaf_nodes": 63,
        "l2_regularization": 0.05,
    },
]


def evaluate_parameter_candidate(
    train_open: pd.DataFrame,
    holdout_open: pd.DataFrame,
    numeric_features: list[str],
    categorical_features: list[str],
    model_params: dict[str, Any],
) -> dict[str, Any]:
    """Train and evaluate one HistGradientBoostingRegressor parameter candidate."""

    feature_columns = numeric_features + categorical_features
    pipeline = make_hist_gradient_boosting_pipeline(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        model_params=model_params,
    )
    pipeline.fit(train_open[feature_columns], train_open[TARGET_COLUMN])

    predictions = pd.Series(
        pipeline.predict(holdout_open[feature_columns]),
        index=holdout_open.index,
    ).clip(lower=0)
    metrics = regression_metrics(holdout_open[TARGET_COLUMN], predictions)

    return {
        "params": model_params,
        "metrics": metrics,
    }


def tune_hist_gradient_boosting(
    train_df: pd.DataFrame,
    store_df: pd.DataFrame,
    holdout_days: int = 42,
    param_grid: Iterable[dict[str, Any]] = DEFAULT_PARAM_GRID,
) -> dict[str, Any]:
    """Evaluate a small parameter grid on the same temporal holdout."""

    train_open, holdout_open = make_training_frame(
        train_df,
        store_df,
        holdout_days=holdout_days,
    )
    numeric_features, categorical_features = select_feature_columns(train_open)

    candidate_results = [
        evaluate_parameter_candidate(
            train_open=train_open,
            holdout_open=holdout_open,
            numeric_features=numeric_features,
            categorical_features=categorical_features,
            model_params=params,
        )
        for params in param_grid
    ]
    candidate_results = sorted(
        candidate_results,
        key=lambda result: result["metrics"]["wape"],
    )

    return {
        "model": "HistGradientBoostingRegressor",
        "selection_metric": "wape",
        "holdout_days": holdout_days,
        "train_open_rows": int(train_open.shape[0]),
        "holdout_open_rows": int(holdout_open.shape[0]),
        "candidate_count": len(candidate_results),
        "best_candidate": candidate_results[0],
        "candidate_results": candidate_results,
    }

