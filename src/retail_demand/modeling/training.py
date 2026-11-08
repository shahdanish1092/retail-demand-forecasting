from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from typing import Any

import mlflow
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from retail_demand.evaluation.metrics import pinball_loss, regression_metrics
from retail_demand.features.build import build_features, build_prediction_features
from retail_demand.modeling.split import (
    WalkForwardSplit,
    make_final_days_holdout,
    make_walk_forward_splits,
)


TARGET_COLUMN = "Sales"
LEAKAGE_COLUMNS = {"Customers"}
NON_FEATURE_COLUMNS = {
    "Date",
    "Sales",
    "Customers",
    "Open",
    "CompetitionOpenSinceMonth",
    "CompetitionOpenSinceYear",
    "Promo2SinceWeek",
    "Promo2SinceYear",
    "PromoInterval",
}

NUMERIC_FEATURES = [
    "DayOfWeek",
    "Promo",
    "SchoolHoliday",
    "CompetitionDistance",
    "Promo2",
    "year",
    "month",
    "day",
    "week_of_year",
    "is_month_start",
    "is_month_end",
    "competition_open_months",
    "promo2_active",
    "sales_lag_7",
    "sales_lag_14",
    "sales_lag_28",
    "sales_rolling_mean_7",
    "sales_rolling_std_7",
    "sales_rolling_mean_28",
    "sales_rolling_std_28",
]

CATEGORICAL_FEATURES = [
    "Store",
    "StateHoliday",
    "StoreType",
    "Assortment",
]


@dataclass(frozen=True)
class ModelRun:
    """Fitted model and holdout predictions from one training run."""

    pipeline: Pipeline
    train_open: pd.DataFrame
    holdout_predictions: pd.DataFrame
    numeric_features: list[str]
    categorical_features: list[str]
    metrics: dict[str, float]

    @property
    def feature_columns(self) -> list[str]:
        return self.numeric_features + self.categorical_features


def make_training_frame(
    train_df: pd.DataFrame,
    store_df: pd.DataFrame,
    holdout_days: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build features and return open-day train and holdout frames."""

    split = make_final_days_holdout(train_df, holdout_days=holdout_days)
    train_features = build_features(split.train, store_df)
    holdout_features = build_prediction_features(
        target_df=split.holdout,
        history_df=split.train,
        store_df=store_df,
    )

    train_open = train_features.loc[train_features["Open"].eq(1)].copy()
    holdout_open = holdout_features.loc[holdout_features["Open"].eq(1)].copy()

    return train_open, holdout_open


def make_walk_forward_training_frame(
    fold: WalkForwardSplit,
    store_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build one leakage-safe train/validation frame pair."""

    train_features = build_features(fold.train, store_df)
    validation_features = build_prediction_features(
        target_df=fold.validation,
        history_df=fold.train,
        store_df=store_df,
    )
    train_open = train_features.loc[train_features["Open"].eq(1)].copy()
    validation_open = validation_features.loc[
        validation_features["Open"].eq(1)
    ].copy()
    return train_open, validation_open


def select_feature_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Return available numeric and categorical features in stable order."""

    numeric_features = [column for column in NUMERIC_FEATURES if column in df.columns]
    categorical_features = [
        column for column in CATEGORICAL_FEATURES if column in df.columns
    ]

    selected_features = set(numeric_features + categorical_features)
    forbidden_selected = selected_features.intersection(NON_FEATURE_COLUMNS)
    if forbidden_selected:
        raise ValueError(f"Forbidden feature columns selected: {sorted(forbidden_selected)}")

    return numeric_features, categorical_features


def make_hist_gradient_boosting_pipeline(
    numeric_features: list[str],
    categorical_features: list[str],
    model_params: dict[str, Any] | None = None,
) -> Pipeline:
    """Create the first sklearn regression pipeline."""

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                SimpleImputer(strategy="median"),
                numeric_features,
            ),
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "encoder",
                            OrdinalEncoder(
                                handle_unknown="use_encoded_value",
                                unknown_value=-1,
                                encoded_missing_value=-1,
                            ),
                        ),
                    ]
                ),
                categorical_features,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    params = {
        "loss": "squared_error",
        "learning_rate": 0.08,
        "max_iter": 150,
        "l2_regularization": 0.01,
        "early_stopping": True,
        "random_state": 42,
    }
    if model_params is not None:
        params.update(model_params)

    model = HistGradientBoostingRegressor(**params)

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


def make_lightgbm_pipeline(
    numeric_features: list[str],
    categorical_features: list[str],
    model_params: dict[str, Any] | None = None,
) -> Pipeline:
    """Create the primary LightGBM regression pipeline."""

    try:
        from lightgbm import LGBMRegressor
    except ImportError as exc:
        raise ImportError(
            "LightGBM is required for this model. Install project dependencies "
            "with `pip install -e .`."
        ) from exc

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", SimpleImputer(strategy="median"), numeric_features),
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "encoder",
                            OrdinalEncoder(
                                handle_unknown="use_encoded_value",
                                unknown_value=-1,
                                encoded_missing_value=-1,
                            ),
                        ),
                    ]
                ),
                categorical_features,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    params = {
        "objective": "regression",
        "n_estimators": 300,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "reg_lambda": 0.1,
        "random_state": 42,
        "n_jobs": -1,
        "verbosity": -1,
    }
    if model_params is not None:
        params.update(model_params)

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", LGBMRegressor(**params)),
        ]
    )


def make_lightgbm_quantile_pipeline(
    quantile: float,
    numeric_features: list[str],
    categorical_features: list[str],
    model_params: dict[str, Any] | None = None,
) -> Pipeline:
    """Create a LightGBM model for one conditional demand quantile."""

    if not 0.0 < quantile < 1.0:
        raise ValueError("quantile must be strictly between 0 and 1.")
    params = {"objective": "quantile", "alpha": quantile}
    if model_params is not None:
        params.update(model_params)
    return make_lightgbm_pipeline(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        model_params=params,
    )


def fit_first_model(
    train_df: pd.DataFrame,
    store_df: pd.DataFrame,
    holdout_days: int = 42,
    model_params: dict[str, Any] | None = None,
) -> ModelRun:
    """Fit the first model and return holdout predictions for analysis."""

    train_open, holdout_open = make_training_frame(
        train_df,
        store_df,
        holdout_days=holdout_days,
    )
    numeric_features, categorical_features = select_feature_columns(train_open)
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
        name="prediction",
    ).clip(lower=0)

    metrics = regression_metrics(holdout_open[TARGET_COLUMN], predictions)
    holdout_predictions = holdout_open.copy()
    holdout_predictions["prediction"] = predictions
    holdout_predictions["error"] = (
        holdout_predictions[TARGET_COLUMN] - holdout_predictions["prediction"]
    )
    holdout_predictions["absolute_error"] = holdout_predictions["error"].abs()

    return ModelRun(
        pipeline=pipeline,
        train_open=train_open,
        holdout_predictions=holdout_predictions,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        metrics=metrics,
    )


def summarize_model_run(model_run: ModelRun, holdout_days: int) -> dict[str, Any]:
    """Summarize a fitted model run for CLI output and documentation."""

    return {
        "model": "HistGradientBoostingRegressor",
        "holdout_days": holdout_days,
        "train_open_rows": int(model_run.train_open.shape[0]),
        "holdout_open_rows": int(model_run.holdout_predictions.shape[0]),
        "numeric_features": model_run.numeric_features,
        "categorical_features": model_run.categorical_features,
        "metrics": model_run.metrics,
        "baseline_to_beat": {
            "name": "pred_store_mean",
            "wape": 0.208699072533352,
        },
        "beats_current_best_baseline": bool(
            model_run.metrics["wape"] < 0.208699072533352
        ),
    }


def train_and_evaluate_first_model(
    train_df: pd.DataFrame,
    store_df: pd.DataFrame,
    holdout_days: int = 42,
    model_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Train the first HistGradientBoostingRegressor and evaluate holdout metrics."""

    model_run = fit_first_model(
        train_df,
        store_df,
        holdout_days=holdout_days,
        model_params=model_params,
    )
    return summarize_model_run(model_run, holdout_days=holdout_days)


def evaluate_walk_forward_model(
    train_df: pd.DataFrame,
    store_df: pd.DataFrame,
    initial_train_days: int = 365,
    validation_days: int = 42,
    step_days: int = 42,
    n_splits: int = 3,
    model_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Train and evaluate independent models on expanding time-series folds."""

    folds = make_walk_forward_splits(
        train_df,
        initial_train_days=initial_train_days,
        validation_days=validation_days,
        step_days=step_days,
        n_splits=n_splits,
    )
    fold_reports: list[dict[str, Any]] = []
    all_actuals: list[pd.Series] = []
    all_predictions: list[pd.Series] = []

    for fold_number, fold in enumerate(folds, start=1):
        train_open, validation_open = make_walk_forward_training_frame(
            fold, store_df
        )
        numeric_features, categorical_features = select_feature_columns(train_open)
        feature_columns = numeric_features + categorical_features
        pipeline = make_hist_gradient_boosting_pipeline(
            numeric_features=numeric_features,
            categorical_features=categorical_features,
            model_params=model_params,
        )
        pipeline.fit(train_open[feature_columns], train_open[TARGET_COLUMN])
        predictions = pd.Series(
            pipeline.predict(validation_open[feature_columns]),
            index=validation_open.index,
        ).clip(lower=0)
        actuals = validation_open[TARGET_COLUMN]
        metrics = regression_metrics(actuals, predictions)
        fold_reports.append(
            {
                "fold": fold_number,
                "train_end": fold.train_end,
                "validation_start": fold.validation_start,
                "validation_end": fold.validation_end,
                "train_open_rows": int(train_open.shape[0]),
                "validation_open_rows": int(validation_open.shape[0]),
                "metrics": metrics,
            }
        )
        all_actuals.append(actuals)
        all_predictions.append(predictions)

    combined_metrics = regression_metrics(
        pd.concat(all_actuals, ignore_index=True),
        pd.concat(all_predictions, ignore_index=True),
    )
    return {
        "model": "HistGradientBoostingRegressor",
        "validation_strategy": "walk_forward_expanding_window",
        "initial_train_days": initial_train_days,
        "validation_days": validation_days,
        "step_days": step_days,
        "fold_count": len(fold_reports),
        "folds": fold_reports,
        "combined_metrics": combined_metrics,
    }


def evaluate_walk_forward_lightgbm(
    train_df: pd.DataFrame,
    store_df: pd.DataFrame,
    initial_train_days: int = 365,
    validation_days: int = 42,
    step_days: int = 42,
    n_splits: int = 3,
    model_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate LightGBM on the same leakage-safe expanding folds."""

    folds = make_walk_forward_splits(
        train_df,
        initial_train_days=initial_train_days,
        validation_days=validation_days,
        step_days=step_days,
        n_splits=n_splits,
    )
    fold_reports: list[dict[str, Any]] = []
    all_actuals: list[pd.Series] = []
    all_predictions: list[pd.Series] = []

    for fold_number, fold in enumerate(folds, start=1):
        train_open, validation_open = make_walk_forward_training_frame(
            fold, store_df
        )
        numeric_features, categorical_features = select_feature_columns(train_open)
        feature_columns = numeric_features + categorical_features
        pipeline = make_lightgbm_pipeline(
            numeric_features=numeric_features,
            categorical_features=categorical_features,
            model_params=model_params,
        )
        pipeline.fit(train_open[feature_columns], train_open[TARGET_COLUMN])
        predictions = pd.Series(
            pipeline.predict(validation_open[feature_columns]),
            index=validation_open.index,
        ).clip(lower=0)
        actuals = validation_open[TARGET_COLUMN]
        metrics = regression_metrics(actuals, predictions)
        fold_reports.append(
            {
                "fold": fold_number,
                "train_end": fold.train_end,
                "validation_start": fold.validation_start,
                "validation_end": fold.validation_end,
                "train_open_rows": int(train_open.shape[0]),
                "validation_open_rows": int(validation_open.shape[0]),
                "metrics": metrics,
            }
        )
        all_actuals.append(actuals)
        all_predictions.append(predictions)

    return {
        "model": "LightGBM",
        "validation_strategy": "walk_forward_expanding_window",
        "initial_train_days": initial_train_days,
        "validation_days": validation_days,
        "step_days": step_days,
        "fold_count": len(fold_reports),
        "folds": fold_reports,
        "combined_metrics": regression_metrics(
            pd.concat(all_actuals, ignore_index=True),
            pd.concat(all_predictions, ignore_index=True),
        ),
    }


def evaluate_walk_forward_quantiles(
    train_df: pd.DataFrame,
    store_df: pd.DataFrame,
    quantiles: tuple[float, ...] = (0.1, 0.5, 0.9),
    initial_train_days: int = 365,
    validation_days: int = 42,
    step_days: int = 42,
    n_splits: int = 3,
    model_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate lower, median, and upper LightGBM demand forecasts."""

    if not quantiles or any(not 0.0 < quantile < 1.0 for quantile in quantiles):
        raise ValueError("quantiles must contain values strictly between 0 and 1.")
    if tuple(sorted(set(quantiles))) != quantiles:
        raise ValueError("quantiles must be sorted and unique.")

    folds = make_walk_forward_splits(
        train_df,
        initial_train_days=initial_train_days,
        validation_days=validation_days,
        step_days=step_days,
        n_splits=n_splits,
    )
    fold_reports: list[dict[str, Any]] = []
    all_actuals: list[pd.Series] = []
    all_predictions = {quantile: [] for quantile in quantiles}

    for fold_number, fold in enumerate(folds, start=1):
        train_open, validation_open = make_walk_forward_training_frame(
            fold, store_df
        )
        numeric_features, categorical_features = select_feature_columns(train_open)
        feature_columns = numeric_features + categorical_features
        predictions_by_quantile: dict[float, pd.Series] = {}
        for quantile in quantiles:
            pipeline = make_lightgbm_quantile_pipeline(
                quantile,
                numeric_features,
                categorical_features,
                model_params,
            )
            predictions_by_quantile[quantile] = pd.Series(
                pipeline.fit(
                    train_open[feature_columns], train_open[TARGET_COLUMN]
                ).predict(validation_open[feature_columns]),
                index=validation_open.index,
            ).clip(lower=0)

        actuals = validation_open[TARGET_COLUMN]
        fold_reports.append(
            {
                "fold": fold_number,
                "train_end": fold.train_end,
                "validation_start": fold.validation_start,
                "validation_end": fold.validation_end,
                "train_open_rows": int(train_open.shape[0]),
                "validation_open_rows": int(validation_open.shape[0]),
                "quantile_metrics": {
                    str(quantile): {
                        "pinball_loss": pinball_loss(
                            actuals, predictions_by_quantile[quantile], quantile
                        )
                    }
                    for quantile in quantiles
                },
                "interval_coverage": float(
                    (
                        (actuals >= predictions_by_quantile[quantiles[0]])
                        & (actuals <= predictions_by_quantile[quantiles[-1]])
                    ).mean()
                ),
            }
        )
        all_actuals.append(actuals)
        for quantile in quantiles:
            all_predictions[quantile].append(predictions_by_quantile[quantile])

    combined_actuals = pd.concat(all_actuals, ignore_index=True)
    combined_predictions = {
        quantile: pd.concat(predictions, ignore_index=True)
        for quantile, predictions in all_predictions.items()
    }
    return {
        "model": "LightGBM quantile regression",
        "quantiles": list(quantiles),
        "validation_strategy": "walk_forward_expanding_window",
        "fold_count": len(fold_reports),
        "folds": fold_reports,
        "combined_quantile_metrics": {
            str(quantile): {
                "pinball_loss": pinball_loss(
                    combined_actuals, combined_predictions[quantile], quantile
                )
            }
            for quantile in quantiles
        },
        "combined_interval_coverage": float(
            (
                (combined_actuals >= combined_predictions[quantiles[0]])
                & (combined_actuals <= combined_predictions[quantiles[-1]])
            ).mean()
        ),
    }


def train_and_log_to_mlflow(
    train_df: pd.DataFrame,
    store_df: pd.DataFrame,
    holdout_days: int = 42,
    model_params: dict[str, Any] | None = None,
    experiment_name: str = "Phase 4: First Model",
    run_name: str | None = None,
) -> tuple[dict[str, Any], ModelRun]:
    """Train model and log to MLflow.
    
    Returns:
        - summary report (same as train_and_evaluate_first_model)
        - ModelRun object (with fitted pipeline)
    """
    
    # Set experiment
    mlflow.set_experiment(experiment_name)
    
    # Start a new run
    with mlflow.start_run(run_name=run_name):
        # Log parameters
        params = {
            "loss": "squared_error",
            "learning_rate": 0.08,
            "max_iter": 150,
            "l2_regularization": 0.01,
            "early_stopping": True,
            "random_state": 42,
            "holdout_days": holdout_days,
        }
        if model_params is not None:
            params.update(model_params)
        
        for key, value in params.items():
            mlflow.log_param(key, value)
        
        # Train model
        model_run = fit_first_model(
            train_df,
            store_df,
            holdout_days=holdout_days,
            model_params=model_params,
        )
        
        # Log metrics
        for metric_name, metric_value in model_run.metrics.items():
            mlflow.log_metric(metric_name, metric_value)
        
        # Get summary report
        report = summarize_model_run(model_run, holdout_days=holdout_days)
        
        # Log report as artifact
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(report, f, indent=2, default=str)
            temp_path = f.name
        mlflow.log_artifact(temp_path, artifact_path="summary")
        
    return report, model_run
