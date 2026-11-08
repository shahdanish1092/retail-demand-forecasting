import pandas as pd

from retail_demand.modeling.baselines import (
    add_calendar_lag_baseline,
    add_store_mean_baseline,
    run_walk_forward_baseline_experiment,
)


def test_add_calendar_lag_baseline_uses_same_store_prior_calendar_day() -> None:
    history = pd.DataFrame(
        {
            "Store": [1, 1, 2],
            "Date": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-01"]),
            "Sales": [100, 120, 200],
        }
    )
    target_rows = pd.DataFrame(
        {
            "Store": [1, 2],
            "Date": pd.to_datetime(["2020-01-02", "2020-01-02"]),
        }
    )

    result = add_calendar_lag_baseline(
        target_rows,
        history=history,
        lag_days=1,
        prediction_column="pred",
    )

    assert result["pred"].tolist() == [100, 200]


def test_add_store_mean_baseline_uses_training_rows_only() -> None:
    training_rows = pd.DataFrame(
        {
            "Store": [1, 1, 2],
            "Sales": [100, 140, 300],
        }
    )
    target_rows = pd.DataFrame({"Store": [1, 2]})

    result = add_store_mean_baseline(target_rows, training_rows, "pred")

    assert result["pred"].tolist() == [120, 300]


def test_walk_forward_baselines_use_only_fold_history() -> None:
    dates = pd.date_range("2020-01-01", periods=30, freq="D")
    train_df = pd.DataFrame(
        {
            "Store": [1] * len(dates),
            "Date": dates,
            "Sales": range(100, 100 + len(dates)),
            "Open": [1] * len(dates),
        }
    )

    report = run_walk_forward_baseline_experiment(
        train_df,
        initial_train_days=14,
        validation_days=5,
        step_days=5,
        n_splits=2,
    )

    assert report["fold_count"] == 2
    assert report["folds"][0]["baselines"]["pred_store_mean"]["rows_evaluated"] == 5
    assert report["combined_baselines"]["pred_store_mean"]["rows_evaluated"] == 10
