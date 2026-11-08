import pandas as pd

from retail_demand.modeling.training import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    evaluate_walk_forward_model,
    select_feature_columns,
)


def test_select_feature_columns_excludes_target_and_customers() -> None:
    df = pd.DataFrame(
        columns=[
            "Sales",
            "Customers",
            "Date",
            "Open",
            *NUMERIC_FEATURES,
            *CATEGORICAL_FEATURES,
        ]
    )

    numeric_features, categorical_features = select_feature_columns(df)
    selected_features = set(numeric_features + categorical_features)

    assert "Sales" not in selected_features
    assert "Customers" not in selected_features
    assert "Date" not in selected_features
    assert "Open" not in selected_features
    assert "Promo" in selected_features
    assert "StoreType" in selected_features


def test_evaluate_walk_forward_model_returns_fold_and_combined_metrics() -> None:
    dates = pd.date_range("2020-01-01", periods=50, freq="D")
    rows = []
    for store in [1, 2]:
        for offset, date in enumerate(dates):
            rows.append(
                {
                    "Store": store,
                    "Date": date,
                    "Sales": 100 + store * 10 + offset,
                    "Open": 1,
                    "Promo": offset % 2,
                    "SchoolHoliday": 0,
                    "StateHoliday": "0",
                    "StoreType": "a",
                    "Assortment": "a",
                }
            )
    train_df = pd.DataFrame(rows)
    store_df = pd.DataFrame(
        {
            "Store": [1, 2],
            "CompetitionOpenSinceMonth": [1, 1],
            "CompetitionOpenSinceYear": [2019, 2019],
            "Promo2": [0, 0],
            "Promo2SinceWeek": [pd.NA, pd.NA],
            "Promo2SinceYear": [pd.NA, pd.NA],
            "PromoInterval": [pd.NA, pd.NA],
        }
    )

    report = evaluate_walk_forward_model(
        train_df,
        store_df,
        initial_train_days=30,
        validation_days=5,
        step_days=5,
        n_splits=2,
    )

    assert report["fold_count"] == 2
    assert len(report["folds"]) == 2
    assert set(report["combined_metrics"]) == {"mae", "wape"}
