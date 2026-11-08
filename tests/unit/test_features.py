import pandas as pd

from retail_demand.features.build import (
    add_calendar_features,
    add_lag_features,
    add_rolling_features,
    add_store_features,
    build_prediction_features,
)


def test_add_calendar_features_uses_date_parts() -> None:
    df = pd.DataFrame({"Date": pd.to_datetime(["2020-01-31"])})

    result = add_calendar_features(df)

    assert result.loc[0, "year"] == 2020
    assert result.loc[0, "month"] == 1
    assert result.loc[0, "day"] == 31
    assert result.loc[0, "is_month_end"] == 1


def test_add_store_features_joins_store_attributes_and_business_features() -> None:
    df = pd.DataFrame(
        {
            "Store": [1],
            "Date": pd.to_datetime(["2015-03-15"]),
        }
    )
    store_df = pd.DataFrame(
        {
            "Store": [1],
            "StoreType": ["a"],
            "Assortment": ["c"],
            "CompetitionDistance": [100.0],
            "CompetitionOpenSinceMonth": [1.0],
            "CompetitionOpenSinceYear": [2015.0],
            "Promo2": [1],
            "Promo2SinceWeek": [1.0],
            "Promo2SinceYear": [2015.0],
            "PromoInterval": ["Jan,Mar,May"],
        }
    )

    result = add_store_features(df, store_df)

    assert result.loc[0, "StoreType"] == "a"
    assert result.loc[0, "competition_open_months"] == 2
    assert result.loc[0, "promo2_active"] == 1


def test_add_lag_features_uses_same_store_past_sales_only() -> None:
    df = pd.DataFrame(
        {
            "Store": [1, 1, 1, 2],
            "Date": pd.to_datetime(
                ["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-01"]
            ),
            "Sales": [100, 120, 140, 999],
        }
    )

    result = add_lag_features(df, lags=(1,))

    store_1_result = result.loc[result["Store"].eq(1)].reset_index(drop=True)
    assert pd.isna(store_1_result.loc[0, "sales_lag_1"])
    assert store_1_result.loc[1, "sales_lag_1"] == 100
    assert store_1_result.loc[2, "sales_lag_1"] == 120


def test_add_rolling_features_shift_before_rolling_to_prevent_leakage() -> None:
    df = pd.DataFrame(
        {
            "Store": [1, 1, 1],
            "Date": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"]),
            "Sales": [100, 120, 1000],
        }
    )

    result = add_rolling_features(df, windows=(2,))

    assert pd.isna(result.loc[0, "sales_rolling_mean_2"])
    assert result.loc[1, "sales_rolling_mean_2"] == 100
    assert result.loc[2, "sales_rolling_mean_2"] == 110


def test_build_prediction_features_ignores_future_target_labels() -> None:
    history = pd.DataFrame(
        {
            "Store": [1, 1, 1],
            "Date": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"]),
            "Sales": [100, 120, 140],
            "Open": [1, 1, 1],
        }
    )
    target = pd.DataFrame(
        {
            "Store": [1, 1],
            "Date": pd.to_datetime(["2020-01-04", "2020-01-05"]),
            "Sales": [160, 999999],
            "Open": [1, 1],
        }
    )
    store = pd.DataFrame({"Store": [1]})

    first = build_prediction_features(
        target, history, store, lags=(1,), windows=(2,)
    )
    changed_target = target.copy()
    changed_target.loc[1, "Sales"] = -123456
    second = build_prediction_features(
        changed_target, history, store, lags=(1,), windows=(2,)
    )

    assert first["sales_lag_1"].equals(second["sales_lag_1"])
    assert first["sales_rolling_mean_2"].equals(
        second["sales_rolling_mean_2"]
    )
    assert first.loc[1, "sales_rolling_mean_2"] == 130
