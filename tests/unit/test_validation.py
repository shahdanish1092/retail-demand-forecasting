import pandas as pd

from retail_demand.data.validation import (
    validate_store_data,
    validate_train_data,
    validate_train_store_relationship,
)


def valid_train_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Store": [1, 1],
            "DayOfWeek": [1, 2],
            "Date": pd.to_datetime(["2015-01-01", "2015-01-02"]),
            "Sales": [100, 120],
            "Customers": [10, 12],
            "Open": [1, 1],
            "Promo": [0, 1],
            "StateHoliday": ["0", "0"],
            "SchoolHoliday": [0, 0],
        }
    )


def valid_store_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Store": [1],
            "StoreType": ["a"],
            "Assortment": ["c"],
            "CompetitionDistance": [100.0],
            "CompetitionOpenSinceMonth": [1.0],
            "CompetitionOpenSinceYear": [2014.0],
            "Promo2": [0],
            "Promo2SinceWeek": [None],
            "Promo2SinceYear": [None],
            "PromoInterval": [None],
        }
    )


def test_validate_train_data_accepts_valid_minimal_data() -> None:
    assert validate_train_data(valid_train_df()) == []


def test_validate_train_data_rejects_duplicate_store_date() -> None:
    df = valid_train_df()
    df.loc[1, "Date"] = df.loc[0, "Date"]

    issues = validate_train_data(df)

    assert any(issue.check == "train_store_date_unique" for issue in issues)


def test_validate_store_data_accepts_valid_minimal_data() -> None:
    assert validate_store_data(valid_store_df()) == []


def test_validate_train_store_relationship_rejects_missing_store_lookup() -> None:
    train_df = valid_train_df()
    store_df = valid_store_df()
    store_df.loc[0, "Store"] = 2

    issues = validate_train_store_relationship(train_df, store_df)

    assert any(issue.check == "train_stores_exist_in_store" for issue in issues)

