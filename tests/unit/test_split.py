import pandas as pd

from retail_demand.modeling.split import (
    make_final_days_holdout,
    make_walk_forward_splits,
)


def test_make_final_days_holdout_uses_final_calendar_days() -> None:
    df = pd.DataFrame(
        {
            "Date": pd.to_datetime(
                ["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04"]
            ),
            "Sales": [10, 20, 30, 40],
        }
    )

    split = make_final_days_holdout(df, holdout_days=2)

    assert split.cutoff_date == pd.Timestamp("2020-01-03")
    assert split.train["Date"].tolist() == [
        pd.Timestamp("2020-01-01"),
        pd.Timestamp("2020-01-02"),
    ]
    assert split.holdout["Date"].tolist() == [
        pd.Timestamp("2020-01-03"),
        pd.Timestamp("2020-01-04"),
    ]


def test_make_walk_forward_splits_keeps_validation_after_training() -> None:
    dates = pd.date_range("2020-01-01", periods=12, freq="D")
    df = pd.DataFrame({"Date": dates, "Sales": range(12)})

    folds = make_walk_forward_splits(
        df,
        initial_train_days=6,
        validation_days=2,
        step_days=2,
        n_splits=2,
    )

    assert len(folds) == 2
    assert folds[0].train["Date"].max() < folds[0].validation["Date"].min()
    assert folds[1].train["Date"].max() < folds[1].validation["Date"].min()
    assert folds[1].train.shape[0] > folds[0].train.shape[0]
