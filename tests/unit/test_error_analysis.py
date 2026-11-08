import pandas as pd
import pytest

from retail_demand.evaluation.error_analysis import (
    segment_error_report,
    top_absolute_errors,
)


def test_segment_error_report_orders_segments_by_wape_descending() -> None:
    prediction_df = pd.DataFrame(
        {
            "Promo": [0, 0, 1, 1],
            "Sales": [100, 100, 100, 100],
            "prediction": [100, 90, 50, 50],
        }
    )

    report = segment_error_report(prediction_df, "Promo")

    assert report[0]["segment"] == "1"
    assert report[0]["wape"] == pytest.approx(0.5)
    assert report[1]["segment"] == "0"
    assert report[1]["wape"] == pytest.approx(0.05)


def test_top_absolute_errors_returns_largest_errors_first() -> None:
    prediction_df = pd.DataFrame(
        {
            "Store": [1, 2],
            "Date": pd.to_datetime(["2020-01-01", "2020-01-01"]),
            "Sales": [100, 500],
            "prediction": [90, 200],
        }
    )

    report = top_absolute_errors(prediction_df, n=1)

    assert report[0]["Store"] == 2
    assert report[0]["absolute_error"] == 300

