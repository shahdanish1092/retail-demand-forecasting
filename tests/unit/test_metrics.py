import pandas as pd
import pytest

from retail_demand.evaluation.metrics import mae, pinball_loss, wape


def test_mae_returns_average_absolute_error() -> None:
    y_true = pd.Series([100, 200, 300])
    y_pred = pd.Series([90, 220, 330])

    assert mae(y_true, y_pred) == pytest.approx(20.0)


def test_wape_returns_total_absolute_error_over_total_actuals() -> None:
    y_true = pd.Series([100, 200, 300])
    y_pred = pd.Series([90, 220, 330])

    assert wape(y_true, y_pred) == pytest.approx(60 / 600)


def test_wape_rejects_zero_denominator() -> None:
    with pytest.raises(ValueError, match="undefined"):
        wape(pd.Series([0, 0]), pd.Series([1, 2]))


def test_pinball_loss_penalizes_under_and_over_prediction_asymmetrically() -> None:
    assert pinball_loss(pd.Series([10]), pd.Series([8]), 0.9) == pytest.approx(1.8)
    assert pinball_loss(pd.Series([10]), pd.Series([12]), 0.9) == pytest.approx(0.2)

