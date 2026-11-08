import pandas as pd

from retail_demand.modeling.tuning import evaluate_parameter_candidate


def test_evaluate_parameter_candidate_returns_metrics() -> None:
    train_open = pd.DataFrame(
        {
            "Store": [1, 1, 1, 1],
            "Promo": [0, 1, 0, 1],
            "Sales": [100, 120, 110, 130],
        }
    )
    holdout_open = pd.DataFrame(
        {
            "Store": [1, 1],
            "Promo": [0, 1],
            "Sales": [115, 135],
        }
    )

    result = evaluate_parameter_candidate(
        train_open=train_open,
        holdout_open=holdout_open,
        numeric_features=["Promo"],
        categorical_features=["Store"],
        model_params={
            "max_iter": 5,
            "learning_rate": 0.1,
            "early_stopping": False,
            "min_samples_leaf": 1,
        },
    )

    assert result["params"]["max_iter"] == 5
    assert set(result["metrics"]) == {"mae", "wape"}

