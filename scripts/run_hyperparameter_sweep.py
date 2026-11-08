"""
Phase 5: Run multiple experiments with different hyperparameters.

This demonstrates the power of MLflow: systematically trying different
parameters and comparing results without losing any data.
"""

from pathlib import Path
import json

from retail_demand.data.ingestion import RawDataPaths, load_store, load_train
from retail_demand.modeling.training import train_and_log_to_mlflow


def main():
    paths = RawDataPaths(Path("data/raw"))
    train_df = load_train(paths)
    store_df = load_store(paths)

    # Define hyperparameter combinations to try
    experiments = [
        {
            "name": "baseline_default",
            "params": {"learning_rate": 0.08, "max_iter": 150},
            "description": "Default parameters from Phase 4",
        },
        {
            "name": "aggressive_learning",
            "params": {"learning_rate": 0.15, "max_iter": 100},
            "description": "Higher learning rate, fewer iterations",
        },
        {
            "name": "conservative_learning",
            "params": {"learning_rate": 0.04, "max_iter": 200},
            "description": "Lower learning rate, more iterations",
        },
        {
            "name": "high_regularization",
            "params": {"learning_rate": 0.08, "max_iter": 150, "l2_regularization": 0.1},
            "description": "Stronger L2 regularization to prevent overfitting",
        },
    ]

    results = {}

    print("=" * 80)
    print("PHASE 5: HYPERPARAMETER SWEEP")
    print("=" * 80)
    print()

    for exp in experiments:
        print(f"Running: {exp['name']}")
        print(f"  Description: {exp['description']}")
        print(f"  Parameters: {exp['params']}")

        try:
            report, _ = train_and_log_to_mlflow(
                train_df,
                store_df,
                holdout_days=42,
                model_params=exp["params"],
                experiment_name="Phase 4: First Model",
                run_name=exp["name"],
            )

            metrics = report["metrics"]
            results[exp["name"]] = metrics

            print(f"  ✓ Completed")
            print(f"    MAE:  {metrics['mae']:.2f}")
            print(f"    WAPE: {metrics['wape']:.4f}")
            print()

        except Exception as e:
            print(f"  ✗ Failed: {e}")
            print()

    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()
    print("All results are now logged in MLflow.")
    print("Visit http://127.0.0.1:5000 to compare runs.")
    print()
    print("Results by WAPE (lower is better):")
    print()

    sorted_results = sorted(results.items(), key=lambda x: x[1]["wape"])
    for rank, (name, metrics) in enumerate(sorted_results, 1):
        print(f"{rank}. {name:25s} - WAPE: {metrics['wape']:.4f}, MAE: {metrics['mae']:.2f}")

    print()


if __name__ == "__main__":
    main()
