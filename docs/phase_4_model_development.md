# Phase 4 Model Development

This phase trains the first machine learning model and compares it against the
Phase 2 baselines.

## Model

Initial model:

```text
sklearn.ensemble.HistGradientBoostingRegressor
```

Why this model:

- Handles nonlinear relationships.
- Handles interactions between promotions, calendar effects, store attributes,
  and recent sales history.
- Trains faster than many tree ensembles on medium-sized tabular data.
- Is available directly in scikit-learn.

## Pipeline

The model is wrapped in a sklearn `Pipeline`.

```text
raw train.csv + store.csv
  -> leakage-safe feature table
  -> final 42-day temporal holdout
  -> Open == 1 filter
  -> ColumnTransformer preprocessing
  -> HistGradientBoostingRegressor
  -> holdout predictions
  -> MAE and WAPE
```

Numeric preprocessing:

- Median imputation.

Categorical preprocessing:

- Most-frequent imputation.
- Ordinal encoding with unknown-category handling.

## Features Used

Numeric:

- `DayOfWeek`
- `Promo`
- `SchoolHoliday`
- `CompetitionDistance`
- `Promo2`
- `year`
- `month`
- `day`
- `week_of_year`
- `is_month_start`
- `is_month_end`
- `competition_open_months`
- `promo2_active`
- `sales_lag_7`
- `sales_lag_14`
- `sales_lag_28`
- `sales_rolling_mean_7`
- `sales_rolling_std_7`
- `sales_rolling_mean_28`
- `sales_rolling_std_28`

Categorical:

- `Store`
- `StateHoliday`
- `StoreType`
- `Assortment`

Excluded:

- `Sales`: target, not a feature.
- `Customers`: leakage risk.
- `Date`: raw timestamp excluded after calendar features are extracted.
- `Open`: filtered to open days for the first regression model.

## Result

Final 42-day temporal holdout:

- Train open rows: 804,110
- Holdout open rows: 40,282

| Model | MAE | WAPE |
|---|---:|---:|
| Store historical mean baseline | 1,456.44 | 0.2087 |
| HistGradientBoostingRegressor | 622.46 | 0.0892 |

The first ML model beats the current best baseline.

## Important Caveat

This is not a final production model yet.

The result is promising, but we still need:

- broader validation,
- error analysis,
- hyperparameter tuning,
- uncertainty estimates,
- experiment tracking,
- model persistence,
- batch prediction behavior,
- and later monitoring.

## Notebook Timing

We started using scikit-learn in Phase 4 source code first.

The notebook should come after this first reproducible training path exists. The
notebook's job will be explanation and controlled experimentation, not becoming
the only place where training works.

