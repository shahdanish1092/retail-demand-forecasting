# Phase 3 Feature Engineering

This phase creates the first leakage-safe feature table for model development.

## Feature Groups Added

| Group | Columns | Available at Prediction Time? |
|---|---|---|
| Calendar | `year`, `month`, `day`, `week_of_year`, `is_month_start`, `is_month_end` | Yes |
| Store attributes | `StoreType`, `Assortment`, competition fields, Promo2 fields | Yes, assuming store master data is current |
| Business timing | `competition_open_months`, `promo2_active` | Yes, derived from date and store metadata |
| Sales lags | `sales_lag_7`, `sales_lag_14`, `sales_lag_28` | Yes, if the lagged actual sales date has already passed |
| Rolling sales | `sales_rolling_mean_7`, `sales_rolling_std_7`, `sales_rolling_mean_28`, `sales_rolling_std_28` | Yes, because rolling windows are shifted before calculation |

## Leakage Rule

Target-derived features must only use past sales.

The rolling features use:

```text
group by Store
shift Sales by 1 day
then calculate rolling statistics
```

This prevents the current row's `Sales` value from appearing inside its own
features.

## Feature Table Summary

- Rows: 1,017,209
- Columns after feature engineering: 33
- Engineered feature columns: 15

Expected missing values:

| Column | Missing Rows | Why |
|---|---:|---|
| `sales_lag_7` | 7,805 | First 7 rows per store have no 7-day history |
| `sales_lag_14` | 15,610 | First 14 rows per store have no 14-day history |
| `sales_lag_28` | 31,220 | First 28 rows per store have no 28-day history |
| `sales_rolling_mean_7` | 1,115 | First row per store has no previous sales |
| `sales_rolling_std_7` | 2,230 | Rolling std needs at least two prior values |
| `sales_rolling_mean_28` | 1,115 | First row per store has no previous sales |
| `sales_rolling_std_28` | 2,230 | Rolling std needs at least two prior values |

These missing values are expected and are a consequence of leakage-safe feature
generation.

## Important Design Decisions

- `Customers` remains excluded from forecasting features because it is not known
  before the sales day finishes.
- Calendar and store features are safe because they are known before prediction.
- Lag and rolling features are safe only when computed from earlier observations.
- Kaggle `test.csv` still requires special handling later because it does not
  contain future `Sales`, so target-derived features for multi-day horizons cannot
  naively depend on unknown future actuals.

## Next Step

Use this feature table to train the first scikit-learn model and compare it
against the Phase 2 baseline WAPE of 0.2087.

