# Phase 1 Data Investigation

## Current Findings

- `train.csv` has 1,017,209 rows and 9 columns.
- `store.csv` has 1,115 rows and 10 columns.
- The training date range is 2013-01-01 to 2015-07-31.
- `test.csv` covers 2015-08-01 to 2015-09-17.
- `train.csv` contains 1,115 unique stores.
- `store.csv` contains 1,115 unique stores.
- There are no duplicate rows in `train.csv` or `store.csv`.

## Important Early ML Decisions

- The regression target is `Sales`.
- Initial training should focus on `Open == 1` days.
- Closed-store rows are structurally different from normal demand rows.
- `Customers` should not be used as a forecasting feature because the number of
  customers is only known after the business day happens.
- Kaggle `test.csv` is useful for future submission-style inference, but not for
  honest local evaluation because it does not contain `Sales`.

## Missing Values

`train.csv` has no missing values.

`store.csv` has missing values in competition and Promo2-related fields. This is
expected because not every store has an active known competitor opening date or
continuing promotion.

`test.csv` has missing values in `Open`. We will handle this later during batch
inference.

## Next Phase 1 Tasks

- Add lightweight validation rules for raw data.
- Decide the first local temporal holdout window.
- Build baseline forecasts before machine learning.

