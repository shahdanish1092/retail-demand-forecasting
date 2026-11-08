# Phase 2 Baselines

This phase establishes simple forecasting benchmarks before training any machine
learning model.

## Local Temporal Holdout

We use the final 42 calendar days of `train.csv` as the first local holdout.

- Holdout cutoff date: 2015-06-20
- Training rows before cutoff: 970,379
- Holdout rows: 46,830
- Open holdout rows evaluated: 40,282

The regression evaluation focuses on `Open == 1` rows because closed-store sales
are structural zero-demand rows, not normal demand observations.

## Baselines

| Baseline | Meaning |
|---|---|
| `pred_naive_yesterday` | Same store's sales from 1 calendar day earlier |
| `pred_seasonal_naive_7` | Same store's sales from 7 calendar days earlier |
| `pred_store_mean` | Same store's average open-day sales from the training period |

## Results

| Baseline | Rows Evaluated | MAE | WAPE |
|---|---:|---:|---:|
| `pred_naive_yesterday` | 40,282 | 2,159.73 | 0.3095 |
| `pred_seasonal_naive_7` | 40,282 | 2,191.87 | 0.3141 |
| `pred_store_mean` | 40,282 | 1,456.44 | 0.2087 |

Lower MAE and WAPE are better.

## Interpretation

The best initial baseline is store historical mean with WAPE of 0.2087.

This gives the first model promotion target: an ML model must beat this baseline,
not merely train successfully.

## Evaluation Assumption

The lag baselines are evaluated as retrospective daily forecasts. For a validation
date, they may use sales from earlier dates, but not from the same date or future
dates.

This is acceptable as an initial benchmark. Later, when we implement true
multi-day batch forecasting, we must distinguish this from forecasting an entire
six-week horizon at once from a single cutoff date.

