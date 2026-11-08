# Phase 4 Error Analysis

After the first model beat the baseline, we inspected where the model performs
better and worse.

## Overall Holdout Metrics

| Metric | Value |
|---|---:|
| MAE | 622.46 |
| WAPE | 0.0892 |

## Segment Findings

Worst WAPE by selected segments:

| Segment | Worst Group | Rows | WAPE |
|---|---|---:|---:|
| `Promo` | `0` | 23,581 | 0.0907 |
| `DayOfWeek` | `6` | 6,682 | 0.1015 |
| `SchoolHoliday` | `0` | 26,978 | 0.0913 |
| `StoreType` | `c` | 5,328 | 0.0898 |
| `Assortment` | `a` | 21,359 | 0.0898 |

The segment metrics do not show a severe broad failure in the selected groups.
The bigger issue appears to be individual high-error store/date events.

## Largest Individual Errors

The largest errors include repeated appearances by stores `909` and `876`.

Examples:

| Store | Date | Actual Sales | Prediction | Absolute Error |
|---:|---|---:|---:|---:|
| 909 | 2015-06-22 | 41,551 | 10,975.01 | 30,575.99 |
| 292 | 2015-06-29 | 29,161 | 9,004.65 | 20,156.35 |
| 876 | 2015-07-06 | 27,330 | 8,708.26 | 18,621.74 |

## Interpretation

The first model is strong on average, but a production system cannot stop at the
overall WAPE.

The next modeling work should investigate whether these large errors are caused
by:

- unusual store-specific events,
- promotion schedule effects,
- missing or weak holiday/event features,
- store closures or reopening behavior,
- or target-history features that do not capture sudden demand changes.

## Commands

Run the error analysis:

```powershell
.\.venv\Scripts\analyze-model-errors.exe
```

