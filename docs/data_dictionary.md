# Data Dictionary

This document records what each Rossmann source file represents and how we
should think about its columns for forecasting.

## `train.csv`

Historical daily store-level observations from 2013-01-01 to 2015-07-31.

| Column | Meaning | Initial Use Decision |
|---|---|---|
| `Store` | Store identifier | Feature and join key |
| `DayOfWeek` | Day of week for the row date | Feature |
| `Date` | Observation date | Time index and feature source |
| `Sales` | Daily turnover | Target |
| `Customers` | Number of customers that day | Leakage risk, exclude from forecasting features |
| `Open` | Whether the store was open | Filter/control feature |
| `Promo` | Whether a promotion was active | Feature if known ahead of forecast date |
| `StateHoliday` | Public holiday indicator | Feature |
| `SchoolHoliday` | School holiday indicator | Feature |

## `store.csv`

Static or slowly changing store attributes, one row per store.

| Column | Meaning | Initial Use Decision |
|---|---|---|
| `Store` | Store identifier | Join key |
| `StoreType` | Store format category | Feature |
| `Assortment` | Assortment level | Feature |
| `CompetitionDistance` | Distance to nearest competitor | Feature with missing-value handling |
| `CompetitionOpenSinceMonth` | Month competitor opened | Feature engineering candidate |
| `CompetitionOpenSinceYear` | Year competitor opened | Feature engineering candidate |
| `Promo2` | Whether continuing promotion is active | Feature |
| `Promo2SinceWeek` | Week continuing promotion started | Feature engineering candidate |
| `Promo2SinceYear` | Year continuing promotion started | Feature engineering candidate |
| `PromoInterval` | Months where Promo2 is active | Feature engineering candidate |

## `test.csv`

Future rows for Kaggle scoring from 2015-08-01 to 2015-09-17. It does not
contain `Sales`.

We will not use Kaggle `test.csv` as our first model test set. For honest local
evaluation, we will create a temporal holdout from `train.csv` where `Sales` is
known.

