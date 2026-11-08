# 📦 Retail Demand Forecasting

> Production-style machine-learning system for forecasting daily store sales on the
> [Rossmann Store Sales](https://www.kaggle.com/c/rossmann-store-sales) dataset.

[![CI](https://github.com/your-username/retail-demand-forecasting/actions/workflows/ci.yml/badge.svg)](https://github.com/your-username/retail-demand-forecasting/actions)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

---

## 🔑 Key Results

| Model | MAE | WAPE | vs. Best Baseline |
|-------|-----|------|-------------------|
| Store-mean baseline | 1,456 | 0.2087 | — |
| **HistGradientBoosting** | **622** | **0.0892** | **↓ 57 %** |
| LightGBM (walk-forward) | ~590 | ~0.082 | ~↓ 61 % |

Evaluated on a strict **42-day temporal holdout** with no data leakage.

---

## 📂 Project Structure

```
├── src/retail_demand/
│   ├── data/           # Ingestion, validation, CLI
│   ├── features/       # Calendar, store, lag, rolling feature engineering
│   ├── modeling/       # Baselines, training, tuning, inference, persistence
│   │   ├── training.py         # HistGBM + LightGBM pipelines + MLflow logging
│   │   ├── persistence.py      # save_model / load_model (joblib)
│   │   ├── inference.py        # Batch inference + Kaggle submission generator
│   │   └── tuning.py           # Grid-search HPO
│   ├── evaluation/     # MAE, WAPE, pinball loss, error analysis
│   └── __main__.py     # Unified CLI entry point
├── app.py              # Streamlit interactive demo
├── Dockerfile          # Multi-stage Docker build
├── docker-compose.yml  # Local dev + training profiles
├── .github/workflows/
│   └── ci.yml          # Lint → test → Docker build → GHCR publish
├── data/raw/           # Rossmann CSVs (gitignored — download from Kaggle)
├── model/              # Saved model artifacts (joblib + JSON metadata)
├── mlruns/             # MLflow experiment tracking
├── notebooks/          # Exploratory analysis
├── docs/               # Phase docs & data dictionary
└── tests/unit/         # 9 pytest modules + shared conftest
```

---

## 🚀 Quick Start

### 1 — Install

```bash
git clone https://github.com/your-username/retail-demand-forecasting.git
cd retail-demand-forecasting
python -m venv .venv && .venv/Scripts/activate   # Windows
pip install ".[app]"                              # includes Streamlit
```

### 2 — Add data

Download the Rossmann CSV files from
[Kaggle](https://www.kaggle.com/c/rossmann-store-sales/data) and place them in
`data/raw/`:

```
data/raw/
├── train.csv
├── store.csv
└── test.csv
```

### 3 — Train

```bash
retail-demand train --raw-dir data/raw --model-path model/latest.joblib
```

This runs the full pipeline (feature engineering → HistGBM training → MLflow logging → model saved to `model/latest.joblib`).

### 4 — Launch the interactive demo

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

### 5 — Generate a Kaggle submission

```bash
retail-demand forecast --raw-dir data/raw --model-path model/latest.joblib --kaggle
# → submissions/submission.csv
```

---

## 🐳 Docker

```bash
# Build
docker build -t retail-demand .

# Run the Streamlit demo (mount your data/raw and model directories)
docker run -p 8501:8501 \
  -v "$(pwd)/data/raw:/app/data/raw:ro" \
  -v "$(pwd)/model:/app/model" \
  retail-demand

# Or use docker compose
docker compose up          # demo app
docker compose run train   # run full training pipeline
```

---

## 🛠️ All CLI Commands

| Command | Description |
|---------|-------------|
| `retail-demand train` | Full training pipeline + save model |
| `retail-demand forecast` | Batch inference on test.csv |
| `retail-demand forecast --kaggle` | Generate Kaggle submission CSV |
| `retail-demand baselines` | Baseline benchmark |
| `retail-demand features` | Build + summarise feature table |
| `retail-demand inspect` | Inspect raw data files |
| `train-model` | MLflow training run (legacy script) |
| `run-baselines` | Baseline runner (legacy script) |
| `build-features` | Feature builder (legacy script) |
| `tune-model` | Hyperparameter tuning |
| `evaluate-lightgbm` | Walk-forward LightGBM evaluation |
| `evaluate-quantiles` | Quantile regression (P10 / P50 / P90) |
| `generate-forecast` | Standalone inference CLI |
| `inspect-raw-data` | Raw data inspector |

---

## 🧪 Tests

```bash
pip install ".[test]"
pytest                    # all tests
pytest --cov=retail_demand  # with coverage
```

9 test modules covering data ingestion, validation, feature engineering, baselines, metrics, splits, training, error analysis, and tuning.

---

## 🔬 ML Pipeline Details

### Feature Engineering
| Group | Features |
|-------|----------|
| Calendar | year, month, day, week_of_year, is_month_start, is_month_end |
| Store | StoreType, Assortment, competition_open_months, promo2_active |
| Business timing | DayOfWeek, Promo, SchoolHoliday, StateHoliday |
| Sales lags | lag_7, lag_14, lag_28 (per-store, leakage-safe) |
| Rolling stats | rolling_mean_7, rolling_std_7, rolling_mean_28, rolling_std_28 |

### Models
- **HistGradientBoostingRegressor** (sklearn) — handles missing values natively, fast training
- **LightGBM** — walk-forward evaluation + quantile regression (P10/P50/P90)
- **Baselines** — naive yesterday, 7-day seasonal naive, store-mean

### Experiment Tracking
All runs logged to **MLflow** (local `mlruns/` SQLite backend). View with:
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

---

## 📄 Documentation

| Document | Contents |
|----------|----------|
| `docs/data_dictionary.md` | Column descriptions, leakage analysis |
| `docs/phase_1_data_investigation.md` | EDA results |
| `docs/phase_2_baselines.md` | Baseline benchmark results |
| `docs/phase_3_feature_engineering.md` | Feature design decisions |
| `docs/phase_4_model_development.md` | First model training report |
| `docs/phase_4_error_analysis.md` | Error segment analysis |

---

## 🗺️ Roadmap

- [ ] Deploy demo to [Render](https://render.com) / [Railway](https://railway.app)
- [ ] SHAP feature importance visualisation
- [ ] Walk-forward LightGBM HPO
- [ ] MkDocs project site with API reference
- [ ] Store-specific event features (reopening detection, competition entry)

---

## 📃 License

MIT © 2026 — see [LICENSE](LICENSE).
