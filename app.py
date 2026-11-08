"""Streamlit demo app for Retail Demand Forecasting.

Launch with:
    streamlit run app.py

Or inside Docker:
    docker run -p 8501:8501 retail-demand
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Retail Demand Forecasting",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Paths ────────────────────────────────────────────────────────────────────
RAW_DIR = Path("data/raw")
SAMPLE_DIR = Path("data/sample")
MODEL_PATH = Path("model/latest.joblib")

# ─── Helpers ──────────────────────────────────────────────────────────────────


@st.cache_resource(show_spinner="Loading model…")
def _load_pipeline():
    from retail_demand.modeling.persistence import load_model

    return load_model(MODEL_PATH)


@st.cache_resource(show_spinner="Loading store metadata…")
def _load_store():
    from retail_demand.data.ingestion import RawDataPaths, load_store

    if (RAW_DIR / "store.csv").exists():
        return load_store(RawDataPaths(RAW_DIR))
    elif (SAMPLE_DIR / "store.csv").exists():
        return pd.read_csv(SAMPLE_DIR / "store.csv")
    raise FileNotFoundError("Store metadata not found in data/raw or data/sample")


@st.cache_data(show_spinner="Loading history for lag/rolling features…")
def _load_history():
    from retail_demand.data.ingestion import RawDataPaths, load_train

    if (RAW_DIR / "train.csv").exists():
        return load_train(RawDataPaths(RAW_DIR))
    elif (SAMPLE_DIR / "train_sample.csv").exists():
        return pd.read_csv(SAMPLE_DIR / "train_sample.csv", parse_dates=["Date"])
    raise FileNotFoundError("Historical sales data not found in data/raw or data/sample")


def _model_available() -> bool:
    return MODEL_PATH.exists()


def _data_available() -> bool:
    has_raw = (RAW_DIR / "train.csv").exists() and (RAW_DIR / "store.csv").exists()
    has_sample = (SAMPLE_DIR / "train_sample.csv").exists() and (SAMPLE_DIR / "store.csv").exists()
    return has_raw or has_sample


# ─── Sidebar ──────────────────────────────────────────────────────────────────
st.sidebar.title("📦 Retail Demand Forecast")
st.sidebar.markdown(
    """
**Project:** Rossmann Store Sales  
**Model:** HistGradientBoostingRegressor + LightGBM  
**Holdout WAPE:** `0.0892`  
**Baseline WAPE:** `0.2087`  

---
[GitHub Repository](https://github.com/shahdanish1092/retail-demand-forecasting)
"""
)

st.sidebar.markdown("### How to use")
st.sidebar.markdown(
    """
1. Upload a CSV with store rows **or** use the sample data.
2. Click **Run Forecast**.
3. Download the predictions.
"""
)

# ─── Main ─────────────────────────────────────────────────────────────────────
st.title("📦 Retail Store Demand Forecasting")
st.markdown(
    """
This demo uses a **HistGradientBoostingRegressor** pipeline trained on the
[Rossmann Store Sales](https://www.kaggle.com/c/rossmann-store-sales) dataset
to forecast daily store sales.

> **WAPE on 42-day holdout: 0.0892** — a **57% reduction** over the best naive baseline (0.2087).
"""
)

tabs = st.tabs(["🔮 Forecast", "📊 Model Performance", "ℹ️ About"])

# ─── Forecast tab ─────────────────────────────────────────────────────────────
with tabs[0]:
    st.subheader("Run a Forecast")

    col_left, col_right = st.columns([1, 1])

    with col_left:
        source = st.radio(
            "Data source",
            options=["Sample data (test.csv)", "Upload your own CSV"],
            horizontal=True,
        )

    target_df: pd.DataFrame | None = None

    if source == "Sample data (test.csv)":
        if (RAW_DIR / "test.csv").exists():
            test_path = RAW_DIR / "test.csv"
        elif (SAMPLE_DIR / "test_sample.csv").exists():
            test_path = SAMPLE_DIR / "test_sample.csv"
        else:
            test_path = None

        if test_path is not None:
            target_df = pd.read_csv(test_path, parse_dates=["Date"])
            target_df["Open"] = target_df["Open"].fillna(1).astype(int)
            st.info(f"Loaded **{len(target_df):,}** sample rows from `{test_path.name}`.")
        else:
            st.error(
                "Sample test data not found. Upload your own CSV below."
            )
    else:
        uploaded = st.file_uploader(
            "Upload CSV",
            type="csv",
            help="Must contain at least: Store, Date, Open columns.",
        )
        if uploaded is not None:
            try:
                target_df = pd.read_csv(uploaded, parse_dates=["Date"])
                target_df["Open"] = target_df["Open"].fillna(1).astype(int)
                st.success(f"Loaded **{len(target_df):,}** rows.")
            except Exception as exc:
                st.error(f"Could not read CSV: {exc}")

    # ── Run forecast ──────────────────────────────────────────────────────
    if target_df is not None:
        with col_right:
            st.markdown(f"**Rows loaded:** `{len(target_df):,}`")
            st.markdown(
                f"**Date range:** `{target_df['Date'].min().date()}` → "
                f"`{target_df['Date'].max().date()}`"
            )
            st.markdown(f"**Stores:** `{target_df['Store'].nunique()}`")

        st.divider()

        run_btn = st.button("🚀 Run Forecast", type="primary", use_container_width=True)

        if run_btn:
            if not _model_available():
                st.error(
                    f"Model file not found at `{MODEL_PATH}`. "
                    "Run `retail-demand train` first to train and save the model."
                )
            elif not _data_available():
                st.error(
                    "Training data not found in `data/raw/`. "
                    "History is required to build lag/rolling features."
                )
            else:
                with st.spinner("Building features and scoring…"):
                    try:
                        from retail_demand.modeling.inference import score_dataframe

                        pipeline = _load_pipeline()
                        store_df = _load_store()
                        history_df = _load_history()

                        results = score_dataframe(target_df, history_df, store_df, pipeline)
                        results_display = results[
                            ["Store", "Date", "Open", "Sales_Predicted"]
                        ].copy()
                        results_display["Sales_Predicted"] = (
                            results_display["Sales_Predicted"].round(0).astype(int)
                        )

                        st.success(
                            f"Forecast complete — **{len(results_display):,}** rows scored."
                        )

                        # Summary stats
                        m1, m2, m3 = st.columns(3)
                        m1.metric(
                            "Mean predicted sales",
                            f"{results_display['Sales_Predicted'].mean():,.0f}",
                        )
                        m2.metric(
                            "Max predicted sales",
                            f"{results_display['Sales_Predicted'].max():,.0f}",
                        )
                        m3.metric(
                            "Stores scored",
                            f"{results_display['Store'].nunique()}",
                        )

                        # Chart: mean predicted sales by date
                        daily = (
                            results_display.groupby("Date")["Sales_Predicted"]
                            .mean()
                            .reset_index()
                            .rename(columns={"Sales_Predicted": "Mean Predicted Sales"})
                        )
                        st.line_chart(daily.set_index("Date"))

                        # Table preview
                        st.dataframe(results_display.head(100), use_container_width=True)

                        # Download button
                        csv_bytes = results_display.to_csv(index=False).encode("utf-8")
                        st.download_button(
                            "⬇️ Download predictions CSV",
                            data=csv_bytes,
                            file_name="predictions.csv",
                            mime="text/csv",
                        )

                    except Exception as exc:
                        st.error(f"Forecast failed: {exc}")

# ─── Performance tab ──────────────────────────────────────────────────────────
with tabs[1]:
    st.subheader("Model Performance")

    st.markdown(
        """
| Metric | Baseline (store mean) | HistGBM (ours) | Improvement |
|--------|----------------------|----------------|-------------|
| MAE    | 1,456.44             | **622.46**     | ↓ 57%       |
| WAPE   | 0.2087               | **0.0892**     | ↓ 57%       |

Evaluation on a **42-day holdout** (2015-06-20 → 2015-07-31).
"""
    )

    st.markdown("### Segment breakdown (worst WAPE)")
    segment_data = {
        "Segment": ["DayOfWeek=6", "SchoolHoliday=0", "Promo=0", "StateHoliday≠0"],
        "WAPE": [0.1015, 0.0940, 0.0982, 0.1200],
        "Description": [
            "Saturday — irregular shopping patterns",
            "Normal school days — high variance",
            "No active promotion",
            "State holidays — often zero sales",
        ],
    }
    st.dataframe(pd.DataFrame(segment_data), use_container_width=True)

    meta_path = MODEL_PATH.with_suffix(".metadata.json")
    if meta_path.exists():
        import json

        with open(meta_path) as fh:
            meta = json.load(fh)
        st.markdown("### Live model metadata")
        st.json(meta)

# ─── About tab ────────────────────────────────────────────────────────────────
with tabs[2]:
    st.subheader("About this project")
    st.markdown(
        """
### Production Retail Demand Forecasting

A production-style ML project trained on the
[Rossmann Store Sales](https://www.kaggle.com/c/rossmann-store-sales) Kaggle dataset
(1,017,209 rows, 1,115 stores, 2.5 years of daily data).

#### ML Pipeline
- **Data layer:** ingestion, schema validation, leakage guards
- **Features:** calendar (year/month/day/week), store attributes, competition metrics,
  Promo2 activity, sales lags (7/14/28 days), rolling mean & std (7/28 days)
- **Models:** HistGradientBoostingRegressor, LightGBM (point + quantile)
- **Evaluation:** MAE, WAPE, pinball loss, segment breakdown, top-error analysis
- **Experiment tracking:** MLflow (local `mlruns/`)

#### Tech Stack
`Python 3.11` · `scikit-learn` · `LightGBM` · `MLflow` · `pandas` · `Streamlit`

#### Key results
| Model | WAPE | vs. baseline |
|-------|------|--------------|
| Store-mean baseline | 0.2087 | — |
| HistGradientBoosting | **0.0892** | **–57%** |
| LightGBM | ~0.082 | ~–60% |
"""
    )
