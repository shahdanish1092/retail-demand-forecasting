# ─── Stage 1: builder ────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app

# Install build tools
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Copy dependency files first for layer caching
COPY pyproject.toml .
COPY src ./src

# Install the package + app extras (streamlit, plotly, altair) + mlflow + joblib
RUN pip install --no-cache-dir ".[app]"

# ─── Stage 2: runtime ────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code and sample/model assets needed for demo
COPY src ./src
COPY app.py .
COPY model ./model
COPY data/raw/store.csv ./data/raw/store.csv
COPY data/raw/test.csv ./data/raw/test.csv

# Ensure required runtime directories exist
RUN mkdir -p data/raw model submissions

# Non-root user for security
RUN useradd --create-home appuser && chown -R appuser /app
USER appuser

ENV PORT=8501
EXPOSE 8501

# Default: run the Streamlit demo, binding to Railway's dynamic $PORT (falling back to 8501)
CMD ["sh", "-c", "streamlit run app.py --server.port=${PORT:-8501} --server.address=0.0.0.0 --server.enableCORS=false --server.enableXsrfProtection=false"]
