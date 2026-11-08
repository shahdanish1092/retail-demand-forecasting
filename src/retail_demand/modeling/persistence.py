"""Model persistence: save and load trained sklearn pipelines.

Usage::

    from retail_demand.modeling.persistence import save_model, load_model

    save_model(model_run.pipeline, path="model/latest.joblib")
    pipeline = load_model("model/latest.joblib")
    predictions = pipeline.predict(X)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)

_DEFAULT_MODEL_DIR = Path("model")
_DEFAULT_MODEL_PATH = _DEFAULT_MODEL_DIR / "latest.joblib"
_DEFAULT_METADATA_PATH = _DEFAULT_MODEL_DIR / "metadata.json"


def save_model(
    pipeline: Pipeline,
    path: str | Path = _DEFAULT_MODEL_PATH,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Persist a fitted sklearn Pipeline to disk using joblib.

    Args:
        pipeline: Fitted sklearn Pipeline to save.
        path: Destination file path (created with parent directories).
        metadata: Optional dict of metadata (metrics, feature names, etc.)
                  saved alongside the model as a JSON file.

    Returns:
        The resolved path where the model was saved.
    """
    import json

    dest = Path(path).resolve()
    dest.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(pipeline, dest)
    logger.info("Model saved to %s", dest)

    if metadata is not None:
        meta_path = dest.with_suffix(".metadata.json")
        with open(meta_path, "w", encoding="utf-8") as fh:
            json.dump(metadata, fh, indent=2, default=str)
        logger.info("Metadata saved to %s", meta_path)

    return dest


def load_model(path: str | Path = _DEFAULT_MODEL_PATH) -> Pipeline:
    """Load a fitted sklearn Pipeline from disk.

    Args:
        path: Path to the saved ``*.joblib`` file.

    Returns:
        The deserialized sklearn Pipeline.

    Raises:
        FileNotFoundError: If the model file does not exist at *path*.
    """
    src = Path(path).resolve()
    if not src.exists():
        raise FileNotFoundError(
            f"No model file found at '{src}'. "
            "Run 'retail-demand --train' or 'train-model' to create one."
        )
    pipeline: Pipeline = joblib.load(src)
    logger.info("Model loaded from %s", src)
    return pipeline


def load_model_metadata(path: str | Path = _DEFAULT_MODEL_PATH) -> dict[str, Any]:
    """Load the JSON metadata saved alongside a model file.

    Args:
        path: Path to the ``*.joblib`` model file. The metadata file is
              expected at ``<path>.metadata.json``.

    Returns:
        Dict of metadata, or an empty dict if the metadata file does not exist.
    """
    import json

    meta_path = Path(path).resolve().with_suffix(".metadata.json")
    if not meta_path.exists():
        return {}
    with open(meta_path, encoding="utf-8") as fh:
        return json.load(fh)
