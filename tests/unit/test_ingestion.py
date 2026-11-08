from pathlib import Path

import pandas as pd

from retail_demand.data.ingestion import RawDataPaths, summarize_dataframe


def test_raw_data_paths_reports_missing_required_files(tmp_path: Path) -> None:
    paths = RawDataPaths(tmp_path)

    assert paths.missing_required_files() == [
        str(tmp_path / "train.csv"),
        str(tmp_path / "store.csv"),
    ]


def test_summarize_dataframe_reports_shape_missing_values_and_duplicates() -> None:
    df = pd.DataFrame(
        {
            "Store": [1, 1, 2],
            "Sales": [100, 100, None],
        }
    )

    summary = summarize_dataframe(df)

    assert summary["shape"] == {"rows": 3, "columns": 2}
    assert summary["missing_values"] == {"Sales": 1}
    assert summary["duplicate_rows"] == 1
