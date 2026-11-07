"""Tests for chunked MetroPT-3 validation."""

from copy import deepcopy
from pathlib import Path

import pandas as pd

from machinepulse.data.validate import validate_dataset


def _validation_config() -> dict:
    columns = ["timestamp", "sensor", "switch"]
    return {
        "source": {"raw_path": "unused.csv"},
        "validation": {
            "timestamp_column": "timestamp",
            "timestamp_format": "%Y-%m-%d %H:%M:%S",
            "chunk_rows": 2,
            "expected_rows": {"min": 3, "max": 3},
            "optional_index_columns": ["index"],
            "required_columns": columns,
            "numeric_columns": ["sensor", "switch"],
            "ranges": {"sensor": {"min": 0.0, "max": 10.0}},
            "binary_columns": ["switch"],
        },
    }


def test_valid_dataset_passes(tmp_path: Path) -> None:
    """A complete, ordered dataset with valid sensor values passes."""
    csv_path = tmp_path / "valid.csv"
    report_path = tmp_path / "report.json"
    pd.DataFrame(
        {
            "timestamp": [
                "2020-01-01 00:00:00",
                "2020-01-01 00:00:10",
                "2020-01-01 00:00:20",
            ],
            "sensor": [1.0, 2.0, 3.0],
            "switch": [0, 1, 0],
        }
    ).to_csv(csv_path, index=False)

    report = validate_dataset(csv_path, report_path, _validation_config())

    assert report["status"] == "passed"
    assert report["row_count"] == 3
    assert report["errors"] == []
    assert report_path.is_file()


def test_invalid_dataset_reports_failures(tmp_path: Path) -> None:
    """Duplicate timestamps and invalid values are reported rather than hidden."""
    csv_path = tmp_path / "invalid.csv"
    report_path = tmp_path / "report.json"
    frame = pd.DataFrame(
        {
            "timestamp": [
                "2020-01-01 00:00:00",
                "2020-01-01 00:00:00",
                "not-a-timestamp",
            ],
            "sensor": [1.0, 99.0, "bad"],
            "switch": [0, 2, 1],
        }
    )
    frame.to_csv(csv_path, index=False)
    config = deepcopy(_validation_config())

    report = validate_dataset(csv_path, report_path, config)

    assert report["status"] == "failed"
    assert report["timestamps"]["duplicate_count"] == 1
    assert report["timestamps"]["invalid_count"] == 1
    assert report["numeric_stats"]["sensor"]["non_numeric_count"] == 1
    assert report["numeric_stats"]["sensor"]["range_violation_count"] == 1
    assert report["numeric_stats"]["switch"]["range_violation_count"] == 1
