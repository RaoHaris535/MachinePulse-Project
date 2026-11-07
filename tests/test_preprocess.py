"""Tests for temporal labeling and chronological preprocessing."""

from pathlib import Path

import pandas as pd

from machinepulse.data.preprocess import assign_split, label_failure_windows, preprocess


def test_failure_and_warning_labels() -> None:
    """The warning horizon precedes the active failure interval without overlap."""
    timestamps = pd.Series(
        pd.to_datetime(
            [
                "2020-01-01 10:59:00",
                "2020-01-01 11:00:00",
                "2020-01-01 11:59:00",
                "2020-01-01 12:00:00",
                "2020-01-01 12:30:00",
                "2020-01-01 13:01:00",
            ]
        )
    )
    events = [
        {
            "id": "failure_01",
            "start": "2020-01-01 12:00:00",
            "end": "2020-01-01 13:00:00",
            "type": "air_leak",
        }
    ]

    labels = label_failure_windows(timestamps, events, warning_horizon_minutes=60)

    assert labels["warning_label"].tolist() == [0, 1, 1, 0, 0, 0]
    assert labels["failure_label"].tolist() == [0, 0, 0, 1, 1, 0]
    assert labels["target"].tolist() == [0, 1, 1, 1, 1, 0]
    assert labels["event_id"].tolist() == [
        "none",
        "failure_01",
        "failure_01",
        "failure_01",
        "failure_01",
        "none",
    ]


def test_chronological_split_boundaries() -> None:
    """Boundary timestamps belong to the earlier chronological partition."""
    timestamps = pd.Series(
        pd.to_datetime(
            [
                "2020-05-31 23:59:59",
                "2020-06-01 00:00:00",
                "2020-06-30 23:59:59",
                "2020-07-01 00:00:00",
            ]
        )
    )

    split = assign_split(timestamps, "2020-05-31 23:59:59", "2020-06-30 23:59:59")

    assert split.tolist() == ["train", "validation", "validation", "test"]


def test_preprocess_writes_all_chronological_artifacts(tmp_path: Path) -> None:
    """The full preprocessing step writes model-ready splits and a fitted scaler."""
    raw_path = tmp_path / "raw.csv"
    output_dir = tmp_path / "processed"
    report_path = tmp_path / "preprocessing.json"
    timestamps = pd.date_range("2020-01-01", periods=240, freq="10s")
    pd.DataFrame(
        {
            "timestamp": timestamps,
            "sensor": range(len(timestamps)),
            "switch": [index % 2 for index in range(len(timestamps))],
        }
    ).to_csv(raw_path, index=False)
    config = {
        "preprocessing": {
            "input_path": raw_path.as_posix(),
            "output_dir": output_dir.as_posix(),
            "report_path": report_path.as_posix(),
            "timestamp_column": "timestamp",
            "resample_rule": "5min",
            "csv_index_columns": [],
            "analog_features": ["sensor"],
            "digital_features": ["switch"],
            "split": {
                "train_end": "2020-01-01 00:14:59",
                "validation_end": "2020-01-01 00:24:59",
            },
            "warning_horizon_minutes": 5,
            "failure_events": [
                {
                    "id": "failure_01",
                    "start": "2020-01-01 00:10:00",
                    "end": "2020-01-01 00:10:00",
                    "type": "test_failure",
                }
            ],
        }
    }

    report = preprocess(config)

    assert report["raw_rows"] == 240
    assert report["splits"]["train"]["rows"] == 3
    assert report["splits"]["validation"]["rows"] == 2
    assert report["splits"]["test"]["rows"] == 3
    assert (output_dir / "train.parquet").is_file()
    assert (output_dir / "validation.parquet").is_file()
    assert (output_dir / "test.parquet").is_file()
    assert (output_dir / "scaler.joblib").is_file()
    assert (output_dir / "metadata.json").is_file()
    assert report_path.is_file()
