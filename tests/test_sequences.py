"""Tests for gap-safe time-series sequence generation."""

import numpy as np
import pandas as pd

from machinepulse.data.sequences import build_split_sequences, valid_window_starts


def test_windows_do_not_cross_time_gaps() -> None:
    """A missing interval divides the data into independent sequence segments."""
    timestamps = pd.Series(
        pd.to_datetime(
            [
                "2020-01-01 00:00:00",
                "2020-01-01 00:05:00",
                "2020-01-01 00:10:00",
                "2020-01-01 01:00:00",
                "2020-01-01 01:05:00",
                "2020-01-01 01:10:00",
            ]
        )
    )

    starts = valid_window_starts(
        timestamps,
        window_rows=3,
        stride_rows=1,
        expected_interval_minutes=5,
    )

    assert starts.tolist() == [0, 3]


def test_normal_only_sequences_filter_positive_windows() -> None:
    """Autoencoder training data excludes every window containing a target."""
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2020-01-01", periods=6, freq="5min"),
            "sensor": np.arange(6, dtype=np.float32),
            "target": [0, 0, 0, 1, 0, 0],
            "failure_label": [0, 0, 0, 1, 0, 0],
        }
    )

    arrays = build_split_sequences(
        frame,
        ["sensor"],
        window_rows=3,
        stride_rows=1,
        expected_interval_minutes=5,
        normal_only=True,
    )

    assert arrays["X"].shape == (1, 3, 1)
    assert arrays["y"].tolist() == [0]
    assert arrays["failure"].tolist() == [0]
