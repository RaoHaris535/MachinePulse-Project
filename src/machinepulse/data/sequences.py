"""Build fixed-length LSTM sequences without crossing time-series gaps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from machinepulse.config import load_yaml


def valid_window_starts(
    timestamps: pd.Series,
    *,
    window_rows: int,
    stride_rows: int,
    expected_interval_minutes: int,
) -> np.ndarray:
    """Return window starts whose consecutive timestamps have the expected interval."""
    if window_rows < 2:
        raise ValueError("window_rows must be at least 2")
    if stride_rows < 1:
        raise ValueError("stride_rows must be positive")

    timestamp_ns = timestamps.astype("int64").to_numpy()
    expected_delta_ns = pd.Timedelta(minutes=expected_interval_minutes).value
    gap_positions = np.flatnonzero(np.diff(timestamp_ns) != expected_delta_ns) + 1
    segment_starts = np.concatenate(([0], gap_positions))
    segment_ends = np.concatenate((gap_positions, [len(timestamps)]))

    starts: list[int] = []
    for segment_start, segment_end in zip(segment_starts, segment_ends, strict=True):
        latest_start = int(segment_end) - window_rows
        if latest_start >= int(segment_start):
            starts.extend(range(int(segment_start), latest_start + 1, stride_rows))
    return np.asarray(starts, dtype=np.int64)


def build_split_sequences(
    frame: pd.DataFrame,
    features: list[str],
    *,
    window_rows: int,
    stride_rows: int,
    expected_interval_minutes: int,
    normal_only: bool,
) -> dict[str, np.ndarray]:
    """Convert a chronological tabular split into arrays for sequence modeling."""
    starts = valid_window_starts(
        frame["timestamp"],
        window_rows=window_rows,
        stride_rows=stride_rows,
        expected_interval_minutes=expected_interval_minutes,
    )
    values = frame[features].to_numpy(dtype=np.float32)
    target = frame["target"].to_numpy(dtype=np.int8)
    failure = frame["failure_label"].to_numpy(dtype=np.int8)
    timestamp_ns = frame["timestamp"].astype("int64").to_numpy()

    target_labels = np.asarray(
        [target[start : start + window_rows].max() for start in starts],
        dtype=np.int8,
    )
    failure_labels = np.asarray(
        [failure[start : start + window_rows].max() for start in starts],
        dtype=np.int8,
    )
    if normal_only:
        keep = target_labels == 0
        starts = starts[keep]
        target_labels = target_labels[keep]
        failure_labels = failure_labels[keep]

    if not len(starts):
        raise ValueError("No valid sequences were generated")

    sequences = np.stack(
        [values[start : start + window_rows] for start in starts],
    ).astype(np.float32, copy=False)
    end_timestamp_ns = timestamp_ns[starts + window_rows - 1]
    return {
        "X": sequences,
        "y": target_labels,
        "failure": failure_labels,
        "end_timestamp_ns": end_timestamp_ns,
    }


def build_sequences(config: dict[str, Any]) -> dict[str, Any]:
    """Create and persist train, validation, and test sequence arrays."""
    preprocessing = config["preprocessing"]
    sequence_config = config["sequences"]
    input_dir = Path(sequence_config["input_dir"])
    output_dir = Path(sequence_config["output_dir"])
    report_path = Path(sequence_config["report_path"])
    features = list(preprocessing["analog_features"]) + list(preprocessing["digital_features"])
    window_rows = int(sequence_config["window_rows"])
    stride_rows = int(sequence_config["stride_rows"])
    interval_minutes = int(sequence_config["expected_interval_minutes"])

    output_dir.mkdir(parents=True, exist_ok=True)
    split_reports: dict[str, Any] = {}
    for split_name in ("train", "validation", "test"):
        frame = pd.read_parquet(input_dir / f"{split_name}.parquet")
        arrays = build_split_sequences(
            frame,
            features,
            window_rows=window_rows,
            stride_rows=stride_rows,
            expected_interval_minutes=interval_minutes,
            normal_only=bool(sequence_config["train_normal_only"] and split_name == "train"),
        )
        np.savez_compressed(output_dir / f"{split_name}.npz", **arrays)
        split_reports[split_name] = {
            "sequences": int(arrays["X"].shape[0]),
            "shape": list(arrays["X"].shape),
            "positive_targets": int(arrays["y"].sum()),
            "failure_sequences": int(arrays["failure"].sum()),
            "normal_only": bool(sequence_config["train_normal_only"] and split_name == "train"),
        }
        print(
            f"- {split_name}: {arrays['X'].shape[0]:,} sequences, "
            f"shape={tuple(arrays['X'].shape)}"
        )

    report = {
        "features": features,
        "window_rows": window_rows,
        "window_minutes": window_rows * interval_minutes,
        "stride_rows": stride_rows,
        "expected_interval_minutes": interval_minutes,
        "splits": split_reports,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(f"{json.dumps(report, indent=2, sort_keys=True)}\n", encoding="utf-8")
    (output_dir / "metadata.json").write_text(
        f"{json.dumps(report, indent=2, sort_keys=True)}\n",
        encoding="utf-8",
    )
    return report


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/preprocess.yaml", help="YAML config path")
    return parser.parse_args()


def main() -> None:
    """Run sequence generation."""
    args = parse_args()
    print("Building one-hour sequences")
    build_sequences(load_yaml(args.config))


if __name__ == "__main__":
    main()
