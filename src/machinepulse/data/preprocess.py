"""Create labeled, chronological, model-ready MetroPT-3 tabular splits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from machinepulse.config import load_yaml


def label_failure_windows(
    timestamps: pd.Series,
    failure_events: list[dict[str, str]],
    warning_horizon_minutes: int,
) -> pd.DataFrame:
    """Derive active-failure, pre-failure warning, target, and event labels."""
    failure = np.zeros(len(timestamps), dtype=np.int8)
    warning = np.zeros(len(timestamps), dtype=np.int8)
    event_id = np.full(len(timestamps), "none", dtype=object)
    warning_delta = pd.Timedelta(minutes=warning_horizon_minutes)

    for event in failure_events:
        start = pd.Timestamp(event["start"])
        end = pd.Timestamp(event["end"])
        failure_mask = timestamps.between(start, end, inclusive="both").to_numpy()
        warning_mask = timestamps.between(start - warning_delta, start, inclusive="left").to_numpy()
        failure[failure_mask] = 1
        warning[warning_mask] = 1
        event_id[failure_mask | warning_mask] = event["id"]

    return pd.DataFrame(
        {
            "failure_label": failure,
            "warning_label": warning,
            "target": np.maximum(failure, warning),
            "event_id": event_id,
        }
    )


def assign_split(
    timestamps: pd.Series,
    train_end: str,
    validation_end: str,
) -> pd.Series:
    """Assign chronological train, validation, and test partitions."""
    train_boundary = pd.Timestamp(train_end)
    validation_boundary = pd.Timestamp(validation_end)
    split = np.full(len(timestamps), "test", dtype=object)
    split[timestamps.to_numpy() <= train_boundary.to_datetime64()] = "train"
    validation_mask = (timestamps > train_boundary) & (timestamps <= validation_boundary)
    split[validation_mask.to_numpy()] = "validation"
    return pd.Series(split, index=timestamps.index, name="split")


def _read_and_resample(config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, int]]:
    timestamp_column = config["timestamp_column"]
    analog_features = list(config["analog_features"])
    digital_features = list(config["digital_features"])
    features = analog_features + digital_features
    use_columns = [timestamp_column, *features]
    dtypes = {feature: "float32" for feature in features}

    raw = pd.read_csv(
        config["input_path"],
        usecols=use_columns,
        dtype=dtypes,
        parse_dates=[timestamp_column],
        low_memory=False,
    )
    raw = raw.sort_values(timestamp_column).set_index(timestamp_column)

    analog = raw[analog_features].resample(config["resample_rule"]).mean()
    digital = raw[digital_features].resample(config["resample_rule"]).max()
    resampled = pd.concat([analog, digital], axis=1)
    rows_before_drop = len(resampled)
    resampled = resampled.dropna(subset=features, how="any").reset_index()
    resampled[features] = resampled[features].astype("float32")
    return resampled, {
        "raw_rows": len(raw),
        "resampled_rows_before_drop": rows_before_drop,
        "empty_intervals_dropped": rows_before_drop - len(resampled),
    }


def _split_summary(frame: pd.DataFrame) -> dict[str, Any]:
    return {
        "rows": len(frame),
        "start": frame["timestamp"].min().isoformat(),
        "end": frame["timestamp"].max().isoformat(),
        "target_positive_rows": int(frame["target"].sum()),
        "failure_rows": int(frame["failure_label"].sum()),
        "warning_rows": int(frame["warning_label"].sum()),
        "target_positive_rate": float(frame["target"].mean()),
    }


def preprocess(config: dict[str, Any]) -> dict[str, Any]:
    """Run resampling, labeling, splitting, scaling, and artifact persistence."""
    preprocessing = config["preprocessing"]
    output_dir = Path(preprocessing["output_dir"])
    report_path = Path(preprocessing["report_path"])
    analog_features = list(preprocessing["analog_features"])
    digital_features = list(preprocessing["digital_features"])
    features = analog_features + digital_features

    frame, ingestion_stats = _read_and_resample(preprocessing)
    labels = label_failure_windows(
        frame[preprocessing["timestamp_column"]],
        preprocessing["failure_events"],
        int(preprocessing["warning_horizon_minutes"]),
    )
    frame = pd.concat([frame, labels], axis=1)
    frame["split"] = assign_split(
        frame[preprocessing["timestamp_column"]],
        preprocessing["split"]["train_end"],
        preprocessing["split"]["validation_end"],
    )

    train_normal = frame[(frame["split"] == "train") & (frame["target"] == 0)]
    if train_normal.empty:
        raise ValueError("No normal training rows are available for fitting the scaler")

    scaler = StandardScaler()
    scaler.fit(train_normal[analog_features])
    frame[analog_features] = scaler.transform(frame[analog_features]).astype("float32")
    frame[digital_features] = frame[digital_features].astype("float32")

    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, output_dir / "scaler.joblib")

    split_summaries: dict[str, Any] = {}
    for split_name in ("train", "validation", "test"):
        split_frame = frame[frame["split"] == split_name].drop(columns="split")
        if split_frame.empty:
            raise ValueError(f"Chronological split is empty: {split_name}")
        split_frame.to_parquet(
            output_dir / f"{split_name}.parquet",
            index=False,
            compression="zstd",
        )
        split_summaries[split_name] = _split_summary(split_frame)

    report = {
        **ingestion_stats,
        "resample_rule": preprocessing["resample_rule"],
        "warning_horizon_minutes": int(preprocessing["warning_horizon_minutes"]),
        "analog_features": analog_features,
        "digital_features": digital_features,
        "model_features": features,
        "scaler_fit": "normal training rows only",
        "scaler_fit_rows": len(train_normal),
        "splits": split_summaries,
        "failure_events": preprocessing["failure_events"],
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(f"{json.dumps(report, indent=2, sort_keys=True)}\n", encoding="utf-8")
    (output_dir / "metadata.json").write_text(
        f"{json.dumps(report, indent=2, sort_keys=True)}\n",
        encoding="utf-8",
    )
    print(
        f"Preprocessed {ingestion_stats['raw_rows']:,} raw rows into "
        f"{sum(summary['rows'] for summary in split_summaries.values()):,} intervals"
    )
    for split_name, summary in split_summaries.items():
        print(
            f"- {split_name}: {summary['rows']:,} rows, "
            f"{summary['target_positive_rows']:,} positive targets"
        )
    return report


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/preprocess.yaml", help="YAML config path")
    return parser.parse_args()


def main() -> None:
    """Run the tabular preprocessing command."""
    args = parse_args()
    preprocess(load_yaml(args.config))


if __name__ == "__main__":
    main()
