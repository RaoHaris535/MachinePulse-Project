"""Validate the raw MetroPT-3 schema and sensor values before preprocessing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from machinepulse.config import load_yaml


def _empty_column_stats(columns: list[str]) -> dict[str, dict[str, float | int | None]]:
    return {
        column: {
            "min": None,
            "max": None,
            "null_count": 0,
            "non_numeric_count": 0,
            "infinite_count": 0,
            "range_violation_count": 0,
        }
        for column in columns
    }


def _update_numeric_stats(
    stats: dict[str, float | int | None],
    values: pd.Series,
    original: pd.Series,
    limits: dict[str, float] | None,
    *,
    binary: bool,
) -> None:
    non_numeric = values.isna() & original.notna()
    finite_mask = np.isfinite(values.fillna(0).to_numpy(dtype=float))
    infinite_count = int((~finite_mask & values.notna().to_numpy()).sum())
    finite_values = values[np.isfinite(values.to_numpy(dtype=float, na_value=np.nan))]

    stats["null_count"] = int(stats["null_count"]) + int(original.isna().sum())
    stats["non_numeric_count"] = int(stats["non_numeric_count"]) + int(non_numeric.sum())
    stats["infinite_count"] = int(stats["infinite_count"]) + infinite_count

    if not finite_values.empty:
        chunk_min = float(finite_values.min())
        chunk_max = float(finite_values.max())
        stats["min"] = chunk_min if stats["min"] is None else min(float(stats["min"]), chunk_min)
        stats["max"] = chunk_max if stats["max"] is None else max(float(stats["max"]), chunk_max)

    violation_count = 0
    if binary:
        violation_count += int((values.notna() & ~values.isin([0, 1])).sum())
    if limits is not None:
        violation_count += int(
            (
                values.notna()
                & ((values < float(limits["min"])) | (values > float(limits["max"])))
            ).sum()
        )
    stats["range_violation_count"] = int(stats["range_violation_count"]) + violation_count


def _build_errors(
    *,
    missing_columns: list[str],
    unexpected_columns: list[str],
    row_count: int,
    row_limits: dict[str, int],
    timestamp_summary: dict[str, Any],
    column_stats: dict[str, dict[str, float | int | None]],
) -> list[str]:
    errors: list[str] = []
    if missing_columns:
        errors.append(f"Missing required columns: {', '.join(missing_columns)}")
    if unexpected_columns:
        errors.append(f"Unexpected columns: {', '.join(unexpected_columns)}")
    if not int(row_limits["min"]) <= row_count <= int(row_limits["max"]):
        errors.append(
            f"Row count {row_count:,} is outside the expected range "
            f"[{int(row_limits['min']):,}, {int(row_limits['max']):,}]"
        )
    if timestamp_summary["invalid_count"]:
        errors.append(f"Invalid timestamps: {timestamp_summary['invalid_count']:,}")
    if timestamp_summary["duplicate_count"]:
        errors.append(f"Duplicate timestamps: {timestamp_summary['duplicate_count']:,}")
    if not timestamp_summary["monotonic_increasing"]:
        errors.append("Timestamps are not monotonically increasing")

    for column, stats in column_stats.items():
        for key, label in (
            ("null_count", "null values"),
            ("non_numeric_count", "non-numeric values"),
            ("infinite_count", "infinite values"),
            ("range_violation_count", "range violations"),
        ):
            if stats[key]:
                errors.append(f"{column}: {int(stats[key]):,} {label}")
    return errors


def validate_dataset(
    input_path: Path,
    report_path: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Validate a CSV in chunks and persist a machine-readable JSON report."""
    validation = config["validation"]
    required_columns = list(validation["required_columns"])
    numeric_columns = list(validation["numeric_columns"])
    optional_index_columns = set(validation["optional_index_columns"])
    binary_columns = set(validation["binary_columns"])
    ranges = validation["ranges"]

    header = pd.read_csv(input_path, nrows=0)
    actual_columns = list(header.columns)
    missing_columns = sorted(set(required_columns) - set(actual_columns))
    allowed_columns = set(required_columns) | optional_index_columns
    unexpected_columns = sorted(set(actual_columns) - allowed_columns)

    column_stats = _empty_column_stats(numeric_columns)
    null_counts = {column: 0 for column in required_columns}
    row_count = 0
    previous_timestamp: pd.Timestamp | None = None
    timestamp_min: pd.Timestamp | None = None
    timestamp_max: pd.Timestamp | None = None
    invalid_timestamp_count = 0
    duplicate_timestamp_count = 0
    monotonic_increasing = True

    if not missing_columns:
        for chunk in pd.read_csv(
            input_path,
            chunksize=int(validation["chunk_rows"]),
            low_memory=False,
        ):
            row_count += len(chunk)
            for column in required_columns:
                null_counts[column] += int(chunk[column].isna().sum())

            timestamps = pd.to_datetime(
                chunk[validation["timestamp_column"]],
                format=validation["timestamp_format"],
                errors="coerce",
            )
            invalid_timestamp_count += int(timestamps.isna().sum())
            valid_timestamps = timestamps.dropna()
            duplicate_timestamp_count += int(valid_timestamps.duplicated().sum())
            if not valid_timestamps.empty:
                if previous_timestamp is not None:
                    if valid_timestamps.iloc[0] == previous_timestamp:
                        duplicate_timestamp_count += 1
                    if valid_timestamps.iloc[0] < previous_timestamp:
                        monotonic_increasing = False
                if not valid_timestamps.is_monotonic_increasing:
                    monotonic_increasing = False
                chunk_min = valid_timestamps.min()
                chunk_max = valid_timestamps.max()
                timestamp_min = (
                    chunk_min if timestamp_min is None else min(timestamp_min, chunk_min)
                )
                timestamp_max = (
                    chunk_max if timestamp_max is None else max(timestamp_max, chunk_max)
                )
                previous_timestamp = valid_timestamps.iloc[-1]

            for column in numeric_columns:
                original = chunk[column]
                values = pd.to_numeric(original, errors="coerce")
                _update_numeric_stats(
                    column_stats[column],
                    values,
                    original,
                    ranges.get(column),
                    binary=column in binary_columns,
                )

    timestamp_summary = {
        "column": validation["timestamp_column"],
        "min": timestamp_min.isoformat() if timestamp_min is not None else None,
        "max": timestamp_max.isoformat() if timestamp_max is not None else None,
        "invalid_count": invalid_timestamp_count,
        "duplicate_count": duplicate_timestamp_count,
        "monotonic_increasing": monotonic_increasing,
    }
    errors = _build_errors(
        missing_columns=missing_columns,
        unexpected_columns=unexpected_columns,
        row_count=row_count,
        row_limits=validation["expected_rows"],
        timestamp_summary=timestamp_summary,
        column_stats=column_stats,
    )
    report = {
        "status": "passed" if not errors else "failed",
        "input_path": input_path.as_posix(),
        "row_count": row_count,
        "column_count": len(actual_columns),
        "columns": actual_columns,
        "missing_columns": missing_columns,
        "unexpected_columns": unexpected_columns,
        "null_counts": null_counts,
        "timestamps": timestamp_summary,
        "numeric_stats": column_stats,
        "errors": errors,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(f"{json.dumps(report, indent=2, sort_keys=True)}\n", encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/data.yaml", help="Data YAML path")
    parser.add_argument("--input", help="Override the raw CSV path from the configuration")
    parser.add_argument(
        "--report",
        default="reports/metrics/data_validation.json",
        help="Validation JSON output path",
    )
    return parser.parse_args()


def main() -> None:
    """Run raw-data validation and fail when any check fails."""
    args = parse_args()
    config = load_yaml(args.config)
    input_path = Path(args.input or config["source"]["raw_path"])
    report = validate_dataset(input_path, Path(args.report), config)
    print(f"Validation {report['status']}: {report['row_count']:,} rows checked")
    if report["errors"]:
        for error in report["errors"]:
            print(f"- {error}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
