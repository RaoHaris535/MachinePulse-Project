"""End-to-end tests for the Logistic Regression baseline."""

from pathlib import Path

import numpy as np
import pandas as pd

from machinepulse.models.baseline import train_baseline


def _write_split(path: Path, negative: np.ndarray, positive: np.ndarray) -> None:
    features = np.concatenate([negative, positive])
    labels = np.concatenate(
        [
            np.zeros(len(negative), dtype=np.int8),
            np.ones(len(positive), dtype=np.int8),
        ]
    )
    pd.DataFrame({"sensor": features, "target": labels}).to_parquet(path, index=False)


def test_baseline_writes_model_metrics_and_figures(tmp_path: Path) -> None:
    """Local baseline execution produces every artifact without contacting MLflow."""
    input_dir = tmp_path / "input"
    model_dir = tmp_path / "model"
    figure_dir = tmp_path / "figures"
    metrics_path = tmp_path / "metrics.json"
    input_dir.mkdir()
    _write_split(input_dir / "train.parquet", np.linspace(-3, -1, 40), np.linspace(1, 3, 10))
    _write_split(
        input_dir / "validation.parquet",
        np.linspace(-2.5, -0.5, 20),
        np.linspace(0.5, 2.5, 8),
    )
    _write_split(input_dir / "test.parquet", np.linspace(-2, -0.2, 20), np.linspace(0.2, 2, 8))
    config = {
        "baseline": {
            "input_dir": input_dir.as_posix(),
            "model_dir": model_dir.as_posix(),
            "figure_dir": figure_dir.as_posix(),
            "metrics_path": metrics_path.as_posix(),
            "model_filename": "baseline.joblib",
            "random_seed": 42,
            "max_iter": 200,
            "solver": "liblinear",
            "class_weight": "balanced",
            "regularization_c": 1.0,
            "features": ["sensor"],
        }
    }

    report = train_baseline(config, tracking_config=None)

    assert report["test"]["f1"] >= 0.8
    assert report["test"]["roc_auc"] == 1.0
    assert report["test"]["average_precision"] == 1.0
    assert metrics_path.is_file()
    assert (model_dir / "baseline.joblib").is_file()
    assert (figure_dir / "coefficients.csv").is_file()
    assert (figure_dir / "confusion_matrices.png").is_file()
    assert (figure_dir / "test_curves.png").is_file()
