"""Train and evaluate the Logistic Regression failure-prediction baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import matplotlib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay

from machinepulse.config import load_yaml
from machinepulse.models.evaluation import (
    binary_metrics,
    flatten_numeric_metrics,
    select_f1_threshold,
)
from machinepulse.tracking import configure_tracking

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402


def _load_split(input_dir: Path, split: str, features: list[str]) -> tuple[np.ndarray, np.ndarray]:
    frame = pd.read_parquet(input_dir / f"{split}.parquet", columns=[*features, "target"])
    return (
        frame[features].to_numpy(dtype=np.float32),
        frame["target"].to_numpy(dtype=np.int8),
    )


def _save_coefficients(
    model: LogisticRegression,
    features: list[str],
    output_path: Path,
) -> None:
    coefficients = pd.DataFrame(
        {
            "feature": features,
            "coefficient": model.coef_[0],
            "absolute_coefficient": np.abs(model.coef_[0]),
        }
    ).sort_values("absolute_coefficient", ascending=False)
    coefficients.to_csv(output_path, index=False)


def _save_confusion_matrices(
    y_validation: np.ndarray,
    validation_scores: np.ndarray,
    y_test: np.ndarray,
    test_scores: np.ndarray,
    threshold: float,
    output_path: Path,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    for axis, title, labels, scores in (
        (axes[0], "Validation", y_validation, validation_scores),
        (axes[1], "Test", y_test, test_scores),
    ):
        ConfusionMatrixDisplay.from_predictions(
            labels,
            (scores >= threshold).astype(np.int8),
            display_labels=["normal", "warning/failure"],
            colorbar=False,
            ax=axis,
        )
        axis.set_title(title)
    figure.suptitle(f"Logistic Regression confusion matrices (threshold={threshold:.3f})")
    figure.tight_layout()
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def _save_test_curves(
    y_test: np.ndarray,
    test_scores: np.ndarray,
    output_path: Path,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    RocCurveDisplay.from_predictions(y_test, test_scores, ax=axes[0], name="baseline")
    PrecisionRecallDisplay.from_predictions(y_test, test_scores, ax=axes[1], name="baseline")
    axes[0].set_title("Test ROC curve")
    axes[1].set_title("Test precision-recall curve")
    figure.tight_layout()
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def train_baseline(
    model_config: dict[str, Any],
    tracking_config: dict[str, Any] | None,
) -> dict[str, Any]:
    """Train the baseline, persist deterministic outputs, and optionally log remotely."""
    config = model_config["baseline"]
    input_dir = Path(config["input_dir"])
    model_dir = Path(config["model_dir"])
    figure_dir = Path(config["figure_dir"])
    metrics_path = Path(config["metrics_path"])
    features = list(config["features"])

    x_train, y_train = _load_split(input_dir, "train", features)
    x_validation, y_validation = _load_split(input_dir, "validation", features)
    x_test, y_test = _load_split(input_dir, "test", features)

    model = LogisticRegression(
        C=float(config["regularization_c"]),
        class_weight=config["class_weight"],
        max_iter=int(config["max_iter"]),
        random_state=int(config["random_seed"]),
        solver=config["solver"],
    )
    model.fit(x_train, y_train)
    validation_scores = model.predict_proba(x_validation)[:, 1]
    test_scores = model.predict_proba(x_test)[:, 1]
    threshold = select_f1_threshold(y_validation, validation_scores)

    report: dict[str, Any] = {
        "model": "LogisticRegression",
        "features": features,
        "parameters": {
            "C": float(config["regularization_c"]),
            "class_weight": config["class_weight"],
            "max_iter": int(config["max_iter"]),
            "random_seed": int(config["random_seed"]),
            "solver": config["solver"],
        },
        "training": {
            "rows": len(y_train),
            "positive_rows": int(y_train.sum()),
            "positive_rate": float(y_train.mean()),
        },
        "validation": binary_metrics(y_validation, validation_scores, threshold),
        "test": binary_metrics(y_test, test_scores, threshold),
    }

    model_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / config["model_filename"]
    bundle = {"model": model, "features": features, "threshold": threshold}
    joblib.dump(bundle, model_path)
    metrics_path.write_text(f"{json.dumps(report, indent=2, sort_keys=True)}\n", encoding="utf-8")

    coefficients_path = figure_dir / "coefficients.csv"
    confusion_path = figure_dir / "confusion_matrices.png"
    curves_path = figure_dir / "test_curves.png"
    _save_coefficients(model, features, coefficients_path)
    _save_confusion_matrices(
        y_validation,
        validation_scores,
        y_test,
        test_scores,
        threshold,
        confusion_path,
    )
    _save_test_curves(y_test, test_scores, curves_path)

    if tracking_config is not None:
        tracking_uri = configure_tracking(tracking_config)
        run_name = tracking_config["tracking"]["baseline_run_name"]
        with mlflow.start_run(run_name=run_name) as run:
            mlflow.set_tags(
                {
                    "model_role": "baseline",
                    "model_family": "logistic_regression",
                    "pipeline_phase": "phase_1",
                }
            )
            mlflow.log_params(report["parameters"])
            mlflow.log_param("feature_count", len(features))
            mlflow.log_param("training_rows", len(y_train))
            mlflow.log_metrics(flatten_numeric_metrics(report))
            mlflow.log_dict(report, "evaluation/baseline_metrics.json")
            mlflow.log_artifact(coefficients_path, artifact_path="evaluation")
            mlflow.log_artifact(confusion_path, artifact_path="evaluation")
            mlflow.log_artifact(curves_path, artifact_path="evaluation")
            mlflow.sklearn.log_model(model, artifact_path="model")
            print(f"Logged baseline MLflow run {run.info.run_id} to {tracking_uri}")

    print(
        f"Baseline test F1={report['test']['f1']:.4f}, "
        f"ROC-AUC={report['test']['roc_auc']:.4f}, "
        f"PR-AUC={report['test']['average_precision']:.4f}"
    )
    return report


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/models.yaml", help="Model YAML path")
    parser.add_argument(
        "--tracking-config",
        default="configs/tracking.yaml",
        help="Remote MLflow YAML path",
    )
    parser.add_argument(
        "--skip-mlflow",
        action="store_true",
        help="Create local artifacts without logging a remote run",
    )
    return parser.parse_args()


def main() -> None:
    """Train and evaluate the Logistic Regression baseline."""
    args = parse_args()
    tracking_config = None if args.skip_mlflow else load_yaml(args.tracking_config)
    train_baseline(load_yaml(args.config), tracking_config)


if __name__ == "__main__":
    main()
