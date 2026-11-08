"""Train, evaluate, and register the main LSTM Autoencoder anomaly detector."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import matplotlib
import mlflow
import mlflow.pytorch
import numpy as np
import torch
from mlflow import MlflowClient
from mlflow.models import infer_signature
from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from machinepulse.config import load_yaml
from machinepulse.models.evaluation import (
    binary_metrics,
    flatten_numeric_metrics,
    select_f1_threshold,
)
from machinepulse.tracking import configure_tracking

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402


class LSTMAutoencoder(nn.Module):
    """Encode a multivariate sequence into a latent vector and reconstruct it."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        latent_size: int,
        num_layers: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        recurrent_dropout = dropout if num_layers > 1 else 0.0
        self.encoder = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=recurrent_dropout,
        )
        self.to_latent = nn.Linear(hidden_size, latent_size)
        self.from_latent = nn.Linear(latent_size, hidden_size)
        self.decoder = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=recurrent_dropout,
        )
        self.output_layer = nn.Linear(hidden_size, input_size)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Reconstruct a batch shaped as batch x time x features."""
        _, (hidden, _) = self.encoder(inputs)
        latent = self.to_latent(hidden[-1])
        decoder_seed = self.from_latent(latent)
        repeated = decoder_seed.unsqueeze(1).expand(-1, inputs.shape[1], -1)
        decoded, _ = self.decoder(repeated)
        return self.output_layer(decoded)


def set_reproducible_seed(seed: int, num_threads: int) -> None:
    """Configure deterministic NumPy and CPU PyTorch execution."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(num_threads)
    torch.use_deterministic_algorithms(True)


def _load_sequence_split(input_dir: Path, split: str) -> dict[str, np.ndarray]:
    with np.load(input_dir / f"{split}.npz") as archive:
        return {name: archive[name] for name in archive.files}


def _data_loader(
    values: np.ndarray,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    dataset = TensorDataset(torch.from_numpy(values.astype(np.float32, copy=False)))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        generator=generator,
    )


def fit_autoencoder(
    model: LSTMAutoencoder,
    train_values: np.ndarray,
    validation_normal_values: np.ndarray,
    config: dict[str, Any],
    *,
    epoch_callback: Callable[[int, float, float], None] | None = None,
) -> tuple[dict[str, list[float]], int]:
    """Train with early stopping against normal-only validation reconstruction loss."""
    device = torch.device(config["device"])
    model.to(device)
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    train_loader = _data_loader(
        train_values,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        seed=int(config["random_seed"]),
    )
    validation_loader = _data_loader(
        validation_normal_values,
        batch_size=int(config["batch_size"]),
        shuffle=False,
        seed=int(config["random_seed"]),
    )

    history = {"train_loss": [], "validation_normal_loss": []}
    best_loss = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0

    for epoch in range(1, int(config["epochs"]) + 1):
        model.train()
        train_total = 0.0
        train_rows = 0
        for (batch,) in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            reconstruction = model(batch)
            loss = loss_function(reconstruction, batch)
            loss.backward()
            optimizer.step()
            train_total += float(loss.detach()) * len(batch)
            train_rows += len(batch)
        train_loss = train_total / train_rows

        model.eval()
        validation_total = 0.0
        validation_rows = 0
        with torch.no_grad():
            for (batch,) in validation_loader:
                batch = batch.to(device)
                loss = loss_function(model(batch), batch)
                validation_total += float(loss) * len(batch)
                validation_rows += len(batch)
        validation_loss = validation_total / validation_rows
        history["train_loss"].append(train_loss)
        history["validation_normal_loss"].append(validation_loss)
        print(
            f"Epoch {epoch:02d}: train_loss={train_loss:.6f}, "
            f"validation_normal_loss={validation_loss:.6f}"
        )
        if epoch_callback is not None:
            epoch_callback(epoch, train_loss, validation_loss)

        min_delta = float(config["early_stopping_min_delta"])
        if validation_loss < best_loss - min_delta:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = {
                name: parameter.detach().cpu().clone()
                for name, parameter in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= int(config["early_stopping_patience"]):
                print(f"Early stopping after epoch {epoch}")
                break

    if best_state is None:
        raise RuntimeError("Training did not produce a valid model state")
    model.load_state_dict(best_state)
    model.to("cpu")
    return history, best_epoch


def reconstruction_scores(
    model: LSTMAutoencoder,
    values: np.ndarray,
    *,
    batch_size: int,
) -> np.ndarray:
    """Calculate mean squared reconstruction error for each sequence."""
    loader = _data_loader(values, batch_size=batch_size, shuffle=False, seed=0)
    model.eval()
    scores: list[np.ndarray] = []
    with torch.no_grad():
        for (batch,) in loader:
            reconstruction = model(batch)
            batch_scores = torch.mean((reconstruction - batch) ** 2, dim=(1, 2))
            scores.append(batch_scores.numpy())
    return np.concatenate(scores).astype(np.float64)


def _save_training_curve(history: dict[str, list[float]], output_path: Path) -> None:
    epochs = np.arange(1, len(history["train_loss"]) + 1)
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.plot(epochs, history["train_loss"], marker="o", label="train")
    axis.plot(
        epochs,
        history["validation_normal_loss"],
        marker="o",
        label="validation normal",
    )
    axis.set(xlabel="Epoch", ylabel="Mean squared reconstruction loss", title="Training history")
    axis.legend()
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def _save_score_distributions(
    validation_scores: np.ndarray,
    validation_labels: np.ndarray,
    test_scores: np.ndarray,
    test_labels: np.ndarray,
    threshold: float,
    output_path: Path,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    for axis, title, scores, labels in (
        (axes[0], "Validation", validation_scores, validation_labels),
        (axes[1], "Test", test_scores, test_labels),
    ):
        axis.hist(scores[labels == 0], bins=50, alpha=0.65, label="normal")
        axis.hist(scores[labels == 1], bins=50, alpha=0.65, label="warning/failure")
        axis.axvline(threshold, color="black", linestyle="--", label="threshold")
        axis.set(title=title, xlabel="Reconstruction error", ylabel="Sequences")
        axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def _save_confusion_matrices(
    validation_labels: np.ndarray,
    validation_scores: np.ndarray,
    test_labels: np.ndarray,
    test_scores: np.ndarray,
    threshold: float,
    output_path: Path,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    for axis, title, labels, scores in (
        (axes[0], "Validation", validation_labels, validation_scores),
        (axes[1], "Test", test_labels, test_scores),
    ):
        ConfusionMatrixDisplay.from_predictions(
            labels,
            (scores >= threshold).astype(np.int8),
            display_labels=["normal", "warning/failure"],
            colorbar=False,
            ax=axis,
        )
        axis.set_title(title)
    figure.suptitle(f"LSTM Autoencoder confusion matrices (threshold={threshold:.4f})")
    figure.tight_layout()
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def _save_test_curves(labels: np.ndarray, scores: np.ndarray, output_path: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    RocCurveDisplay.from_predictions(labels, scores, ax=axes[0], name="LSTM autoencoder")
    PrecisionRecallDisplay.from_predictions(labels, scores, ax=axes[1], name="LSTM autoencoder")
    axes[0].set_title("Test ROC curve")
    axes[1].set_title("Test precision-recall curve")
    figure.tight_layout()
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def _register_staging_model(
    model: LSTMAutoencoder,
    input_example: np.ndarray,
    registered_model_name: str,
) -> str:
    model.eval()
    with torch.no_grad():
        output_example = model(torch.from_numpy(input_example)).numpy()
    signature = infer_signature(input_example, output_example)
    logged_model = mlflow.pytorch.log_model(
        model,
        name="model",
        input_example=input_example,
        signature=signature,
        code_paths=["src"],
    )
    registered = mlflow.register_model(
        model_uri=logged_model.model_uri,
        name=registered_model_name,
        await_registration_for=300,
    )
    client = MlflowClient()
    client.transition_model_version_stage(
        name=registered_model_name,
        version=registered.version,
        stage="Staging",
        archive_existing_versions=True,
    )
    return str(registered.version)


def train_lstm_autoencoder(
    model_config: dict[str, Any],
    tracking_config: dict[str, Any] | None,
) -> dict[str, Any]:
    """Train, evaluate, persist, log, and optionally register the main model."""
    config = model_config["lstm_autoencoder"]
    input_dir = Path(config["input_dir"])
    model_dir = Path(config["model_dir"])
    figure_dir = Path(config["figure_dir"])
    metrics_path = Path(config["metrics_path"])
    train = _load_sequence_split(input_dir, "train")
    validation = _load_sequence_split(input_dir, "validation")
    test = _load_sequence_split(input_dir, "test")
    validation_normal = validation["X"][validation["y"] == 0]
    if not len(validation_normal):
        raise ValueError("Validation data has no normal sequences for early stopping")

    set_reproducible_seed(int(config["random_seed"]), int(config["num_threads"]))
    model = LSTMAutoencoder(
        input_size=int(train["X"].shape[2]),
        hidden_size=int(config["hidden_size"]),
        latent_size=int(config["latent_size"]),
        num_layers=int(config["num_layers"]),
        dropout=float(config["dropout"]),
    )

    tracking_enabled = tracking_config is not None
    if tracking_enabled:
        configure_tracking(tracking_config)
        run_name = tracking_config["tracking"]["lstm_run_name"]
        run_context = mlflow.start_run(run_name=run_name)
    else:
        run_context = nullcontext()

    with run_context as active_run:
        if tracking_enabled:
            mlflow.set_tags(
                {
                    "model_role": "main",
                    "model_family": "lstm_autoencoder",
                    "pipeline_phase": "phase_1",
                }
            )
            mlflow.log_params(
                {
                    name: value
                    for name, value in config.items()
                    if isinstance(value, str | int | float | bool)
                    and name not in {"input_dir", "model_dir", "figure_dir", "metrics_path"}
                }
            )
            mlflow.log_param("input_features", int(train["X"].shape[2]))
            mlflow.log_param("sequence_length", int(train["X"].shape[1]))
            mlflow.log_param("training_sequences", int(train["X"].shape[0]))

        def log_epoch(epoch: int, train_loss: float, validation_loss: float) -> None:
            if tracking_enabled:
                mlflow.log_metrics(
                    {
                        "train_reconstruction_loss": train_loss,
                        "validation_normal_reconstruction_loss": validation_loss,
                    },
                    step=epoch,
                )

        history, best_epoch = fit_autoencoder(
            model,
            train["X"],
            validation_normal,
            config,
            epoch_callback=log_epoch,
        )
        validation_scores = reconstruction_scores(
            model,
            validation["X"],
            batch_size=int(config["batch_size"]),
        )
        test_scores = reconstruction_scores(
            model,
            test["X"],
            batch_size=int(config["batch_size"]),
        )
        threshold = select_f1_threshold(validation["y"], validation_scores)
        report: dict[str, Any] = {
            "model": "LSTMAutoencoder",
            "scoring": "mean_squared_reconstruction_error",
            "parameters": {
                name: value
                for name, value in config.items()
                if isinstance(value, str | int | float | bool)
                and name not in {"input_dir", "model_dir", "figure_dir", "metrics_path"}
            },
            "training": {
                "sequences": int(train["X"].shape[0]),
                "best_epoch": best_epoch,
                "best_validation_normal_loss": float(
                    history["validation_normal_loss"][best_epoch - 1]
                ),
                "epochs_completed": len(history["train_loss"]),
            },
            "validation": binary_metrics(validation["y"], validation_scores, threshold),
            "test": binary_metrics(test["y"], test_scores, threshold),
        }

        model_dir.mkdir(parents=True, exist_ok=True)
        figure_dir.mkdir(parents=True, exist_ok=True)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / config["model_filename"]
        torch.save(
            {
                "state_dict": model.state_dict(),
                "input_size": int(train["X"].shape[2]),
                "hidden_size": int(config["hidden_size"]),
                "latent_size": int(config["latent_size"]),
                "num_layers": int(config["num_layers"]),
                "dropout": float(config["dropout"]),
                "threshold": threshold,
            },
            model_path,
        )
        history_path = model_dir / "training_history.json"
        history_path.write_text(
            f"{json.dumps(history, indent=2, sort_keys=True)}\n",
            encoding="utf-8",
        )
        metrics_path.write_text(
            f"{json.dumps(report, indent=2, sort_keys=True)}\n",
            encoding="utf-8",
        )

        training_curve_path = figure_dir / "training_curve.png"
        distribution_path = figure_dir / "score_distributions.png"
        confusion_path = figure_dir / "confusion_matrices.png"
        curves_path = figure_dir / "test_curves.png"
        _save_training_curve(history, training_curve_path)
        _save_score_distributions(
            validation_scores,
            validation["y"],
            test_scores,
            test["y"],
            threshold,
            distribution_path,
        )
        _save_confusion_matrices(
            validation["y"],
            validation_scores,
            test["y"],
            test_scores,
            threshold,
            confusion_path,
        )
        _save_test_curves(test["y"], test_scores, curves_path)

        if tracking_enabled:
            mlflow.log_metrics(flatten_numeric_metrics(report))
            mlflow.log_metric("best_epoch", float(best_epoch))
            mlflow.log_metric(
                "best_validation_normal_loss",
                float(history["validation_normal_loss"][best_epoch - 1]),
            )
            mlflow.log_dict(report, "evaluation/lstm_autoencoder_metrics.json")
            mlflow.log_artifact(history_path, artifact_path="training")
            mlflow.log_artifacts(figure_dir, artifact_path="evaluation")
            if active_run is None:
                raise RuntimeError("MLflow did not return an active run")
            registered_name = tracking_config["tracking"]["registered_model_name"]
            version = _register_staging_model(
                model,
                train["X"][:1],
                registered_name,
            )
            print(f"Registered {registered_name} version {version} in Staging")

    print(
        f"LSTM test F1={report['test']['f1']:.4f}, "
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
        help="Train and evaluate locally without logging or registering",
    )
    return parser.parse_args()


def main() -> None:
    """Train and evaluate the LSTM Autoencoder."""
    args = parse_args()
    tracking_config = None if args.skip_mlflow else load_yaml(args.tracking_config)
    train_lstm_autoencoder(load_yaml(args.config), tracking_config)


if __name__ == "__main__":
    main()
