"""Tests for the LSTM Autoencoder architecture and local training workflow."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from machinepulse.models import lstm_autoencoder as lstm_module
from machinepulse.models.lstm_autoencoder import LSTMAutoencoder, train_lstm_autoencoder


def test_lstm_autoencoder_preserves_input_shape() -> None:
    """The reconstruction has the same batch, time, and feature dimensions."""
    model = LSTMAutoencoder(input_size=3, hidden_size=8, latent_size=4)
    inputs = torch.randn(5, 6, 3)

    reconstruction = model(inputs)

    assert reconstruction.shape == inputs.shape


def test_registration_uses_mlflow_logged_model_uri(monkeypatch) -> None:
    """MLflow 3 registration uses the URI returned by log_model."""
    calls: dict[str, object] = {}

    def fake_log_model(model: LSTMAutoencoder, **kwargs: object) -> SimpleNamespace:
        calls["log_name"] = kwargs["name"]
        return SimpleNamespace(model_uri="models:/m-test-model")

    def fake_register_model(**kwargs: object) -> SimpleNamespace:
        calls["model_uri"] = kwargs["model_uri"]
        calls["registered_name"] = kwargs["name"]
        return SimpleNamespace(version="7")

    class FakeClient:
        def transition_model_version_stage(self, **kwargs: object) -> None:
            calls["transition"] = kwargs

    monkeypatch.setattr(lstm_module.mlflow.pytorch, "log_model", fake_log_model)
    monkeypatch.setattr(lstm_module.mlflow, "register_model", fake_register_model)
    monkeypatch.setattr(lstm_module, "MlflowClient", FakeClient)

    version = lstm_module._register_staging_model(
        LSTMAutoencoder(input_size=3, hidden_size=4, latent_size=2),
        np.zeros((1, 6, 3), dtype=np.float32),
        "MachinePulse-LSTM-Autoencoder",
    )

    assert version == "7"
    assert calls["log_name"] == "model"
    assert calls["model_uri"] == "models:/m-test-model"
    assert calls["registered_name"] == "MachinePulse-LSTM-Autoencoder"
    assert calls["transition"] == {
        "name": "MachinePulse-LSTM-Autoencoder",
        "version": "7",
        "stage": "Staging",
        "archive_existing_versions": True,
    }


def _write_sequence_split(
    output_path: Path,
    normal: np.ndarray,
    anomalies: np.ndarray,
) -> None:
    values = np.concatenate([normal, anomalies]).astype(np.float32)
    labels = np.concatenate(
        [
            np.zeros(len(normal), dtype=np.int8),
            np.ones(len(anomalies), dtype=np.int8),
        ]
    )
    np.savez_compressed(
        output_path,
        X=values,
        y=labels,
        failure=labels,
        end_timestamp_ns=np.arange(len(values), dtype=np.int64),
    )


def test_local_lstm_training_writes_all_artifacts(tmp_path: Path) -> None:
    """A small local run produces weights, metrics, history, and figures."""
    random = np.random.default_rng(42)
    input_dir = tmp_path / "sequences"
    model_dir = tmp_path / "model"
    figure_dir = tmp_path / "figures"
    metrics_path = tmp_path / "metrics.json"
    input_dir.mkdir()
    train_normal = random.normal(0, 0.2, size=(32, 6, 3)).astype(np.float32)
    validation_normal = random.normal(0, 0.2, size=(16, 6, 3)).astype(np.float32)
    validation_anomalies = random.normal(3, 0.2, size=(8, 6, 3)).astype(np.float32)
    test_normal = random.normal(0, 0.2, size=(16, 6, 3)).astype(np.float32)
    test_anomalies = random.normal(3, 0.2, size=(8, 6, 3)).astype(np.float32)
    _write_sequence_split(input_dir / "train.npz", train_normal, np.empty((0, 6, 3)))
    _write_sequence_split(
        input_dir / "validation.npz",
        validation_normal,
        validation_anomalies,
    )
    _write_sequence_split(input_dir / "test.npz", test_normal, test_anomalies)
    config = {
        "lstm_autoencoder": {
            "input_dir": input_dir.as_posix(),
            "model_dir": model_dir.as_posix(),
            "figure_dir": figure_dir.as_posix(),
            "metrics_path": metrics_path.as_posix(),
            "model_filename": "model.pt",
            "random_seed": 42,
            "device": "cpu",
            "num_threads": 1,
            "hidden_size": 8,
            "latent_size": 4,
            "num_layers": 1,
            "dropout": 0.0,
            "batch_size": 8,
            "epochs": 2,
            "learning_rate": 0.01,
            "weight_decay": 0.0,
            "early_stopping_patience": 2,
            "early_stopping_min_delta": 0.0,
        }
    }

    report = train_lstm_autoencoder(config, tracking_config=None)

    assert report["test"]["roc_auc"] == 1.0
    assert report["test"]["f1"] >= 0.9
    assert (model_dir / "model.pt").is_file()
    assert (model_dir / "training_history.json").is_file()
    assert metrics_path.is_file()
    assert (figure_dir / "training_curve.png").is_file()
    assert (figure_dir / "score_distributions.png").is_file()
    assert (figure_dir / "confusion_matrices.png").is_file()
    assert (figure_dir / "test_curves.png").is_file()
