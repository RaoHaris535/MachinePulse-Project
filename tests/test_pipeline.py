"""Structural checks for the reproducible DVC pipeline."""

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_dvc_pipeline_contains_required_phase_one_stages() -> None:
    """Phase 1 keeps ingestion, validation, preparation, and both models named."""
    pipeline = yaml.safe_load((ROOT / "dvc.yaml").read_text(encoding="utf-8"))
    stages = pipeline["stages"]

    assert set(stages) == {
        "ingest",
        "validate",
        "preprocess",
        "sequences",
        "baseline",
        "lstm_autoencoder",
    }


def test_validation_is_a_training_gate_and_reports_are_metrics() -> None:
    """Model preparation cannot run until the raw-data validation report exists."""
    pipeline = yaml.safe_load((ROOT / "dvc.yaml").read_text(encoding="utf-8"))
    stages = pipeline["stages"]
    validation_report = "reports/metrics/data_validation.json"

    assert validation_report in stages["validate"]["metrics"]
    assert validation_report in stages["preprocess"]["deps"]
    assert "reports/metrics/preprocessing.json" in stages["preprocess"]["metrics"]
    assert "reports/metrics/sequences.json" in stages["sequences"]["metrics"]


def test_remote_tracking_config_is_a_dependency_of_both_model_stages() -> None:
    """Changing the remote experiment configuration invalidates both model runs."""
    pipeline = yaml.safe_load((ROOT / "dvc.yaml").read_text(encoding="utf-8"))
    stages = pipeline["stages"]

    assert "configs/tracking.yaml" in stages["baseline"]["deps"]
    assert "configs/tracking.yaml" in stages["lstm_autoencoder"]["deps"]
