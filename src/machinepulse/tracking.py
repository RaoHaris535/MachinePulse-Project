"""Remote MLflow configuration for the MachinePulse DagsHub repository."""

from __future__ import annotations

from typing import Any

import dagshub
import mlflow


def configure_tracking(config: dict[str, Any]) -> str:
    """Authenticate with DagsHub and select the configured remote experiment."""
    tracking = config["tracking"]
    dagshub.init(
        repo_owner=tracking["repo_owner"],
        repo_name=tracking["repo_name"],
        mlflow=True,
        dvc=False,
    )
    mlflow.set_tracking_uri(tracking["tracking_uri"])
    mlflow.set_experiment(tracking["experiment_name"])
    return str(tracking["tracking_uri"])
