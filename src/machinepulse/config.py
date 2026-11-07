"""Helpers for loading versioned MachinePulse configuration files."""

from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML mapping and reject empty or non-mapping documents."""
    config_path = Path(path)
    with config_path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)

    if not isinstance(config, dict):
        message = f"Configuration must be a YAML mapping: {config_path}"
        raise ValueError(message)
    return config
