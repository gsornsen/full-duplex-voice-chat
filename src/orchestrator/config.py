"""Orchestrator configuration loading."""

from pathlib import Path
from typing import Any

import yaml


def load_config(config_path: Path) -> dict[str, Any]:
    """Load orchestrator configuration from YAML file.

    Args:
        config_path: Path to configuration YAML file

    Returns:
        Configuration dictionary
    """
    with config_path.open("r") as f:
        config: dict[str, Any] = yaml.safe_load(f)
    return config
