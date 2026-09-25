"""YAML configuration loader with environment variable overrides."""

import os
from pathlib import Path
from typing import Any

import yaml


def load_config(config_path: str = "configs/environment.yaml") -> dict[str, Any]:
    """Load a YAML configuration file and return as a nested dict.

    Environment variables with prefix VOIX_ override config values.

    Args:
        config_path: Relative or absolute path to the YAML config file.

    Returns:
        Dictionary of configuration parameters.

    Raises:
        FileNotFoundError: If config_path does not exist.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path.resolve()}")

    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if env_ckpt := os.environ.get("VOIX_CHECKPOINT_DIR"):
        config["checkpoint_dir"] = env_ckpt
    if env_device := os.environ.get("VOIX_DEVICE"):
        config["device"] = env_device
    if env_seed := os.environ.get("VOIX_SEED"):
        config["seed"] = int(env_seed)

    return config