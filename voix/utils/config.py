"""YAML configuration loader with environment variable overrides.

Provides a centralized, type-safe config loader for all VOIX modules.
All hardcoded paths and hyperparameters are sourced from configs/environment.yaml.
Environment variables with prefix VOIX_ override YAML values at runtime.

Usage:
    from voix.utils.config import load_config, get_cfg

    cfg = load_config()                       # Load default config
    root = cfg["paths"]["project_root"]       # Access nested keys
    beta = cfg["training"]["cvae"]["beta_max"]
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

# Default config path relative to project root
_DEFAULT_CONFIG = Path(__file__).resolve().parent.parent.parent / "configs" / "environment.yaml"

# Module-level cached config (loaded once per process)
_CACHED_CONFIG: dict[str, Any] | None = None


def load_config(config_path: str | Path | None = None, force_reload: bool = False) -> dict[str, Any]:
    """Load YAML configuration and apply environment variable overrides.

    Args:
        config_path: Path to YAML config file. Defaults to configs/environment.yaml.
        force_reload: If True, reload even if already cached.

    Returns:
        Nested dict of configuration parameters.

    Raises:
        FileNotFoundError: If config file does not exist.
    """
    global _CACHED_CONFIG
    if _CACHED_CONFIG is not None and not force_reload and config_path is None:
        return _CACHED_CONFIG

    path = Path(config_path) if config_path else _DEFAULT_CONFIG

    if not path.exists():
        if config_path is not None:
            # Explicit path given and does not exist — raise immediately (no fallback)
            raise FileNotFoundError(
                f"Config not found at {path}. "
                "Provide a valid path or leave as None to use the default."
            )
        # Default path not found — try relative to cwd as last-resort fallback
        alt = Path.cwd() / "configs" / "environment.yaml"
        if alt.exists():
            path = alt
        else:
            raise FileNotFoundError(
                f"Config not found at {path}. "
                "Run from the project root or set VOIX_CONFIG env var."
            )

    with open(path, "r", encoding="utf-8") as f:
        config: dict[str, Any] = yaml.safe_load(f) or {}

    # --- Environment variable overrides ---
    if env_val := os.environ.get("VOIX_CONFIG"):
        # Allow completely overriding config path at runtime
        with open(env_val, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}

    # Top-level scalar overrides
    _apply_env_override(config, "VOIX_CHECKPOINT_DIR", "checkpoint_dir")
    _apply_env_override(config, "VOIX_DEVICE", "device")
    _apply_env_override(config, "VOIX_SEED", "seed", cast=int)

    # Inference overrides
    _apply_nested_env_override(config, "VOIX_BETA", ["inference", "beta"], cast=float)
    _apply_nested_env_override(config, "VOIX_BLEND_WEIGHT", ["inference", "blend_weight"], cast=float)
    _apply_nested_env_override(config, "VOIX_DIFFUSION_STEPS", ["inference", "diffusion_steps"], cast=int)
    _apply_nested_env_override(config, "VOIX_NUM_VOICES", ["inference", "num_voices"], cast=int)

    # Training overrides
    _apply_nested_env_override(config, "VOIX_BETA_MAX", ["training", "cvae", "beta_max"], cast=float)
    _apply_nested_env_override(config, "VOIX_CVAE_EPOCHS", ["training", "cvae", "epochs"], cast=int)
    _apply_nested_env_override(config, "VOIX_CVAE_LR", ["training", "cvae", "lr"], cast=float)

    _CACHED_CONFIG = config
    return config


def get_cfg(key_path: str, config: dict[str, Any] | None = None) -> Any:
    """Convenience accessor for dot-separated nested config keys.

    Args:
        key_path: Dot-separated key path, e.g. "training.cvae.beta_max".
        config: Optional pre-loaded config dict. If None, uses cached config.

    Returns:
        The value at the specified key path.

    Raises:
        KeyError: If key path does not exist in config.

    Example:
        beta = get_cfg("training.cvae.beta_max")
        librispeech = get_cfg("paths.librispeech_root")
    """
    cfg = config if config is not None else load_config()
    keys = key_path.split(".")
    node = cfg
    for k in keys:
        if not isinstance(node, dict) or k not in node:
            raise KeyError(f"Config key '{key_path}' not found (missing segment: '{k}')")
        node = node[k]
    return node


def get_anchor_path(anchor_key: str, config: dict[str, Any] | None = None) -> str:
    """Resolve a named anchor to its full absolute path.

    Args:
        anchor_key: One of: male_young, male_mid, male_senior,
                             female_young, female_mid.
        config: Optional pre-loaded config dict.

    Returns:
        Absolute path to the anchor FLAC file.

    Raises:
        KeyError: If anchor_key is not in config.
        FileNotFoundError: If resolved path does not exist.
    """
    cfg = config if config is not None else load_config()
    anchors = cfg.get("anchors", {})
    librispeech_root = cfg.get("paths", {}).get("librispeech_root", "")

    if anchor_key not in anchors:
        raise KeyError(
            f"Anchor '{anchor_key}' not in config. "
            f"Valid keys: {[k for k in anchors if not k.endswith('threshold')]}"
        )

    relative = anchors[anchor_key]
    full_path = os.path.join(librispeech_root, relative)

    if not os.path.exists(full_path):
        raise FileNotFoundError(f"Anchor file not found: {full_path}")

    return full_path


def select_anchor_key(gender: int, age: float, config: dict[str, Any] | None = None) -> str:
    """Select the appropriate anchor key based on detected gender and age.

    Args:
        gender: 1 = Male, 0 = Female (from InsightFace).
        age: Estimated age in years (from InsightFace).
        config: Optional pre-loaded config dict.

    Returns:
        Anchor key string (e.g. "male_mid") for use with get_anchor_path().
    """
    cfg = config if config is not None else load_config()
    anchors = cfg.get("anchors", {})

    male_young_thresh = anchors.get("male_young_age_threshold", 35)
    male_senior_thresh = anchors.get("male_senior_age_threshold", 60)
    female_mid_thresh = anchors.get("female_mid_age_threshold", 40)

    if gender == 1:  # Male
        if age < male_young_thresh:
            return "male_young"    # ~148 Hz, energetic
        elif age < male_senior_thresh:
            return "male_mid"      # ~130 Hz, classic adult male
        else:
            return "male_senior"   # ~118 Hz, deep baritone
    else:           # Female
        if age < female_mid_thresh:
            return "female_young"  # ~202 Hz
        else:
            return "female_mid"    # ~221 Hz


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _apply_env_override(
    config: dict,
    env_var: str,
    config_key: str,
    cast: type = str,
) -> None:
    if val := os.environ.get(env_var):
        try:
            config[config_key] = cast(val)
        except (ValueError, TypeError):
            pass


def _apply_nested_env_override(
    config: dict,
    env_var: str,
    key_path: list[str],
    cast: type = str,
) -> None:
    val = os.environ.get(env_var)
    if val is None:
        return
    try:
        cast_val = cast(val)
    except (ValueError, TypeError):
        return

    node = config
    for k in key_path[:-1]:
        if k not in node or not isinstance(node[k], dict):
            node[k] = {}
        node = node[k]
    node[key_path[-1]] = cast_val
