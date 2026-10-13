"""YAML configuration loader with environment variable overrides and dynamic hardware resolution.

Provides a centralized, type-safe config loader for all VOIX modules.
All hardcoded paths and hyperparameters are sourced from configs/environment.yaml.
Hardware auto-configuration is loaded from configs/hardware.yaml.
Environment variables with prefix VOIX_ override YAML values at runtime.

Usage:
    from voix.utils.config import load_config, get_cfg, get_hardware_profile

    cfg = load_config()                       # Load default config
    root = cfg["paths"]["project_root"]       # Access nested keys
    hw = get_hardware_profile()               # Access detected hardware profile
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

import yaml

from voix.utils.hardware import HardwareProfile, detect_hardware

# Default config paths relative to project root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_CONFIG = _PROJECT_ROOT / "configs" / "environment.yaml"
_DEFAULT_HW_CONFIG = _PROJECT_ROOT / "configs" / "hardware.yaml"

# Module-level cached config and hardware profile
_CACHED_CONFIG: dict[str, Any] | None = None
_CACHED_HARDWARE: HardwareProfile | None = None


def get_hardware_profile(
    config_path: str | Path | None = None,
    force_reload: bool = False,
) -> HardwareProfile:
    """Retrieve the detected cross-platform hardware profile.

    Args:
        config_path: Optional path to configs/hardware.yaml.
        force_reload: If True, re-detect environment even if already cached.

    Returns:
        HardwareProfile: The active resolved hardware profile.
    """
    global _CACHED_HARDWARE
    if _CACHED_HARDWARE is not None and not force_reload and config_path is None:
        return _CACHED_HARDWARE

    path = Path(config_path) if config_path else _DEFAULT_HW_CONFIG
    hw = detect_hardware(path if path.exists() else None)
    _CACHED_HARDWARE = hw
    return hw


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
            raise FileNotFoundError(
                f"Config not found at {path}. "
                "Provide a valid path or leave as None to use the default."
            )
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

    # Attach dynamic hardware profile into config dictionary
    hw = get_hardware_profile(force_reload=force_reload)
    config["hardware"] = {
        "tier_id": hw.tier_id,
        "name": hw.name,
        "accelerator": hw.accelerator,
        "device": str(hw.device),
        "total_memory_gb": hw.total_memory_gb,
        "pipeline_mode": hw.pipeline_mode,
        "precision": hw.precision,
        "tts_device": hw.tts_device,
        "diffusion_steps": hw.diffusion_steps,
        "batch_voices": hw.batch_voices,
        "cpu_threads": hw.cpu_threads,
    }

    _CACHED_CONFIG = config
    return config


def get_cfg(key_path: str, config: dict[str, Any] | None = None) -> Any:
    """Convenience accessor for dot-separated nested config keys."""
    cfg = config if config is not None else load_config()
    keys = key_path.split(".")
    node = cfg
    for k in keys:
        if not isinstance(node, dict) or k not in node:
            raise KeyError(f"Config key '{key_path}' not found (missing segment: '{k}')")
        node = node[k]
    return node


def get_anchor_path(anchor_key: str, config: dict[str, Any] | None = None) -> str:
    """Resolve a named anchor to its full absolute path."""
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
    """Select the appropriate anchor key based on detected gender and age."""
    cfg = config if config is not None else load_config()
    anchors = cfg.get("anchors", {})

    male_young_thresh = anchors.get("male_young_age_threshold", 35)
    male_senior_thresh = anchors.get("male_senior_age_threshold", 60)
    female_mid_thresh = anchors.get("female_mid_age_threshold", 40)

    if gender == 1:  # Male
        if age < male_young_thresh:
            return "male_young"
        elif age < male_senior_thresh:
            return "male_mid"
        else:
            return "male_senior"
    else:           # Female
        if age < female_mid_thresh:
            return "female_young"
        else:
            return "female_mid"


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
