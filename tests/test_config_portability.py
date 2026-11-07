"""Configuration portability and contract tests.

The audit found ``configs/environment.yaml`` hardcoded to ``D:/voix`` and
carrying removed controls (anchor blending, a fixed diffusion-step count).
These tests pin the corrected contract: relative config, absolute resolved
paths, and no dead synthesis knobs.
"""
from __future__ import annotations

from pathlib import Path

import yaml

from voix.utils.config import load_config
from voix.utils.paths import PROJECT_ROOT


def _raw_config() -> dict:
    config_path = PROJECT_ROOT / "configs" / "environment.yaml"
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def test_environment_yaml_is_checkout_relative():
    raw = _raw_config()
    for key, value in raw["paths"].items():
        if isinstance(value, str) and key not in {
            "arcface_model",
            "ecapa_model",
            "whisper_model",
            "audioseal_generator",
            "audioseal_detector",
        }:
            assert not Path(value).is_absolute(), f"paths.{key} is absolute: {value}"
            assert "D:/voix" not in value and "D:\\voix" not in value


def test_load_config_resolves_paths_under_project_root():
    config = load_config(force_reload=True)
    root = PROJECT_ROOT.resolve()
    for key in (
        "checkpoint_dir",
        "cache_dir",
    ):
        assert Path(config[key]).is_absolute()
        assert Path(config[key]).is_relative_to(root)
    for key in ("data_root", "checkpoints", "artifacts", "features_h5"):
        resolved = Path(config["paths"][key])
        assert resolved.is_absolute()
        assert resolved.is_relative_to(root)


def test_removed_controls_are_absent_from_config():
    raw = _raw_config()
    inference = raw.get("inference", {})
    assert "blend_weight" not in inference
    assert "anchors" not in raw
    assert "f0_variance_scale" not in inference
    # null means "use the hardware profile", not a hardcoded override.
    assert inference.get("diffusion_steps") is None


def test_v2_artifacts_are_the_configured_defaults():
    config = load_config(force_reload=True)
    paths = config["paths"]
    assert paths["features_h5"].endswith("features_v2.h5")
    assert paths["training_index_csv"].endswith("index.csv")
    assert paths["feature_scaler"].endswith("face_scaler_v2.pt")
    assert paths["cfvn_checkpoint"].endswith("cfvn_v2_best.pt")
    assert "face_fusion_checkpoint" not in paths
