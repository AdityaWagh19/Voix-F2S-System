"""CLI tests for the CREMA-D generator (no GPU)."""
from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "demo" / "generate_crema_e0.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("generate_crema_e0", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_default_condition_is_fixed_delta():
    module = _load_module()
    args = module.parse_args([])
    assert args.condition == "fixed_delta"
    assert args.residual_variant == "prosody_only"
    assert args.match_fixed_norm is True
    assert args.emotions == "happy,sad,angry"


def test_class_mean_flags():
    module = _load_module()
    args = module.parse_args(
        [
            "--condition",
            "class_mean",
            "--residual-variant",
            "full",
            "--no-match-fixed-norm",
            "--residual-scale",
            "0.2",
            "--emotions",
            "happy,sad,angry,fear,disgust",
            "--condition-label",
            "class_mean_full_ablation",
        ]
    )
    assert args.condition == "class_mean"
    assert args.residual_variant == "full"
    assert args.match_fixed_norm is False
    assert args.residual_scale == 0.2
    assert "fear" in args.emotions
    assert args.condition_label == "class_mean_full_ablation"


def test_condition_label_is_optional():
    module = _load_module()
    args = module.parse_args(["--condition", "class_mean"])
    assert args.condition_label is None
