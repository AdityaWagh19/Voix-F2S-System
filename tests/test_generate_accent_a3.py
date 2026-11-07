"""Tests for the A3 accent style-residual generator."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "demo" / "generate_accent_a3.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("generate_accent_a3", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_apply_residual_weights_timbre_and_prosody_halves():
    module = _load_module()
    style = torch.zeros(1, 256)
    residual = torch.ones(256)
    adjusted = module.apply_residual(
        style, residual, scale=0.5, timbre_weight=0.25
    )
    assert torch.allclose(adjusted[0, :128], torch.full((128,), 0.125))
    assert torch.allclose(adjusted[0, 128:], torch.full((128,), 0.5))


def test_apply_residual_zero_scale_is_noop():
    module = _load_module()
    style = torch.randn(1, 256)
    residual = torch.randn(256)
    adjusted = module.apply_residual(
        style, residual, scale=0.0, timbre_weight=0.25
    )
    assert torch.allclose(adjusted, style)


def test_apply_residual_rejects_wrong_dimension():
    module = _load_module()
    with pytest.raises(ValueError):
        module.apply_residual(
            torch.zeros(1, 255), torch.zeros(256), scale=0.2, timbre_weight=0.25
        )
