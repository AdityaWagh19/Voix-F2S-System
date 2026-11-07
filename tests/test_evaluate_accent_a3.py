"""Tests for the A3 scale-selection rule."""
from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "evaluation" / "evaluate_accent_a3.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("evaluate_accent_a3", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _candidate(condition, scale, accuracy, wer_delta, identity_delta):
    return {
        "condition": condition,
        "scale": scale,
        "trainable_accuracy": accuracy,
        "wer_delta": wer_delta,
        "identity_delta": identity_delta,
    }


def test_invalid_proxy_selects_nothing():
    module = _load_module()
    result = module.select_scale(
        [_candidate("style_s0.2", 0.2, 0.9, 0.0, 0.0)],
        proxy_valid=False,
        baseline_accuracy=0.4,
    )
    assert result["status"] == "unavailable"
    assert result["selected_scale"] is None


def test_selects_highest_accuracy_inside_budgets():
    module = _load_module()
    result = module.select_scale(
        [
            _candidate("style_s0.1", 0.1, 0.50, 0.01, -0.01),
            _candidate("style_s0.2", 0.2, 0.60, 0.02, -0.02),
            _candidate("style_s0.3", 0.3, 0.70, 0.09, -0.02),
        ],
        proxy_valid=True,
        baseline_accuracy=0.45,
    )
    assert result["status"] == "selected"
    assert result["selected_scale"] == 0.2


def test_tie_goes_to_smaller_scale():
    module = _load_module()
    result = module.select_scale(
        [
            _candidate("style_s0.2", 0.2, 0.60, 0.0, 0.0),
            _candidate("style_s0.3", 0.3, 0.60, 0.0, 0.0),
        ],
        proxy_valid=True,
        baseline_accuracy=0.45,
    )
    assert result["selected_scale"] == 0.2


def test_no_gain_is_a_null_result():
    module = _load_module()
    result = module.select_scale(
        [
            _candidate("style_s0.2", 0.2, 0.45, 0.0, 0.0),
            _candidate("style_s0.3", 0.3, 0.40, 0.0, 0.0),
        ],
        proxy_valid=True,
        baseline_accuracy=0.45,
    )
    assert result["status"] == "null"
    assert result["selected_scale"] is None


def test_identity_collapse_disqualifies_scale():
    module = _load_module()
    result = module.select_scale(
        [
            _candidate("style_s0.2", 0.2, 0.80, 0.0, -0.20),
            _candidate("style_s0.1", 0.1, 0.55, 0.0, -0.01),
        ],
        proxy_valid=True,
        baseline_accuracy=0.45,
    )
    assert result["selected_scale"] == 0.1
