"""Tests for the post-hoc emotion Holm sensitivity script."""
from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "evaluation" / "emotion_holm_sensitivity.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("emotion_holm_sensitivity", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _effect(mean: float, sd: float, expected: int, n: int = 12) -> dict:
    return {
        "n_pairs": n,
        "mean_delta": mean,
        "sd_delta": sd,
        "expected_direction": expected,
        "direction_match": mean * expected > 0,
        "ci_excludes_zero_in_expected_direction": mean * expected > 0,
    }


def _report(sad_f0: float = -1.0) -> dict:
    effects = {}
    for emotion, sign in (("happy", 1), ("sad", -1), ("angry", 1)):
        f0 = sad_f0 if emotion == "sad" else 2.0 * sign
        duration = 0.05 * sign
        effects[f"condition:{emotion}"] = {
            "f0_mean_hz": _effect(f0, 0.5, sign),
            "duration_seconds": _effect(duration, 0.005, sign),
        }
    return {"paired_effects": {"effects": effects}}


def test_holm_adjust_is_step_down_and_monotone():
    module = _load_module()
    adjusted = module.holm_adjust({"a": 0.01, "b": 0.04, "c": 0.03})
    assert adjusted == {"a": 0.03, "c": 0.06, "b": 0.06}


def test_analyse_run_requires_expected_direction():
    module = _load_module()

    passing = module.analyse_run(_report(sad_f0=-1.0))
    assert passing["n_checks"] == 6
    assert passing["all_holm_pass_expected_direction"] is True

    wrong_direction = module.analyse_run(_report(sad_f0=1.0))
    assert wrong_direction["n_checks"] == 6
    assert wrong_direction["all_holm_pass_expected_direction"] is False
    sad_f0 = wrong_direction["checks"]["sad:f0_mean_hz"]
    assert sad_f0["holm_pass_0_05"] is True
    assert sad_f0["holm_pass_expected_direction"] is False
