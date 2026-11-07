"""Tests for the CFVN representation comparison rule."""
from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "evaluation" / "compare_cfvn.py"

spec = importlib.util.spec_from_file_location("compare_cfvn", SCRIPT)
assert spec and spec.loader
compare_cfvn = importlib.util.module_from_spec(spec)
spec.loader.exec_module(compare_cfvn)


def _split(r1: float, gender_r1: float) -> dict:
    return {
        "available": True,
        "val_r1": r1,
        "val_gender_controlled_r1": gender_r1,
    }


class TestRepresentationRecommendation:
    def test_missing_test_split_keeps_fused(self):
        result = compare_cfvn._recommend({}, _split(0.5, 0.4))
        assert result["recommended_face_key"] == "face_features"

    def test_small_gain_keeps_fused(self):
        result = compare_cfvn._recommend(
            _split(0.20, 0.15), _split(0.24, 0.18)
        )
        assert result["recommended_face_key"] == "face_features"

    def test_large_gain_on_both_metrics_switches(self):
        result = compare_cfvn._recommend(
            _split(0.10, 0.06), _split(0.20, 0.12)
        )
        assert result["recommended_face_key"] == "arcface"

    def test_unrestricted_gain_without_gender_gain_keeps_fused(self):
        result = compare_cfvn._recommend(
            _split(0.10, 0.06), _split(0.30, 0.06)
        )
        assert result["recommended_face_key"] == "face_features"

    def test_both_zero_does_not_switch(self):
        result = compare_cfvn._recommend(_split(0.0, 0.0), _split(0.0, 0.0))
        assert result["recommended_face_key"] == "face_features"
