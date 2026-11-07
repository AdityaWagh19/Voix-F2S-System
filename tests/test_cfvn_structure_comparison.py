"""Tests for the validation-only CFVN structure comparison rule."""
from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "evaluation" / "compare_cfvn_structure.py"

spec = importlib.util.spec_from_file_location("compare_cfvn_structure", SCRIPT)
assert spec and spec.loader
compare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(compare)


def _run(
    name: str,
    val_r1: float,
    val_r10: float,
    face_face: float,
    raw: float = 0.88,
    *,
    test_face_face: float = 0.0,
    test_raw: float = 0.9153,
):
    return {
        "checkpoint": name,
        "best_epoch": 1,
        "validation": {
            "val_r1": val_r1,
            "val_r10": val_r10,
            "cfvn_face_face_r1": face_face,
            "raw_face_face_r1": raw,
        },
        "test": {
            "val_r1": 0.0,
            "val_r10": 0.0,
            "cfvn_face_face_r1": test_face_face,
            "raw_face_face_r1": test_raw,
        },
    }


def test_eligible_run_with_best_validation_score_wins():
    runs = [
        _run("a", 0.05, 0.20, 0.89),
        _run("b", 0.06, 0.21, 0.885),
        _run("c", 0.10, 0.30, 0.80),  # better val score, but fails identity gate
    ]
    decision = compare.choose(runs)
    assert decision["identity_gate"] is True
    assert decision["winner"] == "b"
    assert decision["winner_best_epoch"] == 1


def test_validation_eligible_winner_can_still_fail_test_identity():
    runs = [
        _run("s3", 0.0233, 0.1683, 0.8867, test_face_face=0.9153),
        _run("s3_seed43", 0.0267, 0.1633, 0.8900, test_face_face=0.8814),
    ]
    decision = compare.choose(runs)
    assert decision["identity_gate"] is True
    assert decision["winner"] == "s3_seed43"
    assert decision["test_identity_gate_winner"] is False
    assert decision["validation_eligible_runs"] == 2
    assert decision["test_identity_pass_runs"] == 1
    assert decision["test_identity_pass"] == ["s3"]


def test_no_eligible_run_reports_identity_gate_false():
    runs = [
        _run("a", 0.10, 0.30, 0.80),
        _run("b", 0.09, 0.28, 0.84),
    ]
    decision = compare.choose(runs)
    assert decision["identity_gate"] is False
    assert decision["winner"] == "a"
