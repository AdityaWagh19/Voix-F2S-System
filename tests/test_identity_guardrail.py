"""Tests for the cross-track identity guardrail."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from voix.evaluation.identity_guardrail import (
    CANONICAL_CHECKPOINTS,
    CANONICAL_METRICS,
    GuardrailThresholds,
    evaluate_cfvn_drift,
    evaluate_identity_guardrail,
    neutral_zero_check,
    neutral_zero_error,
    verify_canonical_hashes,
    write_canonical_hashes,
)


def _write_fake_canonical(root: Path) -> Path:
    for relative in CANONICAL_CHECKPOINTS.values():
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"canonical-" + relative.encode("utf-8"))
    path = root / "results" / "canonical_hashes.json"
    write_canonical_hashes(path, root=root)
    return path


def _rows(condition: str = "happy", neutral: float = 0.90):
    rows = []
    for index, speaker in enumerate(["a", "b", "c", "d"]):
        rows.append(
            {
                "condition": condition,
                "speaker_id": speaker,
                "ecapa_neutral": neutral + index * 0.001,
                "ecapa_reference": 0.88 + index * 0.001,
                "canonical_ecapa": 0.90 + index * 0.001,
                "pairwise_diversity": 0.40 + index * 0.001,
            }
        )
    return rows


def _cfvn_metrics():
    return {"recall_at_1": 0.0932, "recall_at_10": 0.4068}


def test_write_and_verify_canonical_hashes(tmp_path: Path):
    path = _write_fake_canonical(tmp_path)
    result = verify_canonical_hashes(path, root=tmp_path)
    assert result["passed"] is True
    assert result["mismatched"] == []
    assert set(result["files"]) == set(CANONICAL_CHECKPOINTS)


def test_verify_detects_tampering(tmp_path: Path):
    path = _write_fake_canonical(tmp_path)
    target = tmp_path / CANONICAL_CHECKPOINTS["cfvn"]
    target.write_bytes(b"tampered")
    result = verify_canonical_hashes(path, root=tmp_path)
    assert result["passed"] is False
    assert result["mismatched"] == ["cfvn"]


def test_verify_missing_reference(tmp_path: Path):
    result = verify_canonical_hashes(tmp_path / "missing.json", root=tmp_path)
    assert result["status"] == "unavailable"
    assert result["passed"] is False


def test_neutral_zero_check():
    style = [1.0, 2.0, 3.0]
    assert neutral_zero_error(style, [0.0, 0.0, 0.0]) == 0.0
    assert neutral_zero_check(style, [0.0, 0.0, 0.0])["passed"] is True
    assert neutral_zero_check(style, [0.0, 1e-4, 0.0])["passed"] is False
    with pytest.raises(ValueError):
        neutral_zero_error(style, [0.0, 0.0])


def test_cfvn_drift_pass_and_fail():
    passing = evaluate_cfvn_drift(_cfvn_metrics())
    assert passing["passed"] is True
    failing = evaluate_cfvn_drift({"recall_at_1": 0.10, "recall_at_10": 0.4068})
    assert failing["passed"] is False
    assert "recall_at_1" in failing["failures"][0]


def test_guardrail_passes_with_identity_and_hashes(tmp_path: Path):
    path = _write_fake_canonical(tmp_path)
    result = evaluate_identity_guardrail(
        _rows(),
        cfvn_metrics=_cfvn_metrics(),
        canonical_hashes_path=path,
        root=tmp_path,
        n_resamples=200,
        seed=1,
    )
    assert result["passed"] is True
    assert result["conditions"]["happy"]["passed"] is True
    assert result["cfvn"]["passed"] is True
    assert result["canonical_hashes"]["passed"] is True


def test_guardrail_fails_on_identity_drop(tmp_path: Path):
    path = _write_fake_canonical(tmp_path)
    result = evaluate_identity_guardrail(
        _rows(neutral=0.70),
        cfvn_metrics=_cfvn_metrics(),
        canonical_hashes_path=path,
        root=tmp_path,
        n_resamples=200,
    )
    assert result["passed"] is False
    assert any("ECAPA neutral" in failure for failure in result["failures"])


def test_guardrail_fails_when_measurements_are_missing(tmp_path: Path):
    path = _write_fake_canonical(tmp_path)
    rows = [{"condition": "happy", "speaker_id": "a"}]
    result = evaluate_identity_guardrail(
        rows,
        cfvn_metrics=_cfvn_metrics(),
        canonical_hashes_path=path,
        root=tmp_path,
        n_resamples=50,
    )
    assert result["passed"] is False
    assert any("unavailable" in failure for failure in result["failures"])


def test_guardrail_fails_without_cfvn_or_hashes():
    result = evaluate_identity_guardrail(_rows(), n_resamples=100)
    assert result["passed"] is False
    assert result["cfvn"]["status"] == "unavailable"
    assert result["canonical_hashes"]["status"] == "unavailable"


def test_guardrail_can_mark_cfvn_not_applicable(tmp_path: Path):
    path = _write_fake_canonical(tmp_path)
    result = evaluate_identity_guardrail(
        _rows(),
        require_cfvn=False,
        canonical_hashes_path=path,
        root=tmp_path,
        n_resamples=100,
    )
    assert result["cfvn"]["status"] == "not_applicable"
    assert result["cfvn"]["passed"] is True
    assert result["passed"] is True


def test_thresholds_are_serializable():
    thresholds = GuardrailThresholds(min_ecapa_neutral=0.75)
    assert thresholds.to_dict()["min_ecapa_neutral"] == 0.75


def _neutral_and_emotion_rows(
    *,
    neutral_reference: float = 0.60,
    emotion_reference: float = 0.55,
    emotion_neutral: float = 0.90,
):
    rows = []
    for index, speaker in enumerate(["a", "b", "c"]):
        rows.append(
            {
                "condition": "neutral",
                "speaker_id": speaker,
                "ecapa_neutral": 1.0,
                "ecapa_reference": neutral_reference + index * 0.001,
            }
        )
        rows.append(
            {
                "condition": "happy",
                "speaker_id": speaker,
                "ecapa_neutral": emotion_neutral + index * 0.001,
                "ecapa_reference": emotion_reference + index * 0.001,
            }
        )
    return rows


def test_neutral_condition_is_the_reference_baseline(tmp_path: Path):
    path = _write_fake_canonical(tmp_path)
    result = evaluate_identity_guardrail(
        _neutral_and_emotion_rows(),
        require_cfvn=False,
        canonical_hashes_path=path,
        root=tmp_path,
        n_resamples=100,
    )
    neutral = result["conditions"]["neutral"]
    assert neutral["reference_drop"]["status"] == "not_applicable"
    assert neutral["passed"] is True
    happy = result["conditions"]["happy"]
    assert happy["reference_drop"]["status"] == "computed"
    assert happy["reference_drop"]["mean"] == pytest.approx(0.05, abs=1e-9)
    assert happy["passed"] is True


def test_reference_drop_uses_the_neutral_baseline(tmp_path: Path):
    path = _write_fake_canonical(tmp_path)
    result = evaluate_identity_guardrail(
        _neutral_and_emotion_rows(neutral_reference=0.60, emotion_reference=0.45),
        require_cfvn=False,
        canonical_hashes_path=path,
        root=tmp_path,
        n_resamples=100,
    )
    assert result["conditions"]["happy"]["passed"] is False
    assert any("reference drop" in failure for failure in result["failures"])


def test_neutral_sanity_warns_on_domain_shift(tmp_path: Path):
    path = _write_fake_canonical(tmp_path)
    result = evaluate_identity_guardrail(
        _neutral_and_emotion_rows(neutral_reference=0.20, emotion_reference=0.18),
        thresholds=GuardrailThresholds(min_neutral_reference=0.50),
        require_cfvn=False,
        canonical_hashes_path=path,
        root=tmp_path,
        n_resamples=100,
    )
    assert result["neutral_sanity"]["status"] == "computed"
    assert result["neutral_sanity"]["passed"] is False
    assert any("relative only" in warning for warning in result["warnings"])
    # A domain-shift warning does not by itself discard a relative comparison.
    assert result["passed"] is True


def test_neutral_sanity_can_be_disabled(tmp_path: Path):
    path = _write_fake_canonical(tmp_path)
    result = evaluate_identity_guardrail(
        _neutral_and_emotion_rows(neutral_reference=0.20, emotion_reference=0.18),
        thresholds=GuardrailThresholds(min_neutral_reference=None),
        require_cfvn=False,
        canonical_hashes_path=path,
        root=tmp_path,
        n_resamples=100,
    )
    assert result["neutral_sanity"]["status"] == "not_applicable"
    assert result["warnings"] == []
