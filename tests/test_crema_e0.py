"""Tests for the CREMA-D E0 metric aggregation."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from voix.evaluation.crema_e0 import (
    aggregate_conditions,
    classification_metrics,
    condition_gate_summary,
    feature_domain_shift,
    identity_rows,
    paired_delta_effects,
)


def _frame(*, actors=("a", "b", "c")) -> pd.DataFrame:
    rows = []
    for actor in actors:
        for sentence in ("S1", "S2"):
            rows.append(
                {
                    "condition": "neutral",
                    "emotion": "neutral",
                    "actor_id": actor,
                    "sentence": sentence,
                    "seed": 0,
                    "f0_mean_hz": 100.0,
                    "f0_std_hz": 20.0,
                    "duration_seconds": 2.0,
                    "energy_mean": 0.10,
                    "energy_std": 0.02,
                    "ecapa_generated_neutral": 1.0,
                    "ecapa_identity_reference": 0.30,
                }
            )
            for emotion, f0, duration, energy in (
                ("happy", 130.0, 1.8, 0.12),
                ("sad", 80.0, 2.4, 0.08),
                ("angry", 140.0, 1.7, 0.14),
            ):
                rows.append(
                    {
                        "condition": "fixed_delta",
                        "emotion": emotion,
                        "actor_id": actor,
                        "sentence": sentence,
                        "seed": 0,
                        "f0_mean_hz": f0,
                        "f0_std_hz": 22.0,
                        "duration_seconds": duration,
                        "energy_mean": energy,
                        "energy_std": 0.03,
                        "ecapa_generated_neutral": 0.85,
                        "ecapa_identity_reference": 0.28,
                    }
                )
    return pd.DataFrame(rows)


def test_aggregate_conditions_reports_mean_and_speaker_cluster_ci():
    result = aggregate_conditions(
        _frame(),
        metrics=("f0_mean_hz", "duration_seconds"),
        n_resamples=200,
        seed=1,
    )
    neutral = result["neutral:neutral"]
    assert neutral["n"] == 6
    assert neutral["actors"] == 3
    assert neutral["metrics"]["f0_mean_hz"]["mean"] == pytest.approx(100.0)
    assert neutral["metrics"]["f0_mean_hz"]["ci_status"] == "computed"
    assert neutral["metrics"]["f0_mean_hz"]["ci_low"] <= 100.0
    assert neutral["metrics"]["f0_mean_hz"]["ci_high"] >= 100.0
    assert result["fixed_delta:happy"]["metrics"]["f0_mean_hz"]["mean"] == pytest.approx(130.0)


def test_aggregate_conditions_marks_missing_metric_unavailable():
    result = aggregate_conditions(_frame(), metrics=("not_a_column",), n_resamples=50)
    metric = result["neutral:neutral"]["metrics"]["not_a_column"]
    assert metric["status"] == "unavailable"
    assert "missing" in metric["reason"]


def test_aggregate_conditions_marks_single_speaker_ci_unavailable():
    result = aggregate_conditions(
        _frame(actors=("a",)),
        metrics=("f0_mean_hz",),
        n_resamples=50,
    )
    metric = result["neutral:neutral"]["metrics"]["f0_mean_hz"]
    assert metric["status"] == "computed"
    assert metric["ci_status"] == "unavailable"
    assert metric["ci_reason"] == "fewer than two speakers"


def test_aggregate_conditions_requires_core_columns():
    with pytest.raises(ValueError, match="missing columns"):
        aggregate_conditions(pd.DataFrame({"condition": ["neutral"]}), metrics=("f0_mean_hz",))


def test_paired_delta_effects_uses_generated_neutral():
    result = paired_delta_effects(
        _frame(),
        metrics=("f0_mean_hz", "duration_seconds"),
        n_resamples=200,
        seed=2,
    )
    assert result["status"] == "computed"
    happy_f0 = result["effects"]["fixed_delta:happy"]["f0_mean_hz"]
    happy_duration = result["effects"]["fixed_delta:happy"]["duration_seconds"]
    assert happy_f0["mean_delta"] == pytest.approx(30.0)
    assert happy_f0["direction_match"] is True
    assert happy_f0["ci_low"] > 0.0
    assert happy_duration["mean_delta"] == pytest.approx(-0.2)
    assert happy_duration["direction_match"] is True
    sad_f0 = result["effects"]["fixed_delta:sad"]["f0_mean_hz"]
    assert sad_f0["mean_delta"] == pytest.approx(-20.0)
    assert sad_f0["ci_high"] < 0.0


def test_paired_delta_effects_flags_wrong_direction():
    frame = _frame()
    frame.loc[
        (frame["condition"] == "fixed_delta") & (frame["emotion"] == "happy"),
        "f0_mean_hz",
    ] = 90.0
    result = paired_delta_effects(frame, metrics=("f0_mean_hz",), n_resamples=100)
    effect = result["effects"]["fixed_delta:happy"]["f0_mean_hz"]
    assert effect["direction_match"] is False
    assert effect["ci_excludes_zero_in_expected_direction"] is False


def test_paired_delta_effects_without_neutral_is_unavailable():
    frame = _frame()
    result = paired_delta_effects(
        frame[frame["condition"] != "neutral"],
        metrics=("f0_mean_hz",),
        n_resamples=50,
    )
    assert result["status"] == "unavailable"


def test_condition_gate_passes_when_directions_and_cis_hold():
    effects = paired_delta_effects(
        _frame(),
        metrics=("f0_mean_hz", "duration_seconds"),
        n_resamples=200,
    )
    gate = condition_gate_summary(effects)
    assert gate["status"] == "computed"
    assert gate["passed"] is True
    assert gate["checks"]["happy"]["f0_mean_hz"]["passed"] is True


def test_condition_gate_infers_non_fixed_condition_prefix():
    effects = paired_delta_effects(
        _frame(),
        metrics=("f0_mean_hz", "duration_seconds"),
        n_resamples=200,
    )
    renamed = {
        "status": effects["status"],
        "effects": {
            key.replace("fixed_delta:", "class_mean_prosody:"): value
            for key, value in effects["effects"].items()
        },
    }
    gate = condition_gate_summary(renamed)
    assert gate["status"] == "computed"
    assert gate["passed"] is True
    assert gate["checks"]["happy"]["f0_mean_hz"]["status"] == "computed"


def test_condition_gate_fails_when_ci_includes_zero():
    effects = paired_delta_effects(
        _frame(actors=("a",)),
        metrics=("f0_mean_hz", "duration_seconds"),
        n_resamples=50,
    )
    gate = condition_gate_summary(effects)
    assert gate["passed"] is False
    assert any("CI includes zero" in failure for failure in gate["failures"])


def test_classification_metrics_computes_per_class_and_cluster_ci():
    result = classification_metrics(
        ["neutral", "happy", "sad", "neutral", "happy", "sad"],
        ["neutral", "happy", "angry", "neutral", "sad", "sad"],
        ["a", "a", "a", "b", "b", "b"],
        n_resamples=200,
        seed=3,
    )
    assert result["status"] == "computed"
    assert result["n"] == 6
    assert result["accuracy"] == pytest.approx(4 / 6)
    assert result["per_class"]["neutral"]["recall"] == pytest.approx(1.0)
    assert result["per_class"]["happy"]["recall"] == pytest.approx(0.5)
    assert result["per_class"]["angry"]["recall"] == pytest.approx(0.0)
    assert result["speaker_cluster_ci"]["status"] == "computed"
    assert result["confusion_matrix"]["labels"] == ["neutral", "happy", "sad", "angry"]


def test_classification_metrics_empty_is_unavailable():
    result = classification_metrics([], [], [])
    assert result["status"] == "unavailable"


def test_identity_rows_use_neutral_and_emotion_labels():
    frame = pd.DataFrame(
        [
            {
                "condition": "neutral",
                "emotion": "neutral",
                "actor_id": "a",
                "ecapa_generated_neutral": 1.0,
                "ecapa_identity_reference": 0.3,
            },
            {
                "condition": "fixed_delta",
                "emotion": "happy",
                "actor_id": "a",
                "ecapa_generated_neutral": 0.85,
                "ecapa_identity_reference": 0.28,
            },
            {
                "condition": "fixed_delta",
                "emotion": "sad",
                "actor_id": "a",
                "ecapa_generated_neutral": None,
                "ecapa_identity_reference": None,
            },
        ]
    )
    rows = identity_rows(frame)
    assert len(rows) == 2
    assert rows[0] == {
        "condition": "neutral",
        "speaker_id": "a",
        "ecapa_neutral": 1.0,
        "ecapa_reference": 0.3,
    }
    assert rows[1]["condition"] == "happy"
    assert rows[1]["ecapa_neutral"] == pytest.approx(0.85)


def test_non_finite_metrics_are_ignored_in_aggregation():
    frame = _frame(actors=("a", "b"))
    frame.loc[0, "f0_mean_hz"] = math.nan
    result = aggregate_conditions(frame, metrics=("f0_mean_hz",), n_resamples=50)
    metric = result["neutral:neutral"]["metrics"]["f0_mean_hz"]
    assert metric["status"] == "computed"
    assert metric["n"] == 3


def test_feature_domain_shift_accepts_matching_distributions():
    rng = np.random.default_rng(0)
    train = rng.normal(size=(200, 8))
    real = rng.normal(size=(50, 8))
    generated = rng.normal(size=(50, 8))
    result = feature_domain_shift(train, real, generated)
    assert result["status"] == "computed"
    assert result["valid"] is True
    assert result["ratio"] < 3.0


def test_feature_domain_shift_rejects_shifted_generated_features():
    rng = np.random.default_rng(0)
    train = rng.normal(size=(200, 8))
    real = rng.normal(size=(50, 8))
    generated = rng.normal(size=(50, 8)) + 10.0
    result = feature_domain_shift(train, real, generated)
    assert result["status"] == "computed"
    assert result["valid"] is False
    assert result["ratio"] > 3.0
    assert "not a valid emotion measure" in result["reason"]


def test_feature_domain_shift_handles_missing_rows():
    result = feature_domain_shift(np.ones((1, 3)), np.ones((1, 3)), np.ones((1, 3)))
    assert result["status"] == "unavailable"
