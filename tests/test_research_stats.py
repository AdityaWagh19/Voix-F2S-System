"""Tests for the research statistics utilities."""
from __future__ import annotations

import math

import pytest

from voix.evaluation.stats import (
    bootstrap_mean_ci,
    holm_bonferroni,
    paired_bootstrap_difference,
    paired_effect_size,
    speaker_cluster_bootstrap,
)


def test_bootstrap_mean_ci_is_deterministic_and_contains_estimate():
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    first = bootstrap_mean_ci(values, n_resamples=2000, seed=7)
    second = bootstrap_mean_ci(values, n_resamples=2000, seed=7)
    assert first.to_dict() == second.to_dict()
    assert first.estimate == pytest.approx(3.0)
    assert first.n == 5
    assert first.ci_low <= first.estimate <= first.ci_high


def test_bootstrap_mean_ci_rejects_empty_and_non_finite():
    with pytest.raises(ValueError):
        bootstrap_mean_ci([])
    with pytest.raises(ValueError):
        bootstrap_mean_ci([math.nan, math.inf])


def test_paired_bootstrap_difference_drops_incomplete_pairs():
    result = paired_bootstrap_difference(
        [1.0, 2.0, math.nan, 4.0],
        [0.0, 0.0, 0.0, 1.0],
        n_resamples=1000,
        seed=3,
    )
    assert result.n == 3
    assert result.estimate == pytest.approx(2.0)
    assert result.ci_low <= result.estimate <= result.ci_high


def test_paired_bootstrap_difference_requires_equal_lengths():
    with pytest.raises(ValueError):
        paired_bootstrap_difference([1.0, 2.0], [1.0])


def test_paired_effect_size():
    result = paired_effect_size([2.0, 4.0, 6.0], [1.0, 1.0, 1.0])
    assert result["n"] == 3
    assert result["mean_difference"] == pytest.approx(3.0)
    assert result["cohens_dz"] == pytest.approx(1.5)


def test_speaker_cluster_bootstrap_resamples_speakers():
    rows = [
        {"speaker_id": "a", "value": 1.0},
        {"speaker_id": "a", "value": 3.0},
        {"speaker_id": "b", "value": 5.0},
        {"speaker_id": "b", "value": 7.0},
    ]

    def mean_value(selected):
        return sum(float(row["value"]) for row in selected) / len(selected)

    result = speaker_cluster_bootstrap(rows, mean_value, n_resamples=500, seed=11)
    assert result.estimate == pytest.approx(4.0)
    assert result.n == 2
    assert result.ci_low <= result.estimate <= result.ci_high


def test_speaker_cluster_bootstrap_requires_two_speakers():
    with pytest.raises(ValueError):
        speaker_cluster_bootstrap(
            [{"speaker_id": "a", "value": 1.0}],
            lambda rows: float(rows[0]["value"]),
            n_resamples=10,
        )


def test_holm_bonferroni_step_down():
    result = holm_bonferroni({"a": 0.01, "b": 0.04, "c": 0.03}, alpha=0.05)
    assert result["a"]["adjusted_p_value"] == pytest.approx(0.03)
    assert result["c"]["adjusted_p_value"] == pytest.approx(0.06)
    assert result["b"]["adjusted_p_value"] == pytest.approx(0.06)
    assert result["a"]["rejected"] is True
    assert result["b"]["rejected"] is False
    assert result["c"]["rejected"] is False


def test_holm_bonferroni_rejects_invalid_pvalues():
    with pytest.raises(ValueError):
        holm_bonferroni({"a": 1.5})
    with pytest.raises(ValueError):
        holm_bonferroni({"a": math.nan})
    with pytest.raises(ValueError):
        holm_bonferroni({})
