"""Tests for the Phase 4 CI gate and negative controls.

A deliberately collapsed or missing-artifact report must fail.  A healthy
synthetic report must pass.  The controls are also exercised directly so a
future change cannot quietly make them score above chance.
"""
from __future__ import annotations

import numpy as np
import torch

from voix.evaluation.gate import GateThresholds, evaluate_gate
from voix.evaluation.retrieval import (
    constant_control,
    embedding_health,
    shuffled_control,
    speaker_retrieval,
)


def _healthy_report() -> dict:
    return {
        "artifacts": {"missing": []},
        "retrieval": {
            "cfvn": {
                "recall_at_1": 0.22,
                "chance_at_1": 0.01,
                "shuffled_recall_at_1": 0.012,
                "constant_recall_at_1": 0.01,
            }
        },
        "health": {
            "cfvn": {
                "within_speaker_cosine": 0.55,
                "between_speaker_cosine": 0.10,
                "collapse_ratio": 1.0,
            }
        },
        "cvae": {
            "decoder_pairwise_cosine": 0.30,
            "face_margin": 0.20,
            "active_units": 32,
            "prior_pairwise_cosine": 0.10,
        },
        "adapter": {
            "output_pairwise_cosine": 0.60,
            "output_per_dim_std": 0.05,
            "output_target_cosine": 0.90,
        },
    }


def test_healthy_report_passes():
    result = evaluate_gate(_healthy_report())
    assert result.passed, [check.message for check in result.failures]


def test_collapsed_embedding_fails():
    report = _healthy_report()
    report["retrieval"]["cfvn"]["recall_at_1"] = 0.01
    report["health"]["cfvn"].update(
        {
            "within_speaker_cosine": 0.9999,
            "between_speaker_cosine": 0.9998,
            "collapse_ratio": 0.01,
        }
    )
    result = evaluate_gate(report)
    assert not result.passed
    failed = {check.name for check in result.failures}
    assert "cfvn_recall_at_1" in failed
    assert "between_speaker_cosine" in failed
    assert "collapse_ratio" in failed
    assert "within_between_margin" in failed


def test_missing_artifact_fails_even_with_good_metrics():
    report = _healthy_report()
    report["artifacts"]["missing"] = ["cfvn"]
    result = evaluate_gate(report)
    assert not result.passed
    assert "artifacts_present" in {check.name for check in result.failures}


def test_missing_metric_fails_instead_of_skipping():
    report = _healthy_report()
    del report["cvae"]["active_units"]
    result = evaluate_gate(report)
    assert not result.passed
    assert "cvae_active_units" in {check.name for check in result.failures}


def test_adapter_that_destroys_the_style_mean_fails():
    report = _healthy_report()
    report["adapter"].update(
        {
            "output_pairwise_cosine": 0.20,
            "output_per_dim_std": 0.04,
            "output_target_cosine": 0.48,
        }
    )
    result = evaluate_gate(report)
    assert not result.passed
    assert "adapter_target_cosine" in {
        check.name for check in result.failures
    }


def test_missing_adapter_target_cosine_fails():
    report = _healthy_report()
    del report["adapter"]["output_target_cosine"]
    result = evaluate_gate(report)
    assert not result.passed
    assert "adapter_target_cosine" in {
        check.name for check in result.failures
    }


def test_high_negative_controls_fail():
    report = _healthy_report()
    report["retrieval"]["cfvn"]["shuffled_recall_at_1"] = 0.50
    report["retrieval"]["cfvn"]["constant_recall_at_1"] = 0.50
    result = evaluate_gate(report)
    failed = {check.name for check in result.failures}
    assert "shuffled_control_recall_at_1" in failed
    assert "constant_control_recall_at_1" in failed


def test_controls_are_near_chance_on_random_embeddings():
    rng = np.random.default_rng(0)
    n_speakers, clips, dim = 30, 4, 32
    speakers = np.repeat(np.arange(n_speakers), clips)
    features = torch.from_numpy(
        rng.normal(size=(len(speakers), dim)).astype(np.float32)
    )
    query = torch.stack(
        [features[i * clips] for i in range(n_speakers)]
    )
    gallery = torch.cat(
        [features[i * clips + 1 : (i + 1) * clips] for i in range(n_speakers)]
    )
    query_speakers = [f"s{i}" for i in range(n_speakers)]
    gallery_speakers = [
        f"s{i}" for i in range(n_speakers) for _ in range(clips - 1)
    ]

    result = speaker_retrieval(
        query, query_speakers, gallery, gallery_speakers
    )
    shuffled = shuffled_control(
        query, query_speakers, gallery, gallery_speakers
    )
    constant = constant_control(gallery, gallery_speakers)
    chance = result["chance_at_1"]
    assert result["recall_at_1"] < 5 * chance
    assert shuffled["recall_at_1"] < 5 * chance
    assert constant["recall_at_1"] < 5 * chance


def test_embedding_health_detects_constant_embeddings():
    features = torch.ones(20, 16)
    speakers = [f"s{i // 2}" for i in range(20)]
    health = embedding_health(features, speakers)
    assert health["collapse_ratio"] < 0.01
    assert abs(health["between_speaker_cosine"] - 1.0) < 1e-5


def test_thresholds_are_configurable():
    report = _healthy_report()
    result = evaluate_gate(
        report, GateThresholds(min_recall_at_1=0.5)
    )
    assert not result.passed
    assert "cfvn_recall_at_1" in {c.name for c in result.failures}
