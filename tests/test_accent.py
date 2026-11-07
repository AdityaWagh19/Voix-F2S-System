"""Tests for accent metrics and diagnostics."""
from __future__ import annotations

import numpy as np
import pytest

from voix.evaluation.accent import (
    classifier_metrics,
    majority_baseline,
    nearest_centroid_speaker_probe,
    speaker_centered_diagnostic,
)

CLASSES = ("us", "england", "australian", "canadian", "india_south_asia")


def test_classifier_metrics_uses_fixed_label_order_and_cluster_ci():
    result = classifier_metrics(
        ["us", "us", "england", "england"],
        ["us", "england", "england", "england"],
        ["a", "a", "b", "b"],
        labels=CLASSES,
        n_resamples=100,
    )
    assert result["status"] == "computed"
    assert result["accuracy"] == pytest.approx(0.75)
    assert result["confusion_matrix"]["labels"] == list(CLASSES)
    assert result["per_class"]["us"]["recall"] == pytest.approx(0.5)
    assert result["per_class"]["australian"]["recall"] == pytest.approx(0.0)
    assert result["speaker_cluster_ci"]["status"] == "computed"


def test_classifier_metrics_empty_is_unavailable():
    assert classifier_metrics([], [], [], labels=CLASSES)["status"] == "unavailable"


def test_majority_baseline_breaks_ties_deterministically():
    result = majority_baseline(["us", "england"], ["us", "australian"])
    assert result["label"] == "england"
    assert result["accuracy"] == pytest.approx(0.0)
    assert result["predictions"] == ["england", "england"]


def _speaker_clusters() -> tuple[np.ndarray, list[str]]:
    rng = np.random.default_rng(0)
    features = []
    speakers = []
    for speaker_index, speaker in enumerate(("a", "b", "c")):
        center = np.zeros(8)
        center[speaker_index] = 10.0
        for _ in range(5):
            features.append(center + rng.normal(scale=0.1, size=8))
            speakers.append(speaker)
    return np.stack(features), speakers


def test_nearest_centroid_probe_recovers_separable_speakers():
    features, speakers = _speaker_clusters()
    result = nearest_centroid_speaker_probe(features, speakers, seed=1)
    assert result["status"] == "computed"
    assert result["n_speakers"] == 3
    assert result["accuracy"] == pytest.approx(1.0)
    assert result["chance"] == pytest.approx(1 / 3)


def test_nearest_centroid_probe_reports_single_clip_speakers_unavailable():
    features = np.ones((2, 4))
    result = nearest_centroid_speaker_probe(features, ["a", "b"])
    assert result["status"] == "unavailable"


def test_speaker_centered_diagnostic_returns_metrics():
    rng = np.random.default_rng(2)
    features = []
    speakers = []
    labels = []
    for speaker_index, speaker in enumerate(("a", "b", "c", "d")):
        for label in ("us", "england"):
            center = np.zeros(6)
            center[speaker_index] = 5.0
            center[4 if label == "us" else 5] = 1.0
            for _ in range(4):
                features.append(center + rng.normal(scale=0.05, size=6))
                speakers.append(speaker)
                labels.append(label)
    features = np.stack(features)
    speakers = np.asarray(speakers)
    labels = np.asarray(labels)
    train_mask = np.isin(speakers, ["a", "b"])
    val_mask = np.isin(speakers, ["c", "d"])
    result = speaker_centered_diagnostic(
        features,
        speakers,
        labels,
        train_mask,
        val_mask,
        classes=("us", "england"),
        n_resamples=50,
    )
    assert result["status"] == "computed"
    assert result["n_speakers"] == 2
    assert "transductive diagnostic" in result["note"]
