"""Tests for the pilot CREMA-D emotion classifier.

These pin the parts that are easy to get quietly wrong: the smoke cap must not
empty the validation/test splits, and the final report must not present a
train+val refit score as held-out validation.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "training" / "train_emotion_classifier.py"

spec = importlib.util.spec_from_file_location("train_emotion_classifier", SCRIPT)
assert spec and spec.loader
emotion = importlib.util.module_from_spec(spec)
spec.loader.exec_module(emotion)

CLASSES = ["neutral", "happy", "sad", "angry", "fear", "disgust"]


def _manifest() -> pd.DataFrame:
    rows = []
    for split in ("train", "val", "test"):
        for emotion_index, label in enumerate(CLASSES):
            for index in range(3):
                rows.append(
                    {
                        "path": f"data/fake/{split}_{label}_{index}.wav",
                        "intended_emotion": label,
                        "split": split,
                        "actor_id": f"{split}_{emotion_index}_{index}",
                        "duration_sec": 2.0,
                    }
                )
    return pd.DataFrame(rows)


def test_cap_keeps_every_split(monkeypatch):
    manifest = _manifest()
    monkeypatch.setattr(
        emotion, "extract_features", lambda path, **kwargs: np.zeros(8, dtype=np.float32)
    )

    matrix, frame = emotion.build_feature_matrix(manifest, workers=2, max_per_class=1)

    assert matrix.shape == (18, 8)
    assert set(frame["split"]) == {"train", "val", "test"}
    assert set(frame["intended_emotion"]) == set(CLASSES)
    assert frame.groupby(["split", "intended_emotion"]).size().max() == 1


def test_train_classifier_reports_test_only_and_beats_chance():
    rng = np.random.default_rng(0)
    manifest = _manifest()
    centroids = rng.normal(size=(len(CLASSES), 8))
    features = np.stack(
        [
            centroids[CLASSES.index(row.intended_emotion)]
            + 0.05 * rng.normal(size=8)
            for row in manifest.itertuples()
        ]
    )

    model, report, _ = emotion.train_classifier(
        manifest,
        features,
        grid=[{"C": 1.0, "gamma": "scale"}],
    )

    assert "val" not in report
    assert report["test"]["n"] == 18
    assert report["test"]["accuracy"] >= 0.95
    assert report["split_speakers"] == {"train": 18, "val": 18, "test": 18}
    assert model.predict(features[:3]).shape == (3,)


def test_bootstrap_is_speaker_clustered_and_bounded():
    y_true = np.array(["a", "a", "b", "b", "c", "c"])
    y_pred = np.array(["a", "b", "b", "b", "c", "a"])
    groups = np.array(["s1", "s1", "s2", "s2", "s3", "s3"])

    result = emotion._bootstrap_speaker_ci(y_true, y_pred, groups, iterations=200)

    assert result["status"] == "computed"
    assert result["speakers"] == 3
    low, high = result["accuracy_95ci"]
    assert 0.0 <= low <= high <= 1.0


def test_majority_baseline_is_measured():
    train = np.array(["a", "a", "b"])
    test = np.array(["a", "b", "b"])
    assert emotion._baseline_accuracy(train, test) == 0.3333
