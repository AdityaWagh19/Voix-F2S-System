"""Tests for class-mean emotion residuals."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from voix.models.emotion_residual import (
    PROSODY_START,
    STYLE_DIM,
    apply_style_residual,
    build_pairs,
    class_mean_residuals,
    load_artifact,
    residual_from_artifact,
    save_artifact,
    select_one_per_group,
    shuffled_residuals,
)


def _clips() -> pd.DataFrame:
    rows = []
    for actor in ("a", "b"):
        for sentence in ("S1", "S2"):
            rows.append(
                {
                    "actor_id": actor,
                    "sentence": sentence,
                    "intended_emotion": "neutral",
                    "intensity": "unspecified",
                    "clip_id": f"{actor}_{sentence}_NEU",
                    "path": f"{actor}_{sentence}_NEU.wav",
                }
            )
            for emotion in ("happy", "sad"):
                rows.append(
                    {
                        "actor_id": actor,
                        "sentence": sentence,
                        "intended_emotion": emotion,
                        "intensity": "unspecified",
                        "clip_id": f"{actor}_{sentence}_{emotion}",
                        "path": f"{actor}_{sentence}_{emotion}.wav",
                    }
                )
    return pd.DataFrame(rows)


def _vector(first: float, second: float = 0.0) -> np.ndarray:
    vector = np.zeros(STYLE_DIM, dtype=np.float64)
    vector[0] = first
    vector[1] = second
    return vector


def _records() -> list[dict]:
    return [
        {"emotion": "happy", "speaker_id": "a", "delta": _vector(2.0)},
        {"emotion": "happy", "speaker_id": "b", "delta": _vector(0.0, 2.0)},
        {"emotion": "happy", "speaker_id": "b", "delta": _vector(0.0, 2.0)},
        {"emotion": "happy", "speaker_id": "b", "delta": _vector(0.0, 2.0)},
        {"emotion": "sad", "speaker_id": "a", "delta": _vector(-1.0)},
    ]


def test_select_one_per_group_prefers_unspecified_then_high():
    frame = pd.DataFrame(
        {
            "actor_id": ["a"] * 4,
            "sentence": ["S1"] * 4,
            "intended_emotion": ["happy"] * 4,
            "intensity": ["low", "high", "medium", "unspecified"],
            "clip_id": ["low", "high", "medium", "natural"],
            "path": ["low.wav", "high.wav", "medium.wav", "natural.wav"],
        }
    )
    selected = select_one_per_group(frame)
    assert selected["clip_id"].tolist() == ["natural"]


def test_select_one_per_group_is_deterministic_on_ties():
    frame = pd.DataFrame(
        {
            "actor_id": ["a"] * 2,
            "sentence": ["S1"] * 2,
            "intended_emotion": ["happy"] * 2,
            "intensity": ["unspecified", "unspecified"],
            "clip_id": ["b", "a"],
            "path": ["b.wav", "a.wav"],
        }
    )
    selected = select_one_per_group(frame)
    assert selected["clip_id"].tolist() == ["a"]


def test_select_one_per_group_requires_columns():
    with pytest.raises(ValueError, match="missing columns"):
        select_one_per_group(pd.DataFrame({"actor_id": ["a"]}))


def test_build_pairs_joins_neutral_by_actor_and_sentence():
    pairs = build_pairs(_clips())
    assert len(pairs) == 8
    assert set(pairs["emotion"]) == {"happy", "sad"}
    happy = pairs[(pairs["actor_id"] == "a") & (pairs["sentence"] == "S1")]
    happy = happy[happy["emotion"] == "happy"].iloc[0]
    assert happy["neutral_clip_id"] == "a_S1_NEU"
    assert happy["emotion_clip_id"] == "a_S1_happy"


def test_build_pairs_rejects_duplicate_neutral_clips():
    clips = _clips()
    duplicate = clips.iloc[[0]].copy()
    duplicate["clip_id"] = "duplicate"
    with pytest.raises(ValueError, match="not unique"):
        build_pairs(pd.concat([clips, duplicate], ignore_index=True))


def test_class_mean_is_speaker_balanced():
    results = class_mean_residuals(_records(), prosody_only=False, n_resamples=50)
    happy = results["happy"]
    assert happy.n_pairs == 4
    assert happy.n_speakers == 2
    # Speaker a contributes [2, 0], speaker b contributes [0, 2].
    assert happy.delta[0].item() == pytest.approx(1.0)
    assert happy.delta[1].item() == pytest.approx(1.0)
    assert happy.delta.norm().item() == pytest.approx(np.sqrt(2.0), abs=1e-6)
    assert happy.speaker_norm_ci_low is not None


def test_class_mean_prosody_only_zeroes_timbre_half():
    records = _records()
    results = class_mean_residuals(records, prosody_only=True, n_resamples=50)
    delta = results["happy"].delta_prosody_only
    assert torch.count_nonzero(delta[:PROSODY_START]) == 0
    assert torch.count_nonzero(delta[PROSODY_START:]) == 0
    # The fixture only has energy in the timbre half, so the prosody-only
    # variant is exactly zero, not merely small.
    assert float(delta.norm().item()) == 0.0


def test_class_mean_rejects_wrong_width():
    with pytest.raises(ValueError, match="expected"):
        class_mean_residuals(
            [{"emotion": "happy", "speaker_id": "a", "delta": np.zeros(10)}],
            n_resamples=10,
        )


def test_apply_style_residual_is_exact_neutral_zero():
    style = torch.ones(1, STYLE_DIM)
    residual = torch.ones(STYLE_DIM)
    assert apply_style_residual(style, residual, intensity=0.0) is style
    assert apply_style_residual(style, torch.zeros(STYLE_DIM), intensity=1.0) is style


def test_apply_style_residual_applies_scale_and_clamps_norm():
    style = torch.zeros(1, STYLE_DIM)
    residual = torch.zeros(STYLE_DIM)
    residual[0] = 2.0
    result = apply_style_residual(style, residual, intensity=1.0, scale=0.28)
    assert result[0, 0].item() == pytest.approx(0.56)
    clamped = apply_style_residual(style, residual, intensity=1.0, scale=1.0, max_norm=0.5)
    assert clamped.norm().item() == pytest.approx(0.5)


def test_apply_style_residual_preserves_shape():
    style = torch.zeros(2, STYLE_DIM)
    residual = torch.ones(STYLE_DIM)
    result = apply_style_residual(style, residual, intensity=0.5)
    assert result.shape == (2, STYLE_DIM)


def test_shuffled_residuals_are_deterministic_and_change_labels():
    first = shuffled_residuals(_records(), seed=3, prosody_only=False)
    second = shuffled_residuals(_records(), seed=3, prosody_only=False)
    assert set(first) == set(second)
    for emotion in first:
        assert torch.equal(first[emotion].delta, second[emotion].delta)


def test_save_and_load_artifact_roundtrip(tmp_path):
    results = class_mean_residuals(_records(), prosody_only=False, n_resamples=50)
    path = tmp_path / "class_mean.pt"
    save_artifact(path, results, metadata={"split": "train"})
    payload = load_artifact(path)
    assert payload["metadata"]["split"] == "train"
    full = residual_from_artifact(payload, "happy", variant="full")
    prosody = residual_from_artifact(payload, "happy", variant="prosody_only")
    assert full.shape == (STYLE_DIM,)
    assert prosody.shape == (STYLE_DIM,)
    with pytest.raises(KeyError):
        residual_from_artifact(payload, "fear")
