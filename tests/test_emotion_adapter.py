"""Tests for the E2 learned emotion residual adapter."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from voix.models.emotion_adapter import (
    EMOTION_TO_INDEX,
    EMOTIONS,
    NON_NEUTRAL_EMOTIONS,
    STYLE_DIM,
    EmotionResidualAdapter,
    apply_learned_residual,
    build_examples,
    emotion_one_hot,
    load_adapter,
    predict_residual,
    residual_metrics,
    save_adapter,
    shuffle_labels,
    shuffle_style_assignments,
    speaker_disjoint_split,
    train_adapter,
)


def test_neutral_and_zero_intensity_are_exact_noops():
    model = EmotionResidualAdapter(hidden_dim=32, max_norm=0.5).eval()
    style = torch.randn(1, STYLE_DIM)
    neutral = emotion_one_hot("neutral").reshape(1, -1)
    happy = emotion_one_hot("happy").reshape(1, -1)
    with torch.no_grad():
        assert torch.equal(
            model(style, neutral, torch.tensor([1.0])),
            torch.zeros(1, STYLE_DIM),
        )
        assert torch.equal(
            model(style, happy, torch.tensor([0.0])),
            torch.zeros(1, STYLE_DIM),
        )


def test_residual_norm_is_bounded():
    model = EmotionResidualAdapter(hidden_dim=32, max_norm=0.5).eval()
    styles = torch.randn(64, STYLE_DIM)
    onehot = torch.nn.functional.one_hot(
        torch.randint(1, len(EMOTIONS), (64,)), num_classes=len(EMOTIONS)
    ).float()
    intensity = torch.rand(64)
    with torch.no_grad():
        delta = model(styles, onehot, intensity)
    assert torch.all(delta.norm(dim=-1) <= 0.5 + 1e-6)


def test_parameter_count_is_small():
    model = EmotionResidualAdapter(hidden_dim=256)
    assert model.n_parameters < 1_000_000


def test_speaker_disjoint_split_is_deterministic_and_disjoint():
    speakers = [f"spk_{index % 20:02d}" for index in range(100)]
    fit_a, val_a, val_speakers_a = speaker_disjoint_split(
        speakers, val_fraction=0.2, seed=3
    )
    fit_b, val_b, val_speakers_b = speaker_disjoint_split(
        speakers, val_fraction=0.2, seed=3
    )
    assert np.array_equal(fit_a, fit_b)
    assert np.array_equal(val_a, val_b)
    assert val_speakers_a == val_speakers_b
    assert not set(np.asarray(speakers)[fit_a]) & set(np.asarray(speakers)[val_a])
    assert len(val_speakers_a) == 4


def test_build_examples_uses_emotion_minus_neutral_target():
    rng = np.random.default_rng(0)
    neutral = rng.standard_normal(STYLE_DIM).astype(np.float32)
    happy = (neutral + 0.5).astype(np.float32)
    pairs = pd.DataFrame(
        [
            {
                "actor_id": "a1",
                "sentence": "s1",
                "emotion": "happy",
                "neutral_path": "neutral.wav",
                "emotion_path": "happy.wav",
            }
        ]
    )
    examples = build_examples(
        pairs,
        {"neutral.wav": neutral, "happy.wav": happy},
    )
    assert examples["neutral_styles"].shape == (1, STYLE_DIM)
    assert examples["targets"].shape == (1, STYLE_DIM)
    assert np.allclose(
        examples["targets"][0], happy - neutral, atol=1e-6
    )
    assert examples["emotion_indices"][0] == EMOTION_TO_INDEX["happy"]
    assert examples["intensities"][0] == 1.0
    assert examples["emotions"][0] == "happy"


def test_save_load_round_trip_and_predict(tmp_path):
    model = EmotionResidualAdapter(hidden_dim=32, max_norm=0.4)
    style_mean = torch.zeros(STYLE_DIM)
    style_std = torch.ones(STYLE_DIM)
    path = tmp_path / "adapter.pt"
    save_adapter(
        path,
        model,
        style_mean=style_mean,
        style_std=style_std,
        metadata={"test": True},
    )
    loaded, payload = load_adapter(path)
    style = torch.randn(STYLE_DIM)
    before = predict_residual(model, payload, style, "angry", intensity=1.0)
    after = predict_residual(loaded, payload, style, "angry", intensity=1.0)
    assert torch.allclose(before, after)
    assert predict_residual(loaded, payload, style, "neutral").abs().sum() == 0.0
    assert apply_learned_residual(style, torch.zeros(STYLE_DIM)) is style


def _synthetic_examples(
    n_speakers=20,
    per_speaker=6,
    seed=0,
    *,
    constant_styles=False,
):
    rng = np.random.default_rng(seed)
    neutral_styles = rng.standard_normal(
        (n_speakers * per_speaker, STYLE_DIM)
    ).astype(np.float32)
    if constant_styles:
        # The training-loop test uses emotion-only targets.  Random neutral
        # styles would let the small adapter memorize speaker noise instead of
        # the class direction, which is a fixture artifact, not a model bug.
        neutral_styles = np.zeros_like(neutral_styles)
    emotion_indices = np.asarray(
        [1 + (index % len(NON_NEUTRAL_EMOTIONS)) for index in range(len(neutral_styles))],
        dtype=np.int64,
    )
    emotion_directions = rng.standard_normal(
        (len(EMOTIONS), STYLE_DIM)
    ).astype(np.float32)
    targets = np.zeros_like(neutral_styles)
    for index in range(len(neutral_styles)):
        # A deliberately learnable emotion-only relation.  This tests the
        # training loop and generalization, not style personalization.
        direction = emotion_directions[emotion_indices[index]]
        direction = direction / (np.linalg.norm(direction) + 1e-8)
        targets[index] = 0.5 * direction
    return {
        "neutral_styles": neutral_styles,
        "targets": targets,
        "emotion_indices": emotion_indices,
        "intensities": np.ones(len(neutral_styles), dtype=np.float32),
        "speakers": np.asarray(
            [f"spk_{index // per_speaker:02d}" for index in range(len(neutral_styles))],
            dtype=object,
        ),
        "emotions": np.asarray(
            [EMOTIONS[index] for index in emotion_indices], dtype=object
        ),
        "pair_ids": np.asarray(
            [f"pair_{index}" for index in range(len(neutral_styles))], dtype=object
        ),
    }


def test_training_learns_synthetic_emotion_residuals():
    examples = _synthetic_examples(constant_styles=True)
    result = train_adapter(
        examples,
        hidden_dim=64,
        epochs=150,
        batch_size=64,
        learning_rate=3e-3,
        patience=60,
        val_fraction=0.2,
        seed=0,
        device="cpu",
    )
    assert result["n_parameters"] < 1_000_000
    assert result["best_val_loss"] < 10.0
    assert result["val_metrics"]["cosine_mean"] > 0.8
    model = result["model"]
    payload = {
        "style_mean": torch.from_numpy(result["style_mean"]),
        "style_std": torch.from_numpy(result["style_std"]),
    }
    style = torch.from_numpy(examples["neutral_styles"][0])
    assert predict_residual(model, payload, style, "neutral").abs().sum() == 0.0


def test_adapter_output_depends_on_the_neutral_style_input():
    torch.manual_seed(0)
    model = EmotionResidualAdapter(hidden_dim=32).eval()
    happy = emotion_one_hot("happy").reshape(1, -1)
    intensity = torch.tensor([1.0])
    first = torch.randn(1, STYLE_DIM)
    second = torch.randn(1, STYLE_DIM)
    with torch.no_grad():
        left = model(first, happy, intensity)
        right = model(second, happy, intensity)
    assert not torch.allclose(left, right)


def test_residual_metrics_reports_per_emotion():
    prediction = torch.randn(6, STYLE_DIM)
    target = prediction.clone()
    metrics = residual_metrics(
        prediction,
        target,
        emotions=["happy", "happy", "sad", "sad", "angry", "angry"],
    )
    assert metrics["cosine_mean"] == pytest.approx(1.0, abs=1e-5)
    assert set(metrics["per_emotion"]) == {"angry", "happy", "sad"}


def test_shuffle_labels_breaks_label_target_association():
    examples = _synthetic_examples()
    shuffled = shuffle_labels(examples, seed=5)
    assert np.array_equal(examples["neutral_styles"], shuffled["neutral_styles"])
    assert np.array_equal(examples["targets"], shuffled["targets"])
    assert np.array_equal(examples["speakers"], shuffled["speakers"])
    assert sorted(examples["emotion_indices"].tolist()) == sorted(
        shuffled["emotion_indices"].tolist()
    )
    assert not np.array_equal(
        examples["emotion_indices"], shuffled["emotion_indices"]
    )
    # The target still belongs to the original example, but the label is now
    # random.  At least one example must have a label that no longer matches
    # the target it was built from.
    moved = 0
    for index in range(len(examples["emotion_indices"])):
        original = examples["emotions"][index]
        assigned = shuffled["emotions"][index]
        if original != assigned:
            moved += 1
    assert moved > 0
    # The per-label target mean must no longer be the original class mean.
    original_mean = examples["targets"][
        examples["emotion_indices"] == examples["emotion_indices"][0]
    ].mean(axis=0)
    assigned_label = shuffled["emotion_indices"][0]
    shuffled_mean = shuffled["targets"][
        shuffled["emotion_indices"] == assigned_label
    ].mean(axis=0)
    assert not np.allclose(original_mean, shuffled_mean)


def test_shuffle_style_assignments_preserves_label_target_pairs():
    examples = _synthetic_examples()
    shuffled = shuffle_style_assignments(examples, seed=5)
    original_pairs = list(
        zip(
            examples["emotion_indices"].tolist(),
            examples["targets"].tolist(),
        )
    )
    shuffled_pairs = list(
        zip(
            shuffled["emotion_indices"].tolist(),
            shuffled["targets"].tolist(),
        )
    )
    assert original_pairs == shuffled_pairs or sorted(
        map(repr, original_pairs)
    ) == sorted(map(repr, shuffled_pairs))
    assert np.array_equal(examples["neutral_styles"], shuffled["neutral_styles"])
