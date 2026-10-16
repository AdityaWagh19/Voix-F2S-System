"""Tests for Emotional Tone Modulation (Phase 5)."""

import pytest
import torch
import numpy as np

from voix.models.emotion_control import (
    EmotionType,
    EmotionProfile,
    EmotionModulator,
    list_available_emotions,
    EMOTION_REGISTRY,
)
from voix.inference.pipeline import VoixPipeline


class TestEmotionRegistry:
    def test_canonical_emotions_present(self):
        expected = [
            "neutral",
            "happy",
            "sad",
            "angry",
            "whispered",
            "surprised",
        ]
        for em in expected:
            assert em in EMOTION_REGISTRY
            profile = EMOTION_REGISTRY[em]
            assert isinstance(profile, EmotionProfile)
            assert profile.pitch_factor > 0.5
            assert profile.energy_scale > 0.5

    def test_list_available_emotions(self):
        emotions = list_available_emotions()
        assert len(emotions) >= 6
        ids = [e["emotion_id"] for e in emotions]
        assert "happy" in ids
        assert "whispered" in ids


class TestEmotionModulator:
    def test_neutral_is_identity(self):
        mods = EmotionModulator.get_acoustic_modifiers("neutral", intensity=0.8)
        assert mods["pitch_factor"] == 1.0
        assert mods["beta_mult"] == 1.0
        assert mods["duration_scale"] == 1.0
        assert mods["energy_scale"] == 1.0

    def test_happy_elevates_pitch_and_beta(self):
        mods = EmotionModulator.get_acoustic_modifiers("happy", intensity=0.9)
        assert mods["pitch_factor"] > 1.0
        assert mods["beta_mult"] > 1.0
        assert mods["duration_scale"] < 1.0  # Faster tempo

    def test_sad_reduces_pitch_and_beta(self):
        mods = EmotionModulator.get_acoustic_modifiers("sad", intensity=0.9)
        assert mods["pitch_factor"] < 1.0
        assert mods["beta_mult"] < 1.0
        assert mods["duration_scale"] > 1.0  # Slower tempo

    def test_apply_acoustic_filter(self):
        dummy_wav = torch.randn(1, 24000)
        filtered = EmotionModulator.apply_acoustic_filter(
            dummy_wav,
            sample_rate=24000,
            emotion="whispered",
            intensity=0.8,
        )
        assert filtered.shape == dummy_wav.shape
        assert not torch.isnan(filtered).any()
        assert filtered.abs().max() <= 1.0

    def test_invalid_emotion_raises(self):
        with pytest.raises(ValueError, match="Unknown emotion"):
            EmotionModulator.get_acoustic_modifiers("alien_rage")


class TestPipelineEmotionIntegration:
    @pytest.fixture
    def pipeline(self):
        return VoixPipeline(device="cpu", enable_watermarking=False)

    def test_pipeline_generate_with_emotion(self, pipeline):
        res = pipeline.generate(
            face_input="adult_female_warm",
            text="Hello world, I am testing emotions.",
            num_voices=1,
            emotion="happy",
            emotion_intensity=0.8,
        )
        assert len(res) == 1
        assert "waveform" in res[0]
        assert res[0]["waveform"].shape[-1] > 1000

    def test_pipeline_list_emotions(self, pipeline):
        ems = pipeline.list_emotions()
        assert len(ems) >= 6
