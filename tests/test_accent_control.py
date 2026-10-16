"""Tests for Accent / Dialect Controls (Phase 5)."""

import pytest
import torch
import numpy as np

from voix.models.accent_control import (
    AccentType,
    AccentProfile,
    AccentTransformer,
    list_available_accents,
    ACCENT_REGISTRY,
)
from voix.inference.pipeline import VoixPipeline


class TestAccentRegistry:
    def test_canonical_accents_present(self):
        expected = [
            "general_american",
            "british_rp",
            "australian",
            "indian_english",
            "irish",
            "scottish",
        ]
        for acc in expected:
            assert acc in ACCENT_REGISTRY
            profile = ACCENT_REGISTRY[acc]
            assert isinstance(profile, AccentProfile)
            assert profile.duration_scale > 0.5
            assert profile.pitch_factor > 0.5

    def test_list_available_accents(self):
        accents = list_available_accents()
        assert len(accents) >= 6
        ids = [a["accent_id"] for a in accents]
        assert "british_rp" in ids
        assert "indian_english" in ids


class TestAccentTransformer:
    def test_general_american_noop(self):
        text = "Hello, this is standard speech."
        out, mods = AccentTransformer.transform(text, accent="general_american", strength=0.5)
        assert out == text
        assert mods["duration_mult"] == 1.0

    def test_british_rp_transformation(self):
        text = "We must pass the fast task after class."
        out, mods = AccentTransformer.transform(text, accent="british_rp", strength=0.9)
        assert isinstance(out, str)
        assert mods["prosody_mult"] > 1.0

    def test_invalid_accent_raises(self):
        with pytest.raises(ValueError, match="Unknown accent"):
            AccentTransformer.transform("test", accent="martian_alien")

    def test_strength_clamping(self):
        text = "Testing strength clamping."
        _, mods_high = AccentTransformer.transform(text, accent="irish", strength=2.5)
        _, mods_normal = AccentTransformer.transform(text, accent="irish", strength=1.0)
        assert mods_high["pitch_mult"] == mods_normal["pitch_mult"]


class TestPipelineAccentIntegration:
    @pytest.fixture
    def pipeline(self):
        return VoixPipeline(device="cpu", enable_watermarking=False)

    def test_pipeline_generate_with_accent(self, pipeline):
        res = pipeline.generate(
            face_input="adult_female_warm",
            text="Hello world, this is a test.",
            num_voices=1,
            accent="british_rp",
            accent_strength=0.8,
        )
        assert len(res) == 1
        assert "waveform" in res[0]
        assert res[0]["waveform"].shape[-1] > 1000

    def test_pipeline_list_accents(self, pipeline):
        accs = pipeline.list_accents()
        assert len(accs) >= 6
