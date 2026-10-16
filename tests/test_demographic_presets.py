"""Tests for Demographic Presets Rail (Feature 2).

Validates:
  - Canonical preset registry and metadata schema
  - Biometric embedding dimensions (1, 560) and numerical validity
  - Demographic diversity across reference presets
  - VoixPipeline integration via preset ID resolution and generate_from_preset()
"""

import numpy as np
import pytest
import torch

from voix.data.presets import (
    CANONICAL_PRESETS_DEF,
    PresetRegistry,
    get_demographic_preset,
    list_demographic_presets,
)


class TestDemographicPresetsRegistry:
    """Verify preset registry definitions and embeddings."""

    def test_list_presets_completeness(self):
        """Registry must list all canonical presets matching definitions."""
        presets = list_demographic_presets()
        assert len(presets) == len(CANONICAL_PRESETS_DEF)

        expected_ids = {p["preset_id"] for p in CANONICAL_PRESETS_DEF}
        found_ids = {p["preset_id"] for p in presets}
        assert expected_ids == found_ids

    def test_preset_metadata_schema(self):
        """Every preset must contain complete, scientifically sound metadata."""
        presets = list_demographic_presets()
        for p in presets:
            assert isinstance(p["preset_id"], str) and len(p["preset_id"]) > 0
            assert isinstance(p["display_name"], str) and len(p["display_name"]) > 0
            assert isinstance(p["description"], str) and len(p["description"]) > 0
            assert p["gender"] in (0, 1)
            assert 15.0 <= p["age"] <= 85.0
            assert 80.0 <= p["pitch_target_hz"] <= 300.0
            assert isinstance(p["craniofacial_traits"], dict)

    def test_preset_embeddings_validity(self):
        """Every preset must produce a valid 560-D fused tensor without NaNs."""
        for p_def in CANONICAL_PRESETS_DEF:
            preset = get_demographic_preset(p_def["preset_id"])
            emb = preset.embedding
            assert emb is not None
            assert isinstance(emb, torch.Tensor)
            assert emb.shape == (1, 560)
            assert not torch.isnan(emb).any()
            assert not torch.isinf(emb).any()

    def test_preset_distinctness(self):
        """Different presets must have distinct embeddings (non-identical)."""
        female_preset = get_demographic_preset("young_female_resonant")
        male_preset = get_demographic_preset("adult_male_baritone")

        # Distinct presets must have measurable Euclidean separation
        l2_dist = torch.norm(female_preset.embedding - male_preset.embedding).item()
        assert l2_dist > 0.1
        assert not torch.allclose(female_preset.embedding, male_preset.embedding, atol=1e-3)

    def test_unknown_preset_raises_keyerror(self):
        """Querying an unknown preset ID must raise a descriptive KeyError."""
        with pytest.raises(KeyError, match="not found"):
            get_demographic_preset("non_existent_preset_xyz")


class TestPipelinePresetIntegration:
    """Verify end-to-end synthesis using demographic presets."""

    @pytest.fixture(scope="class")
    def pipeline(self):
        from voix.inference.pipeline import VoixPipeline
        return VoixPipeline(device="cpu", enable_watermarking=False)

    def test_pipeline_generate_from_preset(self, pipeline):
        """Verify generate_from_preset synthesizes valid candidate voices."""
        results = pipeline.generate_from_preset(
            preset_id="young_female_resonant",
            text="Testing demographic preset synthesis.",
            num_voices=1,
        )
        assert len(results) == 1
        assert "waveform" in results[0]
        assert results[0]["waveform"].shape[-1] > 0
        assert results[0]["sample_rate"] == 24000

    def test_pipeline_generate_with_preset_id_string(self, pipeline):
        """Passing preset ID directly as face_input string should resolve seamlessly."""
        results = pipeline.generate(
            face_input="adult_male_baritone",
            text="Testing direct preset ID string synthesis.",
            num_voices=1,
        )
        assert len(results) == 1
        assert "waveform" in results[0]
        assert results[0]["waveform"].shape[-1] > 0
