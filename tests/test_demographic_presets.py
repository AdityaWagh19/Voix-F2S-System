"""Tests for Demographic Presets Rail (Feature 2).

Validates:
  - Canonical preset registry and metadata schema
  - Biometric embedding dimensions (1, 560) and numerical validity
  - Demographic diversity across reference presets
  - VoixPipeline integration via preset ID resolution and generate_from_preset()
"""

import itertools

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from voix.data.presets import (
    CANONICAL_PRESETS_DEF,
    PRESET_SCHEMA_VERSION,
    PresetRegistry,
    get_demographic_preset,
    list_demographic_presets,
    load_preset_embedding,
    save_preset_embedding,
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

    def test_preset_embeddings_are_not_v1_collapsed(self):
        """The V1 fusion made every preset cosine ~0.9997; V2 must not."""
        embeddings = [
            get_demographic_preset(p["preset_id"]).embedding.reshape(-1)
            for p in CANONICAL_PRESETS_DEF
        ]
        cosines = [
            float(F.cosine_similarity(a, b, dim=0))
            for a, b in itertools.combinations(embeddings, 2)
        ]
        assert max(cosines) < 0.99, max(cosines)

    def test_unknown_preset_raises_keyerror(self):
        """Querying an unknown preset ID must raise a descriptive KeyError."""
        with pytest.raises(KeyError, match="not found"):
            get_demographic_preset("non_existent_preset_xyz")

    def test_trait_regeneration_changes_geometry_and_embedding(self):
        from voix.data.presets import build_preset_face

        base_embedding, base_geometry = build_preset_face("young_female_resonant")
        slider_embedding, slider_geometry = build_preset_face(
            "young_female_resonant", {"jaw_width": 0.8}
        )
        assert not np.allclose(base_geometry, slider_geometry)
        assert not torch.allclose(base_embedding, slider_embedding)


class TestPresetCacheContract:
    """A bare V1 tensor cache must never be loaded as a V2 preset."""

    def test_versioned_roundtrip(self, tmp_path):
        path = tmp_path / "preset.pt"
        embedding = torch.randn(1, 560)
        save_preset_embedding(path, embedding)
        loaded = load_preset_embedding(path)
        assert torch.allclose(loaded, embedding)
        payload = torch.load(path, map_location="cpu", weights_only=True)
        assert payload["schema_version"] == PRESET_SCHEMA_VERSION

    def test_bare_tensor_is_rejected(self, tmp_path):
        path = tmp_path / "preset.pt"
        torch.save(torch.randn(1, 560), path)
        with pytest.raises(ValueError, match="unversioned"):
            load_preset_embedding(path)

    def test_wrong_schema_is_rejected(self, tmp_path):
        path = tmp_path / "preset.pt"
        torch.save(
            {"schema_version": "FaceFeatureV1", "embedding": torch.randn(1, 560)},
            path,
        )
        with pytest.raises(ValueError, match="expected"):
            load_preset_embedding(path)

    def test_wrong_shape_is_rejected(self, tmp_path):
        path = tmp_path / "preset.pt"
        torch.save(
            {
                "schema_version": PRESET_SCHEMA_VERSION,
                "embedding": torch.randn(1, 512),
            },
            path,
        )
        with pytest.raises(ValueError, match="shape"):
            load_preset_embedding(path)


class TestPipelinePresetIntegration:
    """Verify end-to-end synthesis using demographic presets."""

    @pytest.fixture(scope="class")
    def pipeline(self, v2_artifacts):
        from voix.inference.pipeline import VoixPipeline
        return VoixPipeline(
            device="cpu",
            enable_watermarking=False,
            tts_use_neural=False,
        )

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
