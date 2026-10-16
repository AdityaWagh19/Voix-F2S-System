"""Tests for Interactive Craniofacial Proportion Sliders (Feature 1).

Validates:
  - Availability and metadata of craniofacial sliders
  - Morphological ratio modifications and correlated index updates
  - Physiological bounds clamping (preventing anatomical deformation / NaNs)
  - Identity property (zero/empty slider produces exact original vector)
  - Fusion and pipeline integration
"""

import numpy as np
import pytest
import torch

from voix.data.morphology import (
    CRANIOFACIAL_SLIDER_SPECS,
    apply_craniofacial_sliders,
    get_available_sliders,
)


@pytest.fixture
def baseline_ratios() -> np.ndarray:
    """Return realistic synthetic 32-D baseline craniofacial ratios."""
    ratios = np.ones(32, dtype=np.float32)
    ratios[0] = 2.2   # face height / IOC
    ratios[1] = 1.8   # face width / IOC
    ratios[2] = 1.4   # jaw width / IOC
    ratios[3] = 85.0  # mandible angle degrees
    ratios[4] = 0.55  # nose width / IOC
    ratios[5] = 0.80  # nose height / IOC
    ratios[6] = 0.70  # nose bridge / IOC
    ratios[7] = 1.8   # bizygomatic / IOC
    ratios[8] = 0.35  # inner eye sep / IOC
    ratios[11] = 0.90 # lip width / IOC
    ratios[12] = 0.18 # upper lip height / IOC
    ratios[13] = 0.22 # lower lip height / IOC
    ratios[24] = 1.22 # face aspect ratio
    ratios[25] = 2.54 # mandible to nose ratio
    ratios[26] = 0.64 # lip to jaw ratio
    ratios[27] = 0.61 # nose to lip ratio
    ratios[30] = 0.05 # chin protrusion
    return ratios


class TestCraniofacialSliders:
    """Verify craniofacial proportion slider logic and anatomical bounds."""

    def test_slider_specs_and_metadata(self):
        """Verify all expected sliders exist with proper schema and valid defaults."""
        sliders = get_available_sliders()
        expected = [
            "jaw_width",
            "face_height",
            "midface_length",
            "lip_fullness",
            "lip_width",
            "nose_width",
            "face_width",
            "chin_prominence",
        ]
        for key in expected:
            assert key in sliders, f"Missing slider: {key}"
            assert sliders[key]["default"] == 0.0
            assert sliders[key]["min_val"] == -1.0
            assert sliders[key]["max_val"] == 1.0
            assert len(sliders[key]["target_indices"]) > 0

    def test_identity_preservation_empty_sliders(self, baseline_ratios):
        """Empty or None slider dict must return identical array copy."""
        res_none = apply_craniofacial_sliders(baseline_ratios, None)
        res_empty = apply_craniofacial_sliders(baseline_ratios, {})
        res_zero = apply_craniofacial_sliders(baseline_ratios, {"jaw_width": 0.0})

        np.testing.assert_array_equal(res_none, baseline_ratios)
        np.testing.assert_array_equal(res_empty, baseline_ratios)
        np.testing.assert_allclose(res_zero, baseline_ratios, rtol=1e-6)

    def test_jaw_width_slider_modulation(self, baseline_ratios):
        """Increasing jaw width expands mandibular width and shifts correlated ratios."""
        expanded = apply_craniofacial_sliders(baseline_ratios, {"jaw_width": 0.8})
        reduced = apply_craniofacial_sliders(baseline_ratios, {"jaw_width": -0.8})

        # Index 2 is jaw width
        assert expanded[2] > baseline_ratios[2]
        assert reduced[2] < baseline_ratios[2]

        # Correlated index 26 is lip-to-jaw ratio (should decrease as jaw expands)
        assert expanded[26] < baseline_ratios[26]
        assert reduced[26] > baseline_ratios[26]

    def test_lip_fullness_modulation(self, baseline_ratios):
        """Lip fullness expands upper (idx 12) and lower (idx 13) lip heights."""
        fuller = apply_craniofacial_sliders(baseline_ratios, {"lip_fullness": 0.75})
        thinner = apply_craniofacial_sliders(baseline_ratios, {"lip_fullness": -0.75})

        assert fuller[12] > baseline_ratios[12]
        assert fuller[13] > baseline_ratios[13]
        assert thinner[12] < baseline_ratios[12]
        assert thinner[13] < baseline_ratios[13]

    def test_anatomical_bounds_clamping(self, baseline_ratios):
        """Extreme slider inputs (+10.0, -10.0) must clamp cleanly without NaNs or invalid ratios."""
        extreme_pos = apply_craniofacial_sliders(
            baseline_ratios,
            {"jaw_width": 10.0, "face_height": 10.0, "lip_fullness": 10.0},
        )
        extreme_neg = apply_craniofacial_sliders(
            baseline_ratios,
            {"jaw_width": -10.0, "face_height": -10.0, "lip_fullness": -10.0},
        )

        assert not np.isnan(extreme_pos).any()
        assert not np.isnan(extreme_neg).any()
        assert not np.isinf(extreme_pos).any()
        assert not np.isinf(extreme_neg).any()

        # Check bounded limits for jaw_width
        jaw_spec = CRANIOFACIAL_SLIDER_SPECS["jaw_width"]
        low, high = jaw_spec["anatomical_bounds"]
        assert low <= extreme_neg[2] <= high
        assert low <= extreme_pos[2] <= high

    def test_multi_slider_interaction(self, baseline_ratios):
        """Simultaneous multiple sliders must modify their respective indices independently."""
        sliders = {
            "jaw_width": 0.5,
            "lip_fullness": -0.4,
            "nose_width": 0.6,
        }
        res = apply_craniofacial_sliders(baseline_ratios, sliders)

        assert res[2] > baseline_ratios[2]    # jaw expanded
        assert res[12] < baseline_ratios[12]  # upper lip reduced
        assert res[4] > baseline_ratios[4]    # nose widened


class TestCraniofacialPipelineIntegration:
    """Verify integration of sliders into VoixPipeline and feature fusion."""

    def test_pipeline_generate_with_sliders(self):
        """Verify pipeline accepts craniofacial_sliders without error and produces valid waveforms."""
        from voix.inference.pipeline import VoixPipeline

        pipeline = VoixPipeline(device="cpu", enable_watermarking=False)

        # Create synthetic 560-D face vector
        face_560 = torch.randn(1, 560)
        results = pipeline.generate(
            face_input=face_560,
            text="Testing craniofacial proportion modulation.",
            num_voices=1,
            craniofacial_sliders={"jaw_width": 0.75, "lip_fullness": -0.5},
        )

        assert len(results) == 1
        assert "waveform" in results[0]
        assert results[0]["waveform"].shape[-1] > 0
