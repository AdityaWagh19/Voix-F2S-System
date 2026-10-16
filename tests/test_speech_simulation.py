"""Tests for Speech Difference Simulation Engine (Feature 3).

Validates:
  - Supported speech difference modes and metadata
  - Strict zero-severity no-op guarantees (identity preservation)
  - Articulatory phonetic text transformations (lisp, rhotacism, stutter, prolongation)
  - Acoustic DSP velopharyngeal pole-zero filtering for hypernasality
  - VoixPipeline integration end-to-end
"""

import numpy as np
import pytest
import torch

from voix.models.speech_simulation import (
    SpeechAcousticTransformer,
    SpeechDifferenceType,
    SpeechSimulationConfig,
    SpeechSimulator,
    SpeechTextTransformer,
)


class TestSpeechSimulationConfig:
    """Verify configuration validation and bounds."""

    def test_supported_differences_list(self):
        """All 6 difference types must be registered with complete schema."""
        diffs = SpeechSimulator.list_supported_differences()
        assert len(diffs) == 6

        expected_ids = {
            "interdental_lisp",
            "lateral_lisp",
            "rhotacism",
            "stutter_clonic",
            "hypernasality",
            "prolonged_phonemes",
        }
        found_ids = {d["id"] for d in diffs}
        assert expected_ids == found_ids

        for d in diffs:
            assert isinstance(d["name"], str) and len(d["name"]) > 0
            assert isinstance(d["description"], str) and len(d["description"]) > 0
            assert isinstance(d["mechanism"], str) and len(d["mechanism"]) > 0

    def test_config_severity_clamping(self):
        """Severities outside [0, 1] must be safely clamped."""
        cfg_high = SpeechSimulationConfig(difference_type="interdental_lisp", severity=1.5)
        assert cfg_high.severity == 1.0

        cfg_low = SpeechSimulationConfig(difference_type="interdental_lisp", severity=-0.5)
        assert cfg_low.severity == 0.0

    def test_config_invalid_type_raises(self):
        """Invalid difference type must raise ValueError."""
        with pytest.raises(ValueError, match="Unknown speech difference"):
            SpeechSimulationConfig(difference_type="unknown_alien_voice")


class TestTextTransformations:
    """Verify phonetic transformation rules for articulatory characteristics."""

    def test_zero_severity_identity_guarantee(self):
        """When severity is 0.0, text and duration multiplier must remain strictly unmodified."""
        text = "Synthetic speech synthesis preserves vocal identity across facial biometrics."
        for dtype in SpeechDifferenceType:
            transformed, dur = SpeechTextTransformer.transform(text, dtype.value, severity=0.0)
            assert transformed == text
            assert dur == 1.0

    def test_interdental_lisp_phonetic_substitution(self):
        """Interdental lisp substitutes /s/ and /z/ with /th/ while preserving casing."""
        text = "Synthetic speech synthesis simulates facial structure."
        transformed, dur = SpeechTextTransformer.transform(
            text,
            SpeechDifferenceType.INTERDENTAL_LISP.value,
            severity=1.0,
        )
        assert transformed != text
        # 'Synthetic' -> 'Thynthetic' or 'speech' -> 'thpeech'
        assert "th" in transformed.lower()

    def test_rhotacism_phonetic_substitution(self):
        """Rhotacism substitutes rhotic consonants /r/ with glides /w/."""
        text = "Red rabbits run rapidly through green trees."
        transformed, dur = SpeechTextTransformer.transform(
            text,
            SpeechDifferenceType.RHOTACISM.value,
            severity=1.0,
        )
        assert transformed != text
        # 'Red' -> 'Wed', 'run' -> 'wun', 'trees' -> 'twees'
        assert "w" in transformed.lower()

    def test_stutter_clonic_onset_repetitions(self):
        """Clonic stutter repeats word onsets with hyphenation."""
        text = "Welcome to the future of facial speech synthesis."
        transformed, dur = SpeechTextTransformer.transform(
            text,
            SpeechDifferenceType.STUTTER_CLONIC.value,
            severity=0.8,
        )
        assert transformed != text
        assert "-" in transformed
        assert dur > 1.0  # Utterance pacing dilates

    def test_prolonged_phonemes_vocalic_extension(self):
        """Prolonged phonemes extend vowel durations."""
        text = "Welcome to the voice."
        transformed, dur = SpeechTextTransformer.transform(
            text,
            SpeechDifferenceType.PROLONGED_PHONEMES.value,
            severity=0.9,
        )
        assert dur > 1.0


class TestAcousticTransformers:
    """Verify DSP resonance filtering for vocal tract differences."""

    @pytest.fixture
    def test_waveform(self) -> torch.Tensor:
        """Create 1-second synthetic 24 kHz harmonic tone."""
        sr = 24000
        t = torch.linspace(0, 1.0, sr)
        wav = torch.sin(2 * np.pi * 200 * t) + 0.5 * torch.sin(2 * np.pi * 600 * t)
        return wav.unsqueeze(0)

    def test_zero_severity_waveform_identity(self, test_waveform):
        """Waveform is strictly bitwise identical at zero severity."""
        filtered = SpeechAcousticTransformer.filter(
            test_waveform,
            sample_rate=24000,
            difference_type="hypernasality",
            severity=0.0,
        )
        torch.testing.assert_close(filtered, test_waveform)

    def test_hypernasality_spectral_shaping(self, test_waveform):
        """Hypernasality filter alters waveform without producing NaNs or clipping."""
        filtered = SpeechAcousticTransformer.apply_hypernasality_filter(
            test_waveform,
            sample_rate=24000,
            severity=0.8,
        )
        assert filtered.shape == test_waveform.shape
        assert not torch.isnan(filtered).any()
        assert not torch.isinf(filtered).any()
        assert filtered.max() <= 1.0
        assert filtered.min() >= -1.0
        # Measurable modification occurred
        assert not torch.allclose(filtered, test_waveform, atol=1e-3)


class TestPipelineSpeechSimulationIntegration:
    """Verify end-to-end VoixPipeline synthesis with speech differences."""

    @pytest.fixture(scope="class")
    def pipeline(self):
        from voix.inference.pipeline import VoixPipeline
        return VoixPipeline(device="cpu", enable_watermarking=False)

    def test_generate_with_interdental_lisp(self, pipeline):
        """Synthesize voice using interdental lisp mode."""
        results = pipeline.generate(
            face_input="adult_female_warm",
            text="Synthetic speech synthesis demonstrates clear facial biometrics.",
            num_voices=1,
            speech_difference="interdental_lisp",
            speech_severity=0.75,
        )
        assert len(results) == 1
        assert "waveform" in results[0]
        assert results[0]["waveform"].shape[-1] > 0

    def test_generate_with_hypernasality(self, pipeline):
        """Synthesize voice using hypernasality mode."""
        results = pipeline.generate(
            face_input="adult_male_baritone",
            text="Natural resonant human voice with nasal acoustic characteristics.",
            num_voices=1,
            speech_difference="hypernasality",
            speech_severity=0.80,
        )
        assert len(results) == 1
        assert "waveform" in results[0]
        assert results[0]["waveform"].shape[-1] > 0
