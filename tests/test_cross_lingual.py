"""Unit tests for VOIX Phase 6: Cross-Lingual Voice Transfer (Feature 5)."""

import inspect
import pytest
from voix.models.cross_lingual import (
    LanguageType,
    LanguageProfile,
    MultilingualPhoneticAdapter,
    list_supported_languages,
    LANGUAGE_REGISTRY,
)
from voix.inference.pipeline import VoixPipeline


class TestCrossLingualRegistry:
    """Test suite for language definitions and acoustic registries."""

    def test_language_enum_coverage(self):
        """Verify all 5 target languages are defined in the enum."""
        assert LanguageType.ENGLISH.value == "en"
        assert LanguageType.SPANISH.value == "es"
        assert LanguageType.FRENCH.value == "fr"
        assert LanguageType.GERMAN.value == "de"
        assert LanguageType.ITALIAN.value == "it"

    def test_registry_contains_all_profiles(self):
        """Verify registry contains profiles for all defined languages."""
        for lang in LanguageType:
            assert lang.value in LANGUAGE_REGISTRY
            profile = LANGUAGE_REGISTRY[lang.value]
            assert isinstance(profile, LanguageProfile)
            assert profile.code == lang.value
            assert len(profile.name) > 0
            assert profile.rhythm_type in ("syllable-timed", "stress-timed")
            assert 0.8 <= profile.duration_scale <= 1.3
            assert 0.8 <= profile.pitch_dynamics_factor <= 1.3

    def test_list_supported_languages(self):
        """Verify listing function returns structured dictionary list."""
        langs = list_supported_languages()
        assert len(langs) == 5
        codes = [item["code"] for item in langs]
        assert "en" in codes
        assert "es" in codes
        assert "fr" in codes
        assert "de" in codes
        assert "it" in codes


class TestMultilingualPhoneticAdapter:
    """Test suite for phonetic transformation and prosodic factor generation."""

    def test_english_passthrough(self):
        """Verify English text passes through unchanged with baseline multipliers."""
        text = "Hello world, this is a biometric voice test."
        adapted, mods = MultilingualPhoneticAdapter.transform(text, "en")
        assert adapted == text
        assert mods["duration_scale"] == 1.00
        assert mods["pitch_dynamics_factor"] == 1.00

    def test_spanish_adaptation(self):
        """Verify Spanish phonetic normalization and diacritic handling."""
        text = "Buenos días, sí, muchas gracias por su atención."
        adapted, mods = MultilingualPhoneticAdapter.transform(text, "es")
        assert "deeas" in adapted.lower()
        assert "see" in adapted.lower()
        assert "grahtsyas" in adapted.lower()
        assert "í" not in adapted
        assert "ó" not in adapted
        assert mods["duration_scale"] == 1.02
        assert mods["pitch_dynamics_factor"] == 1.08

    def test_french_adaptation(self):
        """Verify French phonetic normalization."""
        text = "Bonjour, ceci est une démonstration vocale, merci."
        adapted, mods = MultilingualPhoneticAdapter.transform(text, "fr")
        assert "bohn-zhoor" in adapted.lower()
        assert "mair-see" in adapted.lower()
        assert "é" not in adapted
        assert mods["duration_scale"] == 0.98
        assert mods["pitch_dynamics_factor"] == 1.04

    def test_german_adaptation(self):
        """Verify German phonetic normalization."""
        text = "Guten Tag, dies ist eine schnelle Überprüfung, danke."
        adapted, mods = MultilingualPhoneticAdapter.transform(text, "de")
        assert "goo-ten tahg" in adapted.lower()
        assert "dahn-keh" in adapted.lower()
        assert "Ü" not in adapted
        assert mods["duration_scale"] == 1.06
        assert mods["pitch_dynamics_factor"] == 0.95

    def test_italian_adaptation(self):
        """Verify Italian phonetic normalization."""
        text = "Buongiorno, questa è la voce sintetizzata, grazie mille."
        adapted, mods = MultilingualPhoneticAdapter.transform(text, "it")
        assert "bwon-jorno" in adapted.lower()
        assert "graht-syeh" in adapted.lower()
        assert "è" not in adapted
        assert mods["duration_scale"] == 1.00
        assert mods["pitch_dynamics_factor"] == 1.14

    def test_unsupported_language_raises(self):
        """Verify attempting an unsupported language raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported language"):
            MultilingualPhoneticAdapter.transform("Hello", "klingon")


class TestPipelineCrossLingualIntegration:
    """Test suite verifying VoixPipeline signatures and methods for cross-lingual synthesis."""

    def test_generate_signature_has_language(self):
        """Verify generate signature exposes 'language' argument."""
        sig = inspect.signature(VoixPipeline.generate)
        assert "language" in sig.parameters
        assert sig.parameters["language"].default == "en"

    def test_generate_from_preset_signature_has_language(self):
        """Verify generate_from_preset signature exposes 'language' argument."""
        sig = inspect.signature(VoixPipeline.generate_from_preset)
        assert "language" in sig.parameters
        assert sig.parameters["language"].default == "en"

    def test_list_languages_method(self):
        """Verify pipeline expose list_languages method."""
        sig = inspect.signature(VoixPipeline.list_languages)
        assert sig is not None
