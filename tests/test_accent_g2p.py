"""Tests for the experimental IPA-level accent G2P."""
from __future__ import annotations

import pytest

from voix.models.accent_g2p import (
    ACCENT_IDS,
    ACCENT_PROMPT_SET,
    ACCENT_SENSITIVE_WORDS,
    RULE_TABLE,
    WORD_RULES,
    AccentPhonemizer,
    apply_accent_rules,
    normalize_accent,
    normalize_ipa,
    phoneme_inventory,
    phonemize_accent,
    phonemize_base,
    rule_table_report,
    unknown_characters,
    use_accent_phonemizer,
    validate_inventory,
)


def test_prompt_set_covers_accent_sensitive_words():
    text = " ".join(ACCENT_PROMPT_SET).lower()
    for word in ACCENT_SENSITIVE_WORDS:
        assert word in text, word


def test_rule_table_is_documented_and_unique():
    ids = [rule.rule_id for rule in RULE_TABLE] + [rule.rule_id for rule in WORD_RULES]
    assert len(ids) == len(set(ids))
    for rule in RULE_TABLE:
        assert rule.examples
        assert rule.description
        assert rule.pattern
        assert rule.accents
    for rule in WORD_RULES:
        assert rule.examples
        assert rule.words


def test_base_phonemizer_returns_valid_ipa():
    base = phonemize_base("The bath was better than the dance in the car.")
    assert base
    assert unknown_characters(base) == []
    assert "bˈæθ" in base


def test_us_flaps_intervocalic_t():
    assert "ɾ" in phonemize_accent("better", "us")
    assert "ɾ" in phonemize_accent("water", "us")


def test_england_is_nonrhotic_and_has_trap_bath_split():
    car = phonemize_accent("car", "england")
    assert "ɹ" not in car
    assert "ɑː" in car
    assert "ɑː" in phonemize_accent("bath", "england")
    assert "ɑː" in phonemize_accent("dance", "england")


def test_england_retains_yod_in_lexical_exceptions():
    tune = phonemize_accent("tune", "england")
    assert "ju" in tune
    assert "ju" in phonemize_accent("student", "england")


def test_australian_diphthong_shifts_do_not_cascade():
    assert "ɔɪ" in phonemize_accent("time", "australian")
    assert "aɪ" in phonemize_accent("day", "australian")
    assert "əʉ" in phonemize_accent("goat", "australian")


def test_australian_is_nonrhotic():
    assert "ɹ" not in phonemize_accent("car", "australian")


def test_india_uses_conservative_stop_substitutions():
    think = phonemize_accent("think", "indian_english")
    assert "t" in think
    assert "θ" not in think
    this = phonemize_accent("this", "indian_english")
    assert "d" in this
    assert "ð" not in this


def test_india_monophthongizes_face_and_goat():
    day = phonemize_accent("day", "indian_english")
    assert "eː" in day
    assert "eɪ" not in day
    goat = phonemize_accent("goat", "indian_english")
    assert "oː" in goat
    assert "oʊ" not in goat


def test_disabled_rules_are_skipped_for_ablation():
    # Ablation path: disable the two most aggressive rules and verify that the
    # rest of the ordered table still applies.
    australian = phonemize_accent(
        "time", "australian", disabled_rules=("australian_price",)
    )
    assert "aɪ" in australian
    assert "ɔɪ" not in australian
    assert "aː" in phonemize_accent(
        "bath", "australian", disabled_rules=("australian_price",)
    )

    india = phonemize_accent(
        "think",
        "indian_english",
        disabled_rules=("india_theta_stop",),
    )
    assert "θ" in india
    this = phonemize_accent(
        "this",
        "indian_english",
        disabled_rules=("india_eth_stop",),
    )
    assert "ð" in this


def test_disabled_rules_are_carried_by_phonemizer_object():
    converter = AccentPhonemizer(
        "australian", disabled_rules=("australian_price",)
    )
    assert "aɪ" in converter.phonemize("time")
    assert converter.disabled_rules == ("australian_price",)


@pytest.mark.parametrize("accent", ACCENT_IDS)
def test_all_accent_outputs_are_in_styletts2_inventory(accent):
    for prompt in ACCENT_PROMPT_SET:
        output = phonemize_accent(prompt, accent)
        assert output
        assert unknown_characters(output) == []
        validate_inventory(output)


def test_strength_zero_returns_base_phonemes():
    base = phonemize_base("better")
    assert apply_accent_rules(base, "us", strength=0.0) == base


def test_unknown_accent_raises():
    with pytest.raises(ValueError, match="unsupported accent"):
        normalize_accent("klingon")


def test_tie_bars_and_sentence_marks_are_normalized():
    assert normalize_ipa("t͡ʃ‖") == "ʧ"


def test_inventory_matches_styletts2_textcleaner():
    inventory = phoneme_inventory()
    assert "ˈ" in inventory
    assert "ɾ" in inventory
    assert "ʈ" in inventory
    assert "ʋ" in inventory


def test_rule_table_report_is_serializable():
    report = rule_table_report()
    assert report["trainable_accents"] == list(ACCENT_IDS)
    assert len(report["rules"]) == len(RULE_TABLE)
    assert len(report["word_rules"]) == len(WORD_RULES)


class _FakeModel:
    def __init__(self):
        self.phoneme_converter = object()


class _FakeBackend:
    def __init__(self):
        self._model = _FakeModel()
        self._use_fallback = False
        self.loaded = False

    def load(self):
        self.loaded = True


def test_context_manager_swaps_and_restores_converter():
    backend = _FakeBackend()
    original = backend._model.phoneme_converter
    with use_accent_phonemizer(backend, "england", strength=1.0):
        assert isinstance(backend._model.phoneme_converter, AccentPhonemizer)
        assert "ɑː" in backend._model.phoneme_converter.phonemize("bath")
    assert backend._model.phoneme_converter is original
    assert backend.loaded


def test_context_manager_refuses_fallback_backend():
    backend = _FakeBackend()
    backend._use_fallback = True
    with pytest.raises(RuntimeError, match="neural"):
        with use_accent_phonemizer(backend, "england"):
            pass
