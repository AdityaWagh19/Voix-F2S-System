"""Rule-based, IPA-level accent phonemization for the experimental path.

The canonical VOIX path keeps StyleTTS2's gruut phonemizer.  This module is the
optional A2 path: it wraps gruut to get base IPA, then applies explicit,
documented accent rules to the phoneme string.

Only the four supported accents are implemented:

    us, england, indian_english, australian

The Indian English rule set is deliberately conservative.  Earlier global
retroflex, tap, and approximant rewrites produced a 0.2332 Whisper WER and were
removed.  The remaining rules substitute the two interdental fricatives and
monophthongize FACE/GOAT; they are ablatable through ``disabled_rules``.

These are rules, not a learned G2P.  They are expected to be imperfect, and no
perceptual accent-identification claim is attached to them.
"""
from __future__ import annotations

import re
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

ACCENT_IDS: Tuple[str, ...] = (
    "us",
    "england",
    "indian_english",
    "australian",
)
TRAINABLE_ACCENTS: Tuple[str, ...] = ACCENT_IDS
ACCENT_ALIASES: Dict[str, str] = {
    "us": "us",
    "general_american": "us",
    "american": "us",
    "en-us": "us",
    "england": "england",
    "british_rp": "england",
    "british": "england",
    "en-gb": "england",
    "indian_english": "indian_english",
    "india_south_asia": "indian_english",
    "india": "indian_english",
    "australian": "australian",
    "australia": "australian",
}

# Shared phonetically balanced prompt set.  It deliberately covers the
# accent-sensitive words named in the research plan.
ACCENT_PROMPT_SET: Tuple[str, ...] = (
    "The bath was better than the dance in the car.",
    "I think this is the right time to tune the new voice.",
    "Last night the fast train passed the path by the water.",
    "Both of us can start the task after the film.",
    "The human voice reflects a personal identity.",
    "Please ask the staff about the chance of rain.",
)
ACCENT_SENSITIVE_WORDS: Tuple[str, ...] = (
    "bath",
    "dance",
    "car",
    "better",
    "tune",
    "new",
    "think",
    "this",
    "right",
    "time",
)

_VOWELS = "aeiouɑɐɒæeɛɜɝəɚɪɔʊʌɤʉyøœɵɘ"
_CONSONANTS = "bcdfghjklmnpqrstvwxyzðθʃʒʧʤŋɹɾɫʔʈɖʋ"
_VOICELESS = "ptkfθsʃʧ"

# gruut sometimes emits a tie bar (U+0361) between the two halves of an
# affricate.  StyleTTS2's TextCleaner inventory has the single-character
# equivalents, so normalize before validating.
_TIE_BAR_REPLACEMENTS: Tuple[Tuple[str, str], ...] = (
    ("t͡ʃ", "ʧ"),
    ("d͡ʒ", "ʤ"),
    ("t͜ʃ", "ʧ"),
    ("d͜ʒ", "ʤ"),
    ("\u0361", ""),
    ("\u035c", ""),
    ("‖", ""),
)


@dataclass(frozen=True)
class AccentRule:
    """A single ordered IPA substitution rule."""

    rule_id: str
    accents: Tuple[str, ...]
    description: str
    examples: Tuple[str, ...]
    pattern: str
    replacement: str
    min_strength: float = 0.5

    def apply(self, word: str) -> str:
        return re.sub(self.pattern, self.replacement, word)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "accents": list(self.accents),
            "description": self.description,
            "examples": list(self.examples),
            "pattern": self.pattern,
            "replacement": self.replacement,
            "min_strength": self.min_strength,
        }


@dataclass(frozen=True)
class WordRule:
    """A rule that only applies to a small lexical exception set."""

    rule_id: str
    accents: Tuple[str, ...]
    description: str
    words: Tuple[str, ...]
    examples: Tuple[str, ...]
    pattern: str
    replacement: str
    min_strength: float = 0.5

    def applies_to_source(self, source_word: Optional[str]) -> bool:
        if not source_word:
            return False
        key = re.sub(r"[^A-Za-z']", "", str(source_word)).lower()
        return key in set(self.words)

    def apply(self, word: str) -> str:
        return re.sub(self.pattern, self.replacement, word)


RULE_TABLE: Tuple[AccentRule, ...] = (
    AccentRule(
        "us_flap_t",
        ("us",),
        "Intervocalic /t/ is realized as an alveolar flap.",
        ("better", "water", "city"),
        rf"(?<=[{_VOWELS}])t(?=[ˈˌ]*[{_VOWELS}])",
        "ɾ",
    ),
    AccentRule(
        "england_nonrhotic_coda_r",
        ("england", "australian"),
        "Coda /ɹ/ is deleted (non-rhotic).",
        ("car", "better", "water"),
        rf"ɹ(?=(?:[ˈˌ]*[{_CONSONANTS}])|$)",
        "",
    ),
    AccentRule(
        "england_lengthen_final_open",
        ("england", "australian"),
        "A final open /ɑ/ or /ɔ/ is lengthened after coda-r deletion.",
        ("car", "law"),
        r"(?<=[ɑɔ])(?=$)",
        "ː",
    ),
    AccentRule(
        "england_trap_bath",
        ("england",),
        "Trap-bath split: /æ/ -> /ɑː/ before ns, nt, nʧ, s, θ, f.",
        ("bath", "dance", "path", "last", "chance"),
        r"æ(?=[ˈˌ]*(?:nʧ|ns|nt|s|θ|f))",
        "ɑː",
    ),
    AccentRule(
        "australian_trap_bath",
        ("australian",),
        "Broad Australian /æ/ -> /aː/ before ns, nt, nʧ, s, θ, f.",
        ("bath", "dance", "path", "last", "chance"),
        r"æ(?=[ˈˌ]*(?:nʧ|ns|nt|s|θ|f))",
        "aː",
    ),
    AccentRule(
        "australian_price",
        ("australian",),
        "Australian /aɪ/ -> /ɔɪ/ (e.g. time -> tɔɪm).",
        ("time", "right", "like"),
        r"aɪ",
        "ɔɪ",
        min_strength=1.0,
    ),
    AccentRule(
        "australian_face",
        ("australian",),
        "Australian /eɪ/ -> /aɪ/ (e.g. day -> daɪ).",
        ("day", "mate", "face"),
        r"eɪ",
        "aɪ",
    ),
    AccentRule(
        "australian_goat",
        ("australian",),
        "Australian /oʊ/ -> /əʉ/ (e.g. goat -> gəʉt).",
        ("goat", "both", "voice"),
        r"oʊ",
        "əʉ",
    ),
    AccentRule(
        "india_theta_stop",
        ("indian_english",),
        "Interdental /θ/ -> dental stop /t/.",
        ("think", "three", "bath"),
        r"θ",
        "t",
    ),
    AccentRule(
        "india_eth_stop",
        ("indian_english",),
        "Interdental /ð/ -> dental stop /d/.",
        ("this", "that", "the"),
        r"ð",
        "d",
    ),
    AccentRule(
        "india_face_monophthong",
        ("indian_english",),
        "Face vowel /eɪ/ -> monophthong /eː/.",
        ("day", "mate", "face"),
        r"eɪ",
        "eː",
    ),
    AccentRule(
        "india_goat_monophthong",
        ("indian_english",),
        "Goat vowel /oʊ/ -> monophthong /oː/.",
        ("goat", "both", "voice"),
        r"oʊ",
        "oː",
    ),
)

WORD_RULES: Tuple[WordRule, ...] = (
    WordRule(
        "england_yod_retention",
        ("england",),
        "RP retains yod after /t, d, n/ in a lexical exception set.",
        (
            "tune",
            "duty",
            "student",
            "during",
            "tube",
            "tuesday",
            "due",
            "dew",
            "duke",
            "nuclear",
        ),
        ("tune", "duty", "student"),
        r"([tdn])([ˈˌ]*)u",
        r"\1\2ju",
    ),
)

# Rules that are on by default for Indian English.  Kept as a named constant so
# an ablation can report exactly which subset was active.
INDIAN_CONSERVATIVE_RULES: Tuple[str, ...] = (
    "india_theta_stop",
    "india_eth_stop",
    "india_face_monophthong",
    "india_goat_monophthong",
)


def normalize_accent(accent: str) -> str:
    """Map UI names and legacy GLOBE labels to a canonical accent id."""
    key = str(accent).strip().lower().replace("-", "_").replace(" ", "_")
    if key not in ACCENT_ALIASES:
        raise ValueError(
            f"unsupported accent {accent!r}; expected one of {sorted(ACCENT_ALIASES)}"
        )
    return ACCENT_ALIASES[key]


def strip_stress(word: str) -> str:
    return word.replace("ˈ", "").replace("ˌ", "")


def normalize_ipa(text: str) -> str:
    """Normalize tie bars and sentence marks before inventory validation."""
    value = str(text)
    for source, replacement in _TIE_BAR_REPLACEMENTS:
        value = value.replace(source, replacement)
    return re.sub(r"\s+", " ", value).strip()


@lru_cache(maxsize=1)
def phoneme_inventory() -> frozenset[str]:
    """Return the StyleTTS2 TextCleaner symbol inventory."""
    from styletts2.text_utils import symbols

    return frozenset(symbols)


def unknown_characters(text: str) -> List[str]:
    """Return characters not present in the StyleTTS2 TextCleaner inventory."""
    inventory = phoneme_inventory()
    return sorted(
        {
            char
            for char in str(text)
            if char not in inventory and char not in {" ", "\t", "\n"}
        }
    )


def validate_inventory(text: str) -> None:
    """Raise if ``text`` contains symbols the StyleTTS2 TextCleaner drops."""
    unknown = unknown_characters(text)
    if unknown:
        raise ValueError(f"IPA text contains unknown symbols: {unknown}")


def sanitize_ipa(text: str) -> str:
    """Drop unknown symbols and collapse whitespace.

    This is the fallback path.  Callers should prefer a valid base phoneme when
    a transformed word contains an unknown symbol.
    """
    inventory = phoneme_inventory()
    cleaned = "".join(
        char for char in str(text) if char in inventory or char in {" ", "\t"}
    )
    return re.sub(r"\s+", " ", cleaned).strip()


def _phonemizer():
    from styletts2.phoneme import GruutPhonemizer

    return GruutPhonemizer()


@lru_cache(maxsize=1)
def _shared_phonemizer():
    return _phonemizer()


def phonemize_base(text: str, *, lang: str = "en-us") -> str:
    """Return normalized base IPA from StyleTTS2's gruut phonemizer."""
    return normalize_ipa(_shared_phonemizer().phonemize(str(text), lang=lang))


def apply_accent_rules(
    base_ipa: str,
    accent: str,
    *,
    strength: float = 1.0,
    source_word: Optional[str] = None,
    disabled_rules: Sequence[str] = (),
) -> str:
    """Apply ordered accent rules to a base IPA string.

    ``strength`` gates rules by their ``min_strength``.  It is a discrete
    on/off schedule, not a continuous phonetic morph; this is deliberate and
    documented so a strength value is not overinterpreted.

    ``disabled_rules`` is for ablation only.  Disabled rule ids are skipped
    while the rest of the ordered table still applies, so an ablation report
    can attribute an effect to a named rule instead of an undocumented edit.
    """
    value = normalize_ipa(base_ipa)
    accent_id = normalize_accent(accent)
    strength = float(strength)
    if strength <= 0.0:
        return value
    if not 0.0 <= strength <= 1.0:
        raise ValueError("strength must be in [0, 1]")
    disabled = frozenset(str(item) for item in disabled_rules)

    words = value.split()
    transformed_words: List[str] = []
    for word in words:
        transformed = word
        for rule in RULE_TABLE:
            if rule.rule_id in disabled:
                continue
            if accent_id in rule.accents and strength >= rule.min_strength:
                transformed = rule.apply(transformed)
        for rule in WORD_RULES:
            if rule.rule_id in disabled:
                continue
            if (
                accent_id in rule.accents
                and strength >= rule.min_strength
                and rule.applies_to_source(source_word)
            ):
                transformed = rule.apply(transformed)
        if unknown_characters(transformed):
            # Requirement: fall back to the base phoneme for unknown symbols.
            transformed = sanitize_ipa(word)
        transformed_words.append(transformed)

    output = " ".join(transformed_words)
    if unknown_characters(output):
        output = sanitize_ipa(output)
    return output


def phonemize_accent(
    text: str,
    accent: str,
    *,
    strength: float = 1.0,
    lang: str = "en-us",
    disabled_rules: Sequence[str] = (),
) -> str:
    """Phonemize ``text`` with the base gruut voice and accent rules.

    Words are phonemized individually so lexical word rules can see the
    English source word.  This is an experimental accent path; the canonical
    path still phonemizes whole sentences.
    """
    parts: List[str] = []
    for token in str(text).split():
        base = phonemize_base(token, lang=lang)
        if not base:
            continue
        parts.append(
            apply_accent_rules(
                base,
                accent,
                strength=strength,
                source_word=token,
                disabled_rules=disabled_rules,
            )
        )
    return " ".join(part for part in parts if part)


class AccentPhonemizer:
    """Drop-in object for StyleTTS2's ``phoneme_converter`` attribute."""

    def __init__(
        self,
        accent: str,
        *,
        strength: float = 1.0,
        lang: str = "en-us",
        disabled_rules: Sequence[str] = (),
    ):
        self.accent = normalize_accent(accent)
        self.strength = float(strength)
        self.lang = str(lang)
        self.disabled_rules = tuple(str(item) for item in disabled_rules)

    def phonemize(self, text: str, lang: Optional[str] = None) -> str:
        return phonemize_accent(
            text,
            self.accent,
            strength=self.strength,
            lang=lang or self.lang,
            disabled_rules=self.disabled_rules,
        )


_ACCENT_SWAP_LOCK = threading.RLock()


@contextmanager
def use_accent_phonemizer(
    backend: Any,
    accent: str,
    *,
    strength: float = 1.0,
    lang: str = "en-us",
    disabled_rules: Sequence[str] = (),
) -> Iterator[Any]:
    """Temporarily swap a loaded StyleTTS2Backend's phoneme converter.

    The canonical path is restored in ``finally``.  The swap is process-local
    and serialized with a lock, because StyleTTS2Backend shares one loaded
    model across instances.  ``disabled_rules`` is for ablation only.
    """
    backend.load()
    model = getattr(backend, "_model", None)
    if getattr(backend, "_use_fallback", True) or model is None:
        raise RuntimeError(
            "the neural StyleTTS2 backend is required for accent phonemization"
        )
    with _ACCENT_SWAP_LOCK:
        original = model.phoneme_converter
        model.phoneme_converter = AccentPhonemizer(
            accent,
            strength=strength,
            lang=lang,
            disabled_rules=disabled_rules,
        )
        try:
            yield model
        finally:
            model.phoneme_converter = original


def rule_table_report() -> Dict[str, Any]:
    """Machine-readable rule table for reports and tests."""
    return {
        "accents": list(ACCENT_IDS),
        "trainable_accents": list(TRAINABLE_ACCENTS),
        "prompt_set": list(ACCENT_PROMPT_SET),
        "sensitive_words": list(ACCENT_SENSITIVE_WORDS),
        "indian_conservative_rules": list(INDIAN_CONSERVATIVE_RULES),
        "rules": [rule.to_dict() for rule in RULE_TABLE],
        "word_rules": [
            {
                "rule_id": rule.rule_id,
                "accents": list(rule.accents),
                "description": rule.description,
                "words": list(rule.words),
                "examples": list(rule.examples),
                "pattern": rule.pattern,
                "replacement": rule.replacement,
                "min_strength": rule.min_strength,
            }
            for rule in WORD_RULES
        ],
    }
