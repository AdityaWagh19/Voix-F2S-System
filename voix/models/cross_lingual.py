"""Cross-lingual voice transfer engine for VOIX.

Synthesizes speech across multiple target languages while preserving the speaker's
biometric identity, vocal tract resonance, and demographic profile from their facial features.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any


class LanguageType(str, Enum):
    """Supported target languages for cross-lingual synthesis."""

    ENGLISH = "en"
    SPANISH = "es"
    FRENCH = "fr"
    GERMAN = "de"
    ITALIAN = "it"


@dataclass(frozen=True)
class LanguageProfile:
    """Linguistic and acoustic profile for cross-lingual synthesis."""

    code: str
    name: str
    family: str
    rhythm_type: str  # 'syllable-timed' vs 'stress-timed'
    duration_scale: float  # Base pacing multiplier
    pitch_dynamics_factor: float  # Beta prosodic multiplier
    description: str


LANGUAGE_REGISTRY: Dict[str, LanguageProfile] = {
    LanguageType.ENGLISH.value: LanguageProfile(
        code="en",
        name="English",
        family="Germanic",
        rhythm_type="stress-timed",
        duration_scale=1.00,
        pitch_dynamics_factor=1.00,
        description="Standard English baseline synthesis.",
    ),
    LanguageType.SPANISH.value: LanguageProfile(
        code="es",
        name="Spanish (Español)",
        family="Romance",
        rhythm_type="syllable-timed",
        duration_scale=1.02,
        pitch_dynamics_factor=1.08,
        description="Syllable-timed Romance language with pure vowels and rapid dental articulation.",
    ),
    LanguageType.FRENCH.value: LanguageProfile(
        code="fr",
        name="French (Français)",
        family="Romance",
        rhythm_type="syllable-timed",
        duration_scale=0.98,
        pitch_dynamics_factor=1.04,
        description="Non-tonal Romance language with nasal vowel dynamics and continuous liaisons.",
    ),
    LanguageType.GERMAN.value: LanguageProfile(
        code="de",
        name="German (Deutsch)",
        family="Germanic",
        rhythm_type="stress-timed",
        duration_scale=1.06,
        pitch_dynamics_factor=0.95,
        description="Stress-timed Germanic language with consonant clusters and distinct word stress.",
    ),
    LanguageType.ITALIAN.value: LanguageProfile(
        code="it",
        name="Italian (Italiano)",
        family="Romance",
        rhythm_type="syllable-timed",
        duration_scale=1.00,
        pitch_dynamics_factor=1.14,
        description="Melodic syllable-timed cadence with open resonant vowels and geminate consonants.",
    ),
}


class MultilingualPhoneticAdapter:
    """Adapts multilingual text orthography and phonology for neural vocoder rendering."""

    @staticmethod
    def _adapt_spanish(text: str) -> str:
        """Adapt Spanish phonology to phonetic orthography."""
        t = text
        subs = [
            (r"(?i)\bbuenos d[íi]as\b", "Buenos deeas"),
            (r"(?i)\bd[íi]as\b", "deeas"),
            (r"(?i)\bs[íi]\b", "see"),
            (r"(?i)\bgracias\b", "grahtsyas"),
            (r"(?i)ci[óo]n\b", "syon"),
            (r"(?i)ci[óo]nes\b", "syones"),
            (r"(?i)ll([aeiou])", r"y\g<1>"),
            (r"(?i)ñ([aeiou])", r"ny\g<1>"),
            (r"(?i)qu([ei])", r"k\g<1>"),
            (r"(?i)gu([ei])", r"g\g<1>"),
            (r"(?i)z([aeiou])", r"s\g<1>"),
            (r"(?i)\bce\b", "seh"),
            (r"(?i)\bci\b", "see"),
        ]
        for pattern, repl in subs:
            t = re.sub(pattern, repl, t)
        t = unicodedata.normalize('NFKD', t).encode('ASCII', 'ignore').decode('utf-8')
        return t

    @staticmethod
    def _adapt_french(text: str) -> str:
        """Adapt French phonology to phonetic orthography."""
        t = text
        subs = [
            (r"(?i)\bbonjour\b", "bohn-zhoor"),
            (r"(?i)\bmerci\b", "mair-see"),
            (r"(?i)\bceci\b", "seh-see"),
            (r"(?i)\best\b", "eh"),
            (r"(?i)\bune\b", "oon"),
            (r"(?i)\bun\b", "uhn"),
            (r"(?i)\bdu\b", "doo"),
            (r"(?i)\bde\b", "deh"),
            (r"(?i)tion\b", "syon"),
            (r"(?i)eau\b", "oh"),
            (r"(?i)eux\b", "uh"),
            (r"(?i)ou([a-z]+)", r"oo\g<1>"),
            (r"(?i)ch([aeiou])", r"sh\g<1>"),
        ]
        for pattern, repl in subs:
            t = re.sub(pattern, repl, t)
        t = unicodedata.normalize('NFKD', t).encode('ASCII', 'ignore').decode('utf-8')
        return t

    @staticmethod
    def _adapt_german(text: str) -> str:
        """Adapt German phonology to phonetic orthography."""
        t = text
        subs = [
            (r"(?i)\bguten tag\b", "goo-ten tahg"),
            (r"(?i)\bdies ist\b", "dees ist"),
            (r"(?i)\beine\b", "eyne"),
            (r"(?i)\bbitte\b", "bit-teh"),
            (r"(?i)\bdanke\b", "dahn-keh"),
            (r"(?i)sch", "sh"),
            (r"(?i)sp([aeiou])", r"shp\g<1>"),
            (r"(?i)st([aeiou])", r"sht\g<1>"),
            (r"(?i)ie", "ee"),
            (r"(?i)ei", "eye"),
            (r"(?i)tion\b", "ts-yohn"),
            (r"(?i)tz", "ts"),
        ]
        for pattern, repl in subs:
            t = re.sub(pattern, repl, t)
        t = unicodedata.normalize('NFKD', t).encode('ASCII', 'ignore').decode('utf-8')
        return t

    @staticmethod
    def _adapt_italian(text: str) -> str:
        """Adapt Italian phonology to phonetic orthography."""
        t = text
        subs = [
            (r"(?i)\bbuongiorno\b", "bwon-jorno"),
            (r"(?i)\bquesta\b", "kwesta"),
            (r"(?i)\bquesto\b", "kwesto"),
            (r"(?i)\bgrazie\b", "graht-syeh"),
            (r"(?i)\bprego\b", "preh-goh"),
            (r"(?i)zione\b", "ts-yoh-neh"),
            (r"(?i)zioni\b", "ts-yoh-nee"),
            (r"(?i)gli([aeiou])", r"ly\g<1>"),
            (r"(?i)gn([aeiou])", r"ny\g<1>"),
            (r"(?i)ch([ei])", r"k\g<1>"),
            (r"(?i)gh([ei])", r"g\g<1>"),
        ]
        for pattern, repl in subs:
            t = re.sub(pattern, repl, t)
        t = unicodedata.normalize('NFKD', t).encode('ASCII', 'ignore').decode('utf-8')
        return t

    @classmethod
    def transform(
        cls,
        text: str,
        language: str = "en",
    ) -> Tuple[str, Dict[str, float]]:
        """Transform text for the target language and compute prosodic multipliers.

        Args:
            text: Input text prompt.
            language: Target ISO-639-1 language code (e.g. 'es', 'fr', 'de', 'it', 'en').

        Returns:
            Tuple of (adapted_text, prosodic_modifiers_dict).
        """
        lang = language.lower() if language else "en"
        if lang not in LANGUAGE_REGISTRY:
            raise ValueError(f"Unsupported language '{language}'. Supported: {list(LANGUAGE_REGISTRY.keys())}")

        profile = LANGUAGE_REGISTRY[lang]

        if lang == "es":
            adapted_text = cls._adapt_spanish(text)
        elif lang == "fr":
            adapted_text = cls._adapt_french(text)
        elif lang == "de":
            adapted_text = cls._adapt_german(text)
        elif lang == "it":
            adapted_text = cls._adapt_italian(text)
        else:
            adapted_text = text

        modifiers = {
            "duration_scale": profile.duration_scale,
            "pitch_dynamics_factor": profile.pitch_dynamics_factor,
        }

        return adapted_text, modifiers


def list_supported_languages() -> List[Dict[str, Any]]:
    """List all supported target languages and profiles for cross-lingual transfer."""
    return [
        {
            "code": p.code,
            "name": p.name,
            "family": p.family,
            "rhythm_type": p.rhythm_type,
            "duration_scale": p.duration_scale,
            "pitch_dynamics_factor": p.pitch_dynamics_factor,
            "description": p.description,
        }
        for p in LANGUAGE_REGISTRY.values()
    ]
