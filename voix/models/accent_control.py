"""Accent and dialect modulation engine for VOIX.

Provides phonological, prosodic, and acoustic dialect transformations
to simulate regional accents while preserving biometric facial voice identity.
"""

from __future__ import annotations
from pathlib import Path

import re
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import scipy.signal
import torch


class AccentType(str, Enum):
    """Supported regional accents and dialects."""

    GENERAL_AMERICAN = "general_american"
    BRITISH_RP = "british_rp"
    AUSTRALIAN = "australian"
    INDIAN_ENGLISH = "indian_english"
    IRISH = "irish"
    SCOTTISH = "scottish"


@dataclass(frozen=True)
class AccentProfile:
    """Acoustic and prosodic parameters characteristic of a regional accent."""

    accent_id: str
    display_name: str
    region: str
    duration_scale: float  # Speech rate multiplier
    pitch_factor: float    # Relative F0 pitch multiplier
    prosody_dynamics: float  # Intonation variance multiplier (beta factor)
    description: str


ACCENT_REGISTRY: Dict[str, AccentProfile] = {
    AccentType.GENERAL_AMERICAN.value: AccentProfile(
        accent_id="general_american",
        display_name="General American",
        region="North America",
        duration_scale=1.00,
        pitch_factor=1.00,
        prosody_dynamics=1.00,
        description="Standard rhotic North American English baseline.",
    ),
    AccentType.BRITISH_RP.value: AccentProfile(
        accent_id="british_rp",
        display_name="British (Received Pronunciation)",
        region="United Kingdom",
        duration_scale=0.96,
        pitch_factor=1.03,
        prosody_dynamics=1.12,
        description="Non-rhotic prestige British English with broad 'a' and crisp plosives.",
    ),
    AccentType.AUSTRALIAN.value: AccentProfile(
        accent_id="australian",
        display_name="Australian English",
        region="Oceania",
        duration_scale=1.04,
        pitch_factor=1.02,
        prosody_dynamics=1.15,
        description="Non-rhotic English characterized by high rising terminals and diphthong shifts.",
    ),
    AccentType.INDIAN_ENGLISH.value: AccentProfile(
        accent_id="indian_english",
        display_name="Indian English",
        region="South Asia",
        duration_scale=1.08,
        pitch_factor=1.05,
        prosody_dynamics=0.92,
        description="Syllable-timed rhythm, retroflex consonant articulation, distinct prosody.",
    ),
    AccentType.IRISH.value: AccentProfile(
        accent_id="irish",
        display_name="Hiberno-English (Irish)",
        region="Ireland",
        duration_scale=0.98,
        pitch_factor=1.04,
        prosody_dynamics=1.22,
        description="Musical melodic intonation, dental stop substitutions, distinct cadence.",
    ),
    AccentType.SCOTTISH.value: AccentProfile(
        accent_id="scottish",
        display_name="Scottish English",
        region="Scotland",
        duration_scale=1.02,
        pitch_factor=0.98,
        prosody_dynamics=1.10,
        description="Rhotic tapped consonants, monophthongized vowels, distinct vowel duration.",
    ),
}


class AccentTransformer:
    """Performs phonetic, lexical, and prosodic text transformation for dialect adaptation."""

    @staticmethod
    def _apply_british_rp(text: str, strength: float) -> str:
        """Apply Received Pronunciation phonological adjustments."""
        if strength <= 0.05:
            return text

        words = text.split(" ")
        out = []
        rng = np.random.default_rng(len(text) + 11)

        for w in words:
            clean = w.lower().strip(".,!?;:\"'")
            # Broad 'a' in words like fast, pass, dance, castle
            if rng.random() < strength * 0.8:
                if clean in ("fast", "last", "past", "cast", "task", "ask", "path", "bath", "after", "class", "grass", "pass"):
                    w = re.sub(r"(?i)a", "ah", w, count=1)
                elif clean in ("dance", "chance", "glance", "can't", "cant"):
                    w = re.sub(r"(?i)a", "ah", w, count=1)
            out.append(w)
        return " ".join(out)

    @staticmethod
    def _apply_australian(text: str, strength: float) -> str:
        """Apply Australian English vowel shifts and prosodic markers."""
        if strength <= 0.05:
            return text

        words = text.split(" ")
        out = []
        rng = np.random.default_rng(len(text) + 22)

        for w in words:
            clean = w.lower().strip(".,!?;:\"'")
            # Australian vowel shifts: long /i:/ -> /eI/, /aI/ -> /oI/
            if rng.random() < strength * 0.7:
                if clean in ("day", "say", "way", "play", "mate", "today"):
                    w = re.sub(r"(?i)ay", "eye", w)
                elif clean in ("right", "night", "like", "time"):
                    w = re.sub(r"(?i)i([a-z]+e?)", r"oi", w)
            out.append(w)
        return " ".join(out)

    @staticmethod
    def _apply_indian_english(text: str, strength: float) -> str:
        """Apply Indian English syllable cadence and unaspirated phonetic cues."""
        if strength <= 0.05:
            return text

        words = text.split(" ")
        out = []
        rng = np.random.default_rng(len(text) + 33)

        for w in words:
            clean = w.lower().strip(".,!?;:\"'")
            if rng.random() < strength * 0.6:
                # Interdental fricative 'th' often realized as dental stop 't' or 'd'
                if clean.startswith("th"):
                    w = re.sub(r"(?i)th", "t", w)
            out.append(w)
        return " ".join(out)

    @staticmethod
    def _apply_irish(text: str, strength: float) -> str:
        """Apply Hiberno-English phonetic adjustments."""
        if strength <= 0.05:
            return text

        words = text.split(" ")
        out = []
        rng = np.random.default_rng(len(text) + 44)

        for w in words:
            clean = w.lower().strip(".,!?;:\"'")
            if rng.random() < strength * 0.75:
                # 'th' -> 't' or 'd' in common words
                if clean in ("think", "thing", "thought", "three", "thanks"):
                    w = re.sub(r"(?i)th", "t", w)
                elif clean in ("that", "this", "these", "those", "there", "them", "then"):
                    w = re.sub(r"(?i)th", "d", w)
            out.append(w)
        return " ".join(out)

    @staticmethod
    def _apply_scottish(text: str, strength: float) -> str:
        """Apply Scottish English phonological adaptations."""
        if strength <= 0.05:
            return text

        words = text.split(" ")
        out = []
        rng = np.random.default_rng(len(text) + 55)

        for w in words:
            clean = w.lower().strip(".,!?;:\"'")
            if rng.random() < strength * 0.7:
                if clean in ("cannot", "can't"):
                    w = "canna"
                elif clean in ("did", "didn't"):
                    w = "didna"
                elif clean in ("house", "about", "out"):
                    w = re.sub(r"(?i)ou", "oo", w)
            out.append(w)
        return " ".join(out)

    @classmethod
    def transform(
        cls,
        text: str,
        accent: Optional[str] = None,
        strength: float = 0.5,
    ) -> Tuple[str, Dict[str, float]]:
        """Transform text and return acoustic parameter modifiers for the specified accent.

        Args:
            text: Input text prompt.
            accent: Target accent identifier (e.g. 'british_rp', 'australian').
            strength: Dialect intensity in [0.0, 1.0].

        Returns:
            Tuple of (transformed_text, acoustic_modifiers_dict).
        """
        if not accent or accent == AccentType.GENERAL_AMERICAN.value:
            return text, {"duration_mult": 1.0, "pitch_mult": 1.0, "prosody_mult": 1.0}

        profile = ACCENT_REGISTRY.get(accent.lower())
        if profile is None:
            raise ValueError(f"Unknown accent '{accent}'. Supported: {list(ACCENT_REGISTRY.keys())}")

        str_val = float(np.clip(strength, 0.0, 1.0))

        # 1. Phonetic Text Transformation
        if accent == AccentType.BRITISH_RP.value:
            text_mod = cls._apply_british_rp(text, str_val)
        elif accent == AccentType.AUSTRALIAN.value:
            text_mod = cls._apply_australian(text, str_val)
        elif accent == AccentType.INDIAN_ENGLISH.value:
            text_mod = cls._apply_indian_english(text, str_val)
        elif accent == AccentType.IRISH.value:
            text_mod = cls._apply_irish(text, str_val)
        elif accent == AccentType.SCOTTISH.value:
            text_mod = cls._apply_scottish(text, str_val)
        else:
            text_mod = text

        # 2. Acoustic Multipliers (scaled by strength)
        dur_mult = 1.0 + (profile.duration_scale - 1.0) * str_val
        pitch_mult = 1.0 + (profile.pitch_factor - 1.0) * str_val
        prosody_mult = 1.0 + (profile.prosody_dynamics - 1.0) * str_val

        modifiers = {
            "duration_mult": float(dur_mult),
            "pitch_mult": float(pitch_mult),
            "prosody_mult": float(prosody_mult),
        }

        return text_mod, modifiers


def list_available_accents() -> List[Dict[str, Any]]:
    """List all supported regional accents and dialect profiles."""
    return [
        {
            "accent_id": p.accent_id,
            "display_name": p.display_name,
            "region": p.region,
            "description": p.description,
            "duration_scale": p.duration_scale,
            "pitch_factor": p.pitch_factor,
            "prosody_dynamics": p.prosody_dynamics,
        }
        for p in ACCENT_REGISTRY.values()
    ]

# ---------------------------------------------------------------------------
# Style-Space Accent Delta Functions
# ---------------------------------------------------------------------------
_ACCENT_DELTA_DIR = Path(__file__).resolve().parent.parent.parent / "checkpoints" / "accent_deltas"
_ACCENT_DELTA_CACHE: dict = {}

def get_accent_style_delta(accent_id: str, device: str = "cpu") -> torch.Tensor | None:
    """Retrieve precomputed 256-D style delta tensor for given accent."""
    if accent_id in _ACCENT_DELTA_CACHE:
        return _ACCENT_DELTA_CACHE[accent_id].to(device)
    path = _ACCENT_DELTA_DIR / f"{accent_id}.pt"
    if not path.exists():
        return None
    try:
        data = torch.load(path, map_location=device, weights_only=False)
        delta = data.get("delta", None)
        if delta is not None and isinstance(delta, torch.Tensor):
            _ACCENT_DELTA_CACHE[accent_id] = delta.cpu()
            return delta.to(device)
    except Exception:
        return None
    return None

def apply_accent_style_delta(ref_s: torch.Tensor, accent_id: str, strength: float = 1.0) -> torch.Tensor:
    """Modulate the prosody half [128:] of a 256-D reference style vector."""
    if strength <= 0.0 or accent_id in ("general_american", "default"):
        return ref_s
    delta = get_accent_style_delta(accent_id, device=str(ref_s.device))
    if delta is None:
        return ref_s
    is_1d = (ref_s.dim() == 1)
    if is_1d:
        ref_s = ref_s.unsqueeze(0)
    s_mod = ref_s.clone()
    if s_mod.shape[-1] >= 256:
        d_prosody = delta[128:256] if delta.shape[0] >= 256 else delta
        s_mod[:, 128:256] = s_mod[:, 128:256] + float(strength) * d_prosody.to(s_mod.device)
    else:
        s_mod = s_mod + float(strength) * delta[:128].to(s_mod.device)
    return s_mod.squeeze(0) if is_1d else s_mod