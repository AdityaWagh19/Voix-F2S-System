"""Accent registry and serving-side accent controls.

The supported set is deliberately small and evidence-bounded:

    us, england, indian_english, australian

Phoneme-level accent transforms live in :mod:`voix.models.accent_g2p` and are
applied to the neural StyleTTS2 phonemizer.  This module owns the canonical
names, the prosody/serving modifiers, and the optional fixed-delta artifacts.
It does not claim perceptual accent identification.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from voix.models.accent_residual import apply_accent_residual


class AccentType(str, Enum):
    """Supported regional accents."""

    US = "us"
    ENGLAND = "england"
    INDIAN_ENGLISH = "indian_english"
    AUSTRALIAN = "australian"


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

# Accent ids used by the GLOBE class-mean artifact.
RESIDUAL_KEYS: Dict[str, str] = {
    "us": "us",
    "england": "england",
    "indian_english": "india_south_asia",
    "australian": "australian",
}


def normalize_accent(accent: str) -> str:
    """Map aliases and legacy names to a canonical accent id."""
    key = str(accent).strip().lower().replace("-", "_").replace(" ", "_")
    if key not in ACCENT_ALIASES:
        raise ValueError(
            f"Unknown accent '{accent}'. Supported: {sorted(ACCENT_ALIASES)}"
        )
    return ACCENT_ALIASES[key]


@dataclass(frozen=True)
class AccentProfile:
    """Prosodic parameters for a regional accent."""

    accent_id: str
    display_name: str
    region: str
    duration_scale: float
    pitch_factor: float
    prosody_dynamics: float
    description: str


ACCENT_REGISTRY: Dict[str, AccentProfile] = {
    "us": AccentProfile(
        accent_id="us",
        display_name="US English",
        region="North America",
        duration_scale=1.00,
        pitch_factor=1.00,
        prosody_dynamics=1.00,
        description="Rhotic General American baseline; no phoneme rewrite.",
    ),
    "england": AccentProfile(
        accent_id="england",
        display_name="English (England)",
        region="United Kingdom",
        duration_scale=1.02,
        pitch_factor=1.00,
        prosody_dynamics=1.06,
        description="Non-rhotic Southern British English with trap-bath splitting.",
    ),
    "indian_english": AccentProfile(
        accent_id="indian_english",
        display_name="Indian English",
        region="South Asia",
        duration_scale=1.04,
        pitch_factor=1.00,
        prosody_dynamics=0.94,
        description="Syllable-timed Indian English with a conservative consonant subset.",
    ),
    "australian": AccentProfile(
        accent_id="australian",
        display_name="Australian English",
        region="Oceania",
        duration_scale=1.03,
        pitch_factor=1.00,
        prosody_dynamics=1.08,
        description="Non-rhotic Australian English with broad FACE/GOAT shifts.",
    ),
}


def accent_residual_key(accent: str) -> str:
    """Return the GLOBE residual/class key for a canonical accent id."""
    return RESIDUAL_KEYS[normalize_accent(accent)]


class AccentTransformer:
    """Return serving-side prosody modifiers for a canonical accent.

    Text is intentionally returned unchanged.  Ad-hoc English respellings were
    removed because they were not phonological and increased word error rate.
    Neural accent phonemization is handled by ``use_accent_phonemizer`` in the
    inference pipeline; the fallback backend has no phoneme stage.
    """

    @classmethod
    def transform(
        cls,
        text: str,
        accent: Optional[str] = None,
        strength: float = 0.5,
    ) -> Tuple[str, Dict[str, float]]:
        if not accent:
            return text, {
                "duration_mult": 1.0,
                "pitch_mult": 1.0,
                "prosody_mult": 1.0,
            }

        accent_id = normalize_accent(accent)
        profile = ACCENT_REGISTRY[accent_id]
        strength = float(np.clip(strength, 0.0, 1.0))
        return text, {
            "duration_mult": float(
                1.0 + (profile.duration_scale - 1.0) * strength
            ),
            "pitch_mult": float(1.0 + (profile.pitch_factor - 1.0) * strength),
            "prosody_mult": float(
                1.0 + (profile.prosody_dynamics - 1.0) * strength
            ),
        }


def list_available_accents() -> List[Dict[str, Any]]:
    """List the supported accents and their serving-side parameters."""
    return [
        {
            "accent_id": profile.accent_id,
            "display_name": profile.display_name,
            "region": profile.region,
            "description": profile.description,
            "duration_scale": profile.duration_scale,
            "pitch_factor": profile.pitch_factor,
            "prosody_dynamics": profile.prosody_dynamics,
        }
        for profile in ACCENT_REGISTRY.values()
    ]


_ACCENT_DELTA_DIR = (
    Path(__file__).resolve().parent.parent.parent / "checkpoints" / "accent_deltas"
)
_ACCENT_DELTA_CACHE: Dict[str, torch.Tensor] = {}
_DELTA_FILENAMES = {
    "us": "general_american.pt",
    "england": "british_rp.pt",
    "indian_english": "indian_english.pt",
    "australian": "australian.pt",
}


def get_accent_style_delta(
    accent_id: str, device: str = "cpu"
) -> Optional[torch.Tensor]:
    """Load the optional fixed accent delta for reproducibility/compatibility."""
    canonical = normalize_accent(accent_id)
    if canonical in _ACCENT_DELTA_CACHE:
        return _ACCENT_DELTA_CACHE[canonical].to(device)
    path = _ACCENT_DELTA_DIR / _DELTA_FILENAMES[canonical]
    if not path.exists():
        return None
    try:
        payload = torch.load(path, map_location=device, weights_only=False)
        delta = payload.get("delta") if isinstance(payload, dict) else None
        if isinstance(delta, torch.Tensor):
            _ACCENT_DELTA_CACHE[canonical] = delta.detach().cpu()
            return delta.to(device)
    except Exception:
        return None
    return None


def apply_accent_style_delta(
    ref_s: torch.Tensor, accent_id: str, strength: float = 1.0
) -> torch.Tensor:
    """Apply the legacy fixed accent delta with identity-preserving scaling."""
    if strength <= 0.0:
        return ref_s
    delta = get_accent_style_delta(accent_id, device=str(ref_s.device))
    if delta is None:
        return ref_s
    is_1d = ref_s.dim() == 1
    if is_1d:
        ref_s = ref_s.unsqueeze(0)
    adjusted = ref_s.clone()
    effective = float(strength) * 0.12
    if adjusted.shape[-1] >= 256:
        adjusted[:, 128:256] = (
            adjusted[:, 128:256] + effective * delta[128:256].to(adjusted.device)
        )
    else:
        adjusted = adjusted + effective * delta[: adjusted.shape[-1]].to(
            adjusted.device
        )
    return adjusted.squeeze(0) if is_1d else adjusted
