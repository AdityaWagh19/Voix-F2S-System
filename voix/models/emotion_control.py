"""Discrete emotion profiles and serving-side prosody controls.

Only ``neutral``, ``happy``, ``sad`` and ``angry`` are supported.  These are
the three non-neutral directions that passed the E1 held-out direction checks
on CREMA-D; ``whispered`` and ``surprised`` were removed because they have no
validated direction.

The profiles deliberately keep pitch factors close to 1.0.  Large post-hoc
pitch shifts damaged ECAPA speaker identity in the audit.  Emotion is carried
mainly by the E1 prosody residual plus bounded duration and energy shaping.
Perceptual distinctness is not claimed without a listener study.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from voix.utils.audio import limit_true_peak


class EmotionType(str, Enum):
    """Supported discrete affective states."""

    NEUTRAL = "neutral"
    HAPPY = "happy"
    SAD = "sad"
    ANGRY = "angry"


EMOTION_ALIASES: Dict[str, str] = {
    "neutral": "neutral",
    "default": "neutral",
    "happy": "happy",
    "joyful": "happy",
    "sad": "sad",
    "angry": "angry",
    "anger": "angry",
}


def normalize_emotion(emotion: Optional[str]) -> Optional[str]:
    """Map aliases to a canonical emotion id; ``None`` means no control."""
    if emotion is None:
        return None
    key = str(emotion).strip().lower()
    if key in ("", "none"):
        return None
    if key not in EMOTION_ALIASES:
        raise ValueError(
            f"Unknown emotion '{emotion}'. Supported: {sorted(EMOTION_ALIASES)}"
        )
    return EMOTION_ALIASES[key]


@dataclass(frozen=True)
class EmotionProfile:
    """Acoustic and prosodic signature for an emotional state."""

    emotion_id: str
    display_name: str
    pitch_factor: float
    pitch_dynamics_factor: float
    duration_scale: float
    energy_scale: float
    description: str


EMOTION_REGISTRY: Dict[str, EmotionProfile] = {
    "neutral": EmotionProfile(
        emotion_id="neutral",
        display_name="Neutral",
        pitch_factor=1.00,
        pitch_dynamics_factor=1.00,
        duration_scale=1.00,
        energy_scale=1.00,
        description="Balanced conversational baseline.",
    ),
    "happy": EmotionProfile(
        emotion_id="happy",
        display_name="Happy",
        pitch_factor=1.04,
        pitch_dynamics_factor=1.25,
        duration_scale=0.94,
        energy_scale=1.10,
        description="Slightly higher register, wider pitch movement, faster and louder.",
    ),
    "sad": EmotionProfile(
        emotion_id="sad",
        display_name="Sad",
        pitch_factor=0.97,
        pitch_dynamics_factor=0.70,
        duration_scale=1.10,
        energy_scale=0.88,
        description="Flatter and slower delivery with reduced vocal effort.",
    ),
    "angry": EmotionProfile(
        emotion_id="angry",
        display_name="Angry",
        pitch_factor=1.05,
        pitch_dynamics_factor=1.15,
        duration_scale=0.93,
        energy_scale=1.15,
        description="Faster, louder delivery with sharper pitch movement.",
    ),
}


class EmotionModulator:
    """Compute intensity-scaled prosody, duration and energy modifiers."""

    @classmethod
    def get_acoustic_modifiers(
        cls,
        emotion: Optional[str] = None,
        intensity: float = 0.5,
    ) -> Dict[str, float]:
        canonical = normalize_emotion(emotion)
        if canonical is None or canonical == "neutral":
            return {
                "pitch_factor": 1.0,
                "beta_mult": 1.0,
                "duration_scale": 1.0,
                "energy_scale": 1.0,
            }

        profile = EMOTION_REGISTRY[canonical]
        weight = float(np.clip(intensity, 0.0, 1.0))
        return {
            "pitch_factor": float(
                1.0 + (profile.pitch_factor - 1.0) * weight
            ),
            "beta_mult": float(
                1.0 + (profile.pitch_dynamics_factor - 1.0) * weight
            ),
            "duration_scale": float(
                1.0 + (profile.duration_scale - 1.0) * weight
            ),
            "energy_scale": float(
                1.0 + (profile.energy_scale - 1.0) * weight
            ),
        }

    @classmethod
    def apply_acoustic_filter(
        cls,
        waveform: torch.Tensor,
        sample_rate: int = 24000,
        emotion: Optional[str] = None,
        intensity: float = 0.5,
    ) -> torch.Tensor:
        """Apply only a true-peak safety limiter.

        Post-synthesis IIR filters smear phase and damage identity.  Duration,
        energy and style-space prosody carry the emotion instead.
        """
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        wav_np = waveform.detach().cpu().numpy().squeeze()
        return (
            torch.from_numpy(limit_true_peak(wav_np))
            .unsqueeze(0)
            .to(waveform.device)
        )


def list_available_emotions() -> List[Dict[str, Any]]:
    """List supported emotion profiles."""
    return [
        {
            "emotion_id": profile.emotion_id,
            "display_name": profile.display_name,
            "pitch_factor": profile.pitch_factor,
            "pitch_dynamics_factor": profile.pitch_dynamics_factor,
            "duration_scale": profile.duration_scale,
            "energy_scale": profile.energy_scale,
            "description": profile.description,
        }
        for profile in EMOTION_REGISTRY.values()
    ]


_EMOTION_DELTA_DIR = (
    Path(__file__).resolve().parent.parent.parent / "checkpoints" / "emotion_deltas"
)
_EMOTION_DELTA_CACHE: Dict[str, torch.Tensor] = {}


def get_emotion_style_delta(
    emotion_id: str, device: str = "cpu"
) -> Optional[torch.Tensor]:
    """Load the optional fixed emotion delta for reproducibility/compatibility."""
    canonical = normalize_emotion(emotion_id)
    if canonical is None or canonical == "neutral":
        return None
    if canonical in _EMOTION_DELTA_CACHE:
        return _EMOTION_DELTA_CACHE[canonical].to(device)
    path = _EMOTION_DELTA_DIR / f"{canonical}.pt"
    if not path.exists():
        return None
    try:
        payload = torch.load(path, map_location=device, weights_only=False)
        delta = payload.get("delta") if isinstance(payload, dict) else None
        if isinstance(delta, torch.Tensor):
            _EMOTION_DELTA_CACHE[canonical] = delta.detach().cpu()
            return delta.to(device)
    except Exception:
        return None
    return None


def apply_emotion_style_delta(
    ref_s: torch.Tensor, emotion_id: str, intensity: float = 1.0
) -> torch.Tensor:
    """Apply the legacy fixed emotion delta with a small prosody-only scale."""
    canonical = normalize_emotion(emotion_id)
    if intensity <= 0.0 or canonical is None or canonical == "neutral":
        return ref_s
    delta = get_emotion_style_delta(canonical, device=str(ref_s.device))
    if delta is None:
        return ref_s
    is_1d = ref_s.dim() == 1
    if is_1d:
        ref_s = ref_s.unsqueeze(0)
    adjusted = ref_s.clone()
    effective = float(intensity) * 0.18
    if adjusted.shape[-1] >= 256:
        adjusted[:, 128:256] = (
            adjusted[:, 128:256] + effective * delta[128:256].to(adjusted.device)
        )
    else:
        adjusted = adjusted + effective * delta[: adjusted.shape[-1]].to(
            adjusted.device
        )
    return adjusted.squeeze(0) if is_1d else adjusted
