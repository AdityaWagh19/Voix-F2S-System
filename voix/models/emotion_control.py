"""Emotion and affective tone modulation engine for VOIX.

Modulates prosodic dynamic range, fundamental frequency F0, speech tempo,
energy intensity, and spectral acoustic coloration across discrete emotional states.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import scipy.signal
import torch


class EmotionType(str, Enum):
    """Supported discrete affective states."""

    NEUTRAL = "neutral"
    HAPPY = "happy"
    SAD = "sad"
    ANGRY = "angry"
    WHISPERED = "whispered"
    SURPRISED = "surprised"


@dataclass(frozen=True)
class EmotionProfile:
    """Acoustic and prosodic signature for an emotional state."""

    emotion_id: str
    display_name: str
    pitch_factor: float          # F0 fundamental frequency multiplier
    pitch_dynamics_factor: float # Intonation and pitch excursion multiplier (beta factor)
    duration_scale: float        # Speech tempo factor (rate multiplier)
    energy_scale: float          # Relative acoustic RMS amplitude multiplier
    description: str


EMOTION_REGISTRY: Dict[str, EmotionProfile] = {
    EmotionType.NEUTRAL.value: EmotionProfile(
        emotion_id="neutral",
        display_name="Neutral",
        pitch_factor=1.00,
        pitch_dynamics_factor=1.00,
        duration_scale=1.00,
        energy_scale=1.00,
        description="Balanced, natural conversational baseline.",
    ),
    EmotionType.HAPPY.value: EmotionProfile(
        emotion_id="happy",
        display_name="Happy / Joyful",
        pitch_factor=1.08,
        pitch_dynamics_factor=1.25,
        duration_scale=0.92,
        energy_scale=1.08,
        description="Elevated pitch, expanded melodic dynamic range, and brisk energetic tempo.",
    ),
    EmotionType.SAD.value: EmotionProfile(
        emotion_id="sad",
        display_name="Sad / Somber",
        pitch_factor=0.93,
        pitch_dynamics_factor=0.72,
        duration_scale=1.14,
        energy_scale=0.88,
        description="Subdued fundamental pitch, reduced melodic variation, and elongated contemplative pacing.",
    ),
    EmotionType.ANGRY.value: EmotionProfile(
        emotion_id="angry",
        display_name="Angry / Assertive",
        pitch_factor=1.10,
        pitch_dynamics_factor=1.18,
        duration_scale=0.88,
        energy_scale=1.15,
        description="Intense vocal tract tension, sharp articulation, high acoustic energy.",
    ),
    EmotionType.WHISPERED.value: EmotionProfile(
        emotion_id="whispered",
        display_name="Whispered / Intimate",
        pitch_factor=0.96,
        pitch_dynamics_factor=0.55,
        duration_scale=1.08,
        energy_scale=0.75,
        description="Unvoiced breathy aspiration, subdued low harmonics, intimate proximity.",
    ),
    EmotionType.SURPRISED.value: EmotionProfile(
        emotion_id="surprised",
        display_name="Surprised / Excited",
        pitch_factor=1.14,
        pitch_dynamics_factor=1.35,
        duration_scale=0.90,
        energy_scale=1.10,
        description="High terminal pitch excursions, wide frequency dynamics, rapid attack.",
    ),
}


class EmotionModulator:
    """Manages affective prosody parameters and post-synthesis spectral filtering."""

    @classmethod
    def get_acoustic_modifiers(
        cls,
        emotion: Optional[str] = None,
        intensity: float = 0.5,
    ) -> Dict[str, float]:
        """Compute prosodic and vocoder multipliers for the given emotion.

        Args:
            emotion: Target emotion identifier (e.g. 'happy', 'sad').
            intensity: Emotional expression strength in [0.0, 1.0].

        Returns:
            Dictionary of multipliers for beta (pitch factor), duration, and energy.
        """
        if not emotion or emotion == EmotionType.NEUTRAL.value:
            return {
                "pitch_factor": 1.0,
                "beta_mult": 1.0,
                "duration_scale": 1.0,
                "energy_scale": 1.0,
            }

        profile = EMOTION_REGISTRY.get(emotion.lower())
        if profile is None:
            raise ValueError(f"Unknown emotion '{emotion}'. Supported: {list(EMOTION_REGISTRY.keys())}")

        int_val = float(np.clip(intensity, 0.0, 1.0))

        # Modulate parameters linearly by intensity
        pitch_factor = 1.0 + (profile.pitch_factor - 1.0) * int_val
        beta_mult = 1.0 + (profile.pitch_dynamics_factor - 1.0) * int_val
        dur_scale = 1.0 + (profile.duration_scale - 1.0) * int_val
        energy_scale = 1.0 + (profile.energy_scale - 1.0) * int_val

        return {
            "pitch_factor": float(pitch_factor),
            "beta_mult": float(beta_mult),
            "duration_scale": float(dur_scale),
            "energy_scale": float(energy_scale),
        }

    @classmethod
    def apply_acoustic_filter(
        cls,
        waveform: torch.Tensor,
        sample_rate: int = 24000,
        emotion: Optional[str] = None,
        intensity: float = 0.5,
    ) -> torch.Tensor:
        """Apply post-synthesis spectral shaping characteristic of the emotion.

        Args:
            waveform: Audio tensor (1, T) or (T,).
            sample_rate: Audio sampling rate in Hz.
            emotion: Target emotion name.
            intensity: Emotional expression strength in [0.0, 1.0].

        Returns:
            Spectrally modulated audio tensor of shape (1, T).
        """
        if not emotion or emotion == EmotionType.NEUTRAL.value or intensity <= 0.01:
            return waveform if waveform.dim() == 2 else waveform.unsqueeze(0)

        int_val = float(np.clip(intensity, 0.0, 1.0))
        wav_np = waveform.squeeze().cpu().numpy().astype(np.float64)

        if emotion == EmotionType.WHISPERED.value:
            # Whisper: High-pass attenuation below 400 Hz (suppress vocal cord vibration)
            # and mild boost in 3 kHz - 6 kHz aspiration band
            b_hp, a_hp = scipy.signal.butter(N=2, Wn=350.0, btype="highpass", fs=sample_rate)
            filtered = scipy.signal.lfilter(b_hp, a_hp, wav_np)

            # High-shelf aspiration boost
            b_asp, a_asp = scipy.signal.iirpeak(w0=4200.0, Q=1.5, fs=sample_rate)
            asp = scipy.signal.lfilter(b_asp, a_asp, wav_np)

            blend = min(0.70, int_val * 0.75)
            output = (1.0 - blend) * wav_np + blend * (filtered + 0.3 * asp)

        elif emotion == EmotionType.ANGRY.value:
            # Angry: Boost aggressive presence in 2.8 kHz - 4.5 kHz band
            b_pres, a_pres = scipy.signal.iirpeak(w0=3400.0, Q=2.0, fs=sample_rate)
            pres = scipy.signal.lfilter(b_pres, a_pres, wav_np)

            blend = min(0.35, int_val * 0.40)
            output = wav_np + blend * pres * 0.6

        elif emotion == EmotionType.SAD.value:
            # Sad: Mild low-pass roll-off above 4.5 kHz (somber, de-emphasized brightness)
            b_lp, a_lp = scipy.signal.butter(N=2, Wn=4200.0, btype="lowpass", fs=sample_rate)
            filtered = scipy.signal.lfilter(b_lp, a_lp, wav_np)

            blend = min(0.40, int_val * 0.45)
            output = (1.0 - blend) * wav_np + blend * filtered

        elif emotion == EmotionType.HAPPY.value:
            # Happy: Mild brightness boost around 3.5 kHz (smiling acoustic resonance)
            b_br, a_br = scipy.signal.iirpeak(w0=3500.0, Q=1.8, fs=sample_rate)
            bright = scipy.signal.lfilter(b_br, a_br, wav_np)

            blend = min(0.25, int_val * 0.30)
            output = wav_np + blend * bright * 0.4

        else:
            output = wav_np

        # Energy scaling from profile
        profile = EMOTION_REGISTRY.get(emotion)
        if profile:
            e_mult = 1.0 + (profile.energy_scale - 1.0) * int_val
            output = output * e_mult

        # Peak normalization safety guard (avoid digital clipping)
        peak = np.abs(output).max()
        if peak > 0.95:
            output = output / peak * 0.92

        out_tensor = torch.from_numpy(output.astype(np.float32)).unsqueeze(0)
        return out_tensor.to(waveform.device)


def list_available_emotions() -> List[Dict[str, Any]]:
    """List all supported affective emotion profiles."""
    return [
        {
            "emotion_id": p.emotion_id,
            "display_name": p.display_name,
            "pitch_factor": p.pitch_factor,
            "pitch_dynamics_factor": p.pitch_dynamics_factor,
            "duration_scale": p.duration_scale,
            "energy_scale": p.energy_scale,
            "description": p.description,
        }
        for p in EMOTION_REGISTRY.values()
    ]
