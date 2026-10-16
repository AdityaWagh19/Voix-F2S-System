"""Speech Difference Simulation Engine for VOIX (Feature 3).

Models 6 distinct speech characteristics grounded in phonetic and acoustic science:
  1. Interdental Lisp: Articulatory substitution of /s/, /z/ with interdental fricatives /th/, /dh/.
  2. Lateral Lisp: Fricative distortion from lateral air emission around tongue margins (4-8 kHz acoustic shaping).
  3. Rhotacism: Rhotic consonant realization difficulty, typically gliding /r/ -> /w/.
  4. Clonic Stuttering: Involuntary repetition of word-onset consonants and syllables.
  5. Hypernasality: Acoustic pole-zero velopharyngeal resonance coupling (300 Hz resonance, 600 Hz anti-resonance).
  6. Prolonged Phonemes: Sound prolongation and extended vocalic durations (1.2x - 1.8x).

Ethical Design Principle:
  These are treated as distinct acoustic and phonetic characteristics for accessibility,
  synthesis diversity, and speech therapy research, NOT interchangeable novelty filters
  or clinical diagnostic instruments.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import scipy.signal
import torch


class SpeechDifferenceType(str, Enum):
    """Enumeration of supported speech difference characteristics."""

    INTERDENTAL_LISP = "interdental_lisp"
    LATERAL_LISP = "lateral_lisp"
    RHOTACISM = "rhotacism"
    STUTTER_CLONIC = "stutter_clonic"
    HYPERNASALITY = "hypernasality"
    PROLONGED_PHONEMES = "prolonged_phonemes"


@dataclass
class SpeechSimulationConfig:
    """Configuration container for speech difference simulation."""

    difference_type: str
    severity: float = 0.5  # Clamped to [0.0, 1.0]

    def __post_init__(self) -> None:
        self.severity = float(np.clip(self.severity, 0.0, 1.0))
        valid_types = [t.value for t in SpeechDifferenceType]
        if self.difference_type not in valid_types:
            raise ValueError(
                f"Unknown speech difference '{self.difference_type}'. Supported: {valid_types}"
            )


class SpeechTextTransformer:
    """Handles phonetic and linguistic text pre-processing for articulatory characteristics."""

    @staticmethod
    def _apply_interdental_lisp(text: str, severity: float) -> str:
        """Substitute sibilants /s/, /z/ with interdental fricatives /th/, /dh/."""
        if severity <= 0.01:
            return text

        words = text.split(" ")
        transformed_words = []

        # Probability of transforming a sibilant-bearing word scales with severity
        p_transform = min(0.95, severity * 1.1)
        rng = np.random.default_rng(len(text))

        for word in words:
            if rng.random() > p_transform:
                transformed_words.append(word)
                continue

            w = word
            # Preserve capitalization while substituting
            # Word-initial 's' -> 'th'
            w = re.sub(r"\b[sS]", lambda m: "Th" if m.group().isupper() else "th", w)
            # Medial/final 'ss' or 's'
            w = re.sub(r"(?<!\b)[sS]{1,2}(?![hH])", lambda m: "Th" if m.group().isupper() else "th", w)
            # 'z' -> 'th'
            w = re.sub(r"[zZ]", lambda m: "Th" if m.group().isupper() else "th", w)
            # 'ce'/'ci' -> 'the'/'thi'
            w = re.sub(r"\b[cC](?=[eEiI])", lambda m: "Th" if m.group().isupper() else "th", w)

            transformed_words.append(w)

        return " ".join(transformed_words)

    @staticmethod
    def _apply_rhotacism(text: str, severity: float) -> str:
        """Substitute rhotic /r/ phonemes with labialized glides /w/."""
        if severity <= 0.01:
            return text

        words = text.split(" ")
        transformed_words = []
        p_transform = min(0.95, severity * 1.1)
        rng = np.random.default_rng(len(text) + 42)

        for word in words:
            if rng.random() > p_transform:
                transformed_words.append(word)
                continue

            w = word
            # Initial 'R' -> 'W'
            w = re.sub(r"\b[rR]", lambda m: "W" if m.group().isupper() else "w", w)
            # Consonant clusters like 'tr', 'cr', 'pr', 'br', 'gr', 'dr'
            w = re.sub(r"([bBcCldDfFgGpPtTvV])[rR]", lambda m: m.group(1) + ("W" if m.group().isupper() else "w"), w)
            # Medial 'r' between vowels (e.g. very -> vewy)
            w = re.sub(r"(?<=[aAeEiIoOuU])[rR](?=[aAeEiIoOuU])", lambda m: "W" if m.group().isupper() else "w", w)

            transformed_words.append(w)

        return " ".join(transformed_words)

    @staticmethod
    def _apply_clonic_stutter(text: str, severity: float) -> str:
        """Simulate human clonic onset repetitions (e.g. 'Hel- hello', 'dem- demonstration')."""
        if severity <= 0.01:
            return text

        words = text.split(" ")
        transformed_words = []
        p_stutter = min(0.50, severity * 0.60)
        rng = np.random.default_rng(len(text) + 101)

        for word in words:
            clean = re.sub(r"[^a-zA-Z]", "", word)
            # Onset syllable repetition: find first vowel position
            m_vowel = re.search(r"[aeiouAEIOU]", clean)
            if len(clean) >= 4 and m_vowel and m_vowel.start() > 0 and rng.random() < p_stutter:
                v_idx = m_vowel.end()
                syl = clean[:min(v_idx + 1, len(clean))]
                is_caps = word[0].isupper()
                syl_formatted = syl.capitalize() if is_caps else syl.lower()
                reps = 1 if severity < 0.6 else 2
                prefix = " ".join([f"{syl_formatted}-" for _ in range(reps)])
                transformed_words.append(f"{prefix} {word}")
            else:
                transformed_words.append(word)

        return " ".join(transformed_words)

    @staticmethod
    def _apply_prolonged_phonemes(text: str, severity: float) -> str:
        """Simulate prolongation on selected vowel sounds."""
        if severity <= 0.01:
            return text

        words = text.split(" ")
        transformed_words = []
        p_prolong = min(0.50, severity * 0.55)
        rng = np.random.default_rng(len(text) + 202)

        for word in words:
            if rng.random() < p_prolong:
                # Lengthen primary vowel in word
                w = re.sub(r"([aAeEiIoOuU])", lambda m: m.group(1) * (2 if severity < 0.6 else 3), word, count=1)
                transformed_words.append(w)
            else:
                transformed_words.append(word)

        return " ".join(transformed_words)

    @classmethod
    def transform(
        cls,
        text: str,
        difference_type: str,
        severity: float,
    ) -> Tuple[str, float]:
        """Apply phonetic text transformation and compute duration dilation scale.

        Returns:
            Tuple of (transformed_text, duration_scale_multiplier).
        """
        sev = float(np.clip(severity, 0.0, 1.0))
        if sev <= 0.001 or not text:
            return text, 1.0

        duration_multiplier = 1.0

        if difference_type == SpeechDifferenceType.INTERDENTAL_LISP.value:
            transformed = cls._apply_interdental_lisp(text, sev)
        elif difference_type == SpeechDifferenceType.RHOTACISM.value:
            transformed = cls._apply_rhotacism(text, sev)
        elif difference_type == SpeechDifferenceType.STUTTER_CLONIC.value:
            transformed = cls._apply_clonic_stutter(text, sev)
            duration_multiplier = 1.0 + (sev * 0.35)
        elif difference_type == SpeechDifferenceType.PROLONGED_PHONEMES.value:
            transformed = cls._apply_prolonged_phonemes(text, sev)
            duration_multiplier = 1.0 + (sev * 0.45)
        elif difference_type == SpeechDifferenceType.LATERAL_LISP.value:
            # Lateral lisp has mild phonetic effect + primary acoustic effect
            transformed = cls._apply_interdental_lisp(text, sev * 0.4)
        elif difference_type == SpeechDifferenceType.HYPERNASALITY.value:
            # Hypernasality is primarily acoustic/spectral
            transformed = text
        else:
            transformed = text

        return transformed, duration_multiplier


class SpeechAcousticTransformer:
    """Handles post-synthesis acoustic signal processing for vocal tract differences."""

    @staticmethod
    def apply_hypernasality_filter(
        waveform: torch.Tensor,
        sample_rate: int = 24000,
        severity: float = 0.5,
    ) -> torch.Tensor:
        """Apply velopharyngeal pole-zero acoustic model for hypernasal speech.

        Acoustic mechanics:
          - Nasal formant pole around 300-350 Hz (increased low-frequency energy).
          - Anti-formant zero (notch) around 600-750 Hz (velopharyngeal port trapping).
          - High-frequency damping due to mucous membrane absorption in nasal cavity.
        """
        if severity <= 0.01:
            return waveform

        wav_np = waveform.squeeze().cpu().numpy().astype(np.float64)

        # 1. Nasal Pole Resonance peaking around 320 Hz
        b_pole, a_pole = scipy.signal.iirpeak(
            w0=320.0,
            Q=2.5,
            fs=sample_rate,
        )

        # 2. Velopharyngeal Anti-Resonance Notch around 650 Hz
        b_notch, a_notch = scipy.signal.iirnotch(
            w0=650.0,
            Q=3.0,
            fs=sample_rate,
        )

        # 3. High-frequency damping (mucous absorption low-pass above 4.5 kHz)
        b_damp, a_damp = scipy.signal.butter(
            N=2,
            Wn=4800.0,
            btype="lowpass",
            fs=sample_rate,
        )

        # Cascaded filtering
        filtered = scipy.signal.lfilter(b_pole, a_pole, wav_np) * 0.6 + wav_np * 0.4
        filtered = scipy.signal.lfilter(b_notch, a_notch, filtered)
        filtered = scipy.signal.lfilter(b_damp, a_damp, filtered)

        # Smooth blend based on severity
        blend = min(0.85, severity * 0.90)
        output = (1.0 - blend) * wav_np + blend * filtered

        # Normalization guard
        max_val = np.abs(output).max()
        if max_val > 1e-4:
            output = output / max_val * min(0.95, np.abs(wav_np).max() * 1.05)

        out_tensor = torch.from_numpy(output.astype(np.float32)).unsqueeze(0)
        return out_tensor.to(waveform.device)

    @staticmethod
    def apply_lateral_lisp_filter(
        waveform: torch.Tensor,
        sample_rate: int = 24000,
        severity: float = 0.5,
    ) -> torch.Tensor:
        """Simulate acoustic lateral air escape in high-frequency fricative bands (4-8 kHz)."""
        if severity <= 0.01:
            return waveform

        wav_np = waveform.squeeze().cpu().numpy().astype(np.float64)

        # Broad diffuse high-frequency dispersion band around 5500 Hz
        b_lat, a_lat = scipy.signal.iirpeak(
            w0=5500.0,
            Q=1.2,
            fs=sample_rate,
        )
        diffuse = scipy.signal.lfilter(b_lat, a_lat, wav_np)

        # Modulate diffuse air component
        blend = min(0.50, severity * 0.55)
        output = wav_np + blend * diffuse * 0.4

        max_val = np.abs(output).max()
        if max_val > 1.0:
            output = output / max_val * 0.95

        out_tensor = torch.from_numpy(output.astype(np.float32)).unsqueeze(0)
        return out_tensor.to(waveform.device)

    @classmethod
    def filter(
        cls,
        waveform: torch.Tensor,
        sample_rate: int,
        difference_type: str,
        severity: float,
    ) -> torch.Tensor:
        """Route to appropriate acoustic filter based on difference type."""
        sev = float(np.clip(severity, 0.0, 1.0))
        if sev <= 0.001:
            return waveform

        if difference_type == SpeechDifferenceType.HYPERNASALITY.value:
            return cls.apply_hypernasality_filter(waveform, sample_rate, sev)
        elif difference_type == SpeechDifferenceType.LATERAL_LISP.value:
            return cls.apply_lateral_lisp_filter(waveform, sample_rate, sev)

        # Other differences are primarily phonetic/duration-driven
        return waveform


class SpeechSimulator:
    """Unified Facade for speech difference modeling across text and acoustics."""

    @staticmethod
    def list_supported_differences() -> List[Dict[str, Any]]:
        """Return structured list of all supported speech characteristics."""
        return [
            {
                "id": SpeechDifferenceType.INTERDENTAL_LISP.value,
                "name": "Interdental Lisp",
                "description": "Phonemic substitution of /s/, /z/ with interdental fricatives /th/, /dh/.",
                "mechanism": "Articulatory text transformation",
            },
            {
                "id": SpeechDifferenceType.LATERAL_LISP.value,
                "name": "Lateral Lisp",
                "description": "Diffuse lateral air escape across fricative frequencies with softened sibilance.",
                "mechanism": "Acoustic 4-8 kHz spectral dispersion + phonetic softening",
            },
            {
                "id": SpeechDifferenceType.RHOTACISM.value,
                "name": "Rhotacism",
                "description": "Difficulty articulating rhotic consonants, realizing /r/ as labial glide /w/.",
                "mechanism": "Context-aware /r/ -> /w/ phonemic transformation",
            },
            {
                "id": SpeechDifferenceType.STUTTER_CLONIC.value,
                "name": "Clonic Stuttering",
                "description": "Involuntary repetitions of word-initial consonant onsets and syllables.",
                "mechanism": "Onset syllable repetition + prosodic duration dilation",
            },
            {
                "id": SpeechDifferenceType.HYPERNASALITY.value,
                "name": "Hypernasality",
                "description": "Excessive nasal resonance from reduced velopharyngeal closure.",
                "mechanism": "Pole-zero IIR acoustic filter (320 Hz resonance, 650 Hz anti-resonance)",
            },
            {
                "id": SpeechDifferenceType.PROLONGED_PHONEMES.value,
                "name": "Prolonged Phonemes",
                "description": "Sound prolongations and extended steady-state vocalic durations.",
                "mechanism": "Vowel lengthening + synthesis duration scaling",
            },
        ]

    @classmethod
    def process_text(
        cls,
        text: str,
        difference_type: Optional[str] = None,
        severity: float = 0.5,
    ) -> Tuple[str, float]:
        """Transform text and compute pacing dilation factor."""
        if not difference_type or severity <= 0.001:
            return text, 1.0
        return SpeechTextTransformer.transform(text, difference_type, severity)

    @classmethod
    def process_waveform(
        cls,
        waveform: torch.Tensor,
        sample_rate: int = 24000,
        difference_type: Optional[str] = None,
        severity: float = 0.5,
    ) -> torch.Tensor:
        """Apply post-synthesis acoustic resonance filters."""
        if not difference_type or severity <= 0.001:
            return waveform
        return SpeechAcousticTransformer.filter(waveform, sample_rate, difference_type, severity)
