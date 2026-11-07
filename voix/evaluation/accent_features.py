"""Cheap accent features for the A0 baseline.

ECAPA captures speaker/timbre information; the prosodic/spectral block captures
the parts of accent that survive a short utterance.  F0 uses YIN rather than
pYIN because A0 runs over ~9.5k clips and pYIN is roughly 35x slower on this
machine.  The method is recorded in the feature cache report.

Nothing here uses ``predicted_accent`` or any GLOBE metadata as a label.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence, Union

import numpy as np

SAMPLE_RATE = 24000
F0_MIN_HZ = 50.0
F0_MAX_HZ = 600.0
CONTRAST_BANDS = 6

PROSODY_FEATURE_NAMES: List[str] = [
    "duration_seconds",
    "f0_mean_hz",
    "f0_std_hz",
    "f0_range_hz",
    "f0_voiced_ratio",
    "f0_smoothness",
    "energy_mean",
    "energy_std",
    "zcr_mean",
    "zcr_std",
    "spectral_centroid_mean",
    "spectral_centroid_std",
    "spectral_bandwidth_mean",
    "spectral_bandwidth_std",
    "spectral_rolloff_mean",
    "spectral_rolloff_std",
    "spectral_flatness_mean",
    "spectral_flatness_std",
]
PROSODY_FEATURE_NAMES += [f"spectral_contrast_mean_{index}" for index in range(CONTRAST_BANDS)]
PROSODY_FEATURE_NAMES += [f"spectral_contrast_std_{index}" for index in range(CONTRAST_BANDS)]


def _safe_stat(values: np.ndarray, reducer) -> float:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 0.0
    return float(reducer(finite))


def extract_prosody_spectral_features(
    audio_path: Union[str, Path],
    *,
    sample_rate: int = SAMPLE_RATE,
) -> np.ndarray:
    """Return the fixed-length prosody/spectral feature vector for one clip."""
    import librosa

    path = Path(audio_path)
    if not path.is_file():
        raise FileNotFoundError(f"audio not found: {path}")
    wav, sr = librosa.load(str(path), sr=sample_rate, mono=True)
    if wav.size == 0:
        raise ValueError(f"empty audio: {path}")
    wav, _ = librosa.effects.trim(wav, top_db=30)
    if wav.size < int(0.05 * sr):
        raise ValueError(f"audio too short after trimming: {path}")

    f0 = librosa.yin(
        wav.astype(np.float64),
        fmin=F0_MIN_HZ,
        fmax=F0_MAX_HZ,
        sr=sr,
        frame_length=2048,
    )
    voiced = f0[np.isfinite(f0) & (f0 > 0.0)]
    if voiced.size > 2:
        f0_mean = float(np.mean(voiced))
        f0_std = float(np.std(voiced))
        f0_range = float(np.max(voiced) - np.min(voiced))
        f0_smooth = float(np.mean(np.abs(np.diff(voiced))))
        voiced_ratio = float(voiced.size) / max(1, f0.size)
    else:
        f0_mean = f0_std = f0_range = f0_smooth = voiced_ratio = 0.0

    rms = librosa.feature.rms(y=wav, frame_length=2048, hop_length=512)[0]
    zcr = librosa.feature.zero_crossing_rate(wav, frame_length=2048, hop_length=512)[0]
    centroid = librosa.feature.spectral_centroid(y=wav, sr=sr, n_fft=2048, hop_length=512)[0]
    bandwidth = librosa.feature.spectral_bandwidth(y=wav, sr=sr, n_fft=2048, hop_length=512)[0]
    rolloff = librosa.feature.spectral_rolloff(
        y=wav, sr=sr, n_fft=2048, hop_length=512, roll_percent=0.85
    )[0]
    flatness = librosa.feature.spectral_flatness(y=wav, n_fft=2048, hop_length=512)[0]
    contrast = librosa.feature.spectral_contrast(
        y=wav, sr=sr, n_fft=2048, hop_length=512, n_bands=CONTRAST_BANDS, fmin=100.0
    )

    values: Dict[str, float] = {
        "duration_seconds": float(wav.size / sr),
        "f0_mean_hz": f0_mean,
        "f0_std_hz": f0_std,
        "f0_range_hz": f0_range,
        "f0_voiced_ratio": voiced_ratio,
        "f0_smoothness": f0_smooth,
        "energy_mean": _safe_stat(rms, np.mean),
        "energy_std": _safe_stat(rms, np.std),
        "zcr_mean": _safe_stat(zcr, np.mean),
        "zcr_std": _safe_stat(zcr, np.std),
        "spectral_centroid_mean": _safe_stat(centroid, np.mean),
        "spectral_centroid_std": _safe_stat(centroid, np.std),
        "spectral_bandwidth_mean": _safe_stat(bandwidth, np.mean),
        "spectral_bandwidth_std": _safe_stat(bandwidth, np.std),
        "spectral_rolloff_mean": _safe_stat(rolloff, np.mean),
        "spectral_rolloff_std": _safe_stat(rolloff, np.std),
        "spectral_flatness_mean": _safe_stat(flatness, np.mean),
        "spectral_flatness_std": _safe_stat(flatness, np.std),
    }
    for band in range(contrast.shape[0]):
        values[f"spectral_contrast_mean_{band}"] = _safe_stat(contrast[band], np.mean)
        values[f"spectral_contrast_std_{band}"] = _safe_stat(contrast[band], np.std)

    vector = np.asarray([values[name] for name in PROSODY_FEATURE_NAMES], dtype=np.float32)
    if not np.isfinite(vector).all():
        raise ValueError(f"non-finite accent features for {path}")
    return vector


def combine_features(
    ecapa: np.ndarray,
    prosody: np.ndarray,
) -> np.ndarray:
    """Concatenate an ECAPA embedding and the prosody/spectral block."""
    ecapa = np.asarray(ecapa, dtype=np.float32).reshape(-1)
    prosody = np.asarray(prosody, dtype=np.float32).reshape(-1)
    if ecapa.size == 0 or prosody.size == 0:
        raise ValueError("both feature blocks must be non-empty")
    vector = np.concatenate([ecapa, prosody]).astype(np.float32)
    if not np.isfinite(vector).all():
        raise ValueError("combined feature vector contains non-finite values")
    return vector


def feature_names(ecapa_dim: int = 192) -> List[str]:
    """Stable feature order for cached matrices."""
    return [f"ecapa_{index}" for index in range(int(ecapa_dim))] + list(
        PROSODY_FEATURE_NAMES
    )
