"""Shared waveform level utilities for synthesis and export.

The TTS backend normalizes every utterance to an integrated-loudness target.
Export code must not peak-normalize again: doing so overrides the loudness
target and raises the noise floor of quiet conditions.  The only export-time
gain change here is a true-peak safety limiter, which leaves signals below the
ceiling untouched.

True peak is estimated with polyphase oversampling (4x by default), following
the spirit of ITU-R BS.1770 / EBU R128.  It is a safety measure, not a
calibrated broadcast meter.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np

DEFAULT_CEILING_DBTP: float = -1.0
TRUE_PEAK_OVERSAMPLE: int = 4
_SILENCE_DB: float = -120.0


def _as_channels(wav: np.ndarray) -> Tuple[np.ndarray, Tuple[int, ...]]:
    """Return a ``(channels, samples)`` view and the original shape.

    Mono input is the common case.  For 2-D input, the short axis is treated as
    channels when it is unambiguous; otherwise the first axis is assumed to be
    channels.
    """
    array = np.asarray(wav, dtype=np.float64)
    original_shape = array.shape

    if array.ndim == 0:
        return array.reshape(1, 1), original_shape
    if array.ndim == 1:
        return array.reshape(1, -1), original_shape
    if array.ndim != 2:
        raise ValueError(f"Expected a 1-D or 2-D waveform, got shape {original_shape}")

    rows, cols = array.shape
    if rows <= 8 and cols > rows:
        return array, original_shape
    if cols <= 8 and rows > cols:
        return array.T, original_shape
    return array, original_shape


def _oversampled_peak(channel: np.ndarray, oversample: int) -> float:
    """Largest absolute sample value, including inter-sample peaks."""
    if channel.size == 0:
        return 0.0

    sample_peak = float(np.max(np.abs(channel)))
    if oversample <= 1 or channel.size < 2:
        return sample_peak

    try:
        from scipy.signal import resample_poly

        upsampled = resample_poly(channel, int(oversample), 1)
        oversampled_peak = float(np.max(np.abs(upsampled)))
    except Exception:
        # scipy is a hard dependency elsewhere, but level safety must never
        # make synthesis fail if the import is unavailable.
        return sample_peak

    return max(sample_peak, oversampled_peak)


def true_peak_dbfs(wav: np.ndarray, oversample: int = TRUE_PEAK_OVERSAMPLE) -> float:
    """Estimate the true peak of a waveform in dBFS.

    Args:
        wav: Mono or multi-channel waveform.  Values are linear float samples.
        oversample: Polyphase oversampling factor used to catch inter-sample
            peaks.  ``1`` disables oversampling.

    Returns:
        Peak level in dBFS.  Digital silence returns ``-120.0`` so callers can
        compare levels without special-casing ``-inf``.
    """
    channels, _ = _as_channels(wav)
    peak = max((_oversampled_peak(channel, oversample) for channel in channels), default=0.0)
    if peak <= 0.0:
        return _SILENCE_DB
    return float(20.0 * np.log10(peak))


def limit_true_peak(
    wav: np.ndarray,
    ceiling_dbtp: float = DEFAULT_CEILING_DBTP,
    oversample: int = TRUE_PEAK_OVERSAMPLE,
) -> np.ndarray:
    """Attenuate a waveform only when its true peak exceeds ``ceiling_dbtp``.

    The gain is applied uniformly across channels so the stereo image and the
    relative dynamics are preserved.  Signals already below the ceiling are
    returned as an unchanged float32 copy; this function never boosts level and
    never hard-clips.
    """
    if ceiling_dbtp > 0.0:
        raise ValueError("ceiling_dbtp must be <= 0 dBFS")

    array = np.asarray(wav, dtype=np.float32)
    if array.size == 0:
        return array.copy()

    peak_db = true_peak_dbfs(array, oversample=oversample)
    if peak_db <= ceiling_dbtp:
        return array.copy()

    ceiling_linear = float(10.0 ** (ceiling_dbtp / 20.0))
    peak_linear = float(10.0 ** (peak_db / 20.0))
    limited = array.astype(np.float64) * (ceiling_linear / peak_linear)
    limited = limited.astype(np.float32)

    # Float32 rounding can leave the result a hair above the ceiling.  One
    # corrective pass is enough and costs nothing for the normal path.
    limited_peak_db = true_peak_dbfs(limited, oversample=oversample)
    if limited_peak_db > ceiling_dbtp:
        corrective = 10.0 ** ((ceiling_dbtp - limited_peak_db) / 20.0)
        limited = (limited.astype(np.float64) * corrective).astype(np.float32)

    return limited
