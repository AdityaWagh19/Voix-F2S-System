"""Tests for true-peak measurement, limiting, and the demo export contract."""
from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf

from voix.utils.audio import (
    DEFAULT_CEILING_DBTP,
    limit_true_peak,
    true_peak_dbfs,
)


def _tone(amplitude: float = 0.5, frequency: float = 220.0, seconds: float = 1.0) -> np.ndarray:
    sample_rate = 24000
    t = np.arange(int(sample_rate * seconds), dtype=np.float32) / sample_rate
    return (amplitude * np.sin(2.0 * np.pi * frequency * t)).astype(np.float32)


def test_true_peak_reports_known_tone_level():
    # A 220 Hz sine at 0.5 linear amplitude is about -6.02 dBFS.
    assert true_peak_dbfs(_tone(0.5)) == pytest.approx(-6.02, abs=0.1)


def test_true_peak_handles_digital_silence():
    assert true_peak_dbfs(np.zeros(24000, dtype=np.float32)) == -120.0


def test_limiter_leaves_audio_below_ceiling_unchanged():
    quiet = _tone(0.2)
    limited = limit_true_peak(quiet)

    assert np.array_equal(limited, quiet)
    assert true_peak_dbfs(limited) < DEFAULT_CEILING_DBTP


def test_limiter_attenuates_hot_audio_without_clipping_or_shape_change():
    hot = _tone(2.5)
    limited = limit_true_peak(hot)

    assert true_peak_dbfs(limited) <= DEFAULT_CEILING_DBTP + 0.01
    assert float(np.max(np.abs(limited))) <= 1.0
    # Uniform gain reduction: the waveform shape (and therefore dynamics) is
    # preserved instead of being hard-clipped.
    mask = np.abs(hot) > 1e-4
    gains = limited[mask] / hot[mask]
    assert float(np.std(gains)) < 1e-4
    assert np.corrcoef(hot, limited)[0, 1] > 0.9999


def test_limiter_preserves_relative_level_of_quiet_signals():
    loud = limit_true_peak(_tone(0.2))
    quiet = limit_true_peak(_tone(0.1))

    loud_rms = float(np.sqrt(np.mean(loud.astype(np.float64) ** 2)))
    quiet_rms = float(np.sqrt(np.mean(quiet.astype(np.float64) ** 2)))
    assert loud_rms / quiet_rms == pytest.approx(2.0, rel=1e-4)


def test_limiter_handles_multichannel_input():
    stereo = np.stack([_tone(0.2), _tone(0.1, frequency=330.0)], axis=0)
    limited = limit_true_peak(stereo)

    assert limited.shape == stereo.shape
    assert np.array_equal(limited, stereo)


def test_save_audio_sample_does_not_boost_quiet_audio(tmp_path):
    from voix.inference.io import save_audio_sample

    quiet = _tone(0.1)
    output = tmp_path / "quiet.wav"
    saved, metrics = save_audio_sample(quiet, output)

    assert output.exists()
    assert np.max(np.abs(saved)) == pytest.approx(0.1, abs=1e-5)
    assert metrics["peak_amplitude"] == pytest.approx(0.1, abs=1e-3)

    written, sample_rate = sf.read(str(output))
    assert sample_rate == 24000
    assert np.max(np.abs(written)) == pytest.approx(0.1, abs=1e-5)


def test_save_audio_sample_limits_hot_audio(tmp_path):
    from voix.inference.io import save_audio_sample

    output = tmp_path / "hot.wav"
    saved, _ = save_audio_sample(_tone(2.5), output)

    assert true_peak_dbfs(saved) <= DEFAULT_CEILING_DBTP + 0.01
    assert float(np.max(np.abs(saved))) <= 1.0


def test_save_audio_sample_preserves_loudness_differences(tmp_path):
    import pyloudnorm as pyln

    from voix.inference.io import save_audio_sample

    loud, _ = save_audio_sample(_tone(0.2), tmp_path / "loud.wav")
    quiet, _ = save_audio_sample(_tone(0.1), tmp_path / "quiet.wav")

    meter = pyln.Meter(24000)
    loud_lufs = meter.integrated_loudness(loud.astype(np.float64))
    quiet_lufs = meter.integrated_loudness(quiet.astype(np.float64))

    # No export-time normalization: the 2x amplitude difference remains ~6 dB.
    assert loud_lufs - quiet_lufs == pytest.approx(6.02, abs=0.2)
