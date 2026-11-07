"""Tests for the controlled audio-quality diagnostic helpers.

The model-backed synthesis path is not exercised here; these tests cover the
pure measurement and normalization functions that the diagnostic relies on.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "audit" / "diagnose_audio_quality.py"

spec = importlib.util.spec_from_file_location("diagnose_audio_quality", SCRIPT)
assert spec and spec.loader
script = importlib.util.module_from_spec(spec)
spec.loader.exec_module(script)


def _tone(frequency: float = 220.0, seconds: float = 1.0, sr: int = 24000) -> np.ndarray:
    t = np.arange(int(sr * seconds), dtype=np.float32) / sr
    return (0.5 * np.sin(2.0 * np.pi * frequency * t)).astype(np.float32)


def test_frame_rms_db_is_finite_for_silence_and_tone():
    silence = np.zeros(24000, dtype=np.float32)
    tone = _tone()

    silence_db = script.frame_rms_db(silence, 24000)
    tone_db = script.frame_rms_db(tone, 24000)

    assert np.all(np.isfinite(silence_db))
    assert np.all(np.isfinite(tone_db))
    assert float(np.max(tone_db)) > float(np.max(silence_db))


def test_peak_normalize_matches_demo_writer_target():
    signal = _tone() * 0.2
    normalized = script.peak_normalize(signal)

    assert np.isclose(float(np.max(np.abs(normalized))), 0.90, atol=1e-6)


def test_peak_normalize_leaves_near_silence_untouched():
    signal = np.zeros(100, dtype=np.float32)
    normalized = script.peak_normalize(signal)

    assert np.array_equal(normalized, signal)


def test_measure_audio_reports_expected_tone_properties():
    metrics = script.measure_audio(_tone(), 24000)

    assert metrics["status"] == "computed"
    assert metrics["duration_seconds"] == 1.0
    assert metrics["peak"] == 0.5
    assert metrics["f0_median_hz"] is not None
    assert 210.0 <= metrics["f0_median_hz"] <= 230.0
    assert metrics["hf_energy_ratio_above_8khz"] is not None
    assert metrics["hf_energy_ratio_above_8khz"] < 0.01
    assert metrics["spectral_flatness_mean"] is not None
    assert metrics["noise_floor_dbfs"] <= metrics["speech_level_dbfs"]


def test_measure_audio_marks_empty_waveform_unavailable():
    metrics = script.measure_audio(np.array([], dtype=np.float32), 24000)

    assert metrics["status"] == "unavailable"
    assert metrics["reason"] == "empty waveform"


def test_pair_reports_normalization_deltas():
    raw = {
        "noise_floor_dbfs": -46.0,
        "speech_level_dbfs": -18.0,
        "snr_proxy_db": 28.0,
        "lufs": -23.0,
        "crest_factor_db": 15.0,
    }
    normalized = {
        "noise_floor_dbfs": -40.0,
        "speech_level_dbfs": -12.0,
        "snr_proxy_db": 28.0,
        "lufs": -17.0,
        "crest_factor_db": 15.0,
    }
    current = dict(raw)

    paired = script._pair(raw, normalized, current)

    assert paired["delta_save_normalization"]["noise_floor_dbfs"] == 6.0
    assert paired["delta_save_normalization"]["lufs"] == 6.0
    assert paired["delta_save_normalization"]["snr_proxy_db"] == 0.0
    assert paired["delta_save_current"]["noise_floor_dbfs"] == 0.0
    assert paired["delta_save_current"]["lufs"] == 0.0


def test_as_mono_averages_stereo_channels():
    stereo = np.stack([_tone(220.0), _tone(220.0)], axis=0)
    mono = script._as_mono(stereo)

    assert mono.ndim == 1
    assert mono.shape[0] == stereo.shape[1]
