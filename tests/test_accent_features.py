"""Tests for the cheap accent feature block."""
from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf

from voix.evaluation.accent_features import (
    PROSODY_FEATURE_NAMES,
    combine_features,
    extract_prosody_spectral_features,
    feature_names,
)


def _write_tone(path, *, seconds=1.0, sr=24000, frequency=180.0):
    t = np.arange(int(seconds * sr), dtype=np.float32) / sr
    wav = 0.3 * np.sin(2 * np.pi * frequency * t)
    wav += 0.01 * np.random.default_rng(0).normal(size=wav.shape).astype(np.float32)
    sf.write(path, wav, sr)
    return path


def test_feature_block_has_expected_names_and_length(tmp_path):
    path = _write_tone(tmp_path / "tone.wav")
    vector = extract_prosody_spectral_features(path)
    assert vector.shape == (len(PROSODY_FEATURE_NAMES),)
    assert np.isfinite(vector).all()
    assert len(PROSODY_FEATURE_NAMES) == 30


def test_feature_block_recovers_approximate_f0(tmp_path):
    path = _write_tone(tmp_path / "tone.wav", frequency=180.0)
    vector = extract_prosody_spectral_features(path)
    f0_mean = vector[PROSODY_FEATURE_NAMES.index("f0_mean_hz")]
    assert f0_mean == pytest.approx(180.0, abs=15.0)


def test_missing_audio_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        extract_prosody_spectral_features(tmp_path / "missing.wav")


def test_too_short_audio_raises(tmp_path):
    path = _write_tone(tmp_path / "short.wav", seconds=0.02)
    with pytest.raises(ValueError, match="too short"):
        extract_prosody_spectral_features(path)


def test_combine_features_concatenates_and_validates():
    ecapa = np.ones(192, dtype=np.float32)
    prosody = np.zeros(30, dtype=np.float32)
    combined = combine_features(ecapa, prosody)
    assert combined.shape == (222,)
    assert combined.dtype == np.float32
    assert np.isfinite(combined).all()
    with pytest.raises(ValueError, match="non-empty"):
        combine_features(np.array([]), prosody)


def test_combine_features_rejects_non_finite():
    ecapa = np.ones(192, dtype=np.float32)
    prosody = np.zeros(30, dtype=np.float32)
    prosody[0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        combine_features(ecapa, prosody)


def test_feature_names_match_combined_width():
    names = feature_names(192)
    assert len(names) == 222
    assert names[0] == "ecapa_0"
    assert names[-1] == PROSODY_FEATURE_NAMES[-1]
