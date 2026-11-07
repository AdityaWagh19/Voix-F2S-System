"""Tests for objective audio metrics (WER, FAD proxy, diversity availability)."""
from __future__ import annotations

import os
import sys
import types
from pathlib import Path

import numpy as np
import soundfile as sf

from voix.evaluation import audio_metrics as am
from voix.evaluation.audio_metrics import (
    embedding_diversity,
    ensure_ffmpeg_on_path,
    fad_proxy,
    frechet_distance,
    normalize_text,
    transcribe_wer,
    word_error_rate,
)


def test_normalize_text():
    assert normalize_text("Hello,   WORLD!") == "hello world"


def test_wer_identical_is_zero():
    assert word_error_rate("the human voice", "The human voice.") == 0.0


def test_wer_substitution_and_insertion():
    assert word_error_rate("a b c", "a x c") == 1 / 3
    assert word_error_rate("a b", "a b c") == 0.5
    assert word_error_rate("a b c", "a b") == 1 / 3


def test_wer_empty_reference_raises():
    try:
        word_error_rate("   ", "anything")
    except ValueError:
        pass
    else:
        raise AssertionError("empty reference must raise")


def test_transcribe_wer_reports_unavailable_without_paths():
    result = transcribe_wer([], "hello")
    assert result["status"] == "unavailable"
    assert result["wer"] is None


def test_transcribe_wer_reports_unavailable_without_ffmpeg(monkeypatch):
    monkeypatch.setattr(am, "ensure_ffmpeg_on_path", lambda: None)
    result = transcribe_wer(["fake.wav"], "hello")
    assert result["status"] == "unavailable"
    assert result["wer"] is None
    assert "ffmpeg" in result["reason"]


def test_ensure_ffmpeg_on_path_prefers_existing_binary(monkeypatch):
    monkeypatch.setattr(am.shutil, "which", lambda name: r"C:\tools\ffmpeg.exe")
    assert ensure_ffmpeg_on_path() == r"C:\tools\ffmpeg.exe"


def test_ensure_ffmpeg_on_path_aliases_imageio_binary(monkeypatch, tmp_path):
    source = tmp_path / "ffmpeg-win-x86_64-v7.1.exe"
    source.write_bytes(b"fake-ffmpeg")
    fake_module = types.SimpleNamespace(get_ffmpeg_exe=lambda: str(source))
    monkeypatch.setitem(sys.modules, "imageio_ffmpeg", fake_module)
    monkeypatch.setattr(am.shutil, "which", lambda name: None)
    monkeypatch.setattr(am.tempfile, "gettempdir", lambda: str(tmp_path))
    monkeypatch.setenv("PATH", "")

    result = ensure_ffmpeg_on_path()

    expected_name = "ffmpeg.exe" if os.name == "nt" else "ffmpeg"
    expected = tmp_path / "voix_ffmpeg" / expected_name
    assert Path(result) == expected
    assert expected.exists()
    assert str(tmp_path / "voix_ffmpeg") in os.environ["PATH"]


def test_frechet_distance_zero_for_identical_gaussians():
    mean = np.zeros(4)
    covariance = np.eye(4)
    assert abs(frechet_distance(mean, covariance, mean, covariance)) < 1e-6


def _write_tone(path, frequency, duration=1.0, sample_rate=16000):
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    wav = 0.4 * np.sin(2 * np.pi * frequency * t).astype(np.float32)
    sf.write(str(path), wav, sample_rate)


def test_fad_proxy_computed_and_positive_for_different_tones(tmp_path):
    refs = []
    gens = []
    for index in range(2):
        ref = tmp_path / f"ref_{index}.wav"
        gen = tmp_path / f"gen_{index}.wav"
        _write_tone(ref, 220.0)
        _write_tone(gen, 880.0)
        refs.append(ref)
        gens.append(gen)
    result = fad_proxy(refs, gens)
    assert result["status"] == "computed"
    assert result["fad_proxy"] is not None
    assert result["fad_proxy"] > 0.0
    assert "not VGGish FAD" in result["metric"]


def test_embedding_diversity_needs_two_files():
    result = embedding_diversity(["only_one.wav"])
    assert result["status"] == "unavailable"
    assert result["pairwise_cosine"] is None

def test_transcribe_wer_rejects_mismatched_reference_count():
    try:
        transcribe_wer(["a.wav", "b.wav"], ["one reference"])
    except ValueError:
        pass
    else:
        raise AssertionError("a reference count mismatch must raise")
