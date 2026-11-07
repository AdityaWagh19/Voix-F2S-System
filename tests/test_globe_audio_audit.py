"""Tests for the post-download GLOBE audio health check.

The audit is the authoritative 2.0 h / 30-speaker gate, so these tests pin the
sample-rate, duration, and trim checks without touching the network.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "audit" / "audit_globe_audio.py"

spec = importlib.util.spec_from_file_location("audit_globe_audio", SCRIPT)
assert spec and spec.loader
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def _row(clip_id: str, duration: float) -> pd.Series:
    return pd.Series(
        {
            "clip_id": clip_id,
            "accent_class": "us",
            "speaker_split": "train",
            "client_id": "speaker-a",
            "duration_sec": duration,
        }
    )


def _write(path: Path, seconds: float, sample_rate: int = 24000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    samples = int(seconds * sample_rate)
    tone = 0.1 * np.sin(2 * np.pi * 220 * np.arange(samples) / sample_rate)
    sf.write(path, tone, sample_rate, subtype="PCM_16")


def test_audit_one_accepts_valid_24k_mono(tmp_path):
    _write(tmp_path / "train" / "us" / "clip.wav", 1.0)
    result = audit.audit_one(
        _row("clip", 1.0),
        audio_dir=tmp_path,
        max_trim_sec=7.0,
        max_trim_fraction=0.85,
        max_overrun_sec=0.1,
        min_actual_sec=1.0,
    )
    assert result["status"] == "ok"
    assert result["sample_rate"] == 24000
    assert result["channels"] == 1
    assert result["trim_sec"] == 0.0


def test_audit_one_rejects_too_short_audio(tmp_path):
    _write(tmp_path / "train" / "us" / "clip.wav", 0.5)
    result = audit.audit_one(
        _row("clip", 1.0),
        audio_dir=tmp_path,
        max_trim_sec=7.0,
        max_trim_fraction=0.85,
        max_overrun_sec=0.1,
        min_actual_sec=1.0,
    )
    assert result["status"] == "error"
    assert "actual duration" in result["error"]


def test_audit_one_rejects_wrong_sample_rate(tmp_path):
    _write(tmp_path / "train" / "us" / "clip.wav", 1.0, sample_rate=16000)
    result = audit.audit_one(
        _row("clip", 1.0),
        audio_dir=tmp_path,
        max_trim_sec=7.0,
        max_trim_fraction=0.85,
        max_overrun_sec=0.1,
        min_actual_sec=1.0,
    )
    assert result["status"] == "error"
    assert "sample_rate=16000" in result["error"]
