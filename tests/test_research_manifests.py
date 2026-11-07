"""Tests for manifest validation and the core-training data firewall."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from voix.evaluation.data_firewall import (
    DataFirewallError,
    assert_no_core_contamination,
    scan_core_training,
    scan_file,
)
from voix.evaluation.manifest_validation import validate_manifest


def _crema_frame():
    return pd.DataFrame(
        {
            "clip_id": ["a1", "a2", "b1", "c1", "c2", "d1"],
            "actor_id": ["a", "a", "b", "c", "c", "d"],
            "split": ["train", "train", "train", "test", "test", "test"],
            "intended_emotion": ["happy", "happy", "sad", "happy", "sad", "sad"],
            "path": ["a1.wav", "a2.wav", "b1.wav", "c1.wav", "c2.wav", "d1.wav"],
            "duration_sec": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        }
    )


def test_crema_manifest_passes(tmp_path: Path):
    manifest = tmp_path / "crema.csv"
    _crema_frame().to_csv(manifest, index=False)
    report = validate_manifest(manifest, kind="crema_d")
    assert report["passed"] is True
    assert report["summary"]["speakers"] == 4
    assert report["summary"]["labels"]["happy"]["train_speakers"] == 1


def test_speaker_overlap_fails(tmp_path: Path):
    frame = _crema_frame()
    frame.loc[3, "actor_id"] = "a"
    manifest = tmp_path / "overlap.csv"
    frame.to_csv(manifest, index=False)
    report = validate_manifest(manifest, kind="crema_d")
    assert report["passed"] is False
    assert any("more than one split" in error for error in report["errors"])


def test_duplicate_clip_fails(tmp_path: Path):
    frame = _crema_frame()
    frame.loc[1, "clip_id"] = "a1"
    manifest = tmp_path / "duplicate.csv"
    frame.to_csv(manifest, index=False)
    report = validate_manifest(manifest, kind="crema_d")
    assert report["passed"] is False
    assert any("duplicate clip" in error for error in report["errors"])


def test_class_thresholds_fail(tmp_path: Path):
    manifest = tmp_path / "threshold.csv"
    _crema_frame().to_csv(manifest, index=False)
    report = validate_manifest(
        manifest,
        kind="crema_d",
        min_speakers_per_class=2,
        min_hours_per_class=1.0,
    )
    assert report["passed"] is False
    assert any("train speakers" in error for error in report["errors"])
    assert any("train hours" in error for error in report["errors"])


def test_globe_manifest_uses_speaker_split(tmp_path: Path):
    frame = pd.DataFrame(
        {
            "clip_id": ["g1", "g2", "g3"],
            "client_id": ["s1", "s2", "s3"],
            "speaker_split": ["train", "val", "test"],
            "accent_class": ["us", "us", "irish"],
            "duration_sec": [1.0, 1.0, 1.0],
        }
    )
    manifest = tmp_path / "globe.csv"
    frame.to_csv(manifest, index=False)
    report = validate_manifest(manifest, kind="globe")
    assert report["passed"] is True
    assert report["summary"]["splits"]["train"]["speakers"] == 1


def test_ami_requires_transcript_and_reference(tmp_path: Path):
    frame = pd.DataFrame(
        {
            "clip_id": ["m1"],
            "speaker_id": ["s1"],
            "split": ["train"],
            "wav_path": ["m1.wav"],
            "duration_sec": [1.0],
            "ref_clip_id": ["m2"],
            "ref_wav_path": ["m2.wav"],
            "text": [""],
        }
    )
    manifest = tmp_path / "ami.csv"
    frame.to_csv(manifest, index=False)
    report = validate_manifest(manifest, kind="ami")
    assert report["passed"] is False
    assert any("empty transcripts" in error for error in report["errors"])


def test_audio_check(tmp_path: Path):
    import soundfile as sf
    import numpy as np

    audio = tmp_path / "a1.wav"
    sf.write(audio, np.zeros(2400, dtype=np.float32), 24000)
    frame = _crema_frame()
    frame["path"] = [str(audio), "missing.wav", "missing.wav", "missing.wav", "missing.wav", "missing.wav"]
    manifest = tmp_path / "audio.csv"
    frame.to_csv(manifest, index=False)
    report = validate_manifest(
        manifest,
        kind="crema_d",
        check_audio=True,
        root=tmp_path,
        max_audio_errors=2,
    )
    assert report["passed"] is False
    assert report["summary"]["audio_checked"] == 6
    assert report["summary"]["audio_errors"] >= 1


def test_firewall_flags_forbidden_corpus(tmp_path: Path):
    source = tmp_path / "trainer.py"
    source.write_text(
        "manifest = 'data/processed/crema_d_manifest.csv'\n"
        "audio = 'data/processed/globe_audio/train/us/x.wav'\n"
        "meetings = 'data/processed/ami_lora/manifest.csv'\n",
        encoding="utf-8",
    )
    violations = scan_file(source)
    datasets = {violation.dataset for violation in violations}
    assert datasets == {"crema_d", "globe", "ami"}


def test_firewall_passes_clean_core_files(tmp_path: Path):
    source = tmp_path / "trainer.py"
    source.write_text("manifest = 'data/processed/voxceleb2_v2/index.csv'\n", encoding="utf-8")
    report = scan_core_training(files=[source])
    assert report["passed"] is True
    assert report["violations"] == []


def test_firewall_raises_on_contamination(tmp_path: Path):
    source = tmp_path / "trainer.py"
    source.write_text("path = 'data/processed/crema_d_manifest.csv'\n", encoding="utf-8")
    with pytest.raises(DataFirewallError):
        assert_no_core_contamination(files=[source])


def test_real_core_training_path_is_clean():
    report = scan_core_training()
    assert report["passed"] is True, report["violations"]
