"""Tests for pruning GLOBE clips that failed the post-download audit.

The prune step is what turns the strict audit into a usable manifest, so these
tests pin the important invariants: failed clips leave the manifest, the
unpruned manifest is preserved, the excluded WAV is moved out of the training
tree, and a re-run does not overwrite the original exclusion report.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "data" / "prune_globe_manifest.py"

spec = importlib.util.spec_from_file_location("prune_globe_manifest", SCRIPT)
assert spec and spec.loader
prune = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prune)

CLASSES = ["us", "england", "australian", "canadian", "india_south_asia", "irish"]


def _write_wav(path: Path, seconds: float = 1.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    samples = int(seconds * 24000)
    tone = 0.1 * np.sin(2 * np.pi * 220 * np.arange(samples) / 24000)
    sf.write(path, tone, 24000, subtype="PCM_16")


def _fixture(tmp_path: Path):
    audio_dir = tmp_path / "globe_audio"
    excluded_dir = tmp_path / "globe_audio_excluded"
    rows = [
        {
            "clip_id": "clip_us_fail",
            "accent_class": "us",
            "speaker_split": "train",
            "client_id": "speaker_us",
            "duration_sec": 1.0,
            "source_file": "data/train-00000.parquet",
            "row_group": 0,
        }
    ]
    _write_wav(audio_dir / "train" / "us" / "clip_us_fail.wav")
    for accent in CLASSES:
        clip_id = f"clip_{accent}"
        rows.append(
            {
                "clip_id": clip_id,
                "accent_class": accent,
                "speaker_split": "train",
                "client_id": f"speaker_{accent}",
                "duration_sec": 1.0,
                "source_file": "data/train-00000.parquet",
                "row_group": 0,
            }
        )
        _write_wav(audio_dir / "train" / accent / f"{clip_id}.wav")
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame(rows).to_csv(manifest, index=False)

    audit = tmp_path / "audit.json"
    audit.write_text(
        json.dumps(
            {
                "failed": 1,
                "failures": [
                    {
                        "clip_id": "clip_us_fail",
                        "accent_class": "us",
                        "error": "clipping_fraction=0.0001",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return manifest, audit, audio_dir, excluded_dir


def _args(tmp_path: Path, manifest, audit, audio_dir, excluded_dir):
    return [
        "--manifest", str(manifest),
        "--audit", str(audit),
        "--audio-dir", str(audio_dir),
        "--excluded-dir", str(excluded_dir),
        "--backup", str(tmp_path / "manifest_unpruned.csv"),
        "--report", str(tmp_path / "prune_report.json"),
        "--min-hours", "0",
        "--min-speakers", "1",
        "--min-eval-speakers", "0",
    ]


def test_prune_removes_failed_clip_and_moves_wav(tmp_path):
    manifest, audit, audio_dir, excluded_dir = _fixture(tmp_path)
    prune.main(_args(tmp_path, manifest, audit, audio_dir, excluded_dir))

    kept = pd.read_csv(manifest)
    assert len(kept) == 6
    assert "clip_us_fail" not in set(kept["clip_id"])
    assert len(pd.read_csv(tmp_path / "manifest_unpruned.csv")) == 7

    assert not (audio_dir / "train" / "us" / "clip_us_fail.wav").exists()
    assert (excluded_dir / "train" / "us" / "clip_us_fail.wav").is_file()

    report = json.loads((tmp_path / "prune_report.json").read_text(encoding="utf-8"))
    assert report["removed_clips"] == 1
    assert report["removed_by_class"] == {"us": 1}
    assert report["removed_by_reason"] == {"clipping_fraction": 1}
    assert report["moved_excluded"] == 1
    assert report["gate_pass"] is True


def test_rerun_keeps_original_report_and_is_idempotent(tmp_path):
    manifest, audit, audio_dir, excluded_dir = _fixture(tmp_path)
    args = _args(tmp_path, manifest, audit, audio_dir, excluded_dir)
    prune.main(args)
    prune.main(args)

    kept = pd.read_csv(manifest)
    assert len(kept) == 6

    report = json.loads((tmp_path / "prune_report.json").read_text(encoding="utf-8"))
    assert report["removed_clips"] == 1
    assert report["moved_excluded"] == 1
    assert report["excluded_wavs_present"] == 1
    assert any("globe_audio_excluded" in note for note in report["notes"])
    assert not any("remain on disk" in note for note in report["notes"])
