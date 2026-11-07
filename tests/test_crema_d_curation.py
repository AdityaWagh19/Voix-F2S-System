"""Tests for CREMA-D curation and the honest emotion-class gate."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "data" / "curate_crema_d.py"

spec = importlib.util.spec_from_file_location("curate_crema_d", SCRIPT)
assert spec and spec.loader
crema = importlib.util.module_from_spec(spec)
spec.loader.exec_module(crema)

EMOTIONS = [
    ("NEU", "neutral"),
    ("HAP", "happy"),
    ("SAD", "sad"),
    ("ANG", "angry"),
    ("FEA", "fear"),
    ("DIS", "disgust"),
]


def _write_wav(path: Path, seconds: float = 1.0, clipped: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    samples = int(seconds * 16000)
    if clipped:
        data = np.ones(samples, dtype=np.float32)
    else:
        data = 0.1 * np.sin(2 * np.pi * 220 * np.arange(samples) / 16000)
    sf.write(path, data, 16000, subtype="PCM_16")


def _fixture(tmp_path: Path) -> Path:
    root = tmp_path / "crema_d"
    audio = root / "AudioWAV"
    rows = []
    votes = []
    for actor in ("1001", "1002", "1003"):
        for code, emotion in EMOTIONS:
            clip_id = f"{actor}_DFA_{code}_XX"
            _write_wav(audio / f"{clip_id}.wav")
            rows.append(
                {
                    "ActorID": actor,
                    "Age": 30,
                    "Sex": "Male" if int(actor) % 2 else "Female",
                    "Race": "Caucasian",
                    "Ethnicity": "Not Hispanic",
                }
            )
            votes.append(
                {
                    "fileName": clip_id,
                    "emoVote": crema.EMOTION_LETTERS[emotion],
                    "agreement": 0.8,
                    "numResponses": 10,
                }
            )
    (root / "VideoDemographics.csv").write_text(
        pd.DataFrame(rows).drop_duplicates("ActorID").to_csv(index=False),
        encoding="utf-8",
    )
    votes_dir = root / "processedResults"
    votes_dir.mkdir(parents=True)
    (votes_dir / "tabulatedVotes.csv").write_text(
        pd.DataFrame(votes).to_csv(index=False), encoding="utf-8"
    )
    return root


def test_parse_clip_name():
    parsed = crema.parse_clip("1001_DFA_ANG_HI")
    assert parsed == {
        "clip_id": "1001_DFA_ANG_HI",
        "actor_id": "1001",
        "sentence": "DFA",
        "intended_emotion": "angry",
        "intensity": "high",
    }
    assert crema.parse_clip("1001_DFA_XXX_HI") is None
    assert crema.parse_clip("not-a-clip") is None


def test_split_is_deterministic_and_speaker_disjoint():
    speakers = [f"{1000 + index}" for index in range(20)]
    first = crema.assign_splits(speakers, val_fraction=0.1, test_fraction=0.1)
    second = crema.assign_splits(speakers, val_fraction=0.1, test_fraction=0.1)
    assert first == second
    assert set(first) == set(speakers)
    assert set(first.values()) == {"train", "val", "test"}
    # Every speaker appears exactly once, so the splits are disjoint by
    # construction; check the counts are non-trivial.
    assert sum(value == "val" for value in first.values()) >= 1
    assert sum(value == "test" for value in first.values()) >= 1


def test_curate_builds_manifest_and_reports_failed_gate(tmp_path):
    root = _fixture(tmp_path)
    frame, report = crema.curate(root, workers=2)

    assert len(frame) == 18
    assert set(frame["intended_emotion"]) == {
        "neutral", "happy", "sad", "angry", "fear", "disgust"
    }
    assert frame["split"].nunique() == 3
    assert frame.groupby("actor_id")["split"].nunique().max() == 1
    assert frame["duration_sec"].min() == pytest.approx(1.0, abs=0.01)
    assert report["clips_kept"] == 18
    assert report["speakers"] == 3
    assert report["gate"]["pass"] is False
    failures = " ".join(report["gate"]["failures"])
    assert "2.000h" in failures
    assert "30" in failures
    assert report["by_emotion"]["neutral"]["perceptual_label_agreement"] == 1.0


def test_load_votes_selects_voice_block_without_duplication(tmp_path):
    root = tmp_path / "crema_d"
    votes_dir = root / "processedResults"
    votes_dir.mkdir(parents=True)
    rows = []
    for block, agreement in (("voice", 0.5), ("face", 0.7), ("multimodal", 0.9)):
        for clip in ("1001_DFA_NEU_XX", "1001_DFA_HAP_XX"):
            rows.append(
                {
                    "fileName": clip,
                    "emoVote": "N",
                    "agreement": agreement,
                    "numResponses": 10,
                }
            )
    (votes_dir / "tabulatedVotes.csv").write_text(
        pd.DataFrame(rows).to_csv(index=False), encoding="utf-8"
    )

    frame = crema.load_votes(root)

    assert len(frame) == 2
    assert set(frame["clip_id"]) == {"1001_DFA_NEU_XX", "1001_DFA_HAP_XX"}
    assert frame["agreement"].tolist() == [0.5, 0.5]


def test_health_check_rejects_clipped_audio(tmp_path):
    path = tmp_path / "clip.wav"
    _write_wav(path, clipped=True)
    result = crema.check_audio(path, 0.5, 15.0)
    assert result["status"] == "error"
    assert "clipping_fraction" in result["error"]
