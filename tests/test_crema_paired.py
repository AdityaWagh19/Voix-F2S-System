"""Tests for the paired CREMA-D evaluation grid."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from voix.evaluation.crema_paired import (
    CREMA_EMOTIONS,
    SENTENCES,
    build_paired_grid,
    validate_paired_grid,
)


def _row(
    actor: str,
    sentence: str,
    emotion: str,
    *,
    split: str = "val",
    intensity: str = "unspecified",
    duration: float = 2.0,
    clipped: float = 0.0,
) -> dict:
    code = {"neutral": "NEU", "happy": "HAP", "sad": "SAD", "angry": "ANG", "fear": "FEA", "disgust": "DIS"}[emotion]
    suffix = {"low": "LO", "medium": "MD", "high": "HI", "unspecified": "XX"}[intensity]
    clip_id = f"{actor}_{sentence}_{code}_{suffix}"
    return {
        "clip_id": clip_id,
        "path": f"data/raw/crema_d/AudioWAV/{clip_id}.wav",
        "actor_id": actor,
        "sentence": sentence,
        "intended_emotion": emotion,
        "intensity": intensity,
        "split": split,
        "duration_sec": duration,
        "clipping_fraction": clipped,
        "finite": True,
        "age": 30,
        "sex": "Female",
    }


def _manifest() -> pd.DataFrame:
    rows = []
    for actor in ("a", "b"):
        for sentence in ("DFA", "IEO"):
            rows.append(_row(actor, sentence, "neutral"))
            rows.append(_row(actor, sentence, "happy"))
            rows.append(_row(actor, sentence, "sad"))
    # angry is missing for actor b / IEO on purpose
    rows.append(_row("a", "DFA", "angry"))
    rows.append(_row("b", "DFA", "angry"))
    return pd.DataFrame(rows)


def test_grid_has_one_neutral_row_per_pair_and_all_emotions():
    grid, summary = build_paired_grid(_manifest(), split="val")
    assert summary["rows"] == 2 * 2 * 6
    assert summary["pairs"] == 4
    neutral = grid[grid["emotion"] == "neutral"]
    assert len(neutral) == 4
    assert (neutral["emotion_path"] == neutral["neutral_path"]).all()
    assert (neutral["intensity"] == 0.0).all()


def test_missing_real_emotion_clip_is_kept_but_flagged():
    grid, _ = build_paired_grid(_manifest(), split="val")
    missing = grid[
        (grid["actor_id"] == "b")
        & (grid["sentence"] == "IEO")
        & (grid["emotion"] == "angry")
    ]
    assert len(missing) == 1
    assert bool(missing.iloc[0]["has_real_emotion_clip"]) is False
    assert missing.iloc[0]["emotion_path"] == ""


def test_identity_clip_prefers_longest_unclipped_neutral():
    frame = _manifest()
    frame.loc[
        (frame["actor_id"] == "a") & (frame["sentence"] == "IEO") & (frame["intended_emotion"] == "neutral"),
        "duration_sec",
    ] = 9.0
    grid, _ = build_paired_grid(frame, split="val")
    actor_a = grid[grid["actor_id"] == "a"]
    assert set(actor_a["identity_clip_id"]) == {"a_IEO_NEU_XX"}


def test_validation_accepts_grid_and_rejects_duplicate():
    grid, _ = build_paired_grid(_manifest(), split="val")
    assert validate_paired_grid(grid) == []
    duplicate = pd.concat([grid, grid.iloc[[0]]], ignore_index=True)
    errors = validate_paired_grid(duplicate)
    assert any("duplicate" in error for error in errors)


def test_validation_rejects_wrong_text_and_neutral_path():
    grid, _ = build_paired_grid(_manifest(), split="val")
    grid.loc[grid["sentence"] == "DFA", "text"] = "wrong text"
    errors = validate_paired_grid(grid)
    assert any("wrong text" in error for error in errors)

    grid, _ = build_paired_grid(_manifest(), split="val")
    neutral = grid["emotion"] == "neutral"
    grid.loc[neutral, "emotion_path"] = "different.wav"
    errors = validate_paired_grid(grid)
    assert any("neutral rows" in error for error in errors)


def test_validation_checks_audio_paths(tmp_path: Path):
    grid, _ = build_paired_grid(_manifest(), split="val")
    errors = validate_paired_grid(grid, root=tmp_path, check_audio=True)
    assert any("missing file" in error for error in errors)


def test_unknown_emotion_is_rejected():
    with pytest.raises(ValueError):
        build_paired_grid(_manifest(), split="val", emotions=["neutral", "bored"])


def test_neutral_is_required():
    with pytest.raises(ValueError):
        build_paired_grid(_manifest(), split="val", emotions=["happy"])


def test_all_sentences_have_verified_text():
    assert len(SENTENCES) == 12
    assert SENTENCES["DFA"] == "Don't forget a jacket"
    assert SENTENCES["IEO"] == "It's eleven o'clock"
    assert SENTENCES["IOM"] == "I'm on my way to the meeting"
    assert set(CREMA_EMOTIONS) == {"neutral", "happy", "sad", "angry", "fear", "disgust"}
