"""Tests for the GLOBE accent selection rules.

These are pure metadata tests: no audio is read and no model is loaded.  They
lock the label resolution, the per-speaker cap, and the speaker-disjoint split
invariant so a later data pass cannot silently weaken the gate.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "data" / "select_globe_accent_subset.py"

spec = importlib.util.spec_from_file_location("select_globe_accent_subset", SCRIPT)
assert spec and spec.loader
script = importlib.util.module_from_spec(spec)
spec.loader.exec_module(script)


def _frame(rows: list[dict]) -> pd.DataFrame:
    defaults = {
        "accent_class": "us",
        "accent_self_reported": "United States English",
        "split": "train",
        "source_file": "data/train-00000-of-00181.parquet",
        "row_group": 0,
        "row_in_group": 0,
        "predicted_accent": "United States English",
        "predicted_gender": "female",
        "predicted_age": "twenties",
        "common_voice_sentence": "Test sentence.",
    }
    return pd.DataFrame([{**defaults, **row} for row in rows])


def test_resolve_accent_accepts_one_target_and_rejects_ambiguity():
    assert script.resolve_accent("United States English") == ("us", "ok")
    assert script.resolve_accent(
        "United States English,Midwestern,Minnesotan"
    ) == ("us", "ok")
    assert script.resolve_accent(
        "England English,India and South Asia (India, Pakistan, Sri Lanka)"
    ) == (None, "ambiguous")
    assert script.resolve_accent("Scottish English") == (None, "not_target")
    assert script.resolve_accent("") == (None, "missing")
    assert script.resolve_accent(None) == (None, "missing")


def test_cap_speaker_clips_is_strict_and_quality_ordered():
    group = _frame(
        [
            {
                "client_id": "speaker-a",
                "duration_sec": 8.0,
                "utmos": 3.0,
                "snr": 30.0,
                "wer": 0.0,
                "common_voice_path": "a.mp3",
            },
            {
                "client_id": "speaker-a",
                "duration_sec": 8.0,
                "utmos": 4.5,
                "snr": 30.0,
                "wer": 0.0,
                "common_voice_path": "b.mp3",
            },
            {
                "client_id": "speaker-a",
                "duration_sec": 8.0,
                "utmos": 4.0,
                "snr": 30.0,
                "wer": 0.0,
                "common_voice_path": "c.mp3",
            },
        ]
    )
    capped = script.cap_speaker_clips(group, cap_sec=20.0, max_clips=10)
    assert list(capped["common_voice_path"]) == ["b.mp3", "c.mp3"]
    assert capped["duration_sec"].sum() == 16.0


def test_select_class_fills_target_and_respects_cap():
    rows = []
    for speaker in range(4):
        for clip in range(2):
            rows.append(
                {
                    "client_id": f"speaker-{speaker}",
                    "duration_sec": 10.0,
                    "utmos": 4.0,
                    "snr": 30.0,
                    "wer": 0.0,
                    "common_voice_path": f"{speaker}-{clip}.mp3",
                }
            )
    frame = _frame(rows)
    selected = script.select_class(
        frame,
        accent="us",
        target_sec=60.0,
        cap_sec=20.0,
        max_clips=10,
        balance_gender=False,
        seed=42,
    )
    assert selected["duration_sec"].sum() >= 60.0
    per_speaker = selected.groupby("client_id")["duration_sec"].sum()
    assert (per_speaker <= 20.0).all()
    assert selected["client_id"].nunique() == 3


def test_row_group_selection_concentrates_the_audio_fetch():
    rows = []
    for row_group, speakers in [(0, 3), (1, 2), (2, 1)]:
        for speaker in range(speakers):
            for clip in range(2):
                rows.append(
                    {
                        "client_id": f"rg{row_group}-speaker-{speaker}",
                        "duration_sec": 10.0,
                        "utmos": 4.0,
                        "snr": 30.0,
                        "wer": 0.0,
                        "common_voice_path": f"{row_group}-{speaker}-{clip}.mp3",
                        "source_file": "data/train-00000-of-00181.parquet",
                        "row_group": row_group,
                        "row_in_group": speaker * 2 + clip,
                    }
                )
    selected = script.select_class_row_group(
        _frame(rows),
        accent="us",
        target_sec=60.0,
        cap_sec=20.0,
        max_clips=10,
        balance_gender=False,
        seed=42,
    )
    assert selected["duration_sec"].sum() >= 60.0
    assert selected.groupby(["source_file", "row_group"]).ngroups == 1
    assert selected.groupby("client_id")["duration_sec"].sum().max() <= 20.0


def test_assign_splits_is_speaker_disjoint_and_has_validation_speakers():
    rows = [
        {
            "client_id": f"speaker-{index}",
            "duration_sec": 5.0,
            "utmos": 4.0,
            "snr": 30.0,
            "wer": 0.0,
            "common_voice_path": f"{index}.mp3",
        }
        for index in range(60)
    ]
    frame = script.assign_splits(_frame(rows), seed=42, min_eval_speakers=5)
    assert frame.groupby("client_id")["speaker_split"].nunique().max() == 1
    counts = frame.groupby("speaker_split")["client_id"].nunique()
    assert counts.get("val", 0) >= 5
    assert counts.get("test", 0) >= 5
    assert counts.get("train", 0) >= 30
