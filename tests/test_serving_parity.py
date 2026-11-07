"""Tests for the serving-parity measurement helpers."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "evaluation" / "measure_serving_parity.py"

spec = importlib.util.spec_from_file_location("measure_serving_parity", SCRIPT)
assert spec and spec.loader
parity = importlib.util.module_from_spec(spec)
spec.loader.exec_module(parity)


def test_balanced_selection_spans_speakers_before_repeats():
    speakers = np.asarray(["a", "a", "a", "b", "b", "c"])
    selected = parity.select_balanced_indices(speakers, 4, clips_per_speaker=2)
    assert selected == [0, 3, 5, 1]


def test_balanced_selection_respects_limit_and_singletons():
    speakers = np.asarray(["a", "b", "c", "d"])
    selected = parity.select_balanced_indices(speakers, 10, clips_per_speaker=2)
    assert selected == [0, 1, 2, 3]


def test_summary_ignores_non_finite_values():
    summary = parity._summary([1.0, 2.0, float("nan")])
    assert summary == {"mean": 1.5, "min": 1.0, "max": 2.0}


def test_summary_all_non_finite_is_nan():
    summary = parity._summary([float("nan")])
    assert np.isnan(summary["mean"])


def test_resolve_clip_path_keeps_dev_prefix(tmp_path):
    target = tmp_path / "dev" / "id00001" / "abc" / "00001.mp4"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"x")
    assert parity.resolve_clip_path(tmp_path, "dev/id00001/abc/00001") == target


def test_resolve_clip_path_falls_back_to_stripped_pilot_layout(tmp_path):
    target = tmp_path / "id00001" / "abc" / "00001.mp4"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"x")
    assert parity.resolve_clip_path(tmp_path, "dev/id00001/abc/00001") == target


def test_resolve_clip_path_missing_returns_none(tmp_path):
    assert parity.resolve_clip_path(tmp_path, "dev/id00001/abc/00001") is None
