"""Tests for the A3 class-contrast residual builder."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "training" / "build_accent_class_mean.py"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "build_accent_class_mean", SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _frame(rows):
    return pd.DataFrame(
        [
            {
                "clip_id": clip_id,
                "speaker_id": speaker,
                "accent_class": accent,
                "split": "train",
                "style": np.asarray(style, dtype=np.float32),
            }
            for clip_id, speaker, accent, style in rows
        ]
    )


def test_class_contrast_is_nonzero_even_though_per_speaker_centering_is_zero():
    module = _load_module()
    frame = _frame(
        [
            ("a1", "A", "us", [1.0, 0.0]),
            ("a2", "A", "us", [3.0, 0.0]),
            ("b1", "B", "us", [2.0, 0.0]),
            ("b2", "B", "us", [4.0, 0.0]),
            ("c1", "C", "england", [0.0, 1.0]),
            ("c2", "C", "england", [0.0, 3.0]),
            ("d1", "D", "england", [0.0, 2.0]),
            ("d2", "D", "england", [0.0, 4.0]),
        ]
    )
    result = module.build_class_means(
        frame, ["us", "england"], min_clips_per_speaker=2
    )
    us = result["residuals"]["us"]
    england = result["residuals"]["england"]
    assert us.norm().item() > 0.0
    assert england.norm().item() > 0.0
    # A class contrast sums to zero over classes by construction.
    assert np.allclose((us + england).numpy(), np.zeros(2), atol=1e-6)
    # Speaker-balanced class means: us = [2.5, 0], england = [0, 2.5],
    # global mean = [1.25, 1.25].
    assert np.allclose(us.numpy(), [1.25, -1.25], atol=1e-6)
    assert np.allclose(england.numpy(), [-1.25, 1.25], atol=1e-6)


def test_speakers_below_minimum_clips_are_dropped():
    module = _load_module()
    frame = _frame(
        [
            ("a1", "A", "us", [1.0, 0.0]),
            ("a2", "A", "us", [3.0, 0.0]),
            ("b1", "B", "england", [0.0, 1.0]),
            ("b2", "B", "england", [0.0, 3.0]),
            ("c1", "C", "us", [100.0, 100.0]),
        ]
    )
    result = module.build_class_means(
        frame, ["us", "england"], min_clips_per_speaker=2
    )
    assert result["speaker_counts"] == {"us": 1, "england": 1}
    assert result["clip_counts"] == {"us": 2, "england": 2}


def test_speaker_spanning_two_accents_is_rejected():
    module = _load_module()
    frame = _frame(
        [
            ("a1", "A", "us", [1.0, 0.0]),
            ("a2", "A", "england", [3.0, 0.0]),
        ]
    )
    with pytest.raises(ValueError, match="spans accents"):
        module.build_class_means(
            frame, ["us", "england"], min_clips_per_speaker=2
        )


def test_main_writes_v2_class_contrast_artifact(tmp_path):
    module = _load_module()
    clip_ids = ["a1", "a2", "b1", "b2", "c1", "c2", "d1", "d2"]
    speaker_ids = ["A", "A", "B", "B", "C", "C", "D", "D"]
    accents = ["us", "us", "us", "us", "england", "england", "england", "england"]
    styles = np.asarray(
        [
            [1.0, 0.0],
            [1.2, 0.0],
            [0.9, 0.1],
            [1.1, 0.1],
            [0.0, 1.0],
            [0.1, 1.2],
            [0.1, 0.9],
            [0.0, 1.1],
        ],
        dtype=np.float32,
    )
    cache = tmp_path / "cache.npz"
    np.savez(
        cache,
        clip_id=np.asarray(clip_ids),
        speaker_id=np.asarray(speaker_ids),
        accent_class=np.asarray(accents),
        split=np.asarray(["train"] * len(clip_ids)),
        style=styles,
    )
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("clip_id\n" + "\n".join(clip_ids) + "\n", encoding="utf-8")
    artifact = tmp_path / "class_mean.pt"
    report = tmp_path / "report.json"
    module.main(
        [
            "--cache",
            str(cache),
            "--manifest",
            str(manifest),
            "--artifact",
            str(artifact),
            "--report",
            str(report),
            "--split",
            "train",
            "--classes",
            "us",
            "england",
        ]
    )
    payload = torch.load(artifact, map_location="cpu", weights_only=False)
    assert payload["schema_version"] == "voix.accent_class_mean.v2"
    assert payload["metadata"]["speaker_centered"] is False
    assert payload["metadata"]["residual_kind"] == "class_contrast"
    assert set(payload["classes"]) == {"us", "england"}
    total = payload["residuals"]["us"] + payload["residuals"]["england"]
    assert np.allclose(total.numpy(), np.zeros(2), atol=1e-6)
    assert report.is_file()
