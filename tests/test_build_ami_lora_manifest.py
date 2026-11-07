"""Tests for the AMI LoRA manifest builder."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "data" / "build_ami_lora_manifest.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("build_ami_lora_manifest", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _row(tmp_path: Path, clip_id: str, speaker: str, split: str, duration: float):
    path = tmp_path / f"{clip_id}.wav"
    path.write_bytes(b"RIFF")
    return {
        "clip_id": clip_id,
        "speaker_id": speaker,
        "split": split,
        "wav_path": str(path),
        "text": f"text for {clip_id}",
        "duration_sec": duration,
        "meeting_id": "M1",
        "source": "AMI",
        "license": "CC BY 4.0",
    }


def test_reference_is_a_different_same_speaker_clip(tmp_path):
    module = _load_module()
    frame = pd.DataFrame(
        [
            _row(tmp_path, "a1", "A", "train", 2.0),
            _row(tmp_path, "a2", "A", "train", 3.0),
            _row(tmp_path, "b1", "B", "val", 2.0),
            _row(tmp_path, "b2", "B", "val", 4.0),
        ]
    )
    output = module.build(frame, min_duration=1.5, max_duration=12.0)
    by_clip = output.set_index("clip_id")
    assert by_clip.loc["a1", "ref_clip_id"] == "a2"
    assert by_clip.loc["a2", "ref_clip_id"] == "a1"
    assert by_clip.loc["b1", "ref_clip_id"] == "b2"
    assert set(output["speaker_id"]) == {"A", "B"}


def test_single_clip_speaker_uses_self_reference(tmp_path):
    module = _load_module()
    frame = pd.DataFrame([_row(tmp_path, "c1", "C", "train", 2.0)])
    output = module.build(frame, min_duration=1.5, max_duration=12.0)
    assert output.iloc[0]["ref_clip_id"] == "c1"


def test_duration_filter_and_missing_text_are_dropped(tmp_path):
    module = _load_module()
    frame = pd.DataFrame(
        [
            _row(tmp_path, "a1", "A", "train", 2.0),
            _row(tmp_path, "a2", "A", "train", 0.5),
            _row(tmp_path, "a3", "A", "train", 20.0),
        ]
    )
    frame.loc[frame["clip_id"] == "a1", "text"] = "   "
    with pytest.raises(ValueError, match="no usable"):
        module.build(frame, min_duration=1.5, max_duration=12.0)
