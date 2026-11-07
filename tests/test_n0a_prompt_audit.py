"""Tests for the N0a prompt-audit data selection and CSV schema."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from voix.evaluation.naturalness import CLIP_METRICS

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "evaluation" / "run_n0a_prompt_audit.py"
spec = importlib.util.spec_from_file_location("run_n0a_prompt_audit", SCRIPT)
assert spec is not None and spec.loader is not None
_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(_module)
CSV_COLUMNS = _module.CSV_COLUMNS
_sha256_array = _module._sha256_array
select_face_clip = _module.select_face_clip


def _write_store(path, rows):
    import h5py

    with h5py.File(path, "w") as store:
        store.create_dataset(
            "clip_ids", data=np.asarray([row["clip_id"] for row in rows], dtype="S")
        )
        store.create_dataset(
            "speaker_ids",
            data=np.asarray([row["speaker_id"] for row in rows], dtype="S"),
        )
        store.create_dataset(
            "split", data=np.asarray([row["split"] for row in rows], dtype="S")
        )
        store.create_dataset(
            "face_features",
            data=np.stack([row["feature"] for row in rows]).astype(np.float32),
        )
        store.create_dataset(
            "face_gender",
            data=np.asarray([row["gender"] for row in rows], dtype=np.int8),
        )
        store.create_dataset(
            "face_age",
            data=np.asarray([[row["age"]] for row in rows], dtype=np.float32),
        )
        store.create_dataset(
            "face_quality",
            data=np.stack([row["quality"] for row in rows]).astype(np.float32),
        )


def _row(speaker, clip, score, *, split="val", offset=0.0):
    feature = np.zeros(560, dtype=np.float32)
    feature[0] = 1.0 + offset
    quality = np.zeros(12, dtype=np.float32)
    quality[3] = 0.5
    quality[6] = 0.2
    quality[10] = score
    return {
        "speaker_id": speaker,
        "clip_id": clip,
        "split": split,
        "feature": feature,
        "gender": 1,
        "age": 30.0,
        "quality": quality,
    }


def test_select_face_clip_is_deterministic_and_prefers_best_score(tmp_path):
    store = tmp_path / "features.h5"
    _write_store(
        store,
        [
            _row("spk_a", "a1", 0.5),
            _row("spk_a", "a2", 0.9),
            _row("spk_b", "b1", 0.8),
            _row("spk_b", "b2", 0.7),
        ],
    )
    first = select_face_clip(store)
    second = select_face_clip(store)
    assert first["speaker_id"] == "spk_a"
    assert first["clip_id"] == "a2"
    assert first["face_features_sha256"] == second["face_features_sha256"]
    assert first["face_gender"] == 1
    assert first["face_age"] == 30.0


def test_select_face_clip_respects_explicit_speaker_and_clip(tmp_path):
    store = tmp_path / "features.h5"
    _write_store(
        store,
        [
            _row("spk_a", "a1", 0.5),
            _row("spk_a", "a2", 0.9),
            _row("spk_b", "b1", 0.8),
            _row("spk_b", "b2", 0.7),
        ],
    )
    by_speaker = select_face_clip(store, speaker_id="spk_b")
    assert by_speaker["speaker_id"] == "spk_b"
    assert by_speaker["clip_id"] == "b1"
    by_clip = select_face_clip(store, clip_id="a1")
    assert by_clip["speaker_id"] == "spk_a"
    assert by_clip["clip_id"] == "a1"


def test_select_face_clip_rejects_unknown_split_or_speaker(tmp_path):
    store = tmp_path / "features.h5"
    _write_store(store, [_row("spk_a", "a1", 0.5)])
    with pytest.raises(ValueError, match="no .* clip matched"):
        select_face_clip(store, split="train")
    with pytest.raises(ValueError, match="no .* clip matched"):
        select_face_clip(store, speaker_id="spk_missing")


def test_csv_schema_covers_every_clip_metric():
    for metric in CLIP_METRICS:
        assert metric in CSV_COLUMNS


def test_array_hash_is_deterministic_and_content_sensitive():
    left = np.arange(8, dtype=np.float32)
    right = left.copy()
    assert _sha256_array(left) == _sha256_array(right)
    right[0] += 1.0
    assert _sha256_array(left) != _sha256_array(right)
