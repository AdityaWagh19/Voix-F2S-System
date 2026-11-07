"""Tests for the FaceFeatureV2 corpus validator."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "audit" / "validate_v2_corpus.py"

spec = importlib.util.spec_from_file_location("validate_v2_corpus", SCRIPT)
assert spec and spec.loader
validator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validator)


def _write_corpus(
    tmp_path: Path,
    *,
    clip_ids=None,
    speakers=None,
    splits=None,
    genders=None,
    shards=None,
    bad_row: int | None = None,
):
    clip_ids = clip_ids or ["c1", "c2", "c3", "c4"]
    speakers = speakers or ["s1", "s1", "s2", "s2"]
    splits = splits or ["train", "train", "val", "val"]
    genders = genders or ["f", "f", "m", "m"]
    n = len(clip_ids)
    features = {
        "face_features": np.arange(n * 4, dtype=np.float32).reshape(n, 4) + 1.0,
        "arcface": np.arange(n * 3, dtype=np.float32).reshape(n, 3) + 1.0,
        "speaker_features": np.arange(n * 2, dtype=np.float32).reshape(n, 2) + 1.0,
    }
    if bad_row is not None:
        features["face_features"][bad_row, 0] = np.nan

    path = tmp_path / "features.h5"
    with h5py.File(path, "w") as store:
        store.create_dataset(
            "clip_ids", data=np.asarray(clip_ids, dtype=object),
            dtype=h5py.string_dtype(),
        )
        store.create_dataset(
            "speaker_ids", data=np.asarray(speakers, dtype=object),
            dtype=h5py.string_dtype(),
        )
        store.create_dataset(
            "split", data=np.asarray(splits, dtype=object),
            dtype=h5py.string_dtype(),
        )
        store.create_dataset(
            "gender", data=np.asarray(genders, dtype=object),
            dtype=h5py.string_dtype(),
        )
        for name, values in features.items():
            store.create_dataset(name, data=values)
        if shards is not None:
            store.create_dataset("shard", data=np.asarray(shards, dtype=np.int32))
    return path


def _write_index(tmp_path: Path, *, clip_ids=None, speakers=None, splits=None):
    clip_ids = clip_ids or ["c1", "c2", "c3", "c4"]
    speakers = speakers or ["s1", "s1", "s2", "s2"]
    splits = splits or ["train", "train", "val", "val"]
    n = len(clip_ids)
    path = tmp_path / "index.csv"
    pd.DataFrame(
        {
            "clip_id": clip_ids,
            "speaker_id": speakers,
            "split": splits,
            "gender": ["f" if i % 2 == 0 else "m" for i in range(n)],
            "shard": [1 if split != "test" else 7 for split in splits],
        }
    ).to_csv(path, index=False)
    return path


class TestCorpusValidation:
    def test_healthy_corpus_passes(self, tmp_path):
        features = _write_corpus(tmp_path)
        index = _write_index(tmp_path)
        report = validator.validate_corpus(
            features, index, expect_complete=True
        )
        assert report["status"] == "pass"
        assert report["hard_errors"] == []
        assert report["clips"] == 4
        assert report["splits"]["train"]["clips"] == 2
        assert report["manifest"]["missing"] == 0
        assert report["feature_dims"] == {
            "face_features": 4,
            "arcface": 3,
            "speaker_features": 2,
        }

    def test_speaker_leakage_is_a_hard_error(self, tmp_path):
        features = _write_corpus(
            tmp_path, speakers=["s1", "s1", "s1", "s2"]
        )
        index = _write_index(tmp_path, speakers=["s1", "s1", "s1", "s2"])
        report = validator.validate_corpus(features, index)
        assert report["status"] == "fail"
        assert any("speaker leakage" in error for error in report["hard_errors"])

    def test_duplicate_clip_ids_are_a_hard_error(self, tmp_path):
        features = _write_corpus(tmp_path, clip_ids=["c1", "c1", "c2", "c3"])
        report = validator.validate_corpus(features)
        assert report["status"] == "fail"
        assert any("duplicate clip_ids" in error for error in report["hard_errors"])

    def test_non_finite_features_are_a_hard_error(self, tmp_path):
        features = _write_corpus(tmp_path, bad_row=2)
        report = validator.validate_corpus(features)
        assert report["status"] == "fail"
        assert any("non-finite" in error for error in report["hard_errors"])

    def test_unknown_extracted_clip_is_a_hard_error(self, tmp_path):
        features = _write_corpus(
            tmp_path, clip_ids=["c1", "c2", "c3", "outside"]
        )
        index = _write_index(tmp_path)
        report = validator.validate_corpus(features, index)
        assert report["status"] == "fail"
        assert any("absent from the manifest" in error for error in report["hard_errors"])

    def test_expect_complete_flags_unaccounted_manifest_clip(self, tmp_path):
        features = _write_corpus(tmp_path)
        index = _write_index(
            tmp_path,
            clip_ids=["c1", "c2", "c3", "c4", "c5"],
            speakers=["s1", "s1", "s2", "s2", "s3"],
            splits=["train", "train", "val", "val", "test"],
        )
        report = validator.validate_corpus(features, index, expect_complete=True)
        assert report["status"] == "fail"
        assert report["failures"]["unaccounted"] == 1
        assert "c5" in report["failures"]["unaccounted_sample"]

    def test_without_expect_complete_unaccounted_is_only_reported(self, tmp_path):
        features = _write_corpus(tmp_path)
        index = _write_index(
            tmp_path,
            clip_ids=["c1", "c2", "c3", "c4", "c5"],
            speakers=["s1", "s1", "s2", "s2", "s3"],
            splits=["train", "train", "val", "val", "test"],
        )
        report = validator.validate_corpus(features, index, expect_complete=False)
        assert report["status"] == "pass"
        assert report["failures"]["unaccounted"] == 1

    def test_failures_are_deduplicated_and_recovered_clips_removed(self, tmp_path):
        features = _write_corpus(tmp_path)
        index = _write_index(
            tmp_path,
            clip_ids=["c1", "c2", "c3", "c4", "c5"],
            speakers=["s1", "s1", "s2", "s2", "s3"],
            splits=["train", "train", "val", "val", "test"],
        )
        failures = tmp_path / "failures.csv"
        pd.DataFrame(
            {
                "clip_id": ["c5", "c5", "c4"],
                "speaker_id": ["s3", "s3", "s2"],
                "reason": ["no_speaking_face", "no_frames", "no_speaking_face"],
            }
        ).to_csv(failures, index=False)

        report = validator.validate_corpus(
            features, index, failures_path=failures, expect_complete=True
        )
        summary = report["failures"]
        # c5 appears twice and is deduplicated; c4 succeeded and is recovered.
        assert summary["rows"] == 3
        assert summary["unique_clips"] == 2
        assert summary["effective_failures"] == 1
        assert summary["recovered_clips"] == 1
        assert summary["by_reason"] == {
            "no_speaking_face": 1,
            "no_frames": 1,
        }
        assert summary["unaccounted"] == 0
        assert report["status"] == "pass"

    def test_gender_conflict_is_a_hard_error(self, tmp_path):
        features = _write_corpus(
            tmp_path, genders=["f", "f", "m", "f"]
        )
        report = validator.validate_corpus(features)
        assert report["status"] == "fail"
        assert any("conflicting gender" in error for error in report["hard_errors"])

    def test_missing_required_dataset_fails_cleanly(self, tmp_path):
        path = tmp_path / "empty.h5"
        with h5py.File(path, "w") as store:
            store.create_dataset(
                "clip_ids", data=np.asarray(["c1"], dtype=object),
                dtype=h5py.string_dtype(),
            )
        report = validator.validate_corpus(path)
        assert report["status"] == "fail"
        assert any("missing datasets" in error for error in report["hard_errors"])
