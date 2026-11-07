"""Tests for structured artifact/runtime provenance."""
from __future__ import annotations

import hashlib

from voix.utils.provenance import (
    artifact_record,
    checkpoint_provenance,
    runtime_provenance,
    sha256_file,
)


def test_sha256_file_matches_hashlib(tmp_path):
    path = tmp_path / "artifact.bin"
    payload = b"voix-provenance-test"
    path.write_bytes(payload)
    assert sha256_file(path) == hashlib.sha256(payload).hexdigest()


def test_sha256_missing_file_is_none(tmp_path):
    assert sha256_file(tmp_path / "missing.bin") is None


def test_artifact_record_includes_hash_and_size(tmp_path):
    path = tmp_path / "checkpoint.pt"
    path.write_bytes(b"12345")
    record = artifact_record(path, "cfvn")
    assert record["role"] == "cfvn"
    assert record["exists"] is True
    assert record["bytes"] == 5
    assert record["sha256"] == hashlib.sha256(b"12345").hexdigest()


def test_checkpoint_provenance_includes_runtime_and_schema(tmp_path):
    path = tmp_path / "scaler.pt"
    path.write_bytes(b"scaler")
    record = checkpoint_provenance(
        {"face_scaler": path},
        device="cpu",
        backend="fallback",
        schema_version="FaceFeatureV2",
    )
    assert record["runtime"]["schema_version"] == "FaceFeatureV2"
    assert record["runtime"]["backend"] == "fallback"
    assert record["runtime"]["device"] == "cpu"
    assert record["artifacts"]["face_scaler"]["sha256"] == hashlib.sha256(
        b"scaler"
    ).hexdigest()


def test_runtime_provenance_reports_torch():
    record = runtime_provenance(device="cpu")
    assert record["torch"]
    assert record["cuda_available"] in (True, False)
