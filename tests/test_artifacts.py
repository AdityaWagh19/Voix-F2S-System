"""Artifact manifest and restore-tooling tests.

The real manifest is checked for its contract, while verify/restore behavior is
exercised against temporary files so the suite passes on a clean checkout.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import yaml

import voix.utils.artifacts as artifacts
from voix.utils.artifacts import (
    ArtifactManifestError,
    load_manifest,
    restore_artifacts,
    verify_artifacts,
)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _entry(rel_path: str, data: bytes, *, required: bool = True, **overrides):
    entry = {
        "id": rel_path.replace("/", "_").replace(".", "_"),
        "name": rel_path,
        "path": rel_path,
        "role": "test",
        "required": required,
        "size_bytes": len(data),
        "sha256": _sha(data),
    }
    entry.update(overrides)
    return entry


def _write_manifest(tmp_path: Path, entries, name: str = "artifacts.yaml") -> Path:
    path = tmp_path / name
    path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "description": "test manifest",
                "artifacts": entries,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return path


@pytest.fixture
def checkout(tmp_path, monkeypatch):
    root = tmp_path / "checkout"
    root.mkdir()
    monkeypatch.setattr(
        artifacts,
        "project_path",
        lambda *parts: root.joinpath(*(str(part) for part in parts)),
    )
    return root


def test_repository_manifest_declares_serving_contract():
    manifest = load_manifest()
    ids = {artifact.id for artifact in manifest.artifacts}
    assert manifest.schema_version == 1
    assert {
        "cfvn_v2",
        "cvae_diversity",
        "style_adapter",
        "face_scaler_v2",
        "mediapipe_landmarker",
        "arcface_detector",
        "arcface_landmark_2d",
        "arcface_landmark_3d",
        "arcface_genderage",
        "arcface_recognition",
    } <= ids
    assert len([artifact for artifact in manifest.artifacts if artifact.id.startswith("arcface_")]) == 5
    for artifact in manifest.artifacts:
        assert artifact.size_bytes is not None, artifact.id
        assert artifact.sha256 is not None, artifact.id


def test_verify_required_and_optional(tmp_path, checkout):
    required_data = b"required-weights"
    manifest = _write_manifest(
        tmp_path,
        [
            _entry("checkpoints/model.pt", required_data),
            _entry("checkpoints/optional.pt", b"optional", required=False),
        ],
    )
    (checkout / "checkpoints").mkdir()
    (checkout / "checkpoints" / "model.pt").write_bytes(required_data)

    report = verify_artifacts(manifest, verify_hash=True)
    assert report["ok"] is True
    assert report["counts"]["required_ok"] == 1
    assert report["counts"]["optional_ok"] == 0
    statuses = {entry["id"]: entry["status"] for entry in report["artifacts"]}
    assert statuses["checkpoints_optional_pt"] == "optional_missing"

    strict_report = verify_artifacts(manifest, verify_hash=True, strict=True)
    assert strict_report["ok"] is False


def test_verify_detects_size_and_hash_mismatch(tmp_path, checkout):
    (checkout / "checkpoints").mkdir()
    target = checkout / "checkpoints" / "model.pt"
    target.write_bytes(b"hello")

    size_manifest = _write_manifest(
        tmp_path,
        [_entry("checkpoints/model.pt", b"hello", size_bytes=3)],
        name="size.yaml",
    )
    size_report = verify_artifacts(size_manifest, verify_hash=False)
    assert size_report["ok"] is False
    assert size_report["artifacts"][0]["status"] == "size_mismatch"

    hash_manifest = _write_manifest(
        tmp_path,
        [_entry("checkpoints/model.pt", b"world")],
        name="hash.yaml",
    )
    hash_report = verify_artifacts(hash_manifest, verify_hash=True)
    assert hash_report["ok"] is False
    assert hash_report["artifacts"][0]["status"] == "hash_mismatch"


def test_restore_copies_valid_files_from_mirror(tmp_path, checkout):
    data = b"audited-checkpoint"
    manifest = _write_manifest(tmp_path, [_entry("checkpoints/model.pt", data)])
    mirror = tmp_path / "mirror"
    (mirror / "checkpoints").mkdir(parents=True)
    (mirror / "checkpoints" / "model.pt").write_bytes(data)

    report = restore_artifacts(mirror, manifest, verify_hash=True)
    assert report["ok"] is True
    assert report["restore"]["restored"] == ["checkpoints_model_pt"]
    assert (checkout / "checkpoints" / "model.pt").read_bytes() == data


def test_restore_rejects_invalid_source(tmp_path, checkout):
    expected = b"correct-weights"
    manifest = _write_manifest(tmp_path, [_entry("checkpoints/model.pt", expected)])
    mirror = tmp_path / "mirror"
    (mirror / "checkpoints").mkdir(parents=True)
    (mirror / "checkpoints" / "model.pt").write_bytes(b"wrong-weights")

    report = restore_artifacts(mirror, manifest, verify_hash=True)
    assert report["ok"] is False
    assert report["restore"]["failed"] == ["checkpoints_model_pt"]
    assert report["restore"]["failed_required"] == ["checkpoints_model_pt"]
    assert not (checkout / "checkpoints" / "model.pt").exists()


def test_restore_optional_failure_does_not_fail_required_gate(tmp_path, checkout):
    required_data = b"required-weights"
    optional_data = b"optional-weights"
    manifest = _write_manifest(
        tmp_path,
        [
            _entry("checkpoints/model.pt", required_data),
            _entry("checkpoints/optional.pt", optional_data, required=False),
        ],
    )
    (checkout / "checkpoints").mkdir()
    (checkout / "checkpoints" / "model.pt").write_bytes(required_data)
    mirror = tmp_path / "mirror"
    mirror.mkdir()

    report = restore_artifacts(mirror, manifest, verify_hash=True)
    assert report["ok"] is True
    assert report["restore"]["failed"] == ["checkpoints_optional_pt"]
    assert report["restore"]["failed_required"] == []
    assert report["failed_optional"] == ["checkpoints_optional_pt"]


def test_manifest_rejects_absolute_paths_and_duplicate_ids(tmp_path):
    absolute_manifest = _write_manifest(
        tmp_path,
        [
            {
                "id": "absolute",
                "name": "absolute",
                "path": str(tmp_path / "model.pt"),
                "role": "test",
                "required": True,
            }
        ],
        name="absolute.yaml",
    )
    with pytest.raises(ArtifactManifestError, match="project-relative"):
        load_manifest(absolute_manifest)

    duplicate_manifest = _write_manifest(
        tmp_path,
        [
            _entry("a/model.pt", b"a"),
            _entry("b/model.pt", b"b", id="a_model_pt"),
        ],
        name="duplicate.yaml",
    )
    with pytest.raises(ArtifactManifestError, match="duplicate artifact id"):
        load_manifest(duplicate_manifest)
