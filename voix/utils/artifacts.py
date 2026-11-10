"""Artifact manifest loading, verification, and restoration.

The serving stack depends on files that are deliberately not committed to git
(``.gitignore`` excludes model weights and processed data).  ``configs/artifacts.yaml``
is the single source of truth for what a checkout needs, including sizes and
SHA-256 digests.  The service preflight and ``scripts/setup/verify_artifacts.py``
both read this manifest so they cannot drift apart.

This module only inspects and copies files.  It never imports a model package
and never triggers a network download.
"""
from __future__ import annotations

import hashlib
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml

from voix.utils.paths import PROJECT_ROOT, project_path

MANIFEST_PATH = PROJECT_ROOT / "configs" / "artifacts.yaml"
_HASH_CHUNK_SIZE = 1024 * 1024


class ArtifactManifestError(ValueError):
    """Raised when the artifact manifest is malformed."""


@dataclass(frozen=True)
class ArtifactSpec:
    """One file declared in ``configs/artifacts.yaml``."""

    id: str
    name: str
    path: str
    role: str
    required: bool
    size_bytes: Optional[int] = None
    sha256: Optional[str] = None
    license: str = ""
    source: str = ""
    notes: str = ""

    @property
    def absolute_path(self) -> Path:
        return project_path(self.path)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "path": self.path,
            "role": self.role,
            "required": self.required,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
            "license": self.license,
            "source": self.source,
            "notes": self.notes,
        }


@dataclass(frozen=True)
class ArtifactManifest:
    """Parsed artifact manifest."""

    schema_version: int
    description: str
    artifacts: List[ArtifactSpec] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "description": self.description,
            "artifacts": [artifact.to_dict() for artifact in self.artifacts],
        }


def _require_mapping(value: Any, context: str) -> Dict[str, Any]:
    if not isinstance(value, dict):
        raise ArtifactManifestError(f"{context} must be a mapping")
    return value


def _validate_relative_path(value: str, artifact_id: str) -> str:
    path = Path(value)
    if path.is_absolute() or path.drive:
        raise ArtifactManifestError(
            f"artifact '{artifact_id}' path must be project-relative: {value}"
        )
    if ".." in path.parts:
        raise ArtifactManifestError(
            f"artifact '{artifact_id}' path must not escape the project root: {value}"
        )
    return path.as_posix()


def _optional_positive_int(value: Any, context: str) -> Optional[int]:
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ArtifactManifestError(f"{context} must be a non-negative integer")
    return value


def _validate_sha256(value: Any, context: str) -> Optional[str]:
    if value is None:
        return None
    digest = str(value).strip().lower()
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise ArtifactManifestError(f"{context} must be a 64-character SHA-256 digest")
    return digest


def load_manifest(path: Union[str, Path, None] = None) -> ArtifactManifest:
    """Load and validate the artifact manifest.

    Validation is intentionally strict: duplicate ids, absolute paths, and
    malformed digests are configuration errors, not warnings.
    """
    manifest_path = Path(path) if path is not None else MANIFEST_PATH
    if not manifest_path.is_file():
        raise ArtifactManifestError(f"Artifact manifest not found: {manifest_path}")

    with manifest_path.open("r", encoding="utf-8") as handle:
        try:
            raw = yaml.safe_load(handle)
        except yaml.YAMLError as exc:
            raise ArtifactManifestError(
                f"Artifact manifest is not valid YAML: {manifest_path}: {exc}"
            ) from exc

    root = _require_mapping(raw, "artifact manifest")
    schema_version = root.get("schema_version")
    if schema_version != 1:
        raise ArtifactManifestError(
            f"Unsupported artifact manifest schema_version: {schema_version!r}"
        )

    raw_artifacts = root.get("artifacts")
    if not isinstance(raw_artifacts, list) or not raw_artifacts:
        raise ArtifactManifestError("artifact manifest must contain a non-empty 'artifacts' list")

    specs: List[ArtifactSpec] = []
    seen_ids = set()
    for index, raw_entry in enumerate(raw_artifacts):
        entry = _require_mapping(raw_entry, f"artifacts[{index}]")
        artifact_id = str(entry.get("id", "")).strip()
        if not artifact_id:
            raise ArtifactManifestError(f"artifacts[{index}] is missing a non-empty 'id'")
        if artifact_id in seen_ids:
            raise ArtifactManifestError(f"duplicate artifact id: {artifact_id}")
        seen_ids.add(artifact_id)

        name = str(entry.get("name", "")).strip()
        if not name:
            raise ArtifactManifestError(f"artifact '{artifact_id}' is missing a 'name'")
        rel_path = _validate_relative_path(str(entry.get("path", "")), artifact_id)
        role = str(entry.get("role", "serving")).strip() or "serving"
        required = entry.get("required", True)
        if not isinstance(required, bool):
            raise ArtifactManifestError(f"artifact '{artifact_id}' required must be a boolean")

        specs.append(
            ArtifactSpec(
                id=artifact_id,
                name=name,
                path=rel_path,
                role=role,
                required=required,
                size_bytes=_optional_positive_int(
                    entry.get("size_bytes"), f"artifact '{artifact_id}' size_bytes"
                ),
                sha256=_validate_sha256(
                    entry.get("sha256"), f"artifact '{artifact_id}' sha256"
                ),
                license=str(entry.get("license", "")),
                source=str(entry.get("source", "")),
                notes=str(entry.get("notes", "")),
            )
        )

    return ArtifactManifest(
        schema_version=schema_version,
        description=str(root.get("description", "")),
        artifacts=specs,
    )


def sha256_file(path: Union[str, Path]) -> str:
    """Return the SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(_HASH_CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_path(
    spec: ArtifactSpec,
    path: Path,
    verify_hash: bool,
) -> Dict[str, Any]:
    """Verify one artifact at an explicit path."""
    exists = path.is_file()
    actual_size = path.stat().st_size if exists else None

    size_ok: Optional[bool] = None
    if spec.size_bytes is not None and actual_size is not None:
        size_ok = actual_size == spec.size_bytes

    hash_ok: Optional[bool] = None
    actual_hash: Optional[str] = None
    if exists and verify_hash and spec.sha256:
        actual_hash = sha256_file(path)
        hash_ok = actual_hash == spec.sha256

    if not exists:
        status = "optional_missing" if not spec.required else "missing"
    elif size_ok is False:
        status = "size_mismatch"
    elif hash_ok is False:
        status = "hash_mismatch"
    else:
        status = "ok"

    return {
        **spec.to_dict(),
        "absolute_path": str(path),
        "exists": exists,
        "actual_size_bytes": actual_size,
        "size_ok": size_ok,
        "hash_verified": hash_ok is not None,
        "hash_ok": hash_ok,
        "actual_sha256": actual_hash,
        "ok": status == "ok",
        "status": status,
    }


def verify_artifact(spec: ArtifactSpec, verify_hash: bool = True) -> Dict[str, Any]:
    """Verify one artifact against its manifest entry."""
    return _verify_path(spec, spec.absolute_path, verify_hash)


def verify_artifacts(
    manifest_path: Union[str, Path, None] = None,
    *,
    verify_hash: bool = True,
    strict: bool = False,
) -> Dict[str, Any]:
    """Verify every manifest artifact.

    ``strict`` also treats a missing optional artifact as a failure.  By
    default only required artifacts determine ``ok``.
    """
    manifest = load_manifest(manifest_path)
    entries = [verify_artifact(spec, verify_hash=verify_hash) for spec in manifest.artifacts]

    required = [entry for entry in entries if entry["required"]]
    optional = [entry for entry in entries if not entry["required"]]
    required_failed = [entry for entry in required if not entry["ok"]]
    optional_failed = [entry for entry in optional if not entry["ok"]]

    ok = not required_failed and (not strict or not optional_failed)
    return {
        "ok": ok,
        "strict": strict,
        "hash_verified": verify_hash,
        "manifest": str(Path(manifest_path) if manifest_path is not None else MANIFEST_PATH),
        "counts": {
            "total": len(entries),
            "required": len(required),
            "required_ok": len(required) - len(required_failed),
            "optional": len(optional),
            "optional_ok": len(optional) - len(optional_failed),
        },
        "failed_required": [entry["id"] for entry in required_failed],
        "failed_optional": [entry["id"] for entry in optional_failed],
        "artifacts": entries,
    }


def _copy_artifact(source: Path, target: Path) -> None:
    """Copy atomically enough that a partial copy is never treated as valid."""
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".partial")
    try:
        shutil.copy2(source, partial)
        os.replace(partial, target)
    finally:
        if partial.exists():
            partial.unlink(missing_ok=True)


def restore_artifacts(
    mirror_root: Union[str, Path],
    manifest_path: Union[str, Path, None] = None,
    *,
    include_optional: bool = True,
    verify_hash: bool = True,
) -> Dict[str, Any]:
    """Restore missing or invalid artifacts from a mirror directory.

    The mirror must use the same project-relative layout as the checkout, for
    example ``<mirror>/checkpoints/style_adapter_cfvn_best.pt``.  Each source
    file is checked against the manifest before it is copied.
    """
    mirror = Path(mirror_root).expanduser().resolve()
    if not mirror.is_dir():
        raise ArtifactManifestError(f"Artifact mirror is not a directory: {mirror}")

    manifest = load_manifest(manifest_path)
    restored: List[str] = []
    skipped: List[str] = []
    failed: List[str] = []
    failed_required: List[str] = []
    details: List[Dict[str, Any]] = []

    for spec in manifest.artifacts:
        before = verify_artifact(spec, verify_hash=verify_hash)
        if before["ok"]:
            continue
        if not spec.required and not include_optional:
            skipped.append(spec.id)
            details.append(
                {"id": spec.id, "action": "skipped_optional", "reason": before["status"]}
            )
            continue

        source = mirror.joinpath(*Path(spec.path).parts)
        if not source.is_file():
            failed.append(spec.id)
            if spec.required:
                failed_required.append(spec.id)
            details.append(
                {"id": spec.id, "action": "failed", "reason": f"source missing: {source}"}
            )
            continue

        source_check = _verify_path(spec, source, verify_hash)
        if not source_check["ok"]:
            failed.append(spec.id)
            if spec.required:
                failed_required.append(spec.id)
            details.append(
                {
                    "id": spec.id,
                    "action": "failed",
                    "reason": f"source invalid ({source_check['status']}): {source}",
                }
            )
            continue

        try:
            _copy_artifact(source, spec.absolute_path)
        except OSError as exc:
            failed.append(spec.id)
            if spec.required:
                failed_required.append(spec.id)
            details.append(
                {"id": spec.id, "action": "failed", "reason": f"copy failed: {exc}"}
            )
            continue

        after = verify_artifact(spec, verify_hash=verify_hash)
        if after["ok"]:
            restored.append(spec.id)
            details.append({"id": spec.id, "action": "restored"})
        else:
            failed.append(spec.id)
            if spec.required:
                failed_required.append(spec.id)
            details.append(
                {
                    "id": spec.id,
                    "action": "failed",
                    "reason": f"post-copy verification failed ({after['status']})",
                }
            )

    report = verify_artifacts(
        manifest_path, verify_hash=verify_hash, strict=False
    )
    report["restore"] = {
        "mirror": str(mirror),
        "include_optional": include_optional,
        "restored": restored,
        "skipped": skipped,
        "failed": failed,
        "failed_required": failed_required,
        "details": details,
        "ok": not failed_required and report["ok"],
    }
    report["ok"] = report["restore"]["ok"]
    return report
