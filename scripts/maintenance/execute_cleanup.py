"""Execute the approved cleanup with a checksum manifest.

Default mode is a dry run.  ``--execute`` archives the approved evidence and
checkpoints to ``D:\\voix_archive\\20261105_cleanup``, verifies every archived
SHA-256, deletes the approved reproducible caches, and writes manifests under
``results/audit/cleanup``.

This script never touches canonical checkpoints, the v1/v2 research demos,
the audit evidence referenced by the final manifest, or user work outside the
explicit lists below.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

ROOT = Path(__file__).resolve().parents[2]
ARCHIVE_ROOT = Path(r"D:\voix_archive\20261105_cleanup")
ARCHIVE_FILES = ARCHIVE_ROOT / "files"
AUDIT_DIR = ROOT / "results" / "audit" / "cleanup"

AUDIO_SUFFIXES = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}

DELETE_DIRS = [
    ROOT / ".tmp",
    ROOT / "results" / "audio_quality",
    ROOT / "results" / "audio_quality_20261029",
    ROOT / "results" / "audio_quality_phase0",
    ROOT / "results" / "audio_samples",
    ROOT / "results" / "control_audit",
    ROOT / "results" / "control_audit_t1",
    ROOT / "results" / "demo_face_audit",
    ROOT / "results" / "step_ablation",
]

ARCHIVE_AUDIO_DIRS = [
    ROOT / "results" / "emotion" / "e0",
    ROOT / "results" / "emotion" / "e0_smoke",
    ROOT / "results" / "emotion" / "e0_smoke_v2",
    ROOT / "results" / "emotion" / "e1",
    ROOT / "results" / "emotion" / "e1_smoke",
    ROOT / "results" / "emotion" / "e1c",
    ROOT / "results" / "emotion" / "e2",
    ROOT / "results" / "emotion" / "e2_smoke",
    ROOT / "results" / "emotion" / "e2c",
    ROOT / "results" / "naturalness" / "n0b",
    ROOT / "results" / "naturalness" / "n1",
    ROOT / "results" / "accent" / "a2",
    ROOT / "results" / "accent" / "a2b",
    ROOT / "results" / "accent" / "a3",
]

ARCHIVE_WHOLE_DIRS = [
    ROOT / "results" / "demo_pre_phase0",
]

# Explicitly retained because an active code path, the serving diagnostics, or
# the audit manifest references them.  The reasons are recorded in the plan.
RETAINED = {
    "results/checkpoints/cfvn_v2_best.pt": "canonical CFVN; hashed by the v2 report",
    "results/checkpoints/cfvn_v2_arcface.pt": "documented raw-ArcFace baseline; scripts/evaluation/compare_cfvn.py references it",
    "results/checkpoints/D_joint_infonce_best.pt": "voix/service.py model-inventory fallback references it",
    "checkpoints/cvae_cfvn_face192_latent128_diversity.pt": "canonical CVAE; hashed by the v2 report",
    "checkpoints/style_adapter_cfvn_best.pt": "canonical StyleAdapter; hashed by the v2 report",
    "results/naturalness/n1/ami_lora_pilot.pt": "N1 negative-result checkpoint; scripts/audit/audit_synthesis_matrix.py uses it",
    "results/naturalness/n1/ami_lora_pilot_last.pt": "last-epoch companion to the retained N1 pilot",
    "results/checkpoints/styletts2_lora_prosody.pt": "StyleTTS2 LoRA pilot checkpoint retained as the 0.741 h result",
    "results/styletts2_lora_eval": "54 WAV paths are referenced by two retained generation-eval JSON files; retaining avoids broken evidence paths",
}

KEEP_CHECKPOINT_NAMES = {
    "cfvn_v2_best.pt",
    "cfvn_v2_arcface.pt",
    "D_joint_infonce_best.pt",
    "cvae_cfvn_face192_latent128_diversity.pt",
    "style_adapter_cfvn_best.pt",
    "ami_lora_pilot.pt",
    "ami_lora_pilot_last.pt",
    "styletts2_lora_prosody.pt",
}

CHECKPOINT_SMOKE_DIRS = [
    ROOT / "results" / "naturalness" / "n1",
    ROOT / "results" / "checkpoints",
    ROOT / "results" / "emotion" / "e2_smoke",
    ROOT / "checkpoints",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rel(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def _record(path: Path, category: str) -> Dict[str, Any]:
    return {
        "source": _rel(path),
        "destination": (ARCHIVE_FILES / _rel(path)).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
        "category": category,
    }


def _iter_files(path: Path) -> Iterable[Path]:
    if path.is_file():
        yield path
    elif path.is_dir():
        for child in sorted(path.rglob("*")):
            if child.is_file():
                yield child


def build_archive_plan() -> tuple[List[Dict[str, Any]], Dict[str, int]]:
    entries: Dict[str, Dict[str, Any]] = {}

    for directory in ARCHIVE_AUDIO_DIRS:
        if not directory.is_dir():
            continue
        for path in sorted(directory.rglob("*")):
            if path.is_file() and path.suffix.lower() in AUDIO_SUFFIXES:
                entries[_rel(path)] = _record(path, "intermediate_audio")

    for directory in ARCHIVE_WHOLE_DIRS:
        if not directory.is_dir():
            continue
        for path in _iter_files(directory):
            entries[_rel(path)] = _record(path, "legacy_demo")

    # Experimental checkpoints.  Keep the explicit allow-list; archive the rest.
    for directory in CHECKPOINT_SMOKE_DIRS:
        if not directory.is_dir():
            continue
        for path in sorted(directory.glob("*.pt")):
            if path.name in KEEP_CHECKPOINT_NAMES:
                continue
            entries[_rel(path)] = _record(path, "experimental_checkpoint")

    # The two top-level checkpoint files below are covered by the loop above.
    # The styletts2_lora_prosody_last.pt file is covered by the same glob.
    totals: Dict[str, int] = {}
    for entry in entries.values():
        totals[entry["category"]] = totals.get(entry["category"], 0) + entry["bytes"]
    return sorted(entries.values(), key=lambda item: item["source"]), totals


def build_delete_plan() -> tuple[List[Dict[str, Any]], int]:
    entries: List[Dict[str, Any]] = []
    for directory in DELETE_DIRS:
        if not directory.exists():
            continue
        for path in sorted(_iter_files(directory)):
            entries.append(
                {
                    "path": _rel(path),
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                    "category": "reproducible_cache",
                }
            )
    return entries, sum(item["bytes"] for item in entries)


def _prune_empty_dirs(root: Path) -> None:
    """Remove directories left empty by a whole-directory archive.

    Only empty directories are removed; any remaining file keeps its parent
    chain in place.  Used for ``ARCHIVE_WHOLE_DIRS`` so the repository does not
    keep an empty skeleton of an archived legacy snapshot.
    """
    if not root.is_dir():
        return
    for child in sorted(root.rglob("*"), key=lambda p: len(p.parts), reverse=True):
        if child.is_dir() and not any(child.iterdir()):
            child.rmdir()
    if root.is_dir() and not any(root.iterdir()):
        root.rmdir()


def _verify(entry: Dict[str, Any]) -> None:
    destination = Path(entry["destination"])
    if not destination.is_file():
        raise RuntimeError(f"archive destination missing: {destination}")
    if destination.stat().st_size != entry["bytes"]:
        raise RuntimeError(f"archive size mismatch: {destination}")
    if sha256(destination) != entry["sha256"]:
        raise RuntimeError(f"archive hash mismatch: {destination}")


def execute(archive_entries: Sequence[Dict[str, Any]], delete_entries: Sequence[Dict[str, Any]]) -> None:
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    ARCHIVE_ROOT.mkdir(parents=True, exist_ok=True)
    ARCHIVE_FILES.mkdir(parents=True, exist_ok=True)

    created = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    archive_manifest = {
        "schema_version": "voix.cleanup_archive.v1",
        "created_at": created,
        "status": "planned",
        "archive_root": str(ARCHIVE_ROOT),
        "source_root": str(ROOT),
        "policy": (
            "Archived files are moved, not deleted. The manifest records the "
            "original repository path, archive path, byte size, and SHA-256. "
            "Non-audio JSON/MD/LOG files stay in the repository except for the "
            "whole legacy demo snapshot."
        ),
        "retained": RETAINED,
        "totals": {
            "files": len(archive_entries),
            "bytes": sum(item["bytes"] for item in archive_entries),
        },
        "entries": archive_entries,
    }
    (AUDIT_DIR / "archive_manifest.json").write_text(
        json.dumps(archive_manifest, indent=2) + "\n", encoding="utf-8"
    )

    deletion_manifest = {
        "schema_version": "voix.cleanup_deletion.v1",
        "created_at": created,
        "status": "planned",
        "policy": (
            "Only reproducible caches and legacy diagnostic output. Every file "
            "is hashed before deletion. No manifest-pinned audit artifact, demo "
            "audio, canonical checkpoint, or user work is included."
        ),
        "totals": {
            "files": len(delete_entries),
            "bytes": sum(item["bytes"] for item in delete_entries),
        },
        "entries": delete_entries,
    }
    (AUDIT_DIR / "deletion_manifest.json").write_text(
        json.dumps(deletion_manifest, indent=2) + "\n", encoding="utf-8"
    )

    moved: List[Dict[str, Any]] = []
    try:
        for entry in archive_entries:
            source = ROOT / entry["source"]
            destination = Path(entry["destination"])
            if destination.exists():
                raise RuntimeError(f"destination already exists: {destination}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(source), str(destination))
            _verify(entry)
            moved.append(entry)
    except Exception:
        # Best effort: move anything already archived back before re-raising.
        for entry in reversed(moved):
            source = ROOT / entry["source"]
            destination = Path(entry["destination"])
            if destination.exists() and not source.exists():
                source.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(destination), str(source))
        raise

    for directory in ARCHIVE_WHOLE_DIRS:
        _prune_empty_dirs(directory)

    executed_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    archive_manifest["status"] = "executed"
    archive_manifest["executed_at"] = executed_at
    (AUDIT_DIR / "archive_manifest.json").write_text(
        json.dumps(archive_manifest, indent=2) + "\n", encoding="utf-8"
    )

    deleted: List[Dict[str, Any]] = []
    for entry in delete_entries:
        path = ROOT / entry["path"]
        if path.exists():
            path.unlink()
            deleted.append(entry)

    # Remove now-empty directories bottom-up.  Keep the root of results/.
    for directory in sorted(
        (d for d in DELETE_DIRS if d.exists()), key=lambda d: len(d.parts), reverse=True
    ):
        if directory.is_dir():
            shutil.rmtree(directory, ignore_errors=False)

    deletion_manifest["status"] = "executed"
    deletion_manifest["executed_at"] = executed_at
    deletion_manifest["entries"] = deleted
    deletion_manifest["totals"] = {
        "files": len(deleted),
        "bytes": sum(item["bytes"] for item in deleted),
    }
    (AUDIT_DIR / "deletion_manifest.json").write_text(
        json.dumps(deletion_manifest, indent=2) + "\n", encoding="utf-8"
    )

    # Verify every archive destination again after all moves.
    failures = []
    for entry in archive_entries:
        try:
            _verify(entry)
        except RuntimeError as exc:
            failures.append(str(exc))
    if failures:
        raise RuntimeError("post-move verification failed: " + "; ".join(failures))

    summary = {
        "archive": {
            "files": len(archive_entries),
            "bytes": sum(item["bytes"] for item in archive_entries),
            "root": str(ARCHIVE_ROOT),
        },
        "deletion": {
            "files": len(deleted),
            "bytes": sum(item["bytes"] for item in deleted),
        },
        "retained": RETAINED,
        "verification": "all archived destinations re-hashed after the move",
    }
    (AUDIT_DIR / "README.md").write_text(
        "# Cleanup execution record\n\n"
        f"Executed: {executed_at}\n\n"
        f"- Archived: {summary['archive']['files']} files, "
        f"{summary['archive']['bytes'] / 1024 / 1024:.1f} MiB to "
        f"`{ARCHIVE_ROOT}`.\n"
        f"- Deleted: {summary['deletion']['files']} reproducible cache files, "
        f"{summary['deletion']['bytes'] / 1024 / 1024:.1f} MiB.\n"
        "- Every archived file was re-hashed at its destination after the move.\n"
        "- Canonical checkpoints, v1/v2 demo audio, final-manifest entries, and "
        "user work were not touched.\n\n"
        "The authoritative file-level records are `archive_manifest.json` and "
        "`deletion_manifest.json` in this directory.\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    archive_entries, archive_totals = build_archive_plan()
    delete_entries, delete_bytes = build_delete_plan()

    print("archive plan")
    print(
        f"  files={len(archive_entries)} "
        f"bytes={sum(item['bytes'] for item in archive_entries)}"
    )
    for category, size in sorted(archive_totals.items()):
        print(f"  {category}: {size / 1024 / 1024:.1f} MiB")
    print("delete plan")
    print(f"  files={len(delete_entries)} bytes={delete_bytes}")
    print("retained by explicit rule")
    for path, reason in RETAINED.items():
        print(f"  {path}: {reason}")

    if not args.execute:
        print("dry run only; pass --execute to move and delete")
        return 0

    execute(archive_entries, delete_entries)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
