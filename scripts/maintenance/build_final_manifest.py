"""Build and verify the final artifact manifest for the VOIX audit.

The manifest is the freeze record for the v2 research demo.  It hashes every
delivered audio file, face image, report table, comparison asset, calibration
artifact, audit document, canonical checkpoint, and the scripts/tests that
produced them.  It also re-checks the hashes already recorded inside the v2
report so a broken chain cannot be hidden by a fresh manifest.

Usage:
    python scripts/maintenance/build_final_manifest.py
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
DEMO = ROOT / "results" / "demo" / "research_v2"
REPORT = DEMO / "report.json"
BEFORE_AFTER = ROOT / "results" / "audit" / "before_after"
WER = ROOT / "results" / "audit" / "wer_crosscheck" / "wer_small.json"
CALIBRATION = ROOT / "results" / "audit" / "style_calibration" / "calibration.json"
OUTPUT = ROOT / "results" / "audit" / "final_manifest.json"

DEMO_FILES = [
    REPORT,
    DEMO / "index.csv",
    DEMO / "README.md",
    DEMO / "index.html",
]
BEFORE_AFTER_FILES = [
    BEFORE_AFTER / "index.html",
    BEFORE_AFTER / "comparison.csv",
    BEFORE_AFTER / "comparison.json",
]
AUDIT_EVIDENCE = [
    ROOT / "results" / "audit" / "face_voice_diversity" / "face_voice_diversity.json",
    ROOT / "results" / "audit" / "condition_response" / "condition_response.json",
    ROOT / "results" / "audit" / "demo_settings" / "demo_settings.json",
    ROOT / "results" / "audit" / "synthesis_matrix" / "synthesis_matrix.json",
    ROOT / "results" / "audit" / "ecapa_reference" / "ecapa_reference.json",
    ROOT / "results" / "audit" / "style_geometry" / "style_geometry.json",
    ROOT / "results" / "audit" / "baseline_audio" / "audio_metrics.json",
    ROOT / "results" / "audit" / "baseline_audio" / "long_term_spectra.json",
    ROOT / "results" / "audit" / "legacy_demo_audio" / "audio_metrics.json",
    ROOT / "results" / "audit" / "legacy_demo_audio" / "long_term_spectra.json",
    ROOT / "results" / "audit" / "reference_audio_metrics" / "audio_metrics.json",
    ROOT / "results" / "audit" / "reference_audio_metrics" / "long_term_spectra.json",
    ROOT / "results" / "audit" / "style_calibration" / "calibration.json",
    ROOT / "results" / "audit" / "style_calibration" / "real_style_bank.json",
    ROOT / "results" / "audit" / "style_calibration" / "style_calibration_audit.json",
    ROOT / "results" / "audit" / "cleanup" / "README.md",
    ROOT / "results" / "audit" / "cleanup" / "archive_manifest.json",
    ROOT / "results" / "audit" / "cleanup" / "deletion_manifest.json",
    WER,
]
AUDIT_DOCUMENTS = [
    ROOT / "results" / "audit" / "AUDIT_REPORT.md",
    ROOT / "results" / "audit" / "BEFORE_AFTER.md",
    ROOT / "results" / "audit" / "REMEDIATION_PLAN.md",
    ROOT / "results" / "audit" / "CLEANUP_INVENTORY.md",
]
SCRIPTS = [
    ROOT / "scripts" / "demo" / "generate_research_demo.py",
    ROOT / "scripts" / "audit" / "build_before_after.py",
    ROOT / "scripts" / "audit" / "audit_wer_crosscheck.py",
    ROOT / "scripts" / "audit" / "audit_audio_quality.py",
    ROOT / "scripts" / "audit" / "audit_face_voice_diversity.py",
    ROOT / "scripts" / "audit" / "audit_synthesis_matrix.py",
    ROOT / "scripts" / "audit" / "audit_style_calibration.py",
    ROOT / "scripts" / "training" / "build_style_calibration.py",
    ROOT / "scripts" / "audit" / "audit_ecapa_reference.py",
    ROOT / "scripts" / "audit" / "audit_condition_response.py",
    ROOT / "scripts" / "audit" / "audit_demo_settings.py",
    ROOT / "scripts" / "maintenance" / "build_final_manifest.py",
    ROOT / "scripts" / "maintenance" / "execute_cleanup.py",
    ROOT / "voix" / "models" / "style_calibration.py",
]
TESTS = [
    ROOT / "tests" / "test_generate_research_demo.py",
    ROOT / "tests" / "test_style_calibration.py",
]
SUPPORTING_EVIDENCE = [
    ROOT / "results" / "emotion" / "FINDINGS.md",
    ROOT / "results" / "accent" / "a3" / "scale_selection.json",
]
V1_SNAPSHOT = ROOT / "results" / "audit" / "baseline_audio" / "snapshot_20261105"

CANONICAL_EXPECTED = {
    "cfvn": "0253bbbf96def21ca9a3a9df3a4466c1e1538ef36986d9eb9fe86bfabd7b2f6b",
    "cvae": "bb7ad29b7b5cebb2141073892b9d5736d2e996c0bd3e81acc4c671c1ea66656b",
    "style_adapter": "97750ee84e50f9878e3d98de6b2e3dc0ca6829f448f2a5e4a31185adafc0a69c",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def entry(path: Path, category: str, note: str | None = None) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    record: Dict[str, Any] = {
        "path": _rel(path),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
        "category": category,
    }
    if note:
        record["note"] = note
    return record


def _git(*args: str) -> str:
    # rstrip only: the first porcelain status line starts with a significant
    # space, and .strip() would turn " M" into "M " and lose the count.
    return subprocess.check_output(
        ["git", *args], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
    ).rstrip("\r\n")


def main() -> int:
    report_bytes = REPORT.read_bytes()
    report = json.loads(report_bytes.decode("utf-8"))
    report_sha = hashlib.sha256(report_bytes).hexdigest()

    entries: Dict[str, List[Dict[str, Any]]] = {}
    entries["demo"] = [entry(path, "demo") for path in DEMO_FILES]
    entries["demo_audio"] = [
        entry(path, "demo_audio", condition["name"])
        for path in sorted((DEMO / "audio").glob("*.wav"))
        for condition in [
            next(c for c in report["conditions"] if c["file_name"] == path.name)
        ]
    ]
    entries["demo_faces"] = [
        entry(
            DEMO / "faces" / f"{voice['id']}{voice['face']['suffix']}",
            "demo_face",
            voice["label"],
        )
        for voice in report["voices"]
    ]
    entries["before_after"] = [
        entry(path, "before_after") for path in BEFORE_AFTER_FILES
    ] + [
        entry(path, "before_after_audio")
        for path in sorted((BEFORE_AFTER / "audio").rglob("*.wav"))
    ]
    entries["audit_evidence"] = [
        entry(path, "audit_evidence") for path in AUDIT_EVIDENCE
    ]
    entries["audit_documents"] = [
        entry(path, "audit_document") for path in AUDIT_DOCUMENTS
    ]
    entries["scripts"] = [entry(path, "script") for path in SCRIPTS]
    entries["tests"] = [entry(path, "test") for path in TESTS]
    entries["supporting_evidence"] = [
        entry(path, "supporting_evidence") for path in SUPPORTING_EVIDENCE
    ]
    entries["v1_snapshot"] = [
        entry(path, "v1_snapshot")
        for path in sorted(V1_SNAPSHOT.rglob("*"))
        if path.is_file()
    ]

    canonical_paths = {
        "cfvn": ROOT / "results" / "checkpoints" / "cfvn_v2_best.pt",
        "cvae": ROOT / "checkpoints" / "cvae_cfvn_face192_latent128_diversity.pt",
        "style_adapter": ROOT / "checkpoints" / "style_adapter_cfvn_best.pt",
    }
    entries["canonical_checkpoints"] = []
    canonical_checks: Dict[str, Any] = {}
    for name, path in canonical_paths.items():
        actual = sha256(path)
        expected = CANONICAL_EXPECTED[name]
        entries["canonical_checkpoints"].append(
            entry(path, "canonical_checkpoint", name)
        )
        canonical_checks[name] = {
            "path": _rel(path),
            "expected_sha256": expected,
            "sha256": actual,
            "match": actual == expected,
        }

    artifacts = report["artifacts"]
    artifact_checks: Dict[str, Any] = {}
    for name, record in artifacts.items():
        path = Path(record["path"])
        actual = sha256(path)
        recorded = record.get("sha256")
        artifact_checks[name] = {
            "path": _rel(path),
            "recorded_sha256": recorded,
            "sha256": actual,
            "match": actual == recorded,
        }

    audio_checks = report["verification"]["audio_hashes"]["checks"]
    audio_dir = DEMO / "audio"
    audio_hashes = []
    for check in audio_checks:
        path = audio_dir / f"{check['name']}.wav"
        actual = sha256(path)
        audio_hashes.append(
            {
                "name": check["name"],
                "recorded_sha256": check["sha256"],
                "sha256": actual,
                "match": actual == check["sha256"],
            }
        )

    face_checks = []
    for voice in report["voices"]:
        source = Path(voice["face"]["path"])
        copied = DEMO / "faces" / f"{voice['id']}{voice['face']['suffix']}"
        actual = sha256(source)
        face_checks.append(
            {
                "label": voice["label"],
                "source": _rel(source),
                "copied": _rel(copied),
                "recorded_sha256": voice["face"]["sha256"],
                "sha256": actual,
                "copied_sha256": sha256(copied),
                "source_match": actual == voice["face"]["sha256"],
                "copy_match": sha256(copied) == actual,
            }
        )

    calibration = json.loads(CALIBRATION.read_text(encoding="utf-8"))
    design_calibration = report["design"]["calibration"]
    calibration_check = {
        "path": _rel(CALIBRATION),
        "sha256": sha256(CALIBRATION),
        "report_sha256": design_calibration["sha256"],
        "match": sha256(CALIBRATION) == design_calibration["sha256"],
        "method": calibration.get("method"),
        "alpha": calibration.get("alpha"),
    }

    wer = json.loads(WER.read_text(encoding="utf-8"))
    wer_audio = {item["name"]: item["sha256"] for item in wer["audio"]}
    wer_check = {
        "path": _rel(WER),
        "source_report_sha256": wer["source_report_sha256"],
        "source_report_match": wer["source_report_sha256"] == report_sha,
        "model": wer["model"],
        "n": wer["n"],
        "mean_wer": wer["mean_wer"],
        "max_wer": wer["max_wer"],
        "audio_hashes_match": all(
            wer_audio.get(item["name"]) == item["sha256"] for item in audio_hashes
        ),
        "audio_count_match": len(wer_audio) == len(audio_hashes),
    }

    status = _git("status", "--porcelain").splitlines()
    dirty_counts = {
        "modified": sum(1 for line in status if line.startswith(" M")),
        "deleted": sum(1 for line in status if line.startswith(" D")),
        "untracked": sum(1 for line in status if line.startswith("??")),
        "total": len(status),
    }

    verification = {
        "canonical_checkpoints": {
            "passed": all(item["match"] for item in canonical_checks.values()),
            "checks": canonical_checks,
        },
        "report_artifacts": {
            "passed": all(item["match"] for item in artifact_checks.values()),
            "checks": artifact_checks,
        },
        "audio_hashes": {
            "passed": len(audio_hashes) == 21
            and all(item["match"] for item in audio_hashes),
            "n": len(audio_hashes),
            "checks": audio_hashes,
        },
        "face_images": {
            "passed": len(face_checks) == 3
            and all(
                item["source_match"] and item["copy_match"] for item in face_checks
            ),
            "n": len(face_checks),
            "checks": face_checks,
        },
        "calibration": calibration_check,
        "wer_cross_check": wer_check,
        "counts": {
            "demo_files": len(entries["demo"]),
            "demo_audio": len(entries["demo_audio"]),
            "demo_faces": len(entries["demo_faces"]),
            "before_after_audio": len(
                [item for item in entries["before_after"] if item["category"] == "before_after_audio"]
            ),
            "audit_evidence": len(entries["audit_evidence"]),
            "audit_documents": len(entries["audit_documents"]),
            "v1_snapshot_files": len(entries["v1_snapshot"]),
        },
    }
    verification["passed"] = all(
        [
            verification["canonical_checkpoints"]["passed"],
            verification["report_artifacts"]["passed"],
            verification["audio_hashes"]["passed"],
            verification["face_images"]["passed"],
            verification["calibration"]["match"],
            verification["wer_cross_check"]["source_report_match"],
            verification["wer_cross_check"]["audio_hashes_match"],
            verification["wer_cross_check"]["audio_count_match"],
        ]
    )

    manifest = {
        "schema_version": "voix.final_manifest.v1",
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "level": "exploratory",
        "source_report": {
            "path": _rel(REPORT),
            "sha256": report_sha,
            "schema_version": report["schema_version"],
            "created_at": report["created_at"],
            "conditions": len(report["conditions"]),
        },
        "design": {
            "voices": [voice["label"] for voice in report["voices"]],
            "emotions": report["design"]["emotions"],
            "accents": report["design"]["accents"],
            "clip_count": report["design"]["clip_count"],
            "seed": report["design"]["seed"],
            "diffusion_steps": report["design"]["diffusion_steps"],
            "calibration": calibration_check,
        },
        "git": {
            "branch": _git("branch", "--show-current"),
            "commit": _git("rev-parse", "HEAD"),
            "dirty_tree": dirty_counts,
            "note": "The dirty tree is pre-existing user work; this audit did not commit or revert it.",
        },
        "verification": verification,
        "entries": entries,
        "known_gaps": [
            "No human MOS, CMOS, emotion-identification, accent-identification, or voice-triad study exists.",
            "The demo noise floor is reduced but remains about 12.7 dB above the human AMI median.",
            "Timothee-Jackie is at the 97.8th percentile of real same-gender between-speaker pairs; the pass uses the observed real maximum.",
            "Accent labels are applied timbre conditions, not perceptually validated accents.",
            "Sad is duration-only on the calibrated face styles.",
            "The config files do not describe the actual run (diffusion_steps null, tts_device cpu).",
            "The report does not embed the git commit, dirty-tree flag, or exact command line.",
            "The approved 2026-11-05 cleanup was executed; v1 remains alongside v2. Per-file records are in results/audit/cleanup/.",
        ],
        "notes": [
            "Every entry was re-hashed from disk when this manifest was built.",
            "The manifest does not include itself, so it is not a fixed point.",
            "The before/after audio files are copies; the original v1 snapshot and source audit audio remain unchanged.",
            "No canonical checkpoint was modified and no experimental adapter was wired into serving.",
        ],
    }

    OUTPUT.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {OUTPUT}")
    print(f"verification passed: {verification['passed']}")
    for key in (
        "canonical_checkpoints",
        "report_artifacts",
        "audio_hashes",
        "face_images",
    ):
        print(f"  {key}: {verification[key]['passed']}")
    print(
        "  calibration: "
        f"{verification['calibration']['match']} | "
        "wer source: "
        f"{verification['wer_cross_check']['source_report_match']} | "
        "wer audio: "
        f"{verification['wer_cross_check']['audio_hashes_match']}"
    )
    print(f"  entries: {sum(len(items) for items in entries.values())}")
    return 0 if verification["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
