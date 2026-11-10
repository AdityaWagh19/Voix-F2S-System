#!/usr/bin/env python3
"""Verify or restore the artifacts declared in ``configs/artifacts.yaml``.

The repository deliberately does not commit model weights or processed data.
This command checks that a checkout has the exact audited files, and can copy
them from a mirror directory that preserves the project-relative layout.

Examples:
    python scripts/setup/verify_artifacts.py
    python scripts/setup/verify_artifacts.py --no-hash --verbose
    python scripts/setup/verify_artifacts.py --restore-from D:\\voix-artifacts
    python scripts/setup/verify_artifacts.py --strict --json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from voix.utils.artifacts import (  # noqa: E402
    ArtifactManifestError,
    restore_artifacts,
    verify_artifacts,
)


def _format_bytes(value: Any) -> str:
    if value is None:
        return "n/a"
    size = float(value)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024.0 or unit == "GB":
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} GB"


def _print_human(report: Dict[str, Any], verbose: bool) -> None:
    print(f"Manifest: {report['manifest']}")
    for entry in report["artifacts"]:
        if entry["ok"] and not verbose:
            continue
        required = "required" if entry["required"] else "optional"
        status = entry["status"]
        if entry["ok"]:
            status = "ok"
        detail = f"{status} ({required})"
        if entry["actual_size_bytes"] is not None:
            detail += f", {_format_bytes(entry['actual_size_bytes'])}"
        print(f"  {entry['id']:<24} {detail:<34} {entry['path']}")
        if entry["status"] == "size_mismatch":
            print(
                f"       expected {_format_bytes(entry['size_bytes'])}, "
                f"found {_format_bytes(entry['actual_size_bytes'])}"
            )
        if entry["status"] == "hash_mismatch":
            print(f"       expected sha256 {entry['sha256']}")
            print(f"       actual   sha256 {entry['actual_sha256']}")

    counts = report["counts"]
    print(
        f"\nRequired: {counts['required_ok']}/{counts['required']} ok; "
        f"optional: {counts['optional_ok']}/{counts['optional']} ok"
    )
    if report.get("restore"):
        restore = report["restore"]
        print(
            f"Restore: {len(restore['restored'])} restored, "
            f"{len(restore['skipped'])} skipped, {len(restore['failed'])} failed"
        )
        for detail in restore["details"]:
            if detail["action"] != "restored":
                print(f"  {detail['id']}: {detail['action']} - {detail.get('reason', '')}")

    if report["ok"]:
        print("Artifact check passed.")
    else:
        print("Artifact check failed.", file=sys.stderr)
        if report["failed_required"]:
            print(
                "Missing or invalid required: "
                + ", ".join(report["failed_required"]),
                file=sys.stderr,
            )


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Verify or restore the VOIX artifact manifest."
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default=None,
        help="Manifest path. Default: configs/artifacts.yaml",
    )
    parser.add_argument(
        "--restore-from",
        type=str,
        default=None,
        metavar="DIR",
        help="Copy missing/invalid files from a mirror with the same layout.",
    )
    parser.add_argument(
        "--skip-optional",
        action="store_true",
        help="Do not restore optional artifacts (they do not affect exit status).",
    )
    parser.add_argument(
        "--no-hash",
        action="store_true",
        help="Check existence and size only. Faster, but not tamper-evident.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat missing optional artifacts as a failure.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print every artifact, including passing ones.",
    )
    parser.add_argument("--json", action="store_true", help="Machine-readable output.")
    args = parser.parse_args(argv)

    verify_hash = not args.no_hash

    try:
        if args.restore_from:
            report = restore_artifacts(
                args.restore_from,
                manifest_path=args.manifest,
                include_optional=not args.skip_optional,
                verify_hash=verify_hash,
            )
            if args.strict:
                failed_optional = report.get("failed_optional", [])
                if failed_optional:
                    report["ok"] = False
        else:
            report = verify_artifacts(
                manifest_path=args.manifest,
                verify_hash=verify_hash,
                strict=args.strict,
            )
    except ArtifactManifestError as exc:
        print(f"Artifact manifest error: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        _print_human(report, verbose=args.verbose)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
