"""Validate a research manifest before training or evaluation.

Examples::

    python scripts/audit/validate_research_manifest.py \
        --manifest data/processed/crema_d_manifest.csv --kind crema_d

    python scripts/audit/validate_research_manifest.py \
        --manifest data/processed/globe_accent_manifest.csv --kind globe \
        --audio-root data/processed/globe_audio --check-audio \
        --min-speakers-per-class 30 --min-hours-per-class 2.0

The validator never repairs the manifest.  A non-zero exit with
``--require-pass`` means the split is not safe to use.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from voix.evaluation.manifest_validation import (  # noqa: E402
    KINDS,
    validate_manifest,
    write_validation_report,
)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a research manifest")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--kind", choices=sorted(KINDS), required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON report path (default: print only)",
    )
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--audio-root", type=Path, default=None)
    parser.add_argument("--check-audio", action="store_true")
    parser.add_argument("--min-speakers-per-class", type=int, default=0)
    parser.add_argument("--min-hours-per-class", type=float, default=None)
    parser.add_argument(
        "--trainable-labels",
        default=None,
        help="Comma-separated labels subject to the class thresholds; others are "
        "reported as evaluation-only",
    )
    parser.add_argument("--require-pass", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    report = validate_manifest(
        args.manifest,
        kind=args.kind,
        root=args.root,
        audio_root=args.audio_root,
        check_audio=args.check_audio,
        min_speakers_per_class=args.min_speakers_per_class,
        min_hours_per_class=args.min_hours_per_class,
        trainable_labels=(
            [label.strip() for label in args.trainable_labels.split(",") if label.strip()]
            if args.trainable_labels
            else None
        ),
    )
    if args.output is not None:
        write_validation_report(report, args.output)

    print(f"Manifest: {report['manifest']}")
    print(f"Kind: {report['kind']}")
    print(f"Passed: {report['passed']}")
    summary = report.get("summary", {})
    if summary:
        print(f"Rows: {summary.get('rows')}")
        print(f"Speakers: {summary.get('speakers')}")
        for split, values in sorted(summary.get("splits", {}).items()):
            print(
                f"  {split}: rows={values['rows']} speakers={values['speakers']} "
                f"hours={values['hours']}"
            )
        for label, values in sorted(summary.get("labels", {}).items()):
            suffix = "" if values.get("trainable", True) else " (evaluation-only)"
            print(
                f"  label {label}: rows={values['rows']} speakers={values['speakers']} "
                f"train_speakers={values['train_speakers']} "
                f"train_hours={values['train_hours']}{suffix}"
            )
    for error in report.get("errors", []):
        print(f"ERROR: {error}")
    for warning in report.get("warnings", []):
        print(f"WARNING: {warning}")
    if args.output is not None:
        print(f"Wrote {args.output}")
    return 0 if report["passed"] or not args.require_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
