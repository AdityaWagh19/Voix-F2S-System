"""Build the paired CREMA-D evaluation grid.

The default is the validation actors.  The test actors are reserved for the
final E3 comparison and are not used by E0/E1/E2.

Examples::

    # Confirmatory-pilot grid on validation actors, all six CREMA-D classes
    python scripts/data/build_crema_paired_eval.py \
        --split val \
        --output data/processed/crema_d_paired_eval_val.csv \
        --summary results/emotion/e0/paired_eval_manifest_summary.json

    # Smoke grid for one actor
    python scripts/data/build_crema_paired_eval.py --split val --max-actors 1 \
        --output results/emotion/smoke/paired_eval.csv
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from voix.evaluation.crema_paired import (  # noqa: E402
    CREMA_EMOTIONS,
    PAIRED_EVAL_SCHEMA,
    build_paired_grid,
    validate_paired_grid,
    write_paired_grid,
)

DEFAULT_MANIFEST = ROOT / "data" / "processed" / "crema_d_manifest.csv"
DEFAULT_OUTPUT = ROOT / "data" / "processed" / "crema_d_paired_eval_val.csv"
DEFAULT_SUMMARY = ROOT / "results" / "emotion" / "e0" / "paired_eval_manifest_summary.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the paired CREMA-D evaluation grid")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--split", default="val", choices=("train", "val", "test"))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument(
        "--emotions",
        default=",".join(CREMA_EMOTIONS),
        help="Comma-separated emotion classes to include; neutral is always required",
    )
    parser.add_argument("--max-actors", type=int, default=None)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--check-audio", action="store_true")
    parser.add_argument("--allow-invalid", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if not args.manifest.is_file():
        raise FileNotFoundError(f"CREMA-D manifest not found: {args.manifest}")

    manifest = pd.read_csv(args.manifest)
    emotions = [item.strip() for item in args.emotions.split(",") if item.strip()]
    grid, summary = build_paired_grid(
        manifest,
        split=args.split,
        emotions=emotions,
        max_actors=args.max_actors,
        root=args.root,
    )
    errors = validate_paired_grid(
        grid,
        root=args.root,
        check_audio=args.check_audio,
    )

    summary.update(
        {
            "schema_version": PAIRED_EVAL_SCHEMA,
            "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
            "source_manifest": str(args.manifest),
            "source_manifest_sha256": _sha256(args.manifest),
            "output": str(args.output),
            "output_sha256": None,
            "validation": {
                "passed": not errors,
                "errors": errors,
                "audio_checked": bool(args.check_audio),
            },
        }
    )

    write_paired_grid(grid, args.output)
    summary["output_sha256"] = _sha256(args.output)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Split: {summary['split']}")
    print(f"Actors: {summary['actors']}  Sentences: {summary['sentences']}  Rows: {summary['rows']}")
    for emotion, values in summary["per_emotion"].items():
        print(
            f"  {emotion}: rows={values['rows']} with_real_clip={values['with_real_clip']} "
            f"real_seconds={values['real_seconds']:.1f}"
        )
    if summary["skipped_missing_neutral"]:
        print(f"Skipped missing neutral: {len(summary['skipped_missing_neutral'])}")
    if summary["duplicate_neutral"]:
        print(f"Duplicate neutral groups: {len(summary['duplicate_neutral'])}")
    if summary["duplicate_emotion"]:
        print(f"Duplicate emotion groups: {len(summary['duplicate_emotion'])}")
    for error in errors:
        print(f"ERROR: {error}")
    print(f"Wrote {args.output}")
    print(f"Wrote {args.summary}")
    if errors and not args.allow_invalid:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
