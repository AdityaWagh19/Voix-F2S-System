"""Select up to three training clips per speaker for the A3 style cache.

The A3 residual averages each speaker's style into a class contrast, so a
stable per-speaker mean needs more than one clip per speaker.  Selecting up to
three clips cuts the style extraction from 7,682 clips to roughly 1,700 while
covering every training speaker.  This is a compute-bounded subset, not a new
split: every selected row is from ``speaker_split=train`` and the val/test
speakers remain untouched.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_MANIFEST = ROOT / "data" / "processed" / "globe_accent_manifest.csv"
DEFAULT_OUTPUT = ROOT / "data" / "processed" / "globe_accent_style_subset.csv"
DEFAULT_REPORT = ROOT / "results" / "accent" / "a3" / "style_subset_report.json"
TRAINABLE = ("us", "england", "australian", "canadian", "india_south_asia")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select A3 style-cache subset")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--classes", nargs="+", default=list(TRAINABLE))
    parser.add_argument(
        "--clips-per-speaker",
        type=int,
        default=3,
        help="Maximum clips per speaker; at least two are required for centering.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if int(args.clips_per_speaker) < 2:
        raise ValueError("--clips-per-speaker must be at least 2 for centering")
    frame = pd.read_csv(args.manifest)
    required = {"clip_id", "client_id", "accent_class", "speaker_split"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"manifest is missing columns: {missing}")
    work = frame[
        (frame["speaker_split"].astype(str) == "train")
        & (frame["accent_class"].astype(str).isin(args.classes))
    ].copy()
    if work.empty:
        raise ValueError("no training rows matched the requested classes")
    work = work.sort_values(["accent_class", "client_id", "clip_id"])
    subset = work.groupby(["accent_class", "client_id"], sort=True).head(
        int(args.clips_per_speaker)
    )
    counts = subset.groupby(["accent_class", "client_id"]).size()
    eligible = set(counts[counts >= 2].index)
    subset = subset[
        subset.set_index(["accent_class", "client_id"]).index.isin(eligible)
    ]
    subset = subset.sort_values(["accent_class", "client_id", "clip_id"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    subset.to_csv(args.output, index=False)
    counts_by_class = {
        str(accent): {
            "clips": int(len(group)),
            "speakers": int(group["client_id"].nunique()),
        }
        for accent, group in subset.groupby("accent_class", sort=True)
    }
    report = {
        "schema_version": "voix.accent_style_subset.v1",
        "source_manifest": str(args.manifest),
        "output": str(args.output),
        "selection": (
            f"first {int(args.clips_per_speaker)} clips (sorted by clip_id) for "
            "every train speaker with at least two selected clips"
        ),
        "split": "train",
        "classes": list(args.classes),
        "rows": int(len(subset)),
        "by_class": counts_by_class,
        "note": (
            "Speaker-balanced compute subset; it is not a new data split and "
            "does not use val or test speakers.  At least two clips per speaker "
            "give the class-contrast baseline a stable speaker mean."
        ),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.output} ({len(subset)} rows)")
    print(f"Wrote {args.report}")
    for accent, entry in counts_by_class.items():
        print(f"  {accent}: {entry['clips']} clips / {entry['speakers']} speakers")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
