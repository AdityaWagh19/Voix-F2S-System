"""Extract real reference speech for the mel-spectrogram FAD proxy.

The FAD proxy in ``voix.evaluation.audio_metrics`` compares the distribution of
generated audio with real speech.  Using generated audio as its own reference
would make the number meaningless, so this script extracts a deterministic
subset of held-out VoxCeleb2 clips (default: the official test split) to WAV.

It never writes into the feature HDF5 and is safe to run while extraction is
working on a different shard.

Usage:
    python scripts/data/extract_reference_audio.py \
        --output-dir data/processed/voxceleb2_v2/reference_wavs \
        --split test --limit 120
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from voix.data.video_io import extract_audio  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract reference speech WAVs")
    parser.add_argument(
        "--index",
        type=Path,
        default=ROOT / "data" / "processed" / "voxceleb2_v2" / "index.csv",
    )
    parser.add_argument(
        "--videos",
        type=Path,
        default=ROOT / "data" / "raw" / "voxceleb2" / "clips",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "data" / "processed" / "voxceleb2_v2" / "reference_wavs",
    )
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=120)
    parser.add_argument("--max-seconds", type=float, default=8.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.index, newline="", encoding="utf-8") as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if str(row.get("split")) == args.split
        ]
    if not rows:
        raise ValueError(f"no rows for split {args.split!r} in {args.index}")

    # Deterministic: sort by clip_id before taking the limit.
    rows = sorted(rows, key=lambda row: str(row["clip_id"]))[: args.limit]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    extracted = []
    failures = []
    for row in rows:
        video = args.videos / str(row["video_path"])
        if not video.exists():
            failures.append({"clip_id": row["clip_id"], "reason": "missing_video"})
            continue
        wav = args.output_dir / (str(row["clip_id"]).replace("/", "_") + ".wav")
        if not wav.exists():
            ok = extract_audio(video, str(wav), max_seconds=args.max_seconds)
            if not ok:
                failures.append(
                    {"clip_id": row["clip_id"], "reason": "audio_extract_failed"}
                )
                continue
        extracted.append(
            {
                "clip_id": row["clip_id"],
                "speaker_id": row["speaker_id"],
                "wav": str(wav.relative_to(ROOT)),
            }
        )

    summary = {
        "split": args.split,
        "requested": len(rows),
        "extracted": len(extracted),
        "failures": failures,
        "output_dir": str(args.output_dir.relative_to(ROOT)),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    if not extracted:
        raise SystemExit("no reference audio was extracted")


if __name__ == "__main__":
    main()
