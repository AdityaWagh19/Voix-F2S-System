"""Build a StyleTTS2 LoRA manifest from the AMI naturalness corpus.

AMI already provides human transcripts and speaker-disjoint splits, so this
script only adds the reference clip columns required by
``scripts/training/train_styletts2_lora.py``.  The reference is a different clip from the
same speaker, used to extract the frozen style/prosody conditioning vector.

The output is a research manifest.  It does not train anything and does not
touch the canonical face-to-voice path.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_MANIFEST = ROOT / "data" / "processed" / "ami_naturalness" / "manifest.csv"
DEFAULT_OUTPUT = ROOT / "data" / "processed" / "ami_lora" / "manifest.csv"
DEFAULT_REPORT = ROOT / "data" / "processed" / "ami_lora" / "manifest_report.json"
OUTPUT_COLUMNS = (
    "clip_id",
    "speaker_id",
    "split",
    "wav_path",
    "ref_clip_id",
    "ref_wav_path",
    "duration_sec",
    "text",
    "meeting_id",
    "gender",
    "source",
    "license",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _pick_reference(group: pd.DataFrame, clip_id: str) -> pd.Series:
    """Pick a deterministic same-speaker reference, preferring a different clip."""
    ordered = group.sort_values(["duration_sec", "clip_id"], ascending=[False, True])
    different = ordered[ordered["clip_id"].astype(str) != str(clip_id)]
    if not different.empty:
        return different.iloc[0]
    return ordered.iloc[0]


def build(
    frame: pd.DataFrame,
    *,
    min_duration: float,
    max_duration: float,
) -> pd.DataFrame:
    required = {
        "clip_id",
        "speaker_id",
        "split",
        "wav_path",
        "text",
        "duration_sec",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"AMI manifest is missing columns: {missing}")
    frame = frame[frame["split"].astype(str).isin(["train", "val"])].copy()
    frame = frame[
        frame["duration_sec"].astype(float).between(min_duration, max_duration)
    ].copy()
    frame = frame[frame["text"].astype(str).str.strip().ne("")]
    if frame.empty:
        raise ValueError("no usable train/val AMI rows")

    rows: List[Dict[str, Any]] = []
    for speaker, group in frame.groupby("speaker_id", sort=True):
        for _, row in group.sort_values("clip_id").iterrows():
            reference = _pick_reference(group, str(row["clip_id"]))
            rows.append(
                {
                    "clip_id": str(row["clip_id"]),
                    "speaker_id": str(speaker),
                    "split": str(row["split"]),
                    "wav_path": str(row["wav_path"]),
                    "ref_clip_id": str(reference["clip_id"]),
                    "ref_wav_path": str(reference["wav_path"]),
                    "duration_sec": float(row["duration_sec"]),
                    "text": str(row["text"]).strip(),
                    "meeting_id": str(row.get("meeting_id", "")),
                    "gender": "unknown",
                    "source": str(row.get("source", "AMI corpus headset microphone")),
                    "license": str(row.get("license", "CC BY 4.0")),
                }
            )
    output = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    train_speakers = set(
        output.loc[output["split"] == "train", "speaker_id"].astype(str)
    )
    val_speakers = set(output.loc[output["split"] == "val", "speaker_id"].astype(str))
    overlap = sorted(train_speakers & val_speakers)
    if overlap:
        raise ValueError(f"train/val speaker overlap: {overlap[:5]}")
    missing_files = sorted(
        {
            value
            for column in ("wav_path", "ref_wav_path")
            for value in output[column]
            if not Path(value).is_file()
        }
    )
    if missing_files:
        raise FileNotFoundError(f"missing AMI audio: {missing_files[:5]}")
    return output


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--min-duration", type=float, default=1.5)
    parser.add_argument("--max-duration", type=float, default=12.0)
    args = parser.parse_args(argv)

    frame = pd.read_csv(args.manifest)
    output = build(
        frame,
        min_duration=float(args.min_duration),
        max_duration=float(args.max_duration),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    report = {
        "schema_version": "voix.ami_lora_manifest.v1",
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "level": "exploratory",
        "track": "naturalness",
        "source_manifest": str(args.manifest),
        "source_manifest_sha256": _sha256(args.manifest),
        "output": str(args.output),
        "output_sha256": _sha256(args.output),
        "rows": int(len(output)),
        "by_split": {
            split: {
                "clips": int((output["split"] == split).sum()),
                "speakers": int(
                    output.loc[output["split"] == split, "speaker_id"].nunique()
                ),
                "hours": round(
                    float(
                        output.loc[output["split"] == split, "duration_sec"].sum()
                    )
                    / 3600.0,
                    3,
                ),
            }
            for split in ("train", "val")
        },
        "reference_policy": (
            "Longest same-speaker clip, preferring a different clip id; "
            "self-reference only when the speaker has one usable clip."
        ),
        "notes": [
            "AMI transcripts are human annotations, not Whisper output.",
            "The reference clip supplies frozen style/prosody conditioning only.",
            "No face, CFVN, CVAE, or canonical style adapter is used.",
        ],
        "limitations": [
            "AMI headset audio is 16 kHz and noisy.",
            "The pilot has 8 train and 4 validation speakers, below the 10-speaker N1 target.",
        ],
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {args.output} ({len(output)} rows)")
    print(f"Wrote {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
