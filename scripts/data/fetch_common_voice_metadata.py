"""Fetch Common Voice 21 English metadata for the GLOBE speaker join.

GLOBE_V3 has no ``speaker_id``.  It does carry ``common_voice_path``, which is
the Common Voice clip filename, so the speaker key is the Common Voice
``client_id``.  The official Common Voice 21 HF repo is gated; this script uses
the public mirror ``fsicoli/common_voice_21_0`` and records the provenance in
the output summary.  Only the TSV metadata is downloaded, never the audio
tarballs.

Outputs (under ``data/processed/``):
  * ``common_voice_21_en_metadata.parquet`` -- path, client_id, accent, age,
    gender, sentence, votes, duration.
  * ``common_voice_21_en_metadata_summary.json`` -- row/speaker/duration counts.

Usage:
    python scripts/data/fetch_common_voice_metadata.py
    python scripts/data/fetch_common_voice_metadata.py --force
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_REPO = "fsicoli/common_voice_21_0"
DEFAULT_LOCALE = "en"
DEFAULT_RAW = ROOT / "data" / "raw" / "common_voice_21"
DEFAULT_OUTPUT = ROOT / "data" / "processed" / "common_voice_21_en_metadata.parquet"

VALIDATED_COLUMNS = [
    "client_id",
    "path",
    "sentence_id",
    "sentence",
    "up_votes",
    "down_votes",
    "age",
    "gender",
    "accents",
    "variant",
    "locale",
    "segment",
]


def download(repo: str, filename: str, raw_dir: Path, token: Optional[str]) -> Path:
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        repo_id=repo,
        filename=filename,
        repo_type="dataset",
        local_dir=str(raw_dir),
        local_dir_use_symlinks=False,
        resume_download=True,
        token=token,
    )
    return Path(path)


def load_durations(path: Path) -> Dict[str, float]:
    frame = pd.read_csv(
        path,
        sep="\t",
        dtype={"clip": str, "duration[ms]": "float64"},
        keep_default_na=False,
        on_bad_lines="warn",
    )
    return {
        str(clip): float(duration_ms) / 1000.0
        for clip, duration_ms in zip(frame["clip"], frame["duration[ms]"])
    }


def parse_validated(
    source: Path,
    destination: Path,
    durations: Dict[str, float],
    *,
    chunksize: int,
) -> int:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp.parquet")
    writer: Optional[pq.ParquetWriter] = None
    rows = 0
    try:
        reader = pd.read_csv(
            source,
            sep="\t",
            dtype=str,
            usecols=VALIDATED_COLUMNS,
            keep_default_na=False,
            on_bad_lines="warn",
            chunksize=chunksize,
        )
        for chunk in reader:
            chunk = chunk.rename(
                columns={
                    "path": "common_voice_path",
                    "accents": "cv_accents",
                    "age": "cv_age",
                    "gender": "cv_gender",
                }
            )
            for column in chunk.columns:
                chunk[column] = chunk[column].astype(str).str.strip()
            chunk["duration_sec"] = (
                chunk["common_voice_path"].map(durations).astype("float64")
            )
            table = pa.Table.from_pandas(chunk, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(temporary, table.schema, compression="zstd")
            writer.write_table(table)
            rows += len(chunk)
    finally:
        if writer is not None:
            writer.close()
    if writer is None:
        raise SystemExit(f"no rows parsed from {source}")
    temporary.replace(destination)
    return rows


def summarize(path: Path, *, source_repo: str) -> Dict[str, object]:
    frame = pq.read_table(
        path,
        columns=[
            "common_voice_path",
            "client_id",
            "cv_accents",
            "cv_age",
            "cv_gender",
            "duration_sec",
        ],
    ).to_pandas()
    accents = frame["cv_accents"].fillna("").astype(str)
    frame["accent_label"] = accents.where(accents.str.len() > 0, "(unspecified)")
    by_accent = (
        frame.groupby("accent_label")
        .agg(
            clips=("common_voice_path", "nunique"),
            speakers=("client_id", "nunique"),
            hours=("duration_sec", lambda values: float(values.sum()) / 3600.0),
        )
        .sort_values("clips", ascending=False)
    )
    return {
        "source_repo": source_repo,
        "rows": int(len(frame)),
        "unique_paths": int(frame["common_voice_path"].nunique()),
        "unique_speakers": int(frame["client_id"].nunique()),
        "hours": round(float(frame["duration_sec"].sum()) / 3600.0, 3),
        "missing_duration": int(frame["duration_sec"].isna().sum()),
        "unspecified_accent_clips": int((frame["accent_label"] == "(unspecified)").sum()),
        "accents": {
            str(index): {
                "clips": int(row["clips"]),
                "speakers": int(row["speakers"]),
                "hours": round(float(row["hours"]), 3),
            }
            for index, row in by_accent.iterrows()
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--locale", default=DEFAULT_LOCALE)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--chunksize", type=int, default=200_000)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--token", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.is_file() and not args.force:
        summary_path = args.output.with_name(args.output.stem + "_summary.json")
        if summary_path.is_file():
            print(summary_path.read_text(encoding="utf-8"))
            print(f"[skip] {args.output} exists; use --force to rebuild")
            return

    transcript = f"transcript/{args.locale}/validated.tsv"
    durations_file = f"transcript/{args.locale}/clip_durations.tsv"
    print(f"[download] {args.repo}/{transcript}")
    validated_path = download(args.repo, transcript, args.raw_dir, args.token)
    print(f"[download] {args.repo}/{durations_file}")
    durations_path = download(args.repo, durations_file, args.raw_dir, args.token)

    print(f"[parse] {durations_path}")
    durations = load_durations(durations_path)
    print(f"[parse] {validated_path} ({len(durations)} durations)")
    rows = parse_validated(
        validated_path,
        args.output,
        durations,
        chunksize=args.chunksize,
    )
    summary = summarize(args.output, source_repo=args.repo)
    summary["validated_rows_parsed"] = rows
    summary["validated_tsv"] = str(validated_path)
    summary["clip_durations_tsv"] = str(durations_path)
    summary_path = args.output.with_name(args.output.stem + "_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
