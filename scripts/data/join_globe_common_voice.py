"""Join GLOBE_V3 metadata to Common Voice speaker metadata.

GLOBE has no speaker column.  Its ``common_voice_path`` is the Common Voice
clip filename, so joining on that key supplies ``client_id`` (the speaker key),
the self-reported accent, and the clip duration.  The join is left-sided: rows
without a Common Voice match stay in the output with null speaker fields so the
match rate is auditable rather than silently dropped.

Outputs:
  * ``data/processed/globe_with_speakers.parquet``
  * ``data/processed/globe_with_speakers_summary.json``

Usage:
    python scripts/data/join_globe_common_voice.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd
import pyarrow.dataset as ds
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_GLOBE = ROOT / "data" / "processed" / "globe_metadata"
DEFAULT_CV = ROOT / "data" / "processed" / "common_voice_21_en_metadata.parquet"
DEFAULT_OUTPUT = ROOT / "data" / "processed" / "globe_with_speakers.parquet"

GLOBE_COLUMNS = [
    "snr",
    "utmos",
    "wer",
    "cer",
    "predicted_gender",
    "predicted_accent",
    "predicted_age",
    "common_voice_path",
    "common_voice_sentence_id",
    "common_voice_sentence",
    "common_voice_age",
    "common_voice_gender",
    "common_voice_accents",
    "source_file",
    "split",
    "row_group",
    "row_in_group",
]

CV_COLUMNS = [
    "common_voice_path",
    "client_id",
    "cv_accents",
    "cv_age",
    "cv_gender",
    "sentence",
    "duration_sec",
]


def _empty_to_null(frame: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    for column in columns:
        if column in frame.columns:
            frame[column] = frame[column].replace({"": None, "nan": None})
    return frame


def summarize(frame: pd.DataFrame) -> Dict[str, object]:
    matched = frame["client_id"].notna()
    self_reported = frame["accent_self_reported"].fillna("").astype(str)
    predicted = frame["predicted_accent"].fillna("").astype(str)

    def counts(series: pd.Series) -> Dict[str, int]:
        return {
            str(key): int(value)
            for key, value in series.value_counts().head(40).items()
        }

    return {
        "rows": int(len(frame)),
        "matched_rows": int(matched.sum()),
        "match_rate": round(float(matched.mean()), 4),
        "unique_paths": int(frame["common_voice_path"].nunique()),
        "unique_speakers": int(frame.loc[matched, "client_id"].nunique()),
        "matched_hours": round(
            float(frame.loc[matched, "duration_sec"].sum()) / 3600.0, 3
        ),
        "missing_duration": int(frame["duration_sec"].isna().sum()),
        "self_reported_accent_present": int((self_reported.str.len() > 0).sum()),
        "predicted_accent_present": int((predicted.str.len() > 0).sum()),
        "self_reported_accents": counts(
            self_reported.where(self_reported.str.len() > 0, "(unspecified)")
        ),
        "predicted_accents": counts(
            predicted.where(predicted.str.len() > 0, "(unspecified)")
        ),
        "by_split": {
            str(key): int(value)
            for key, value in frame["split"].value_counts().items()
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--globe-dir", type=Path, default=DEFAULT_GLOBE)
    parser.add_argument("--common-voice", type=Path, default=DEFAULT_CV)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.is_file() and not args.overwrite:
        print(f"[skip] {args.output} exists; use --overwrite to rebuild")
        return
    if not args.common_voice.is_file():
        raise SystemExit(f"missing Common Voice metadata: {args.common_voice}")

    parquet_files = sorted(
        str(path)
        for path in args.globe_dir.rglob("*.parquet")
        if ".tmp" not in path.name
    )
    if not parquet_files:
        raise SystemExit(f"no Parquet shards under {args.globe_dir}")
    dataset = ds.dataset(parquet_files, format="parquet")
    print(f"[read] GLOBE metadata: {dataset.count_rows()} rows in {len(parquet_files)} shards")
    globe = dataset.to_table(columns=GLOBE_COLUMNS).to_pandas()
    globe = _empty_to_null(globe, ["common_voice_accents", "predicted_accent"])

    print(f"[read] Common Voice metadata: {args.common_voice}")
    cv = pq.read_table(args.common_voice, columns=CV_COLUMNS).to_pandas()
    cv = _empty_to_null(cv, ["cv_accents", "cv_age", "cv_gender"])
    cv = cv.drop_duplicates(subset=["common_voice_path"], keep="first")

    merged = globe.merge(cv, on="common_voice_path", how="left", validate="m:1")
    merged["accent_self_reported"] = merged["cv_accents"].where(
        merged["cv_accents"].notna(), merged["common_voice_accents"]
    )
    merged["accent_source"] = "none"
    merged.loc[merged["common_voice_accents"].notna(), "accent_source"] = "globe"
    merged.loc[merged["cv_accents"].notna(), "accent_source"] = "common_voice"
    mismatch = (
        merged["cv_accents"].notna()
        & merged["common_voice_accents"].notna()
        & (merged["cv_accents"] != merged["common_voice_accents"])
    )
    merged["accent_mismatch"] = mismatch

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".tmp.parquet")
    merged.to_parquet(temporary, index=False, compression="zstd")
    temporary.replace(args.output)

    summary = summarize(merged)
    summary["accent_source_counts"] = {
        str(key): int(value) for key, value in merged["accent_source"].value_counts().items()
    }
    summary["accent_mismatches"] = int(merged["accent_mismatch"].sum())
    summary_path = args.output.with_name(args.output.stem + "_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
