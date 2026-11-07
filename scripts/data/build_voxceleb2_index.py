"""Build the deterministic VoxCeleb2 FaceFeatureV2 manifest.

This script reads only the ZIP central directories over HTTP range requests,
so it can run while the shard downloads are still in progress and without
extracting 267 GB of video.

The resulting manifest is a balanced, speaker-disjoint clip subset:

  * every dev speaker is kept,
  * each speaker contributes at most ``--max-clips-per-speaker`` clips,
  * clips are sampled across distinct videos,
  * the larger gender contributes fewer clips per speaker so the two genders
    contribute roughly equal clip counts,
  * dev speakers are split into train/val by a deterministic gender-stratified
    stride, and official test speakers are held out entirely.

Usage:
    python scripts/data/build_voxceleb2_index.py
    python scripts/data/build_voxceleb2_index.py --refresh-cache
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from huggingface_hub import get_token

from voix.data.remote_zip import RemoteZip, hf_dataset_url
from voix.data.voxceleb2_index import (
    ClipRecord,
    build_manifest,
    load_meta_genders,
    parse_zip_member,
    save_manifest,
)


REPO_ID = "ProgramComputer/voxceleb"
DEV_ZIPS = [f"vox2/vox2_mp4_{i}.zip" for i in range(1, 7)]
TEST_ZIP = "vox2/vox2_test_mp4.zip"
DEFAULT_META = ROOT / "data" / "raw" / "voxceleb2" / "vox2_meta.csv"
DEFAULT_CACHE = ROOT / "data" / "raw" / "voxceleb2" / "zip_names"
DEFAULT_OUTPUT = ROOT / "data" / "processed" / "voxceleb2_v2" / "index.csv"
DEFAULT_SUMMARY = ROOT / "data" / "processed" / "voxceleb2_v2" / "index_summary.json"


def _cache_path(cache_dir: Path, filename: str) -> Path:
    return cache_dir / (Path(filename).stem + ".txt")


def _read_names(
    filename: str,
    *,
    cache_dir: Path,
    refresh: bool,
) -> list[str]:
    cache_path = _cache_path(cache_dir, filename)
    if cache_path.exists() and not refresh:
        print(f"  cache: {cache_path} ({cache_path.stat().st_size / 1e6:.1f} MB)")
        with open(cache_path, "r", encoding="utf-8") as handle:
            return [line.rstrip("\n") for line in handle if line.strip()]

    url = hf_dataset_url(REPO_ID, filename)
    print(f"  reading ZIP directory: {url}")
    started = time.time()
    token = get_token()
    if not token:
        raise RuntimeError(
            "No Hugging Face token found. The VoxCeleb2 repo is gated; "
            "run `hf auth login` or set HF_TOKEN."
        )
    with RemoteZip(url, token=token) as archive:
        names = archive.namelist()
        size_gb = archive.size / 1e9
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w", encoding="utf-8", newline="\n") as handle:
        for name in names:
            handle.write(name + "\n")
    print(
        f"  {len(names):,} members | {size_gb:.1f} GB | "
        f"{time.time() - started:.1f}s"
    )
    return names


def _parse_records(
    names: Iterable[str],
    *,
    set_name: str,
    shard: int,
    source_zip: str,
    genders: dict[str, str],
) -> list[ClipRecord]:
    records: list[ClipRecord] = []
    for name in names:
        record = parse_zip_member(
            name,
            set_name=set_name,
            shard=shard,
            source_zip=source_zip,
            gender=genders.get(_speaker_from_name(name), "u"),
        )
        if record is not None:
            records.append(record)
    return records


def _speaker_from_name(name: str) -> str:
    parts = str(name).replace("\\", "/").strip("/").split("/")
    try:
        index = parts.index("mp4")
    except ValueError:
        return ""
    return parts[index + 1] if len(parts) > index + 1 else ""


def _ensure_meta(meta_path: Path) -> Path:
    if meta_path.exists():
        return meta_path
    from huggingface_hub import hf_hub_download

    meta_path.parent.mkdir(parents=True, exist_ok=True)
    downloaded = hf_hub_download(
        REPO_ID,
        "vox2/vox2_meta.csv",
        repo_type="dataset",
        local_dir=str(meta_path.parent.parent),
    )
    return Path(downloaded)


def build(
    *,
    meta_path: Path,
    cache_dir: Path,
    output_path: Path,
    summary_path: Path,
    max_clips_per_speaker: int,
    val_fraction: float,
    min_clips_per_speaker: int,
    balance_gender: bool,
    refresh_cache: bool,
) -> pd.DataFrame:
    meta_path = _ensure_meta(meta_path)
    genders = load_meta_genders(meta_path)
    print(f"Metadata: {meta_path} ({len(genders):,} speakers)")

    records: list[ClipRecord] = []
    for shard, filename in enumerate(DEV_ZIPS, start=1):
        print(f"Shard {shard}/{len(DEV_ZIPS)}: {filename}")
        names = _read_names(filename, cache_dir=cache_dir, refresh=refresh_cache)
        shard_records = _parse_records(
            names,
            set_name="dev",
            shard=shard,
            source_zip=Path(filename).name,
            genders=genders,
        )
        print(f"  parsed {len(shard_records):,} clips")
        records.extend(shard_records)

    print(f"Test: {TEST_ZIP}")
    test_names = _read_names(TEST_ZIP, cache_dir=cache_dir, refresh=refresh_cache)
    test_records = _parse_records(
        test_names,
        set_name="test",
        shard=7,
        source_zip=Path(TEST_ZIP).name,
        genders=genders,
    )
    print(f"  parsed {len(test_records):,} clips")
    records.extend(test_records)

    print(
        f"Building manifest: {len(records):,} clips, "
        f"{len({r.speaker_id for r in records}):,} speakers"
    )
    frame = build_manifest(
        records,
        val_fraction=val_fraction,
        max_clips_per_speaker=max_clips_per_speaker,
        balance_gender=balance_gender,
        min_clips_per_speaker=min_clips_per_speaker,
    )
    summary = save_manifest(
        frame,
        output_path,
        summary_path,
        val_fraction=val_fraction,
        max_clips_per_speaker=max_clips_per_speaker,
        balance_gender=balance_gender,
    )
    print(json.dumps(summary, indent=2))
    return frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the deterministic VoxCeleb2 FaceFeatureV2 manifest"
    )
    parser.add_argument("--meta", type=Path, default=DEFAULT_META)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--max-clips-per-speaker", type=int, default=8)
    parser.add_argument("--min-clips-per-speaker", type=int, default=4)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--no-gender-balance", action="store_true")
    parser.add_argument("--refresh-cache", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build(
        meta_path=args.meta,
        cache_dir=args.cache_dir,
        output_path=args.output,
        summary_path=args.summary,
        max_clips_per_speaker=args.max_clips_per_speaker,
        val_fraction=args.val_fraction,
        min_clips_per_speaker=args.min_clips_per_speaker,
        balance_gender=not args.no_gender_balance,
        refresh_cache=args.refresh_cache,
    )
