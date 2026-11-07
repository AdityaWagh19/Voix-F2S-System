"""Download audio only for the selected GLOBE manifest rows.

The manifest carries ``source_file``, ``row_group``, and ``row_in_group``.  For
each source file this script opens the remote Parquet once, reads only the
selected row groups' audio chunks, decodes the 24 kHz FLAC payload, and writes
a 24 kHz mono WAV under ``data/processed/globe_audio``.  It never downloads the
full 165.5 GB dataset.

The download is resumable: an existing WAV with a readable duration is skipped.
Each worker owns one source file at a time, so a file's footer is fetched once.

Usage:
    python scripts/data/download_globe_selected.py
    python scripts/data/download_globe_selected.py --workers 8 --limit 2
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import soundfile as sf

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from voix.data.parquet_remote import open_dataset_parquet

DEFAULT_REPO = "MushanW/GLOBE_V3"
DEFAULT_MANIFEST = ROOT / "data" / "processed" / "globe_accent_manifest.csv"
DEFAULT_AUDIO_DIR = ROOT / "data" / "processed" / "globe_audio"
DEFAULT_REPORT = ROOT / "data" / "processed" / "globe_accent_download_report.json"


def target_path(audio_dir: Path, row: pd.Series) -> Path:
    return (
        audio_dir
        / str(row["speaker_split"])
        / str(row["accent_class"])
        / f"{row['clip_id']}.wav"
    )


def decode_and_save(payload: bytes, destination: Path) -> Dict[str, object]:
    audio, sample_rate = sf.read(io.BytesIO(payload), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sample_rate != 24000:
        raise ValueError(f"expected 24 kHz, got {sample_rate}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp.wav")
    sf.write(temporary, audio, sample_rate, subtype="PCM_16")
    temporary.replace(destination)
    return {
        "sample_rate": int(sample_rate),
        "samples": int(len(audio)),
        "duration_sec": round(float(len(audio)) / sample_rate, 4),
        "audio_sha256": hashlib.sha256(payload).hexdigest(),
    }


def download_file(
    *,
    repo: str,
    source_file: str,
    rows: pd.DataFrame,
    audio_dir: Path,
    token: Optional[str],
    overwrite: bool,
) -> List[Dict[str, object]]:
    results: List[Dict[str, object]] = []
    remote = open_dataset_parquet(repo, source_file, token=token)
    try:
        for row_group, group in rows.groupby("row_group", sort=True):
            table = remote.read_audio_row_group(int(row_group))
            audio_column = table.column("audio").to_pylist()
            for _, row in group.iterrows():
                index = int(row["row_in_group"])
                destination = target_path(audio_dir, row)
                if destination.is_file() and not overwrite:
                    try:
                        info = sf.info(destination)
                        results.append(
                            {
                                "clip_id": row["clip_id"],
                                "status": "skip",
                                "path": str(destination),
                                "duration_sec": round(info.duration, 4),
                                "sample_rate": info.samplerate,
                            }
                        )
                        continue
                    except Exception:
                        pass
                if index >= len(audio_column):
                    results.append(
                        {
                            "clip_id": row["clip_id"],
                            "status": "error",
                            "error": f"row {index} outside row group of {len(audio_column)}",
                        }
                    )
                    continue
                item = audio_column[index]
                payload = item["bytes"] if isinstance(item, dict) else item
                if not payload:
                    results.append(
                        {
                            "clip_id": row["clip_id"],
                            "status": "error",
                            "error": "empty audio payload",
                        }
                    )
                    continue
                try:
                    details = decode_and_save(payload, destination)
                except Exception as exc:
                    results.append(
                        {
                            "clip_id": row["clip_id"],
                            "status": "error",
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
                    continue
                results.append(
                    {
                        "clip_id": row["clip_id"],
                        "status": "saved",
                        "path": str(destination),
                        **details,
                    }
                )
    finally:
        remote.close()
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--audio-dir", type=Path, default=DEFAULT_AUDIO_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--token", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.manifest.is_file():
        raise SystemExit(f"missing manifest: {args.manifest}")
    manifest = pd.read_csv(args.manifest)
    if manifest.empty:
        raise SystemExit("manifest is empty")
    required = {"clip_id", "source_file", "row_group", "row_in_group", "accent_class", "speaker_split"}
    missing = required - set(manifest.columns)
    if missing:
        raise SystemExit(f"manifest missing columns: {sorted(missing)}")

    files = sorted(manifest["source_file"].unique().tolist())
    if args.limit is not None:
        files = files[: args.limit]
        manifest = manifest[manifest["source_file"].isin(files)]
    print(
        f"[plan] {len(manifest)} clips across {len(files)} source files; "
        f"{manifest['source_file'].nunique()} files, workers={args.workers}",
        flush=True,
    )

    started = time.time()
    results: List[Dict[str, object]] = []
    failures: List[Dict[str, object]] = []
    lock = threading.Lock()
    completed = 0

    def report(batch: List[Dict[str, object]]) -> None:
        nonlocal completed
        with lock:
            results.extend(batch)
            completed += len(batch)
            errors = sum(1 for item in batch if item["status"] == "error")
            print(
                f"[{completed}/{len(manifest)}] {batch[0].get('clip_id') if batch else ''} "
                f"batch={len(batch)} errors={errors} elapsed={time.time()-started:.0f}s",
                flush=True,
            )
            failures.extend(item for item in batch if item["status"] == "error")

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {
            pool.submit(
                download_file,
                repo=args.repo,
                source_file=source_file,
                rows=manifest[manifest["source_file"] == source_file],
                audio_dir=args.audio_dir,
                token=args.token,
                overwrite=args.overwrite,
            ): source_file
            for source_file in files
        }
        for future in as_completed(futures):
            source_file = futures[future]
            try:
                batch = future.result()
            except Exception as exc:
                rows = manifest[manifest["source_file"] == source_file]
                batch = [
                    {
                        "clip_id": row["clip_id"],
                        "status": "error",
                        "source_file": source_file,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                    for _, row in rows.iterrows()
                ]
            report(batch)

    saved = [item for item in results if item["status"] == "saved"]
    skipped = [item for item in results if item["status"] == "skip"]
    total_seconds = sum(float(item.get("duration_sec", 0) or 0) for item in results)
    summary = {
        "manifest": str(args.manifest),
        "audio_dir": str(args.audio_dir),
        "clips": int(len(manifest)),
        "saved": len(saved),
        "skipped": len(skipped),
        "failed": len(failures),
        "hours": round(total_seconds / 3600.0, 3),
        "source_files": len(files),
        "seconds": round(time.time() - started, 2),
        "failures": failures[:50],
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Wrote {args.report}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
