"""Index GLOBE_V3 metadata without downloading audio.

The dataset is 165.5 GB of Parquet, but every row group stores its metadata
columns in one small contiguous span before the audio chunk.  This script
reads each file's footer and only that span, then writes one metadata shard per
source file.  A later pass can fetch audio for selected row groups only.

The pass is parallel across source files (one HTTP session per worker) and
resumable: an existing shard is skipped unless ``--overwrite`` is set.

Usage:
    python scripts/data/index_globe_metadata.py --workers 8
    python scripts/data/index_globe_metadata.py --splits train --limit 2 --overwrite
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional

import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from voix.data.parquet_remote import list_parquet_files, open_dataset_parquet

DEFAULT_REPO = "MushanW/GLOBE_V3"
DEFAULT_OUTPUT = ROOT / "data" / "processed" / "globe_metadata"


def split_of(path: str) -> str:
    """Return ``train``/``dev``/``test`` from ``data/train-...parquet``."""
    name = Path(path).name
    return name.split("-", 1)[0]


def shard_path(output_dir: Path, source_path: str) -> Path:
    split = split_of(source_path)
    return output_dir / split / f"{Path(source_path).stem}.parquet"


def index_one(
    *,
    repo: str,
    revision: str,
    entry: Dict[str, object],
    output_dir: Path,
    token: Optional[str],
    overwrite: bool,
) -> Dict[str, object]:
    source_path = str(entry["path"])
    destination = shard_path(output_dir, source_path)
    if destination.is_file() and not overwrite:
        rows = int(pq.read_metadata(destination).num_rows)
        return {
            "status": "skip",
            "source_file": source_path,
            "rows": rows,
            "shard": str(destination),
        }

    started = time.time()
    remote = open_dataset_parquet(
        repo,
        source_path,
        revision=revision,
        token=token,
    )
    try:
        tables: List[pa.Table] = []
        for row_group in range(remote.num_row_groups):
            table = remote.read_metadata_row_group(row_group)
            count = table.num_rows
            table = table.append_column(
                "source_file", pa.array([source_path] * count, pa.string())
            )
            table = table.append_column(
                "split", pa.array([split_of(source_path)] * count, pa.string())
            )
            table = table.append_column(
                "row_group", pa.array([row_group] * count, pa.int32())
            )
            table = table.append_column(
                "row_in_group", pa.array(list(range(count)), pa.int32())
            )
            tables.append(table)
        combined = pa.concat_tables(tables)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(".tmp")
        pq.write_table(combined, temporary, compression="zstd")
        os.replace(temporary, destination)
        return {
            "status": "ok",
            "source_file": source_path,
            "rows": int(combined.num_rows),
            "row_groups": remote.num_row_groups,
            "shard": str(destination),
            "seconds": round(time.time() - started, 2),
            "source_bytes": int(entry.get("size", 0) or 0),
        }
    finally:
        remote.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--revision", default="main")
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "dev", "test"],
        choices=["train", "dev", "test"],
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process at most this many source files (for a smoke test).",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--token", default=os.environ.get("HF_TOKEN"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    entries = list_parquet_files(
        args.repo, revision=args.revision, token=args.token
    )
    entries = [entry for entry in entries if split_of(str(entry["path"])) in args.splits]
    entries.sort(key=lambda entry: str(entry["path"]))
    if args.limit is not None:
        entries = entries[: args.limit]
    if not entries:
        raise SystemExit("no Parquet files matched the requested splits")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    total = len(entries)
    done = 0
    rows_total = 0
    counts = {"ok": 0, "skip": 0, "error": 0}
    failures: List[Dict[str, object]] = []
    lock = threading.Lock()
    started = time.time()

    def report(result: Dict[str, object]) -> None:
        nonlocal done, rows_total
        with lock:
            done += 1
            rows_total += int(result.get("rows", 0) or 0)
            counts[result["status"]] = counts.get(result["status"], 0) + 1
            elapsed = time.time() - started
            print(
                f"[{done}/{total}] {result['status']:4s} "
                f"{result['source_file']} rows={result.get('rows', 0)} "
                f"({elapsed:.0f}s)",
                flush=True,
            )
            if result["status"] == "error":
                failures.append(result)

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {
            pool.submit(
                index_one,
                repo=args.repo,
                revision=args.revision,
                entry=entry,
                output_dir=args.output_dir,
                token=args.token,
                overwrite=args.overwrite,
            ): entry
            for entry in entries
        }
        for future in as_completed(futures):
            entry = futures[future]
            try:
                result = future.result()
            except Exception as exc:  # keep the rest of the pass alive
                result = {
                    "status": "error",
                    "source_file": str(entry["path"]),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            report(result)

    summary = {
        "repo": args.repo,
        "revision": args.revision,
        "splits": args.splits,
        "files": total,
        "ok": counts.get("ok", 0),
        "skipped": counts.get("skip", 0),
        "failed": failures,
        "rows": rows_total,
        "seconds": round(time.time() - started, 2),
    }
    summary_path = args.output_dir / "index_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
