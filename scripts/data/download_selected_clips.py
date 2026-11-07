"""Download only the manifest-selected clips from the VoxCeleb2 ZIPs.

The dev shards are ~50 GB each, while a balanced manifest needs only a few
hundred kilobytes per clip.  This script reads each shard's ZIP central
directory once, then fetches each selected member with a single HTTP range
request.  It never stores the full archive.

The downloader is resumable: a clip is skipped when the destination already
exists with the expected uncompressed size.  Each worker owns a requests
session, and the central-directory metadata is shared read-only.

Usage:
    python scripts/data/download_selected_clips.py --index index.csv --output-dir videos
    python scripts/data/download_selected_clips.py --index pilot.csv --workers 8
"""
from __future__ import annotations

import argparse
import json
import os
import struct
import sys
import threading
import time
import zipfile
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:  # huggingface_hub >= 0.22
    from huggingface_hub import get_token
except ImportError:  # pragma: no cover - exercised on older installs
    try:
        from huggingface_hub import HfFolder

        def get_token() -> Optional[str]:
            return HfFolder.get_token()
    except ImportError:
        def get_token() -> Optional[str]:
            return None

from voix.data.remote_zip import RemoteZip, hf_dataset_url
from voix.data.voxceleb2_index import load_manifest


REPO_ID = "ProgramComputer/voxceleb"
SHARD_ZIPS = {
    1: "vox2/vox2_mp4_1.zip",
    2: "vox2/vox2_mp4_2.zip",
    3: "vox2/vox2_mp4_3.zip",
    4: "vox2/vox2_mp4_4.zip",
    5: "vox2/vox2_mp4_5.zip",
    6: "vox2/vox2_mp4_6.zip",
    7: "vox2/vox2_test_mp4.zip",
}
# Over-read the local header by this much so filename + extra fields fit.
LOCAL_HEADER_SLACK = 4096
RETRY_STATUS = {408, 425, 429, 500, 502, 503, 504}


class DownloadError(RuntimeError):
    pass


_thread_local = threading.local()


def _session() -> requests.Session:
    session = getattr(_thread_local, "session", None)
    if session is None:
        session = requests.Session()
        _thread_local.session = session
    return session


def _member_name(row: pd.Series) -> str:
    video_path = str(row["video_path"]).replace("\\", "/").lstrip("/")
    if str(row.get("set", "dev")) == "test":
        return f"mp4/{video_path}"
    return f"dev/mp4/{video_path}"


def _fetch_member(
    url: str,
    token: str,
    info: zipfile.ZipInfo,
    destination: Path,
    *,
    timeout_sec: float = 120.0,
    max_retries: int = 5,
) -> Path:
    """Fetch one ZIP member with a single range request."""
    if destination.exists() and destination.stat().st_size == info.file_size:
        return destination

    start = int(info.header_offset)
    end = start + 30 + LOCAL_HEADER_SLACK + int(info.compress_size) - 1
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept-Encoding": "identity",
        "Range": f"bytes={start}-{end}",
    }

    last_error: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            response = _session().get(url, headers=headers, timeout=timeout_sec)
            if response.status_code == 206:
                data = response.content
                break
            if response.status_code == 416:
                # Already complete or the requested range is past EOF.
                raise DownloadError(f"range not satisfiable for {info.filename}")
            if response.status_code in RETRY_STATUS:
                last_error = DownloadError(f"HTTP {response.status_code}")
                time.sleep(min(10.0, 0.5 * (2**attempt)))
                continue
            raise DownloadError(
                f"HTTP {response.status_code} for {info.filename}: "
                f"{response.text[:200]}"
            )
        except (requests.RequestException, DownloadError) as exc:
            last_error = exc
            time.sleep(min(10.0, 0.5 * (2**attempt)))
    else:
        raise DownloadError(f"failed after {max_retries} attempts: {last_error}")

    if data[:4] != b"PK\x03\x04":
        raise DownloadError(f"bad local header for {info.filename}")
    name_len, extra_len = struct.unpack("<HH", data[26:30])
    payload_start = 30 + name_len + extra_len
    payload = data[payload_start : payload_start + int(info.compress_size)]
    if len(payload) != int(info.compress_size):
        raise DownloadError(
            f"short payload for {info.filename}: "
            f"{len(payload)} != {info.compress_size}"
        )
    if info.compress_type == zipfile.ZIP_DEFLATED:
        payload = zlib.decompress(payload, -15)
    elif info.compress_type != zipfile.ZIP_STORED:
        raise DownloadError(
            f"unsupported compression {info.compress_type} for {info.filename}"
        )
    if len(payload) != int(info.file_size):
        raise DownloadError(
            f"uncompressed size mismatch for {info.filename}: "
            f"{len(payload)} != {info.file_size}"
        )
    if int(info.CRC) and (zlib.crc32(payload) & 0xFFFFFFFF) != int(info.CRC):
        raise DownloadError(f"CRC mismatch for {info.filename}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_suffix(destination.suffix + ".part")
    with open(tmp, "wb") as handle:
        handle.write(payload)
    os.replace(tmp, destination)
    return destination


def download_shard(
    shard: int,
    rows: pd.DataFrame,
    *,
    output_dir: Path,
    token: str,
    workers: int,
    retries: int,
) -> tuple[int, int]:
    filename = SHARD_ZIPS[shard]
    url = hf_dataset_url(REPO_ID, filename)
    print(f"\nShard {shard}: {filename} | {len(rows):,} selected clips")
    print(f"  reading ZIP central directory: {url}")
    started = time.time()
    with RemoteZip(url, token=token) as archive:
        infos: dict[str, zipfile.ZipInfo] = {}
        missing: list[str] = []
        for _, row in rows.iterrows():
            member = _member_name(row)
            try:
                infos[member] = archive._zip.getinfo(member)
            except KeyError:
                missing.append(member)
        print(
            f"  {len(infos):,} members found | {len(missing):,} missing | "
            f"{time.time() - started:.1f}s"
        )
        if missing:
            for member in missing[:5]:
                print(f"    missing: {member}")
            if len(missing) > 5:
                print(f"    ... and {len(missing) - 5:,} more")

        downloaded = 0
        skipped = 0
        failed = 0
        failures: list[dict] = []

        def _run(row: pd.Series) -> None:
            nonlocal downloaded, skipped
            member = _member_name(row)
            info = infos.get(member)
            if info is None:
                raise DownloadError("member not found in ZIP directory")
            destination = output_dir / str(row["video_path"]).replace("\\", "/")
            if destination.exists() and destination.stat().st_size == info.file_size:
                skipped += 1
                return
            _fetch_member(
                url,
                token,
                info,
                destination,
                max_retries=retries,
            )
            downloaded += 1

        with ThreadPoolExecutor(max_workers=max(1, int(workers))) as pool:
            futures = {pool.submit(_run, row): row for _, row in rows.iterrows()}
            done = 0
            for future in as_completed(futures):
                done += 1
                row = futures[future]
                try:
                    future.result()
                except Exception as exc:
                    failed += 1
                    failures.append(
                        {
                            "clip_id": row.get("clip_id", ""),
                            "reason": str(exc),
                        }
                    )
                    if failed <= 10:
                        print(f"  [FAIL] {row.get('clip_id', '')}: {exc}")
                if done % 100 == 0 or done == len(futures):
                    elapsed = max(1e-6, time.time() - started)
                    rate = done / elapsed
                    print(
                        f"  {done:,}/{len(futures):,} | "
                        f"{rate:.2f} clips/s | "
                        f"downloaded={downloaded:,} skipped={skipped:,} "
                        f"failed={failed:,}",
                        flush=True,
                    )

    if failures:
        failure_path = output_dir / f"download_failures_shard{shard}.json"
        failure_path.parent.mkdir(parents=True, exist_ok=True)
        failure_path.write_text(json.dumps(failures, indent=2), encoding="utf-8")
        print(f"  failures written to {failure_path}")
    return downloaded, failed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download only manifest-selected VoxCeleb2 clips"
    )
    parser.add_argument("--index", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--shards",
        default="all",
        help="Comma-separated shard numbers (1-6 dev, 7 test) or 'all'",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--max-speakers", type=int, default=None)
    parser.add_argument("--max-clips-per-speaker", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = load_manifest(args.index)
    if args.max_speakers is not None:
        speakers = sorted(frame["speaker_id"].unique())[: int(args.max_speakers)]
        frame = frame[frame["speaker_id"].isin(speakers)].copy()
    if args.max_clips_per_speaker is not None:
        frame = (
            frame.groupby("speaker_id", group_keys=False)
            .head(int(args.max_clips_per_speaker))
            .copy()
        )
    if frame.empty:
        raise SystemExit("No clips selected")

    token = get_token()
    if not token:
        raise SystemExit(
            "No Hugging Face token found. Run `hf auth login` or set HF_TOKEN."
        )

    if str(args.shards).strip().lower() == "all":
        shards = sorted(frame["shard"].astype(int).unique())
    else:
        shards = [int(part) for part in str(args.shards).split(",") if part.strip()]
    unknown = [s for s in shards if s not in SHARD_ZIPS]
    if unknown:
        raise SystemExit(f"Unknown shards: {unknown}")

    print(
        f"Manifest: {args.index} | {len(frame):,} clips | "
        f"{frame['speaker_id'].nunique():,} speakers | shards={shards}"
    )
    total_downloaded = 0
    total_failed = 0
    started = time.time()
    for shard in shards:
        rows = frame[frame["shard"].astype(int) == shard]
        if rows.empty:
            continue
        downloaded, failed = download_shard(
            shard,
            rows,
            output_dir=args.output_dir,
            token=token,
            workers=args.workers,
            retries=args.retries,
        )
        total_downloaded += downloaded
        total_failed += failed

    elapsed = time.time() - started
    print("\n" + "=" * 72)
    print(
        f"Done: downloaded={total_downloaded:,} failed={total_failed:,} "
        f"elapsed={elapsed / 60:.1f} min"
    )
    print(f"Output: {args.output_dir}")
    print("=" * 72)


if __name__ == "__main__":
    main()
