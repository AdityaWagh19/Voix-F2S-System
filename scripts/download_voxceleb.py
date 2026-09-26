"""Download VoxCeleb2 from HuggingFace (Reverb/voxceleb2) to local disk.

Downloads the full dev split mp4 archives using 4 parallel workers,
then extracts all clips to the target directory. Also downloads
vox2_meta.csv and txt/ utterance annotations.

Usage:
    python scripts/download_voxceleb.py
    python scripts/download_voxceleb.py --out D:\voix\data\raw\voxceleb2 --workers 4

Checkpoint-safe: already-downloaded parts are skipped on re-run.

Disk space:
    ~65 GB during extraction (32 GB archives + 32 GB extracted).
    Archives are deleted after extraction -> ~32 GB final footprint.

Time (depends on internet speed):
    100 Mbps: ~45 min    50 Mbps: ~90 min    20 Mbps: ~4 h
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def _hf_download(repo_id: str, filename: str, local_dir: str) -> str:
    from huggingface_hub import hf_hub_download
    return hf_hub_download(
        repo_id=repo_id, filename=filename,
        repo_type="dataset", local_dir=local_dir,
    )


def download_metadata(out_root: Path, hf_repo: str) -> None:
    """Download vox2_meta.csv and txt/ annotation files in parallel."""
    from huggingface_hub import list_repo_files

    # Speaker metadata CSV
    meta_dst = out_root / "vox2_meta.csv"
    if meta_dst.exists():
        print(f"vox2_meta.csv already present ({meta_dst.stat().st_size // 1024} KB) -- skipping")
    else:
        print("Downloading vox2_meta.csv...")
        local = _hf_download(hf_repo, "vox2_meta.csv", str(out_root))
        if Path(local) != meta_dst:
            shutil.copy(local, meta_dst)
        print(f"  -> {meta_dst}")

    # Utterance txt annotations
    txt_dir = out_root / "txt"
    txt_dir.mkdir(parents=True, exist_ok=True)
    existing = len(list(txt_dir.rglob("*.txt")))
    if existing > 5000:
        print(f"txt annotations already present ({existing} files) -- skipping")
        return

    print("Fetching txt annotation file list from HuggingFace...")
    all_files = list(list_repo_files(hf_repo, repo_type="dataset"))
    txt_files = [f for f in all_files if f.startswith("txt/") and f.endswith(".txt")]
    print(f"Downloading {len(txt_files)} txt files with 32 parallel workers...")

    done = 0
    def _dl(fname):
        try:
            _hf_download(hf_repo, fname, str(out_root))
            return True
        except Exception:
            return False

    with ThreadPoolExecutor(max_workers=32) as pool:
        futures = {pool.submit(_dl, f): f for f in txt_files}
        for fut in as_completed(futures):
            done += 1
            if done % 500 == 0:
                print(f"  {done}/{len(txt_files)}")

    total = len(list(txt_dir.rglob("*.txt")))
    print(f"txt annotations done: {total} files in {txt_dir}")


def download_archives(out_root: Path, hf_repo: str, n_workers: int = 4) -> Path:
    """Download all vox2_dev_mp4_part* archive files in parallel."""
    from huggingface_hub import list_repo_files

    archive_dir = out_root / "archives"
    archive_dir.mkdir(parents=True, exist_ok=True)

    print("Listing mp4 archive parts on HuggingFace...")
    all_files = list(list_repo_files(hf_repo, repo_type="dataset"))
    parts = sorted([f for f in all_files if "vox2_dev_mp4_part" in f])
    print(f"Found {len(parts)} parts. Downloading with {n_workers} workers...")

    def _dl_part(part: str) -> tuple[str, bool, int]:
        dest = archive_dir / Path(part).name
        if dest.exists():
            return str(dest), True, dest.stat().st_size
        local = _hf_download(hf_repo, part, str(archive_dir))
        sz = Path(local).stat().st_size
        return str(dest), False, sz

    done = 0
    total_bytes = 0
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_dl_part, p): p for p in parts}
        for fut in as_completed(futures):
            dest, cached, sz = fut.result()
            total_bytes += sz
            done += 1
            tag = "cached" if cached else f"{sz/1e9:.2f} GB"
            print(f"  [{done}/{len(parts)}] {Path(dest).name} -- {tag}")

    print(f"All parts done. Total on disk: {total_bytes/1e9:.1f} GB")
    return archive_dir


def extract_archives(archive_dir: Path, mp4_dir: Path) -> None:
    """Combine archive parts and extract mp4 files, then delete archives."""
    mp4_dir.mkdir(parents=True, exist_ok=True)

    existing = len(list(mp4_dir.rglob("*.mp4")))
    if existing > 50_000:
        print(f"mp4 files already extracted ({existing:,}) -- skipping extraction")
        return

    combined = archive_dir / "combined.tar"

    if not combined.exists():
        parts = sorted(archive_dir.glob("vox2_dev_mp4_part*"))
        if not parts:
            raise FileNotFoundError(f"No archive parts found in {archive_dir}")
        print(f"Combining {len(parts)} parts (Python binary concat, may take a few min)...")
        with open(combined, "wb") as out_f:
            for i, part in enumerate(parts, 1):
                with open(part, "rb") as in_f:
                    shutil.copyfileobj(in_f, out_f)
                print(f"  {i}/{len(parts)} -- {part.name}", end="\r", flush=True)
        print(f"\nCombined: {combined.stat().st_size/1e9:.1f} GB")

    print(f"Extracting to {mp4_dir} (may take ~15 min)...")
    result = subprocess.run(
        ["tar", "-xf", str(combined), "-C", str(mp4_dir)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"[WARN] tar exit {result.returncode}: {result.stderr[:400]}")

    print("Removing archives to free disk space...")
    combined.unlink(missing_ok=True)
    for part in archive_dir.glob("vox2_dev_mp4_part*"):
        part.unlink(missing_ok=True)

    n_final = len(list(mp4_dir.rglob("*.mp4")))
    print(f"Done: {n_final:,} mp4 files in {mp4_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download VoxCeleb2 from HuggingFace")
    parser.add_argument("--out", default=r"D:\voix\data\raw\voxceleb2")
    parser.add_argument("--workers", type=int, default=4,
                        help="Parallel workers for archive downloads")
    parser.add_argument("--skip-videos", action="store_true",
                        help="Download only metadata, skip mp4 archives")
    args = parser.parse_args()

    HF_REPO  = "Reverb/voxceleb2"
    out_root = Path(args.out)
    mp4_dir  = out_root / "dev" / "mp4"
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"Output root : {out_root}")
    print(f"mp4 target  : {mp4_dir}")
    print()

    download_metadata(out_root, HF_REPO)
    print()

    if not args.skip_videos:
        archive_dir = download_archives(out_root, HF_REPO, n_workers=args.workers)
        print()
        extract_archives(archive_dir, mp4_dir)
        print()

    print("Download complete.")
    print(f"  Metadata : {out_root / 'vox2_meta.csv'}")
    print(f"  Txt dir  : {out_root / 'txt'}")
    if not args.skip_videos:
        print(f"  mp4 dir  : {mp4_dir}")
    print()
    print("Next -- generate the curated index:")
    print(f"  python scripts/curate_voxceleb_index.py ^")
    print(f"    --meta {out_root / 'vox2_meta.csv'} ^")
    print(f"    --txt  {out_root / 'txt'} ^")
    print(f"    --out  D:\\voix\\data\\processed\\curated_index.csv")


if __name__ == "__main__":
    main()