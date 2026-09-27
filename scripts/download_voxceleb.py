r"""Download VoxCeleb2 from HuggingFace (Reverb/voxceleb2) to local disk.

Two modes:
  --stream  (default/recommended): download one archive part at a time,
            stream it into tar immediately, delete it, then move to next part.
            Stops early once --target-clips are extracted.
            Peak disk: ~32 GB (one part) + extracted mp4s.

  --batch:  download all parts first, then extract. Needs ~290 GB disk.

Usage:
    python scripts/download_voxceleb.py --stream
    python scripts/download_voxceleb.py --stream --target-clips 120000
    python scripts/download_voxceleb.py --batch --workers 4

Checkpoint-safe: already-extracted speakers are skipped on re-run.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


HF_REPO = "Reverb/voxceleb2"


def _hf_download_part(repo_id: str, filename: str, local_dir: str) -> Path:
    from huggingface_hub import hf_hub_download
    local = hf_hub_download(
        repo_id=repo_id, filename=filename,
        repo_type="dataset", local_dir=local_dir,
    )
    return Path(local)


def download_metadata(out_root: Path, download_txt: bool = False) -> None:
    """Download vox2_meta.csv. Txt files skipped by default (use --with-txt)."""
    meta_dst = out_root / "vox2_meta.csv"
    if meta_dst.exists():
        print(f"vox2_meta.csv already present ({meta_dst.stat().st_size // 1024} KB) -- skipping")
    else:
        print("Downloading vox2_meta.csv...")
        local = _hf_download_part(HF_REPO, "vox2_meta.csv", str(out_root))
        if local != meta_dst:
            shutil.copy(local, meta_dst)
        print(f"  -> {meta_dst}")

    if not download_txt:
        print("Skipping txt annotations (not needed -- mp4 dir has same structure).")
        return

    # Txt download (rate-limited, 2 workers with backoff)
    from huggingface_hub import list_repo_files
    txt_dir = out_root / "txt"
    txt_dir.mkdir(parents=True, exist_ok=True)
    existing = len(list(txt_dir.rglob("*.txt")))
    if existing > 5000:
        print(f"txt annotations already present ({existing} files) -- skipping")
        return
    print("Fetching txt file list...")
    all_files = list(list_repo_files(HF_REPO, repo_type="dataset"))
    txt_files = [f for f in all_files if f.startswith("txt/") and f.endswith(".txt")]
    print(f"Downloading {len(txt_files)} txt files (2 workers, rate-limit safe)...")
    done = 0
    def _dl(fname):
        for attempt in range(5):
            try:
                _hf_download_part(HF_REPO, fname, str(out_root))
                return True
            except Exception as e:
                if "429" in str(e):
                    time.sleep(30 * (attempt + 1))
        return False
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = {pool.submit(_dl, f): f for f in txt_files}
        for fut in as_completed(futures):
            done += 1
            if done % 200 == 0:
                print(f"  {done}/{len(txt_files)}")
    print(f"txt done: {len(list(txt_dir.rglob('*.txt')))} files")


def _list_parts() -> list[str]:
    from huggingface_hub import list_repo_files
    all_files = list(list_repo_files(HF_REPO, repo_type="dataset"))
    return sorted([f for f in all_files if "vox2_dev_mp4_part" in f])


def _count_mp4s(mp4_dir: Path) -> int:
    return sum(1 for _ in mp4_dir.rglob("*.mp4"))


def stream_download_and_extract(out_root: Path, mp4_dir: Path, target_clips: int) -> None:
    """Download one part at a time, stream to tar, delete. Stops at target_clips.

    Peak disk: ~32 GB (one part on disk) + extracted mp4s.
    Covers all speakers across all parts for diverse sampling.
    """
    mp4_dir.mkdir(parents=True, exist_ok=True)
    archive_dir = out_root / "archives"
    archive_dir.mkdir(parents=True, exist_ok=True)

    current_clips = _count_mp4s(mp4_dir)
    if current_clips >= target_clips:
        print(f"Already have {current_clips:,} clips (target: {target_clips:,}) -- skipping download")
        return

    print(f"Listing archive parts on HuggingFace...")
    parts = _list_parts()
    print(f"Found {len(parts)} parts. Will stop once {target_clips:,} clips are extracted.")
    print(f"Currently have {current_clips:,} clips.")
    print()

    for i, part_name in enumerate(parts, 1):
        if _count_mp4s(mp4_dir) >= target_clips:
            print(f"Reached {target_clips:,} clips -- stopping early after {i-1}/{len(parts)} parts.")
            break

        part_path = archive_dir / Path(part_name).name

        # --- Download part ---
        if part_path.exists():
            print(f"[{i}/{len(parts)}] {part_path.name} already on disk ({part_path.stat().st_size/1e9:.1f} GB) -- using cached")
        else:
            print(f"[{i}/{len(parts)}] Downloading {part_path.name}...")
            _hf_download_part(HF_REPO, part_name, str(archive_dir))
            print(f"  Downloaded: {part_path.stat().st_size/1e9:.1f} GB")

        # --- Stream into tar (no intermediate combined file) ---
        n_before = _count_mp4s(mp4_dir)
        print(f"  Extracting into {mp4_dir} ...")
        tar_proc = subprocess.Popen(
            ["tar", "-x", "--ignore-failed-read", "--warning=no-all", "-C", str(mp4_dir)],
            stdin=subprocess.PIPE,
        )
        try:
            with open(part_path, "rb") as f:
                shutil.copyfileobj(f, tar_proc.stdin, length=8 * 1024 * 1024)
            tar_proc.stdin.close()
            tar_proc.wait()
        except Exception as e:
            tar_proc.kill()
            print(f"  [WARN] tar error: {e}")

        n_after = _count_mp4s(mp4_dir)
        print(f"  Extracted {n_after - n_before:,} clips  |  Total: {n_after:,} / {target_clips:,}")

        # --- Delete archive part to free disk ---
        part_path.unlink(missing_ok=True)
        print(f"  Deleted {part_path.name} to free disk.")
        print()

    final = _count_mp4s(mp4_dir)
    print(f"Stream extraction complete: {final:,} clips in {mp4_dir}")


def batch_download_and_extract(out_root: Path, mp4_dir: Path, n_workers: int) -> None:
    """Download all parts first (needs ~290 GB), then extract. Legacy mode."""
    from huggingface_hub import list_repo_files
    archive_dir = out_root / "archives"
    archive_dir.mkdir(parents=True, exist_ok=True)
    mp4_dir.mkdir(parents=True, exist_ok=True)

    existing = _count_mp4s(mp4_dir)
    if existing > 50_000:
        print(f"mp4 files already extracted ({existing:,}) -- skipping")
        return

    parts = _list_parts()
    print(f"Found {len(parts)} parts. Downloading with {n_workers} workers...")

    def _dl_part(part: str) -> tuple[str, bool, int]:
        dest = archive_dir / Path(part).name
        if dest.exists():
            return str(dest), True, dest.stat().st_size
        local = _hf_download_part(HF_REPO, part, str(archive_dir))
        sz = Path(local).stat().st_size
        return str(dest), False, sz

    done = 0
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_dl_part, p): p for p in parts}
        for fut in as_completed(futures):
            dest, cached, sz = fut.result()
            done += 1
            tag = "cached" if cached else f"{sz/1e9:.2f} GB"
            print(f"  [{done}/{len(parts)}] {Path(dest).name} -- {tag}")

    # Stream all parts into tar (no combined file)
    print(f"\nExtracting all parts -> {mp4_dir}")
    archive_parts = sorted(archive_dir.glob("vox2_dev_mp4_part*"))
    tar_proc = subprocess.Popen(
        ["tar", "-x", "--ignore-failed-read", "--warning=no-all", "-C", str(mp4_dir)],
        stdin=subprocess.PIPE,
    )
    try:
        for part in archive_parts:
            with open(part, "rb") as f:
                shutil.copyfileobj(f, tar_proc.stdin, length=8 * 1024 * 1024)
        tar_proc.stdin.close()
        tar_proc.wait()
    except Exception as e:
        tar_proc.kill()
        print(f"[WARN] tar error: {e}")

    print("Removing archives...")
    for p in archive_parts:
        p.unlink(missing_ok=True)

    print(f"Done: {_count_mp4s(mp4_dir):,} mp4 files")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download VoxCeleb2 from HuggingFace")
    parser.add_argument("--out",           default=r"D:\voix\data\raw\voxceleb2")
    parser.add_argument("--stream",        action="store_true", default=True,
                        help="Stream mode: one part at a time (low disk, recommended)")
    parser.add_argument("--batch",         action="store_true", default=False,
                        help="Batch mode: download all parts then extract (needs ~290 GB)")
    parser.add_argument("--workers",       type=int, default=4,
                        help="Parallel workers (batch mode only)")
    parser.add_argument("--target-clips",  type=int, default=130000,
                        help="Stop stream mode after this many clips (default: 130000)")
    parser.add_argument("--with-txt",      action="store_true",
                        help="Also download txt annotations (slow, rate-limited)")
    args = parser.parse_args()

    out_root = Path(args.out)
    mp4_dir  = out_root / "dev" / "mp4"
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"Output root : {out_root}")
    print(f"mp4 target  : {mp4_dir}")
    print()

    download_metadata(out_root, download_txt=args.with_txt)
    print()

    if args.batch:
        batch_download_and_extract(out_root, mp4_dir, args.workers)
    else:
        stream_download_and_extract(out_root, mp4_dir, target_clips=args.target_clips)

    print()
    print("Download complete.")
    print(f"  Metadata : {out_root / 'vox2_meta.csv'}")
    print(f"  mp4 dir  : {mp4_dir}")
    print()
    print("Next -- generate the curated index:")
    print(f"  python scripts/curate_voxceleb_index.py --meta {out_root / 'vox2_meta.csv'} --mp4 {mp4_dir} --output D:\\voix\\data\\processed\\curated_index.csv")


if __name__ == "__main__":
    main()