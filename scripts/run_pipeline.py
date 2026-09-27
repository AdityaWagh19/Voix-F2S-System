"""End-to-end per-chunk pipeline for VoxCeleb2 feature extraction.

For each of the 9 archive parts:
  1. Download the archive part      (~32 GB)
  2. Extract mp4s from it           (~35 GB, archive deleted)
  3. Build a mini-index for that part
  4. Run ECAPA + ArcFace extraction -> append to global HDF5
  5. Delete the extracted mp4s      (back to ~0 GB)
  6. Move to next part

Peak disk at any time: ~67 GB (1 archive + mp4s from 1 part)
Final output: a single HDF5 with features from all 9 parts (~300 MB for 100K clips)

Usage:
    python scripts/run_pipeline.py

Options:
    --out           D:\voix\data\raw\voxceleb2   (download root)
    --hdf5          D:\voix\data\processed\features.h5
    --clips-per-part  Number of clips to extract per archive part (default: 12000)
                      12000 * 9 parts = 108000 total (>100K target)
    --parts         How many archive parts to process (default: all 9)
    --device        cuda or cpu (default: cuda)
    --resume        Skip parts whose clips are already in the HDF5
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(r"D:\voix")
sys.path.insert(0, str(ROOT))

HF_REPO = "Reverb/voxceleb2"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _count_mp4s(d: Path) -> int:
    return sum(1 for _ in d.rglob("*.mp4")) if d.exists() else 0


def _hf_download_part(filename: str, local_dir: Path) -> Path:
    from huggingface_hub import hf_hub_download
    local = hf_hub_download(
        repo_id=HF_REPO, filename=filename,
        repo_type="dataset", local_dir=str(local_dir),
    )
    return Path(local)


def _list_parts() -> list[str]:
    from huggingface_hub import list_repo_files
    all_files = list(list_repo_files(HF_REPO, repo_type="dataset"))
    return sorted(f for f in all_files if "vox2_dev_mp4_part" in f)


def _hdf5_clip_count(hdf5_path: Path) -> int:
    if not hdf5_path.exists():
        return 0
    try:
        import h5py
        with h5py.File(str(hdf5_path), "r") as f:
            return f["face_features"].shape[0] if "face_features" in f else 0
    except Exception:
        return 0


def _hdf5_processed_ids(hdf5_path: Path) -> set[str]:
    """Return set of clip IDs already written to the HDF5."""
    if not hdf5_path.exists():
        return set()
    try:
        import h5py
        with h5py.File(str(hdf5_path), "r") as f:
            if "clip_ids" in f:
                return set(cid.decode() if isinstance(cid, bytes) else cid
                           for cid in f["clip_ids"][:])
    except Exception:
        pass
    return set()


# ---------------------------------------------------------------------------
# Step 1: download one archive part
# ---------------------------------------------------------------------------

def download_part(part_name: str, archive_dir: Path) -> Path:
    dest = archive_dir / Path(part_name).name
    if dest.exists():
        print(f"  Already on disk: {dest.name} ({dest.stat().st_size/1e9:.1f} GB)")
        return dest
    print(f"  Downloading {dest.name} from HuggingFace...")
    _hf_download_part(part_name, archive_dir)
    print(f"  Downloaded: {dest.stat().st_size/1e9:.1f} GB")
    return dest


# ---------------------------------------------------------------------------
# Step 2: extract archive part to mp4s (stream, no combined.tar)
# ---------------------------------------------------------------------------

def extract_part(part_path: Path, mp4_dir: Path) -> int:
    mp4_dir.mkdir(parents=True, exist_ok=True)
    n_before = _count_mp4s(mp4_dir)
    print(f"  Streaming {part_path.name} -> {mp4_dir} ...")
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
    print(f"  Extracted {n_after - n_before:,} new clips (total on disk: {n_after:,})")
    return n_after - n_before


# ---------------------------------------------------------------------------
# Step 3: build mini index for this part's mp4s
# ---------------------------------------------------------------------------

def build_mini_index(mp4_dir: Path, meta_path: Path,
                     already_done: set[str],
                     clips_wanted: int, seed: int = 42) -> Path:
    """Sample clips_wanted random clips from mp4_dir, excluding already_done."""
    import csv as _csv, random as _random
    from collections import defaultdict

    # Load gender/nationality from meta
    meta: dict[str, dict] = {}
    if meta_path.exists():
        with open(meta_path, newline="", encoding="utf-8") as f:
            for row in _csv.reader(f):
                if not row or row[0].startswith("VoxCeleb"):
                    continue
                parts = [p.strip() for p in row]
                if len(parts) >= 3:
                    meta[parts[0]] = {
                        "gender": parts[2].lower(),
                        "nationality": parts[3] if len(parts) > 3 else "unknown",
                    }

    # Scan available mp4s
    available: list[tuple[str, str, str]] = []  # (speaker_id, video_id, utt_id)
    for spk_dir in mp4_dir.iterdir():
        if not spk_dir.is_dir():
            continue
        spk_id = spk_dir.name
        for vid_dir in spk_dir.iterdir():
            if not vid_dir.is_dir():
                continue
            vid_id = vid_dir.name
            for mp4 in vid_dir.glob("*.mp4"):
                clip_id = f"{spk_id}/{vid_id}/{mp4.stem}"
                if clip_id not in already_done:
                    available.append((spk_id, vid_id, mp4.stem))

    # Sample proportionally across speakers for diversity
    by_speaker: dict[str, list] = defaultdict(list)
    for entry in available:
        by_speaker[entry[0]].append(entry)

    _random.seed(seed)
    selected: list[tuple[str, str, str]] = []
    speakers = sorted(by_speaker.keys())

    # Round-robin across speakers to maximise diversity
    while len(selected) < clips_wanted and speakers:
        _random.shuffle(speakers)
        for spk in list(speakers):
            if not by_speaker[spk]:
                speakers.remove(spk)
                continue
            chosen = by_speaker[spk].pop(
                _random.randrange(len(by_speaker[spk]))
            )
            selected.append(chosen)
            if len(selected) >= clips_wanted:
                break

    # Write mini index CSV
    index_path = mp4_dir.parent / ".mini_index.csv"
    with open(index_path, "w", newline="", encoding="utf-8") as f:
        w = _csv.DictWriter(f, fieldnames=["speaker_id", "video_id", "utterance_id", "gender", "nationality"])
        w.writeheader()
        for spk_id, vid_id, utt_id in selected:
            m = meta.get(spk_id, {"gender": "u", "nationality": "unknown"})
            w.writerow({"speaker_id": spk_id, "video_id": vid_id,
                        "utterance_id": utt_id, **m})

    print(f"  Mini-index: {len(selected)} clips from {len(set(e[0] for e in selected))} speakers")
    return index_path


# ---------------------------------------------------------------------------
# Step 4: run feature extraction on mini index
# ---------------------------------------------------------------------------

def run_extraction(index_path: Path, mp4_dir: Path, hdf5_path: Path, device: str) -> int:
    """Run extract_voxceleb.py on the mini index, appending to global HDF5."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "extract_voxceleb",
        str(ROOT / "scripts" / "extract_voxceleb.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    n_before = _hdf5_clip_count(hdf5_path)
    mod.run_extraction(
        index_path=str(index_path),
        videos_root=str(mp4_dir),
        output_path=str(hdf5_path),
        device=device,
        batch_size=50,
        session_max_hours=999.0,
        progress_callback=None,
    )
    n_after = _hdf5_clip_count(hdf5_path)
    return n_after - n_before


# ---------------------------------------------------------------------------
# Step 5: delete mp4s from this part
# ---------------------------------------------------------------------------

def delete_mp4s(mp4_dir: Path) -> None:
    n = _count_mp4s(mp4_dir)
    if n == 0:
        return
    print(f"  Deleting {n:,} mp4 files to free disk...")
    for spk_dir in list(mp4_dir.iterdir()):
        if spk_dir.is_dir():
            shutil.rmtree(spk_dir, ignore_errors=True)
    print(f"  mp4 directory cleared.")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="VoxCeleb2 per-chunk extraction pipeline")
    parser.add_argument("--out",            default=r"D:\voix\data\raw\voxceleb2")
    parser.add_argument("--hdf5",           default=r"D:\voix\data\processed\features.h5")
    parser.add_argument("--clips-per-part", type=int, default=12000,
                        help="Clips to extract per archive part (default: 12000, 9 parts = 108K total)")
    parser.add_argument("--parts",          type=int, default=9,
                        help="Number of archive parts to process (default: all 9)")
    parser.add_argument("--device",         default="cuda")
    parser.add_argument("--skip-delete",    action="store_true",
                        help="Keep mp4s after extraction (debug)")
    args = parser.parse_args()

    out_root    = Path(args.out)
    mp4_dir     = out_root / "dev" / "mp4"
    archive_dir = out_root / "archives"
    hdf5_path   = Path(args.hdf5)
    meta_path   = out_root / "vox2_meta.csv"

    out_root.mkdir(parents=True, exist_ok=True)
    archive_dir.mkdir(parents=True, exist_ok=True)
    hdf5_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  VOIX Per-Chunk Extraction Pipeline")
    print(f"  HDF5 output  : {hdf5_path}")
    print(f"  Clips/part   : {args.clips_per_part:,}")
    print(f"  Parts to run : {args.parts}")
    print(f"  Target total : ~{args.clips_per_part * args.parts:,} clips")
    print("=" * 60)

    # Download metadata if needed
    if not meta_path.exists():
        print("Downloading vox2_meta.csv...")
        _hf_download_part("vox2_meta.csv", out_root)

    parts = _list_parts()[:args.parts]
    total_extracted = 0

    for i, part_name in enumerate(parts, 1):
        print()
        print(f"{'='*60}")
        print(f"  Part {i}/{len(parts)}: {Path(part_name).name}")
        print(f"  HDF5 so far: {_hdf5_clip_count(hdf5_path):,} clips")
        print(f"{'='*60}")

        # --- Step 1: Download ---
        print("\n[1/5] Download")
        part_path = download_part(part_name, archive_dir)

        # --- Step 2: Extract mp4s ---
        print("\n[2/5] Extract mp4s from archive")
        extract_part(part_path, mp4_dir)

        # --- Step 3: Delete archive immediately ---
        print("\n[3/5] Delete archive to free disk")
        part_path.unlink(missing_ok=True)
        print(f"  Deleted {part_path.name}")

        # --- Step 4: Build mini index ---
        print("\n[4/5] Build mini index")
        already_done = _hdf5_processed_ids(hdf5_path)
        index_path = build_mini_index(mp4_dir, meta_path, already_done,
                                       clips_wanted=args.clips_per_part)

        # --- Step 5: Run extraction ---
        print("\n[5/5] Feature extraction -> HDF5")
        n_extracted = run_extraction(index_path, mp4_dir, hdf5_path, args.device)
        total_extracted += n_extracted
        print(f"  Extracted {n_extracted:,} clips this part | Total: {_hdf5_clip_count(hdf5_path):,}")

        # --- Step 6: Delete mp4s ---
        if not args.skip_delete:
            print("\n[+] Cleanup: delete mp4s")
            delete_mp4s(mp4_dir)

        print(f"\n  Part {i} complete.")

    print()
    print("=" * 60)
    print(f"  PIPELINE COMPLETE")
    print(f"  Total clips in HDF5 : {_hdf5_clip_count(hdf5_path):,}")
    print(f"  HDF5 path           : {hdf5_path}")
    print(f"  HDF5 size           : {hdf5_path.stat().st_size/1e6:.0f} MB")
    print("=" * 60)


if __name__ == "__main__":
    main()