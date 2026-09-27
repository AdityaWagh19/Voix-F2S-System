"""VoxCeleb2 per-chunk extraction pipeline with per-speaker fixed quota.

Bias-free design
----------------
Bias source 1 -- popular speakers have more clips:
    Fixed by taking exactly CLIPS_PER_SPEAKER clips per speaker,
    regardless of how many they actually have. Every speaker contributes
    identically to the feature space.

Bias source 2 -- gender imbalance (VoxCeleb2 is 61% male):
    Fixed by the final selection step: exactly N_MALE + N_FEMALE speakers
    are chosen globally after all parts are processed.

Bias source 3 -- archive part ordering (some parts may cluster by name):
    Fixed by processing all 9 parts, so all 5994 speakers are covered.

Pipeline (per archive part)
---------------------------
  1. Download archive part     ~32 GB
  2. Extract mp4 clips         ~35 GB, then delete archive
  3. Per-speaker quota sample  exactly CLIPS_PER_SPEAKER clips each,
                               spread across different video sessions
  4. ECAPA + ArcFace extract   append features to global HDF5
  5. Delete all mp4s           back to ~0 GB extra disk

Final step (after all parts)
-----------------------------
  6. Gender-stratified select  pick N_SPEAKERS_TOTAL (half male, half female)
                               from the full ~6000 speaker pool in HDF5
  7. Write final training CSV  data/processed/training_index.csv

Disk budget
-----------
  Peak per iteration : ~67 GB  (1 archive + mp4s from 1 part)
  Final HDF5         : ~900 MB (6000 speakers x 50 clips x feature vectors)
  Training CSV       : <1 MB

Usage
-----
    python scripts/run_pipeline.py
    python scripts/run_pipeline.py --clips-per-speaker 50 --speakers 2000
    python scripts/run_pipeline.py --parts 3   # test with first 3 parts only
    python scripts/run_pipeline.py --skip-delete  # keep mp4s for debugging
"""

from __future__ import annotations

import argparse
import csv
import random
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(r"D:\voix")
sys.path.insert(0, str(ROOT))

HF_REPO = "Reverb/voxceleb2"

# ---------------------------------------------------------------------------
# Constants (tunable via CLI args)
# ---------------------------------------------------------------------------
DEFAULT_CLIPS_PER_SPEAKER = 50   # fixed per-speaker quota
DEFAULT_MIN_CLIPS         = 50   # skip speakers with fewer clips
DEFAULT_SPEAKERS_TOTAL    = 2000 # final training set size (half male, half female)
DEFAULT_SEED              = 42


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _count_mp4s(d: Path) -> int:
    return sum(1 for _ in d.rglob("*.mp4")) if d.exists() else 0


def _hf_download(filename: str, local_dir: Path) -> Path:
    from huggingface_hub import hf_hub_download
    return Path(hf_hub_download(
        repo_id=HF_REPO, filename=filename,
        repo_type="dataset", local_dir=str(local_dir),
    ))


def _list_parts() -> list[str]:
    from huggingface_hub import list_repo_files
    return sorted(f for f in list_repo_files(HF_REPO, repo_type="dataset")
                  if "vox2_dev_mp4_part" in f)


def _load_meta(meta_path: Path) -> dict[str, dict]:
    """Load speaker metadata from vox2_meta.csv."""
    speakers: dict[str, dict] = {}
    if not meta_path.exists():
        return speakers
    with open(meta_path, newline="", encoding="utf-8") as f:
        for row in csv.reader(f):
            if not row or row[0].startswith("VoxCeleb"):
                continue
            p = [x.strip() for x in row]
            if len(p) < 3:
                continue
            split = p[4].lower() if len(p) > 4 else "dev"
            if split != "dev":
                continue
            speakers[p[0]] = {
                "gender":      p[2].lower(),
                "nationality": p[3] if len(p) > 3 else "unknown",
            }
    return speakers


def _hdf5_info(hdf5_path: Path) -> tuple[int, set[str]]:
    """Return (total_clips, set_of_processed_clip_ids) from HDF5."""
    if not hdf5_path.exists():
        return 0, set()
    try:
        import h5py
        with h5py.File(str(hdf5_path), "r") as f:
            n = f["face_features"].shape[0] if "face_features" in f else 0
            ids: set[str] = set()
            if "clip_ids" in f:
                ids = set(
                    cid.decode() if isinstance(cid, bytes) else cid
                    for cid in f["clip_ids"][:]
                )
            return n, ids
    except Exception:
        return 0, set()


# ---------------------------------------------------------------------------
# Step 1: Download one archive part
# ---------------------------------------------------------------------------

def download_part(part_name: str, archive_dir: Path) -> Path:
    dest = archive_dir / Path(part_name).name
    if dest.exists():
        print(f"  Cached on disk: {dest.name} ({dest.stat().st_size/1e9:.1f} GB)")
        return dest
    print(f"  Downloading {dest.name}...")
    _hf_download(part_name, archive_dir)
    print(f"  Done: {dest.stat().st_size/1e9:.1f} GB")
    return dest


# ---------------------------------------------------------------------------
# Step 2: Extract archive to mp4s (streaming, no intermediate combined file)
# ---------------------------------------------------------------------------

def extract_part(part_path: Path, mp4_dir: Path) -> int:
    mp4_dir.mkdir(parents=True, exist_ok=True)
    n_before = _count_mp4s(mp4_dir)
    print(f"  Streaming {part_path.name} -> tar -> {mp4_dir.name}/")
    tar = subprocess.Popen(
        ["tar", "-x", "--ignore-failed-read", "--warning=no-all", "-C", str(mp4_dir)],
        stdin=subprocess.PIPE,
    )
    try:
        with open(part_path, "rb") as f:
            shutil.copyfileobj(f, tar.stdin, length=8 * 1024 * 1024)
        tar.stdin.close()
        tar.wait()
    except Exception as e:
        tar.kill()
        print(f"  [WARN] tar: {e}")
    extracted = _count_mp4s(mp4_dir) - n_before
    print(f"  Extracted {extracted:,} clips  (total on disk: {_count_mp4s(mp4_dir):,})")
    return extracted


# ---------------------------------------------------------------------------
# Step 3: Per-speaker fixed quota sampling (cross-session)
# Bias fix: every speaker contributes EXACTLY clips_per_speaker clips,
# regardless of their total clip count. Popular celebrities do not dominate.
# ---------------------------------------------------------------------------

def build_speaker_quota_index(
    mp4_dir:          Path,
    meta:             dict[str, dict],
    already_done:     set[str],
    clips_per_speaker: int,
    min_clips:        int,
    seed:             int,
) -> list[dict]:
    """Return rows for all qualified speakers in this part, fixed quota each.

    Cross-session sampling: clips spread across different video sessions
    to maximise intra-speaker acoustic variation.
    """
    rng = np.random.default_rng(seed)
    rows: list[dict] = []

    for spk_dir in sorted(mp4_dir.iterdir()):
        if not spk_dir.is_dir():
            continue
        spk_id = spk_dir.name

        # Group clips by video session
        by_session: dict[str, list[str]] = defaultdict(list)
        for vid_dir in spk_dir.iterdir():
            if not vid_dir.is_dir():
                continue
            vid_id = vid_dir.name
            for mp4 in vid_dir.glob("*.mp4"):
                clip_id = f"{spk_id}/{vid_id}/{mp4.stem}"
                if clip_id not in already_done:
                    by_session[vid_id].append(mp4.stem)

        total_available = sum(len(v) for v in by_session.values())
        if total_available < min_clips:
            continue   # skip speaker: not enough clips

        # Cross-session round-robin to fill quota
        sessions = sorted(by_session.keys())
        selected: list[tuple[str, str]] = []   # (vid_id, utt_id)
        session_pools = {s: list(by_session[s]) for s in sessions}

        # Shuffle clips within each session for variety
        for pool in session_pools.values():
            rng.shuffle(pool)

        # Round-robin across sessions
        session_idx = 0
        while len(selected) < clips_per_speaker and any(session_pools.values()):
            s = sessions[session_idx % len(sessions)]
            if session_pools[s]:
                utt = session_pools[s].pop()
                selected.append((s, utt))
            session_idx += 1

        m = meta.get(spk_id, {"gender": "u", "nationality": "unknown"})
        for vid_id, utt_id in selected:
            rows.append({
                "speaker_id":   spk_id,
                "video_id":     vid_id,
                "utterance_id": utt_id,
                "gender":       m["gender"],
                "nationality":  m["nationality"],
            })

    n_speakers = len(set(r["speaker_id"] for r in rows))
    print(f"  Quota index: {len(rows):,} clips from {n_speakers} speakers "
          f"({clips_per_speaker} clips/speaker)")
    return rows


def write_index(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["speaker_id", "video_id", "utterance_id", "gender", "nationality"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


# ---------------------------------------------------------------------------
# Step 4: Feature extraction -> HDF5
# ---------------------------------------------------------------------------

def run_extraction(index_path: Path, mp4_dir: Path,
                   hdf5_path: Path, device: str) -> int:
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "extract_voxceleb", str(ROOT / "scripts" / "extract_voxceleb.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    n_before, _ = _hdf5_info(hdf5_path)
    mod.run_extraction(
        index_path=str(index_path),
        videos_root=str(mp4_dir),
        output_path=str(hdf5_path),
        device=device,
        batch_size=50,
        session_max_hours=999.0,
    )
    n_after, _ = _hdf5_info(hdf5_path)
    return n_after - n_before


# ---------------------------------------------------------------------------
# Step 5: Delete mp4s to free disk
# ---------------------------------------------------------------------------

def delete_mp4s(mp4_dir: Path) -> None:
    n = _count_mp4s(mp4_dir)
    if n == 0:
        return
    for d in list(mp4_dir.iterdir()):
        if d.is_dir():
            shutil.rmtree(d, ignore_errors=True)
    print(f"  Deleted {n:,} mp4s from {mp4_dir.name}/")


# ---------------------------------------------------------------------------
# Final step: gender-stratified global speaker selection from HDF5
# Bias fix: exactly N/2 male + N/2 female speakers chosen uniformly,
# regardless of nationality or clip count (already fixed by quota above).
# ---------------------------------------------------------------------------

def finalize_training_index(
    hdf5_path:        Path,
    meta:             dict[str, dict],
    output_csv:       Path,
    speakers_total:   int,
    seed:             int,
) -> int:
    """Select speakers_total speakers (half male, half female) from HDF5.

    All speakers in HDF5 already have identical clip counts (per-speaker quota),
    so uniform random selection over gender groups is fully unbiased.
    """
    import h5py

    rng = random.Random(seed)
    n_per_gender = speakers_total // 2

    with h5py.File(str(hdf5_path), "r") as f:
        if "clip_ids" not in f:
            print("[WARN] HDF5 has no clip_ids -- skipping final selection")
            return 0
        clip_ids = [
            (cid.decode() if isinstance(cid, bytes) else cid)
            for cid in f["clip_ids"][:]
        ]

    # Collect all speakers present in HDF5
    speaker_set: set[str] = set()
    for cid in clip_ids:
        spk_id = cid.split("/")[0]
        speaker_set.add(spk_id)

    males   = sorted(s for s in speaker_set if meta.get(s, {}).get("gender") == "m")
    females = sorted(s for s in speaker_set if meta.get(s, {}).get("gender") == "f")

    print(f"  Available for selection: {len(males)} male, {len(females)} female speakers")

    if len(males) < n_per_gender:
        print(f"  [WARN] Only {len(males)} male speakers, wanted {n_per_gender}")
        n_per_gender = min(len(males), len(females))

    sel_m = rng.sample(males,   min(n_per_gender, len(males)))
    sel_f = rng.sample(females, min(n_per_gender, len(females)))
    selected_speakers = set(sel_m + sel_f)

    # Filter clip_ids to selected speakers
    selected_rows = []
    for cid in clip_ids:
        spk_id = cid.split("/")[0]
        if spk_id in selected_speakers:
            parts = cid.split("/")
            m = meta.get(spk_id, {"gender": "u", "nationality": "unknown"})
            selected_rows.append({
                "speaker_id":   spk_id,
                "video_id":     parts[1] if len(parts) > 1 else "",
                "utterance_id": parts[2] if len(parts) > 2 else "",
                "gender":       m["gender"],
                "nationality":  m["nationality"],
                "clip_id":      cid,
            })

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = ["clip_id", "speaker_id", "video_id", "utterance_id", "gender", "nationality"]
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(selected_rows)

    print(f"  Selected {len(sel_m)} male + {len(sel_f)} female = {len(selected_speakers)} speakers")
    print(f"  Training index: {len(selected_rows):,} clips -> {output_csv}")
    return len(selected_rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="VoxCeleb2 per-chunk extraction (bias-free, per-speaker quota)"
    )
    parser.add_argument("--out",               default=r"D:\voix\data\raw\voxceleb2")
    parser.add_argument("--hdf5",              default=r"D:\voix\data\processed\features.h5")
    parser.add_argument("--training-csv",      default=r"D:\voix\data\processed\training_index.csv")
    parser.add_argument("--clips-per-speaker", type=int, default=DEFAULT_CLIPS_PER_SPEAKER)
    parser.add_argument("--min-clips",         type=int, default=DEFAULT_MIN_CLIPS)
    parser.add_argument("--speakers",          type=int, default=DEFAULT_SPEAKERS_TOTAL,
                        help="Final training set: total speakers (half male, half female)")
    parser.add_argument("--parts",             type=int, default=9)
    parser.add_argument("--device",            default="cuda")
    parser.add_argument("--seed",              type=int, default=DEFAULT_SEED)
    parser.add_argument("--skip-delete",       action="store_true",
                        help="Keep mp4s after extraction (for debugging)")
    parser.add_argument("--finalize-only",     action="store_true",
                        help="Skip extraction, only run final gender-stratified selection")
    args = parser.parse_args()

    out_root    = Path(args.out)
    mp4_dir     = out_root / "dev" / "mp4"
    archive_dir = out_root / "archives"
    hdf5_path   = Path(args.hdf5)
    train_csv   = Path(args.training_csv)
    meta_path   = out_root / "vox2_meta.csv"

    for d in [out_root, archive_dir, hdf5_path.parent]:
        d.mkdir(parents=True, exist_ok=True)

    print("=" * 62)
    print("  VOIX Per-Chunk Extraction Pipeline (bias-free)")
    print(f"  Clips per speaker : {args.clips_per_speaker} (fixed quota)")
    print(f"  Min clips needed  : {args.min_clips}")
    print(f"  Parts to process  : {args.parts}")
    print(f"  Final speakers    : {args.speakers} ({args.speakers//2}M + {args.speakers//2}F)")
    print(f"  Expected total    : ~{args.speakers * args.clips_per_speaker:,} clips")
    print(f"  HDF5              : {hdf5_path}")
    print("=" * 62)

    # Load speaker metadata
    if not meta_path.exists():
        print("Downloading vox2_meta.csv...")
        _hf_download("vox2_meta.csv", out_root)
    meta = _load_meta(meta_path)
    print(f"Loaded metadata: {len(meta)} dev speakers\n")

    if not args.finalize_only:
        parts = _list_parts()[:args.parts]
        n_total, done_ids = _hdf5_info(hdf5_path)
        print(f"HDF5 starting state: {n_total:,} clips already extracted\n")

        for i, part_name in enumerate(parts, 1):
            print()
            print(f"{'='*62}")
            print(f"  Part {i}/{len(parts)}: {Path(part_name).name}")
            print(f"  HDF5 total so far : {_hdf5_info(hdf5_path)[0]:,} clips")
            print(f"{'='*62}")

            # 1. Download
            print("\n[1/5] Download archive part")
            part_path = download_part(part_name, archive_dir)

            # 2. Extract mp4s
            print("\n[2/5] Extract mp4s (streaming, no combined.tar)")
            extract_part(part_path, mp4_dir)

            # 3. Delete archive immediately
            print("\n[3/5] Delete archive")
            part_path.unlink(missing_ok=True)
            print(f"  Deleted {part_path.name}")

            # 4. Build per-speaker quota index
            print("\n[4/5] Build per-speaker quota index")
            _, done_ids = _hdf5_info(hdf5_path)
            rows = build_speaker_quota_index(
                mp4_dir, meta, done_ids,
                clips_per_speaker=args.clips_per_speaker,
                min_clips=args.min_clips,
                seed=args.seed + i,
            )
            if not rows:
                print("  No new speakers -- skipping extraction for this part")
            else:
                index_path = out_root / ".part_index.csv"
                write_index(rows, index_path)

                # 5. Run extraction
                print("\n[5/5] Feature extraction -> HDF5")
                n_new = run_extraction(index_path, mp4_dir, hdf5_path, args.device)
                print(f"  Extracted {n_new:,} clips this part")
                index_path.unlink(missing_ok=True)

            # 6. Delete mp4s
            if not args.skip_delete:
                print("\n[+] Cleanup mp4s")
                delete_mp4s(mp4_dir)

    # Final: gender-stratified speaker selection
    n_total, _ = _hdf5_info(hdf5_path)
    print()
    print("=" * 62)
    print(f"  All parts done. HDF5 total: {n_total:,} clips")
    print(f"  Final gender-stratified speaker selection...")
    print("=" * 62)
    finalize_training_index(
        hdf5_path, meta, train_csv,
        speakers_total=args.speakers,
        seed=args.seed,
    )

    print()
    print("=" * 62)
    print("  PIPELINE COMPLETE")
    print(f"  HDF5 (all speakers) : {hdf5_path}")
    n_total, _ = _hdf5_info(hdf5_path)
    print(f"  HDF5 clip count     : {n_total:,}")
    sz = hdf5_path.stat().st_size / 1e6 if hdf5_path.exists() else 0
    print(f"  HDF5 size           : {sz:.0f} MB")
    print(f"  Training index      : {train_csv}")
    print("=" * 62)


if __name__ == "__main__":
    main()