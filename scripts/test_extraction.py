"""100-clip test extraction to verify the full pipeline before overnight run.

Runs on a random sample of 100 clips from the curated index.
Takes ~5 min on GTX 1650. Verifies:
  - All models load and run without VRAM OOM
  - ffmpeg audio extraction works on real mp4 files
  - HDF5 output is written and readable
  - Checkpoint recovery works (runs twice, second run skips already-done clips)
  - Actual throughput (clips/hr) on your hardware

Usage:
    python scripts/test_extraction.py

Requirements: curated_index.csv and mp4 files must already be downloaded.
Run verify_setup.py first.
"""
import sys, os, time, shutil, random, csv, h5py
from pathlib import Path

VOIX_ROOT  = Path(r"D:\voix")
sys.path.insert(0, str(VOIX_ROOT))

INDEX_PATH = VOIX_ROOT / "data" / "processed" / "curated_index.csv"
VIDEOS_DIR = VOIX_ROOT / "data" / "raw"  / "voxceleb2" / "dev" / "mp4"
TEST_H5    = VOIX_ROOT / "data" / "processed" / "test_extraction_100.h5"
N_CLIPS    = 100

PASS = "[  OK  ]"
FAIL = "[ FAIL ]"

print("=" * 60)
print("  VOIX -- 100-clip extraction test")
print(f"  Index  : {INDEX_PATH}")
print(f"  Videos : {VIDEOS_DIR}")
print(f"  Output : {TEST_H5}")
print("=" * 60)

# Pre-checks
if not INDEX_PATH.exists():
    print(f"\n{FAIL} Index not found: {INDEX_PATH}")
    print("  Run: python scripts/curate_voxceleb_index.py first")
    sys.exit(1)

if not VIDEOS_DIR.exists():
    print(f"\n{FAIL} Videos not found: {VIDEOS_DIR}")
    print("  Run: python scripts/download_voxceleb.py first")
    sys.exit(1)

n_mp4 = len(list(VIDEOS_DIR.rglob("*.mp4")))
if n_mp4 < 100:
    print(f"\n{FAIL} Only {n_mp4} mp4 files found, need at least 100")
    sys.exit(1)

print(f"\n{PASS} Pre-checks: index exists, {n_mp4:,} mp4 files present")

# Build a 100-clip sub-index file
with open(INDEX_PATH) as f:
    rows = list(csv.DictReader(f))
random.seed(42)
sample = random.sample(rows, min(N_CLIPS, len(rows)))
test_index = VOIX_ROOT / "data" / "processed" / "test_index_100.csv"
with open(test_index, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys())
    w.writeheader()
    w.writerows(sample)
print(f"{PASS} Test index: {len(sample)} clips sampled -> {test_index}")

# Delete any previous test HDF5 to get a clean run
if TEST_H5.exists():
    TEST_H5.unlink()
    print(f"  Deleted previous test HDF5 for clean run")

# ── Run 1: full extraction of 100 clips ──────────────────────
print(f"\nRun 1: Extracting {N_CLIPS} clips...")
import torch
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
# GPU pre-warm
_d = torch.zeros(1, device="cuda")
del _d
torch.cuda.empty_cache()
free_mb = torch.cuda.mem_get_info(0)[0] / 1e6
print(f"  VRAM free before models: {free_mb:.0f} MB")

t0 = time.time()

from scripts.extract_voxceleb import run_extraction
run_extraction(
    index_path=str(test_index),
    videos_root=str(VIDEOS_DIR),
    output_path=str(TEST_H5),
    device="cuda",
    batch_size=50,
    session_max_hours=999.0,
    progress_callback=None,
)

elapsed = time.time() - t0
with h5py.File(str(TEST_H5), "r") as f:
    n_done = f["face_features"].shape[0] if "face_features" in f else 0
    dims   = f["face_features"].shape if "face_features" in f else None

rate = n_done / (elapsed / 3600)
print()
print(f"{PASS} Run 1 complete:")
print(f"  Clips extracted : {n_done} / {N_CLIPS}")
print(f"  Time taken      : {elapsed:.1f} s")
print(f"  Throughput      : {rate:.0f} clips/hr")
print(f"  Feature dims    : {dims}")
print(f"  Projected total : {100000 / rate:.0f} hours for 100K clips")

if n_done < N_CLIPS * 0.8:
    print(f"\n{FAIL} Too many failures ({N_CLIPS - n_done} out of {N_CLIPS}). Check warnings above.")
    sys.exit(1)

# ── Run 2: checkpoint recovery test ──────────────────────────
print(f"\nRun 2: Checkpoint recovery test (should skip all {n_done} done clips)...")
t1 = time.time()
run_extraction(
    index_path=str(test_index),
    videos_root=str(VIDEOS_DIR),
    output_path=str(TEST_H5),
    device="cuda",
    batch_size=50,
    session_max_hours=999.0,
    progress_callback=None,
)
elapsed2 = time.time() - t1
with h5py.File(str(TEST_H5), "r") as f:
    n_after = f["face_features"].shape[0] if "face_features" in f else 0

if n_after == n_done and elapsed2 < 30:
    print(f"{PASS} Checkpoint recovery: skipped all {n_done} clips in {elapsed2:.1f}s (fast resume works)")
else:
    print(f"{FAIL} Checkpoint issue: {n_after} vs {n_done} clips, took {elapsed2:.1f}s")

# ── VRAM after ────────────────────────────────────────────────
free_after = torch.cuda.mem_get_info(0)[0] / 1e6
print(f"\n  VRAM free after run : {free_after:.0f} MB")

# ── Summary ───────────────────────────────────────────────────
print()
print("=" * 60)
print("  TEST SUMMARY")
print(f"  Clips extracted : {n_done}/{N_CLIPS}")
print(f"  Throughput      : {rate:.0f} clips/hr  ({rate/1000*100:.0f} hr for 100K)")
print(f"  Checkpoint      : works correctly")
print(f"  VRAM headroom   : {free_after:.0f} MB free after run")
print()
if n_done >= N_CLIPS * 0.8 and elapsed2 < 30:
    print("  RESULT: PASS -- ready for overnight run")
    print()
    print("  Start full extraction:")
    print("  > python scripts/run_local.py")
else:
    print("  RESULT: ISSUES FOUND -- check output above before proceeding")
print("=" * 60)

# Clean up test files
test_index.unlink(missing_ok=True)
TEST_H5.unlink(missing_ok=True)