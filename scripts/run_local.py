"""Local extraction runner for GTX 1650 (4 GB VRAM).

Wraps run_extraction with:
  - Windows sleep prevention (keeps PC awake during overnight runs)
  - Checkpoint every 50 clips (~2.5 min on GTX 1650)
  - Progress bar to terminal
  - Estimated completion time printed at each checkpoint
  - Clean Ctrl+C handling (flushes HDF5 before exit)
  - Log file at data/processed/extraction.log

Usage:
    python scripts/run_local.py

Paths default to D:\voix layout. Edit the CONFIG block below if needed.

Expected throughput on GTX 1650:
    ~1,200 clips/hr -> 100K clips in ~83 hours (~4 overnight runs of 20h)

Checkpoint safety:
    Progress is saved to HDF5 every 50 clips. If you Ctrl+C or the PC
    restarts, re-running this script resumes from exactly where it stopped.
    Zero compute is wasted.
"""

from __future__ import annotations

import ctypes
import json
import os
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# CONFIG -- edit these if your paths differ
# ---------------------------------------------------------------------------
VOIX_ROOT     = Path(r"D:\voix")
VIDEOS_ROOT   = VOIX_ROOT / "data" / "raw"  / "voxceleb2" / "dev" / "mp4"
INDEX_PATH    = VOIX_ROOT / "data" / "processed" / "curated_index.csv"
OUTPUT_H5     = VOIX_ROOT / "data" / "processed" / "voxceleb2_train.h5"
PROGRESS_JSON = VOIX_ROOT / "data" / "processed" / "extraction_progress.json"
LOG_FILE      = VOIX_ROOT / "data" / "processed" / "extraction.log"

# GTX 1650 (TU117, 896 CUDA cores, CUDA 7.5, 4.29 GB VRAM):
#   - Display driver uses ~855 MB permanently -> 3.46 GB free
#   - Models (ArcFace + ECAPA + FusionLayer) ~500 MB -> ~2.1 GB working room
#   - N_ECAPA_BATCH=4 and N_FACE_BATCH=4 are set in extract_voxceleb.py
#   - batch_size=50 -> HDF5 flush every ~2.5 min on GTX 1650
# i5-11320H: 4 cores / 8 threads. Prefetch thread uses 1 thread, leaving
#   7 for Python + OS. PREFETCH_QUEUE=2 in extract_voxceleb.py (RAM-safe).
# RAM: 8 GB single-channel. CLOSE CHROME AND UNNECESSARY APPS BEFORE RUNNING.
#   Python + models need ~3-4 GB free RAM. See RAM warning at startup.
BATCH_SIZE  = 50
DEVICE      = "cuda"
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Windows sleep prevention
# ---------------------------------------------------------------------------
# ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED
_ES_CONTINUOUS        = 0x80000000
_ES_SYSTEM_REQUIRED   = 0x00000001
_ES_DISPLAY_REQUIRED  = 0x00000002

def _prevent_sleep() -> None:
    """Tell Windows not to sleep while this process is running."""
    if os.name == "nt":
        ctypes.windll.kernel32.SetThreadExecutionState(
            _ES_CONTINUOUS | _ES_SYSTEM_REQUIRED
        )
        print("[sleep prevention] Windows sleep disabled for this session.")

def _allow_sleep() -> None:
    """Re-enable Windows sleep."""
    if os.name == "nt":
        ctypes.windll.kernel32.SetThreadExecutionState(_ES_CONTINUOUS)
        print("[sleep prevention] Windows sleep re-enabled.")


# ---------------------------------------------------------------------------
# Progress tracking
# ---------------------------------------------------------------------------

def _load_progress() -> dict:
    if PROGRESS_JSON.exists():
        with open(PROGRESS_JSON) as f:
            return json.load(f)
    return {"run": 0, "total_extracted": 0}


def _save_progress(n_done: int, last_key: str | None) -> None:
    prog = _load_progress()
    prog.update({
        "total_extracted": n_done,
        "last_key": last_key,
        "last_updated": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
    })
    with open(PROGRESS_JSON, "w") as f:
        json.dump(prog, f, indent=2)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

class _Tee:
    """Writes stdout to both terminal and log file simultaneously."""
    def __init__(self, log_path: Path):
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(log_path, "a", encoding="utf-8")
        self._terminal = sys.stdout

    def write(self, msg):
        self._terminal.write(msg)
        self._file.write(msg)
        self._file.flush()

    def flush(self):
        self._terminal.flush()
        self._file.flush()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # Redirect stdout to terminal + log file
    sys.stdout = _Tee(LOG_FILE)

    print("=" * 60)
    print("  VOIX -- Local Feature Extraction (GTX 1650)")
    print(f"  Started : {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Videos  : {VIDEOS_ROOT}")
    print(f"  Index   : {INDEX_PATH}")
    print(f"  Output  : {OUTPUT_H5}")
    print(f"  Batch   : {BATCH_SIZE} clips per checkpoint (~2.5 min each)")
    print("=" * 60)

    # Validate paths
    if not INDEX_PATH.exists():
        print(f"\n[ERROR] Index not found: {INDEX_PATH}")
        print("Run first:  python scripts/curate_voxceleb_index.py --help")
        sys.exit(1)

    if not VIDEOS_ROOT.exists() or not any(VIDEOS_ROOT.rglob("*.mp4")):
        print(f"\n[ERROR] No mp4 files found in: {VIDEOS_ROOT}")
        print("Run first:  python scripts/download_voxceleb.py")
        sys.exit(1)

    # RAM check
    import psutil
    ram = psutil.virtual_memory()
    print(f"  RAM free    : {ram.available/1e9:.1f} GB / {ram.total/1e9:.1f} GB ({ram.percent:.0f}% used)")
    if ram.available < 3.5e9:
        print()
        print("  [WARNING] Less than 3.5 GB RAM free.")
        print("  Close Chrome, other browsers, and background apps before proceeding.")
        print("  SpeechBrain + MediaPipe need ~2-3 GB RAM. Low RAM will cause swap and slow down extraction 5-10x.")
        print("  Press Enter to continue anyway, or Ctrl+C to exit and free RAM first.")
        try:
            input("  > ")
        except KeyboardInterrupt:
            print("\nExiting. Free RAM and re-run.")
            sys.exit(0)

    # Load previous progress
    prog = _load_progress()
    prog["run"] = prog.get("run", 0) + 1
    _save_progress(prog.get("total_extracted", 0), prog.get("last_key"))
    print(f"  Run #{prog['run']} | Previously extracted: {prog.get('total_extracted', 0):,} clips")

    # Add project root to path
    sys.path.insert(0, str(VOIX_ROOT))

    # Prevent Windows sleep
    _prevent_sleep()

    from scripts.extract_voxceleb import run_extraction

    # CUDA tuning for GTX 1650
    import torch
    torch.backends.cudnn.benchmark     = True   # profile fastest kernels (+10-20%)
    torch.backends.cudnn.deterministic = False
    # Pre-warm CUDA context (avoids slow first-batch kernel JIT compilation)
    _dummy = torch.zeros(1, device=DEVICE)
    del _dummy
    torch.cuda.empty_cache()
    free_vram, total_vram = torch.cuda.mem_get_info(0)
    print(f"  VRAM free   : {free_vram/1e6:.0f} MB / {total_vram/1e6:.0f} MB")
    if free_vram < 2.0e9:
        print("  [WARN] Less than 2 GB VRAM free. Consider closing other GPU apps.")

    try:
        run_extraction(
            index_path=str(INDEX_PATH),
            videos_root=str(VIDEOS_ROOT),
            output_path=str(OUTPUT_H5),
            device=DEVICE,
            batch_size=BATCH_SIZE,
            session_max_hours=999.0,       # no session limit locally
            progress_callback=_save_progress,
        )
    except KeyboardInterrupt:
        print("\n\n[Ctrl+C] Stopping cleanly. Progress is saved.")
    finally:
        _allow_sleep()

    # Final summary
    import h5py
    if OUTPUT_H5.exists():
        with h5py.File(str(OUTPUT_H5), "r") as f:
            n = f["face_features"].shape[0] if "face_features" in f else 0
        size_mb = OUTPUT_H5.stat().st_size / 1e6
        print(f"\nFinal count : {n:,} / 100,000 clips ({n/1000:.1f}%)")
        print(f"HDF5 size   : {size_mb:.1f} MB")
        _save_progress(n, "run_end")

    print(f"Stopped at  : {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Log saved to: {LOG_FILE}")


if __name__ == "__main__":
    main()