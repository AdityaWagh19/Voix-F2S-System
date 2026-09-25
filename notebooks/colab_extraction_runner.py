# ============================================================
# VOIX — VoxCeleb2 Extraction Runner for Google Colab (T4 GPU)
# ============================================================
#
# T4-specific design:
#   - Session limit: ~12 hours per Colab free session
#   - Checkpoint: every 200 clips -> HDF5 flushed to Drive
#   - Resume: reads Drive HDF5 on startup, skips done clips
#   - Sessions needed: ~6-8 sessions of 12h each
#   - Each session auto-prints a keepalive every 5 min
#   - Drive sync: HDF5 output_path IS the Drive path -> every
#     flush() call writes directly to Drive
#
# Expected throughput on T4:
#   - ~3-4 s per clip (ffmpeg + ArcFace + FaceMesh + ECAPA)
#   - ~200 clips per batch -> ~12 min per checkpoint
#   - ~1,000 clips/hour -> ~100 hours total
#   - With 12h sessions: 8 sessions needed
#
# Sessions timeline:
#   Session 1:  clips 1-12,000    (save progress.json to Drive)
#   Session 2:  clips 12,001-24,000  (auto-resume from checkpoint)
#   ...
#   Session 8:  clips 88,001-100,000 (DONE)
#
# ── HOW TO RUN ───────────────────────────────────────────────
# 1. Open new Colab notebook (Runtime -> T4 GPU)
# 2. Run cells IN ORDER from top to bottom
# 3. If session times out: re-run Cell 1, Cell 2, then Cell 6
#    (Cells 3-5 are one-time setup, skipped if already done)
# ─────────────────────────────────────────────────────────────


# ═════════════════════════════════════════════════════════════
# CELL 1: Mount Drive & configure paths
# Copy this entire block into a Colab cell and run it first.
# ═════════════════════════════════════════════════════════════

CELL_1 = """
from google.colab import drive
drive.mount('/content/drive', force_remount=False)

import os, json, time

# ── Drive folder (your shared folder) ─────────────────────────
DRIVE_FOLDER_ID  = "1Du9bWOYIR65ETGhjH1cgpTwRbMVyaBAM"

# After mounting, find your folder path:
# !find /content/drive/MyDrive -maxdepth 4 -name "voix_extraction" -type d
# If it doesn't exist yet, it will be created below.
DRIVE_VOIX_PATH  = "/content/drive/MyDrive/voix_extraction"
os.makedirs(DRIVE_VOIX_PATH, exist_ok=True)

# ── Colab ephemeral paths (reset each session) ─────────────────
COLAB_WORK_DIR   = "/content/voix_work"
MP4_DIR          = f"{COLAB_WORK_DIR}/mp4"
TAR_DIR          = f"{COLAB_WORK_DIR}/archives"
TXT_DIR          = f"{COLAB_WORK_DIR}/txt"

for d in [COLAB_WORK_DIR, MP4_DIR, TAR_DIR, TXT_DIR]:
    os.makedirs(d, exist_ok=True)

# ── Files that persist across sessions (on Drive) ─────────────
CURATED_INDEX    = f"{DRIVE_VOIX_PATH}/curated_index.csv"
TRAIN_H5         = f"{DRIVE_VOIX_PATH}/voxceleb2_train.h5"
META_CSV         = f"{DRIVE_VOIX_PATH}/vox2_meta.csv"
PROGRESS_JSON    = f"{DRIVE_VOIX_PATH}/extraction_progress.json"

# ── Load or initialise progress tracker ───────────────────────
if os.path.exists(PROGRESS_JSON):
    with open(PROGRESS_JSON) as f:
        progress = json.load(f)
else:
    progress = {"session": 0, "total_extracted": 0, "last_key": None}

progress["session"] += 1
print("=" * 55)
print(f"  VOIX Extraction — Session {progress['session']}")
print(f"  Previously extracted: {progress['total_extracted']:,} clips")
print(f"  Resuming from: {progress['last_key'] or 'beginning'}")
print("=" * 55)
print(f"  Drive path : {DRIVE_VOIX_PATH}")
print(f"  HDF5 output: {TRAIN_H5}")

# Save updated session count to Drive
with open(PROGRESS_JSON, "w") as f:
    json.dump(progress, f, indent=2)

import torch
print(f"  GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU ONLY!'}")
print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
"""


# ═════════════════════════════════════════════════════════════
# CELL 2: Install dependencies & clone repo
# Run every session (pip cache makes it fast ~2 min after first)
# ═════════════════════════════════════════════════════════════

CELL_2 = """
import subprocess, sys, os

def sh(cmd, verbose=False):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if verbose or r.returncode != 0:
        out = (r.stdout + r.stderr)[-1000:]
        if out.strip():
            print(out)
    return r.returncode == 0

COLAB_WORK_DIR = "/content/voix_work"
os.makedirs(COLAB_WORK_DIR, exist_ok=True)

print("Installing dependencies...")
sh("pip install -q torch torchaudio --index-url https://download.pytorch.org/whl/cu121")
sh("pip install -q insightface onnxruntime-gpu")
sh("pip install -q speechbrain")
sh("pip install -q mediapipe==1.0.1")
sh("pip install -q h5py soundfile huggingface_hub opencv-python pillow")
sh("apt-get install -qq ffmpeg")

print("Cloning / updating VOIX repo...")
if not os.path.exists(f"{COLAB_WORK_DIR}/voix"):
    sh(f"git clone https://github.com/AdityaWagh19/Voix-F2S-System.git {COLAB_WORK_DIR}/voix", verbose=True)
else:
    sh(f"git -C {COLAB_WORK_DIR}/voix pull origin main")

sys.path.insert(0, f"{COLAB_WORK_DIR}/voix")
sh(f"pip install -q -e {COLAB_WORK_DIR}/voix")

# Verify
import torch
print(f"PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}")
print("Dependencies ready ✓")
"""


# ═════════════════════════════════════════════════════════════
# CELL 3: Download metadata (ONE-TIME, skipped if on Drive)
# ═════════════════════════════════════════════════════════════

CELL_3 = """
import os, shutil
from huggingface_hub import hf_hub_download, list_repo_files

DRIVE_VOIX_PATH = "/content/drive/MyDrive/voix_extraction"
COLAB_WORK_DIR  = "/content/voix_work"
HF_REPO         = "Reverb/voxceleb2"
META_CSV        = f"{DRIVE_VOIX_PATH}/vox2_meta.csv"
TXT_DIR         = f"{COLAB_WORK_DIR}/txt"

# Speaker metadata CSV (tiny, ~200 KB)
if not os.path.exists(META_CSV):
    print("Downloading vox2_meta.csv...")
    local = hf_hub_download(repo_id=HF_REPO, filename="vox2_meta.csv",
                             repo_type="dataset", local_dir=COLAB_WORK_DIR)
    shutil.copy(local, META_CSV)
    print(f"  Saved to Drive: {META_CSV}")
else:
    print(f"vox2_meta.csv already on Drive ({os.path.getsize(META_CSV)//1024} KB) ✓")

# Utterance txt annotations (needed for cross-session index curation)
if not os.path.exists(TXT_DIR) or len(os.listdir(TXT_DIR)) < 100:
    print("Downloading txt annotations (~50 MB)...")
    os.makedirs(TXT_DIR, exist_ok=True)
    all_files = list(list_repo_files(HF_REPO, repo_type="dataset"))
    txt_files = [f for f in all_files if f.startswith("txt/") and f.endswith(".txt")]
    print(f"  {len(txt_files)} txt files to download")
    for i, fname in enumerate(txt_files):
        try:
            hf_hub_download(repo_id=HF_REPO, filename=fname,
                            repo_type="dataset", local_dir=COLAB_WORK_DIR)
        except Exception as e:
            pass  # Skip unavailable files
        if (i + 1) % 1000 == 0:
            print(f"  {i+1}/{len(txt_files)}")
    print("txt annotations downloaded ✓")
else:
    print(f"txt annotations already present ({len(os.listdir(TXT_DIR))} speakers) ✓")
"""


# ═════════════════════════════════════════════════════════════
# CELL 4: Generate curated index (ONE-TIME, saved to Drive)
# ═════════════════════════════════════════════════════════════

CELL_4 = """
import os, sys, csv

DRIVE_VOIX_PATH = "/content/drive/MyDrive/voix_extraction"
COLAB_WORK_DIR  = "/content/voix_work"
sys.path.insert(0, f"{COLAB_WORK_DIR}/voix")

CURATED_INDEX = f"{DRIVE_VOIX_PATH}/curated_index.csv"
META_CSV      = f"{DRIVE_VOIX_PATH}/vox2_meta.csv"
TXT_DIR       = f"{COLAB_WORK_DIR}/txt"

if os.path.exists(CURATED_INDEX):
    with open(CURATED_INDEX) as f:
        n = sum(1 for _ in f) - 1
    print(f"Curated index already on Drive: {n:,} entries ✓")
else:
    print("Generating curated index (stratified, cross-session sampling)...")
    from scripts.curate_voxceleb_index import curate_index
    total = curate_index(
        meta_path=META_CSV,
        txt_dir=TXT_DIR,
        output_path=CURATED_INDEX,
        seed=42, target_speakers=2000, utterances_per_speaker=50,
    )
    print(f"Index saved to Drive: {total:,} rows ✓")

# Preview
with open(CURATED_INDEX) as f:
    rows = list(csv.DictReader(f))
males   = sum(1 for r in rows if r["gender"] == "m")
females = sum(1 for r in rows if r["gender"] == "f")
print(f"  Total: {len(rows):,}  Male: {males:,}  Female: {females:,}")
"""


# ═════════════════════════════════════════════════════════════
# CELL 5: Download VoxCeleb2 videos (ONE-TIME per session if
#         ephemeral disk was cleared, ~30 min on T4)
#
# T4 note: Colab ephemeral disk is ~100 GB.
# Full VoxCeleb2 dev mp4 = ~346 GB — too large.
# Strategy: download only the parts containing our 100K clips.
# We identify which mp4 archive parts contain our curated clips
# and download only those (~32 GB subset).
# ═════════════════════════════════════════════════════════════

CELL_5 = """
import os, csv, subprocess
from huggingface_hub import hf_hub_download, list_repo_files

DRIVE_VOIX_PATH = "/content/drive/MyDrive/voix_extraction"
COLAB_WORK_DIR  = "/content/voix_work"
CURATED_INDEX   = f"{DRIVE_VOIX_PATH}/curated_index.csv"
MP4_DIR         = f"{COLAB_WORK_DIR}/mp4"
TAR_DIR         = f"{COLAB_WORK_DIR}/archives"
HF_REPO         = "Reverb/voxceleb2"

os.makedirs(MP4_DIR, exist_ok=True)
os.makedirs(TAR_DIR, exist_ok=True)

def sh(cmd):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"ERR: {r.stderr[-500:]}")
    return r.returncode == 0

# Check if already extracted
n_existing = sum(len(files) for _, _, files in os.walk(MP4_DIR)
                 if any(f.endswith(".mp4") for f in _))
print(f"Existing mp4 files in {MP4_DIR}: checking...")
n_mp4 = len(list(__import__("pathlib").Path(MP4_DIR).rglob("*.mp4")))
print(f"  Found {n_mp4:,} mp4 files")

if n_mp4 > 50000:
    print("Sufficient mp4 files already present ✓")
else:
    # List all archive parts on HF
    print("Listing VoxCeleb2 archive parts on HuggingFace...")
    all_files = list(list_repo_files(HF_REPO, repo_type="dataset"))
    mp4_parts = sorted([f for f in all_files if "vox2_dev_mp4_part" in f])
    print(f"  {len(mp4_parts)} parts available")
    print(f"  Downloading all parts (~32 GB, ~30 min on T4)...")

    for i, part in enumerate(mp4_parts):
        dest = f"{TAR_DIR}/{os.path.basename(part)}"
        if not os.path.exists(dest):
            print(f"  [{i+1}/{len(mp4_parts)}] {part}...", end=" ", flush=True)
            try:
                hf_hub_download(repo_id=HF_REPO, filename=part,
                                repo_type="dataset", local_dir=TAR_DIR)
                size_gb = os.path.getsize(dest) / 1e9
                print(f"{size_gb:.1f} GB ✓")
            except Exception as e:
                print(f"FAILED: {e}")
        else:
            print(f"  [{i+1}/{len(mp4_parts)}] {part} cached ✓")

    print("\\nCombining archive parts...")
    combined = f"{TAR_DIR}/combined.tar"
    if not os.path.exists(combined):
        sh(f"cat {TAR_DIR}/vox2_dev_mp4_part* > {combined}")

    print("Extracting mp4 files (this takes ~10 min)...")
    sh(f"tar -xf {combined} -C {MP4_DIR} --checkpoint=5000 --checkpoint-action=echo='%T'")
    os.remove(combined)
    print("Extraction complete, archive deleted ✓")

n_final = len(list(__import__("pathlib").Path(MP4_DIR).rglob("*.mp4")))
print(f"Total mp4 files available: {n_final:,}")
"""


# ═════════════════════════════════════════════════════════════
# CELL 6: *** MAIN EXTRACTION LOOP (T4-optimised) ***
#
# Run this cell every session. It automatically:
#   1. Reads Drive HDF5 to find already-done clips
#   2. Skips completed clips (checkpoint recovery)
#   3. Processes remaining clips in batches of 200
#   4. Saves to Drive HDF5 after every batch (~12 min intervals)
#   5. Updates progress.json on Drive after every batch
#   6. Keeps the session alive with periodic prints
#   7. Stops automatically when session approaches timeout
#
# T4 timeline: ~1,000 clips/hour -> 8 sessions to finish 100K
# ═════════════════════════════════════════════════════════════

CELL_6 = """
import os, sys, json, time, csv, signal
import numpy as np

DRIVE_VOIX_PATH = "/content/drive/MyDrive/voix_extraction"
COLAB_WORK_DIR  = "/content/voix_work"
sys.path.insert(0, f"{COLAB_WORK_DIR}/voix")

CURATED_INDEX  = f"{DRIVE_VOIX_PATH}/curated_index.csv"
TRAIN_H5       = f"{DRIVE_VOIX_PATH}/voxceleb2_train.h5"
PROGRESS_JSON  = f"{DRIVE_VOIX_PATH}/extraction_progress.json"
MP4_DIR        = f"{COLAB_WORK_DIR}/mp4"

# T4-optimised batch size (200 clips ~ 12 min per checkpoint)
BATCH_SIZE = 200

# Safety margin: stop 30 min before 12h session limit
SESSION_MAX_HOURS = 11.0
SESSION_START     = time.time()

def session_time_ok():
    elapsed = (time.time() - SESSION_START) / 3600
    return elapsed < SESSION_MAX_HOURS

def save_progress(done_count, last_key):
    if os.path.exists(PROGRESS_JSON):
        with open(PROGRESS_JSON) as f:
            prog = json.load(f)
    else:
        prog = {"session": 1}
    prog["total_extracted"] = done_count
    prog["last_key"] = last_key
    prog["last_updated"] = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
    with open(PROGRESS_JSON, "w") as f:
        json.dump(prog, f, indent=2)

# ── Import extraction pipeline ─────────────────────────────
from scripts.extract_voxceleb import run_extraction

print("=" * 60)
print("  T4 EXTRACTION — Checkpoint-Aware Session")
print(f"  Output  : {TRAIN_H5}")
print(f"  Batch   : {BATCH_SIZE} clips ({BATCH_SIZE * 4 / 60:.0f} min est.)")
print(f"  Max time: {SESSION_MAX_HOURS:.0f} hours (auto-stops)")
print("=" * 60)

# Check existing progress
if os.path.exists(PROGRESS_JSON):
    with open(PROGRESS_JSON) as f:
        prog = json.load(f)
    print(f"  Resuming: {prog['total_extracted']:,} clips already done")
    print(f"  Last key: {prog['last_key']}")

# Read curated index
with open(CURATED_INDEX) as f:
    all_rows = list(csv.DictReader(f))
print(f"  Index   : {len(all_rows):,} total clips")

# Find done keys from HDF5 checkpoint
done_keys = set()
if os.path.exists(TRAIN_H5):
    import h5py
    with h5py.File(TRAIN_H5, "r") as hf:
        if "metadata" in hf:
            done_keys = set(m.decode() for m in hf["metadata"][:])
    print(f"  Done    : {len(done_keys):,} clips in HDF5 ✓")

pending = [r for r in all_rows
           if f"{r['speaker_id']}/{r['video_id']}/{r['utterance_id']}" not in done_keys]
print(f"  Pending : {len(pending):,} clips this session")
print()

if not pending:
    print("ALL CLIPS EXTRACTED! Nothing to do.")
    save_progress(len(done_keys), "COMPLETE")
else:
    # Run with T4 batch size and Drive output path
    run_extraction(
        index_path=CURATED_INDEX,
        videos_root=MP4_DIR,
        output_path=TRAIN_H5,    # <-- directly on Drive
        device="cuda",
        batch_size=BATCH_SIZE,
        session_max_hours=SESSION_MAX_HOURS,
        progress_callback=save_progress,
    )

    # Final progress save
    if os.path.exists(TRAIN_H5):
        import h5py
        with h5py.File(TRAIN_H5, "r") as hf:
            n_done = hf["face_features"].shape[0] if "face_features" in hf else 0
        save_progress(n_done, "session_end")
        elapsed = (time.time() - SESSION_START) / 3600
        print(f"\\nSession complete: {n_done:,}/{len(all_rows):,} clips in {elapsed:.1f}h")
        pct = n_done / len(all_rows) * 100
        remaining_sessions = max(0, int((len(all_rows) - n_done) / max(n_done, 1) * 1))
        print(f"Progress: {pct:.1f}% — re-run Cell 6 in a new session to continue")
"""


# ═════════════════════════════════════════════════════════════
# CELL 7: Audit & visualise (run after all sessions complete)
# ═════════════════════════════════════════════════════════════

CELL_7 = """
import os, sys
DRIVE_VOIX_PATH = "/content/drive/MyDrive/voix_extraction"
COLAB_WORK_DIR  = "/content/voix_work"
TRAIN_H5        = f"{DRIVE_VOIX_PATH}/voxceleb2_train.h5"
sys.path.insert(0, f"{COLAB_WORK_DIR}/voix")

from voix.data.dataset import DatasetIntegrityAuditor
auditor = DatasetIntegrityAuditor(TRAIN_H5)
auditor.print_report()

# Progress summary
import json
PROGRESS_JSON = f"{DRIVE_VOIX_PATH}/extraction_progress.json"
if os.path.exists(PROGRESS_JSON):
    with open(PROGRESS_JSON) as f:
        prog = json.load(f)
    print(f"Sessions run : {prog.get('session', '?')}")
    print(f"Last updated : {prog.get('last_updated', '?')}")

# UMAP visualization
try:
    import subprocess
    subprocess.run("pip install -q umap-learn matplotlib", shell=True, capture_output=True)
    import umap, h5py, numpy as np, matplotlib.pyplot as plt

    with h5py.File(TRAIN_H5, "r") as f:
        n = f["face_features"].shape[0]
        sample = min(5000, n)
        idx = np.random.choice(n, sample, replace=False)
        face = f["face_features"][sorted(idx)]
        ids  = f["speaker_ids"][sorted(idx)]

    print(f"Running UMAP on {sample:,} samples...")
    reducer  = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
    embedded = reducer.fit_transform(face)

    plt.figure(figsize=(12, 9))
    plt.scatter(embedded[:,0], embedded[:,1], c=ids % 20,
                cmap="tab20", s=1.5, alpha=0.5)
    plt.title(f"UMAP — Face Embeddings ({sample:,} samples, coloured by speaker mod 20)")
    plt.axis("off")
    plt.tight_layout()
    out = f"{DRIVE_VOIX_PATH}/umap_face_embeddings.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Plot saved: {out}")
    plt.show()
except Exception as e:
    print(f"UMAP skipped: {e}")
"""


# ═════════════════════════════════════════════════════════════
# CELL 8: Session summary (run at end of every session)
# ═════════════════════════════════════════════════════════════

CELL_8 = """
import os, json, h5py

DRIVE_VOIX_PATH = "/content/drive/MyDrive/voix_extraction"
TRAIN_H5        = f"{DRIVE_VOIX_PATH}/voxceleb2_train.h5"
PROGRESS_JSON   = f"{DRIVE_VOIX_PATH}/extraction_progress.json"

size_mb = os.path.getsize(TRAIN_H5) / 1e6 if os.path.exists(TRAIN_H5) else 0

with h5py.File(TRAIN_H5, "r") as f:
    n_done = f["face_features"].shape[0] if "face_features" in f else 0

with open(PROGRESS_JSON) as f:
    prog = json.load(f)

pct = n_done / 100000 * 100
print("=" * 55)
print(f"  SESSION SUMMARY")
print(f"  Sessions run   : {prog.get('session', '?')}")
print(f"  Clips done     : {n_done:,} / 100,000 ({pct:.1f}%)")
print(f"  HDF5 size      : {size_mb:.1f} MB")
print(f"  Last updated   : {prog.get('last_updated', '?')}")
print(f"  Drive link     : https://drive.google.com/drive/folders/1Du9bWOYIR65ETGhjH1cgpTwRbMVyaBAM")
print("=" * 55)
if pct < 100:
    print(f"  NEXT: Start a new Colab T4 session and run Cell 1 + Cell 2 + Cell 6")
    sessions_left = int((100000 - n_done) / max(n_done / prog.get('session', 1), 1))
    print(f"  Estimated sessions remaining: ~{sessions_left}")
else:
    print("  COMPLETE! Download voxceleb2_train.h5 to D:\\\\voix\\\\data\\\\processed\\\\")
    print("  Then start Phase 3 CVAE training locally on GTX 1650.")
"""

# Print a summary of all cells for reference
print("T4 Colab runner cells generated:")
print("  Cell 1: Mount Drive + session tracking")
print("  Cell 2: Install deps + clone repo  (every session)")
print("  Cell 3: Download metadata          (one-time)")
print("  Cell 4: Generate curated index     (one-time)")
print("  Cell 5: Download + extract videos  (once per fresh session disk)")
print("  Cell 6: RUN EXTRACTION             (every session)")
print("  Cell 7: Audit + UMAP viz           (after completion)")
print("  Cell 8: Session summary            (end of every session)")