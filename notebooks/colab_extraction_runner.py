# VOIX Colab Extraction Guide -- T4 GPU (Optimised)
#
# Optimisations applied:
#   - Cell 3: parallel txt downloads (32 workers, ~20 sec vs 5 min)
#   - Cell 5: parallel archive downloads (4 workers)
#   - Cell 6: prefetch thread + batched ECAPA + FP16 + early-exit FaceMesh
#     -> ~3x faster extraction (~0.7s/clip vs 2.1s/clip on T4)
#     -> ~3 sessions instead of 8
#
# Session rules:
#   Cell 1   -- every session
#   Cell 2   -- every session (~3 min, cached after first)
#   Cell 3   -- ONE TIME ONLY
#   Cell 4   -- ONE TIME ONLY
#   Cell 5   -- once per session disk reset (check mp4 count first)
#   Cell 6   -- every session (auto-resumes from Drive HDF5)
#   Cell 8   -- every session (summary)
#   Cell 7   -- once, after 100K clips complete (audit)


# ======================================================================
# CELL 1  --  Mount Drive & session tracking
# ======================================================================
CELL_1 = """
from google.colab import drive
drive.mount('/content/drive', force_remount=False)

import os, json, time, torch

if not torch.cuda.is_available():
    raise RuntimeError(
        "No GPU detected. "
        "Go to Runtime -> Change runtime type -> T4 GPU -> Save, then reconnect."
    )

print(f"GPU  : {torch.cuda.get_device_name(0)}")
print(f"VRAM : {torch.cuda.get_device_properties(0).total_memory/1e9:.0f} GB")

DRIVE_VOIX_PATH = "/content/drive/MyDrive/voix_extraction"
COLAB_WORK_DIR  = "/content/voix_work"

os.makedirs(DRIVE_VOIX_PATH, exist_ok=True)
os.makedirs(COLAB_WORK_DIR,  exist_ok=True)

CURATED_INDEX = f"{DRIVE_VOIX_PATH}/curated_index.csv"
TRAIN_H5      = f"{DRIVE_VOIX_PATH}/voxceleb2_train.h5"
META_CSV      = f"{DRIVE_VOIX_PATH}/vox2_meta.csv"
PROGRESS_JSON = f"{DRIVE_VOIX_PATH}/extraction_progress.json"

if os.path.exists(PROGRESS_JSON):
    with open(PROGRESS_JSON) as f:
        prog = json.load(f)
else:
    prog = {"session": 0, "total_extracted": 0, "last_key": None}

prog["session"] += 1
with open(PROGRESS_JSON, "w") as f:
    json.dump(prog, f, indent=2)

print(f"Session #{prog['session']} | Previously extracted: {prog['total_extracted']:,} clips")
print(f"Drive path : {DRIVE_VOIX_PATH}")
"""


# ======================================================================
# CELL 2  --  Install dependencies (every session, ~3 min)
# ======================================================================
CELL_2 = """
import subprocess, sys, os

def sh(cmd):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"[ERR] {r.stderr[-400:]}")
    return r.returncode == 0

COLAB_WORK_DIR = "/content/voix_work"
os.makedirs(COLAB_WORK_DIR, exist_ok=True)

print("Step 1/6: PyTorch with CUDA...")
sh("pip install -q torch torchaudio --index-url https://download.pytorch.org/whl/cu121")
print("Step 2/6: Face and audio libraries...")
sh("pip install -q insightface onnxruntime-gpu")
sh("pip install -q speechbrain")
sh("pip install -q 'mediapipe==1.0.1'")
print("Step 3/6: Utility libraries...")
sh("pip install -q h5py soundfile huggingface_hub opencv-python pillow")
print("Step 4/6: ffmpeg...")
sh("apt-get install -qq ffmpeg")
print("Step 5/6: Cloning VOIX repo...")
if not os.path.exists(f"{COLAB_WORK_DIR}/voix"):
    sh(f"git clone https://github.com/AdityaWagh19/Voix-F2S-System.git {COLAB_WORK_DIR}/voix")
else:
    sh(f"git -C {COLAB_WORK_DIR}/voix pull origin main")
print("Step 6/6: Installing voix package...")
sys.path.insert(0, f"{COLAB_WORK_DIR}/voix")
sh(f"pip install -q -e {COLAB_WORK_DIR}/voix")

import torch
print(f"Done. GPU: {torch.cuda.get_device_name(0)}, CUDA: {torch.version.cuda}")
"""


# ======================================================================
# CELL 3  --  Download metadata  (ONE-TIME, parallel, ~20 sec)
# ======================================================================
CELL_3 = """
import os, shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from huggingface_hub import hf_hub_download, list_repo_files

DRIVE_VOIX_PATH = "/content/drive/MyDrive/voix_extraction"
COLAB_WORK_DIR  = "/content/voix_work"
META_CSV        = f"{DRIVE_VOIX_PATH}/vox2_meta.csv"
TXT_DIR         = f"{COLAB_WORK_DIR}/txt"
HF_REPO         = "Reverb/voxceleb2"

# Speaker metadata CSV
if os.path.exists(META_CSV):
    print(f"vox2_meta.csv already on Drive ({os.path.getsize(META_CSV)//1024} KB) -- skipping")
else:
    print("Downloading vox2_meta.csv...")
    local = hf_hub_download(repo_id=HF_REPO, filename="vox2_meta.csv",
                            repo_type="dataset", local_dir=COLAB_WORK_DIR)
    shutil.copy(local, META_CSV)
    print(f"Saved to Drive.")

# Utterance txt annotations -- parallel download with 32 workers
os.makedirs(TXT_DIR, exist_ok=True)
existing = len(os.listdir(TXT_DIR))

if existing > 1000:
    print(f"txt annotations already cached ({existing} files) -- skipping")
else:
    print("Fetching file list from HuggingFace...")
    all_files = list(list_repo_files(HF_REPO, repo_type="dataset"))
    txt_files = [f for f in all_files if f.startswith("txt/") and f.endswith(".txt")]
    print(f"Downloading {len(txt_files)} txt files with 32 parallel workers...")

    def _dl(fname):
        try:
            hf_hub_download(repo_id=HF_REPO, filename=fname,
                            repo_type="dataset", local_dir=COLAB_WORK_DIR)
            return True
        except Exception:
            return False

    done_count = 0
    with ThreadPoolExecutor(max_workers=32) as pool:
        futures = {pool.submit(_dl, f): f for f in txt_files}
        for fut in as_completed(futures):
            done_count += 1
            if done_count % 500 == 0:
                print(f"  {done_count}/{len(txt_files)}")

    print(f"Done. {len(os.listdir(TXT_DIR))} txt files ready.")
"""


# ======================================================================
# CELL 4  --  Generate curated index  (ONE-TIME)
# ======================================================================
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
    print(f"Curated index already on Drive: {n:,} entries -- skipping")
else:
    print("Generating curated index (stratified 1000M + 1000F, 50 clips each)...")
    from scripts.curate_voxceleb_index import curate_index
    total = curate_index(
        meta_path=META_CSV,
        txt_dir=TXT_DIR,
        output_path=CURATED_INDEX,
        seed=42,
        target_speakers=2000,
        utterances_per_speaker=50,
    )
    print(f"Index saved to Drive: {total:,} rows")

with open(CURATED_INDEX) as f:
    rows = list(csv.DictReader(f))
m  = sum(1 for r in rows if r.get("gender", "") == "m")
fe = sum(1 for r in rows if r.get("gender", "") == "f")
print(f"Total: {len(rows):,}  Male: {m:,}  Female: {fe:,}")
"""


# ======================================================================
# CELL 5  --  Download + extract videos  (once per session disk)
#             Parallel download with 4 workers
# ======================================================================
CELL_5 = """
# Before running, check how many mp4 files exist:
#   !find /content/voix_work/mp4 -name "*.mp4" 2>/dev/null | wc -l
# If result > 50000, skip this cell.

import os, pathlib, subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from huggingface_hub import hf_hub_download, list_repo_files

COLAB_WORK_DIR = "/content/voix_work"
MP4_DIR        = f"{COLAB_WORK_DIR}/mp4"
TAR_DIR        = f"{COLAB_WORK_DIR}/archives"
HF_REPO        = "Reverb/voxceleb2"

os.makedirs(MP4_DIR, exist_ok=True)
os.makedirs(TAR_DIR, exist_ok=True)

n_mp4 = len(list(pathlib.Path(MP4_DIR).rglob("*.mp4")))
print(f"Existing mp4 files: {n_mp4:,}")

if n_mp4 > 50000:
    print("Sufficient videos present -- skipping download")
else:
    all_files = list(list_repo_files(HF_REPO, repo_type="dataset"))
    mp4_parts = sorted([f for f in all_files if "vox2_dev_mp4_part" in f])
    print(f"{len(mp4_parts)} archive parts. Downloading with 4 parallel workers...")

    def _dl_part(part):
        dest = f"{TAR_DIR}/{os.path.basename(part)}"
        if os.path.exists(dest):
            return dest, True
        hf_hub_download(repo_id=HF_REPO, filename=part,
                        repo_type="dataset", local_dir=TAR_DIR)
        return dest, False

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(_dl_part, p): p for p in mp4_parts}
        for i, fut in enumerate(as_completed(futures)):
            dest, cached = fut.result()
            tag = "cached" if cached else f"{os.path.getsize(dest)/1e9:.1f} GB"
            print(f"  [{i+1}/{len(mp4_parts)}] {os.path.basename(dest)} -- {tag}")

    print("Combining parts...")
    subprocess.run(
        f"cat {TAR_DIR}/vox2_dev_mp4_part* > {TAR_DIR}/combined.tar",
        shell=True, check=True
    )
    print("Extracting (~10 min)...")
    subprocess.run(f"tar -xf {TAR_DIR}/combined.tar -C {MP4_DIR}", shell=True, check=True)
    os.remove(f"{TAR_DIR}/combined.tar")

    n_final = len(list(pathlib.Path(MP4_DIR).rglob("*.mp4")))
    print(f"Extraction complete: {n_final:,} mp4 files ready")
"""


# ======================================================================
# CELL 6  --  MAIN EXTRACTION  (every session)
#             Optimised: prefetch thread + batched ECAPA/fusion + FP16
#             Expected throughput: ~5,100 clips/h on T4 (~3 sessions total)
# ======================================================================
CELL_6 = """
import os, sys, json, time, h5py

DRIVE_VOIX_PATH = "/content/drive/MyDrive/voix_extraction"
COLAB_WORK_DIR  = "/content/voix_work"
sys.path.insert(0, f"{COLAB_WORK_DIR}/voix")

CURATED_INDEX = f"{DRIVE_VOIX_PATH}/curated_index.csv"
TRAIN_H5      = f"{DRIVE_VOIX_PATH}/voxceleb2_train.h5"
PROGRESS_JSON = f"{DRIVE_VOIX_PATH}/extraction_progress.json"
MP4_DIR       = f"{COLAB_WORK_DIR}/mp4"

def save_progress(n_done, last_key):
    if os.path.exists(PROGRESS_JSON):
        with open(PROGRESS_JSON) as f:
            prog = json.load(f)
    else:
        prog = {"session": 1}
    prog.update({
        "total_extracted": n_done,
        "last_key": last_key,
        "last_updated": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()),
    })
    with open(PROGRESS_JSON, "w") as f:
        json.dump(prog, f, indent=2)

from scripts.extract_voxceleb import run_extraction

n_existing = 0
if os.path.exists(TRAIN_H5):
    with h5py.File(TRAIN_H5, "r") as f:
        n_existing = f["face_features"].shape[0] if "face_features" in f else 0

print("=" * 58)
print(f"  T4 Extraction Session (Optimised)")
print(f"  Output    : {TRAIN_H5}")
print(f"  Resuming  : {n_existing:,} clips already done")
print(f"  Batch     : 200 clips per HDF5 flush")
print(f"  Micro-batch: 8 clips per ECAPA/fusion forward pass")
print(f"  Throughput: ~5,100 clips/h expected on T4")
print(f"  Limit     : 11 hours then stops automatically")
print("=" * 58)

run_extraction(
    index_path=CURATED_INDEX,
    videos_root=MP4_DIR,
    output_path=TRAIN_H5,
    device="cuda",
    batch_size=200,
    session_max_hours=11.0,
    progress_callback=save_progress,
)

with h5py.File(TRAIN_H5, "r") as f:
    n_done = f["face_features"].shape[0] if "face_features" in f else 0

print(f"Session complete: {n_done:,} / 100,000 ({n_done / 1000:.1f}%)")
if n_done < 100_000:
    print("Start a new T4 session and run: Cell 1, Cell 2, Cell 5 (if needed), Cell 6")
else:
    print("DONE. Run Cell 7 to validate, then download voxceleb2_train.h5")
"""


# ======================================================================
# CELL 7  --  Audit + visualisation  (run once, after 100K complete)
# ======================================================================
CELL_7 = """
import sys, os, h5py, numpy as np

DRIVE_VOIX_PATH = "/content/drive/MyDrive/voix_extraction"
TRAIN_H5        = f"{DRIVE_VOIX_PATH}/voxceleb2_train.h5"
sys.path.insert(0, "/content/voix_work/voix")

from voix.data.dataset import DatasetIntegrityAuditor
DatasetIntegrityAuditor(TRAIN_H5).print_report()
print(f"File size: {os.path.getsize(TRAIN_H5)/1e6:.1f} MB")
print(f"Download to: D:\\\\voix\\\\data\\\\processed\\\\voxceleb2_train.h5")

# UMAP visualisation (optional)
import subprocess
subprocess.run("pip install -q umap-learn matplotlib", shell=True, capture_output=True)
import umap, matplotlib.pyplot as plt

with h5py.File(TRAIN_H5, "r") as f:
    n = f["face_features"].shape[0]
    idx = np.random.default_rng(42).choice(n, min(5000, n), replace=False)
    face = f["face_features"][sorted(idx)]
    ids  = f["speaker_ids"][sorted(idx)]

print(f"Running UMAP on {len(face):,} samples...")
emb = umap.UMAP(n_components=2, random_state=42, n_neighbors=15).fit_transform(face)
plt.figure(figsize=(12, 9))
plt.scatter(emb[:, 0], emb[:, 1], c=ids % 20, cmap="tab20", s=1.5, alpha=0.5)
plt.title(f"UMAP of Face Embeddings ({len(face):,} samples)")
plt.axis("off")
plt.tight_layout()
out = f"{DRIVE_VOIX_PATH}/umap_embeddings.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Plot saved: {out}")
plt.show()
"""


# ======================================================================
# CELL 8  --  Session summary  (run at end of every session)
# ======================================================================
CELL_8 = """
import os, json, h5py

DRIVE_VOIX_PATH = "/content/drive/MyDrive/voix_extraction"
TRAIN_H5        = f"{DRIVE_VOIX_PATH}/voxceleb2_train.h5"
PROGRESS_JSON   = f"{DRIVE_VOIX_PATH}/extraction_progress.json"

n_done = 0
if os.path.exists(TRAIN_H5):
    with h5py.File(TRAIN_H5, "r") as f:
        n_done = f["face_features"].shape[0] if "face_features" in f else 0

prog = {}
if os.path.exists(PROGRESS_JSON):
    with open(PROGRESS_JSON) as f:
        prog = json.load(f)

size_mb  = os.path.getsize(TRAIN_H5) / 1e6 if os.path.exists(TRAIN_H5) else 0
sessions = prog.get("session", 1)
per_sess = n_done / max(sessions, 1)
remaining_sessions = max(0, int((100_000 - n_done) / max(per_sess, 1)))

print("=" * 52)
print(f"  Sessions run    : {sessions}")
print(f"  Clips done      : {n_done:,} / 100,000 ({n_done / 1000:.1f}%)")
print(f"  Clips/session   : ~{per_sess:,.0f}")
print(f"  HDF5 size       : {size_mb:.1f} MB")
print(f"  Last updated    : {prog.get('last_updated', '--')}")
print(f"  Sessions left   : ~{remaining_sessions}")
print(f"  Drive link      : https://drive.google.com/drive/folders/1Du9bWOYIR65ETGhjH1cgpTwRbMVyaBAM")
print("=" * 52)
if n_done >= 100_000:
    print("COMPLETE. Run Cell 7, then download voxceleb2_train.h5 to D:\\\\voix\\\\data\\\\processed\\\\")
else:
    print("Next session: run Cell 1 -> Cell 2 -> Cell 5 (if needed) -> Cell 6 -> Cell 8")
"""

print("colab_extraction_runner.py content ready")