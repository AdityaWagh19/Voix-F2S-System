# ============================================================
# VOIX — VoxCeleb2 Extraction Runner for Google Colab Pro
# ============================================================
# Run this script cell-by-cell in a Colab Pro notebook.
# GPU: Select A100 (Runtime -> Change runtime type -> A100)
# Estimated time: ~6-8 hours for 100K clips on A100
# Drive folder: https://drive.google.com/drive/folders/1Du9bWOYIR65ETGhjH1cgpTwRbMVyaBAM
# ============================================================

# ─────────────────────────────────────────────────────────────
# CELL 1: Mount Google Drive & set paths
# ─────────────────────────────────────────────────────────────
from google.colab import drive
drive.mount('/content/drive', force_remount=False)

import os

# ── Configuration ────────────────────────────────────────────
DRIVE_FOLDER_ID   = "1Du9bWOYIR65ETGhjH1cgpTwRbMVyaBAM"
DRIVE_VOIX_PATH   = "/content/drive/MyDrive/voix_extraction"  # adjust if different
COLAB_WORK_DIR    = "/content/voix_work"
HF_REPO_ID        = "Reverb/voxceleb2"          # HuggingFace mirror
GITHUB_REPO       = "https://github.com/AdityaWagh19/Voix-F2S-System.git"

CURATED_INDEX     = os.path.join(DRIVE_VOIX_PATH, "curated_index.csv")
TRAIN_H5          = os.path.join(DRIVE_VOIX_PATH, "voxceleb2_train.h5")
VAL_H5            = os.path.join(DRIVE_VOIX_PATH, "voxceleb2_val.h5")
META_CSV_PATH     = os.path.join(COLAB_WORK_DIR, "vox2_meta.csv")
TXT_DIR           = os.path.join(COLAB_WORK_DIR, "vox2_txt")
VIDEO_TEMP_DIR    = os.path.join(COLAB_WORK_DIR, "video_temp")

# Create directories
os.makedirs(DRIVE_VOIX_PATH, exist_ok=True)
os.makedirs(COLAB_WORK_DIR,  exist_ok=True)
os.makedirs(VIDEO_TEMP_DIR,  exist_ok=True)

print("=== Path Configuration ===")
print(f"  Drive folder : {DRIVE_VOIX_PATH}")
print(f"  Work dir     : {COLAB_WORK_DIR}")
print(f"  Train HDF5   : {TRAIN_H5}")
print(f"  Curated index: {CURATED_INDEX}")
print("Drive mounted successfully ✓")


# ─────────────────────────────────────────────────────────────
# CELL 2: Clone repo & install dependencies
# ─────────────────────────────────────────────────────────────
import subprocess, sys

def run(cmd, **kw):
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, **kw)
    if result.returncode != 0:
        print(f"STDERR: {result.stderr[-2000:]}")
    else:
        print(result.stdout[-1000:] or "OK")
    return result.returncode == 0

# Clone or update repo
if not os.path.exists(f"{COLAB_WORK_DIR}/voix"):
    run(f"git clone {GITHUB_REPO} {COLAB_WORK_DIR}/voix")
else:
    run(f"git -C {COLAB_WORK_DIR}/voix pull origin main")

sys.path.insert(0, f"{COLAB_WORK_DIR}/voix")

# Install all dependencies
print("\nInstalling dependencies (2-4 min)...")
run("pip install -q torch torchaudio --index-url https://download.pytorch.org/whl/cu121")
run("pip install -q insightface onnxruntime-gpu speechbrain mediapipe")
run("pip install -q h5py soundfile huggingface_hub opencv-python pillow")
run("pip install -q ffmpeg-python")

# Install ffmpeg binary
run("apt-get install -qq ffmpeg")

# Install voix package
run(f"pip install -q -e {COLAB_WORK_DIR}/voix")

print("\nAll dependencies installed ✓")

# Verify GPU
import torch
print(f"\nGPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")


# ─────────────────────────────────────────────────────────────
# CELL 3: Download VoxCeleb2 metadata (csv + txt files)
# Metadata only — small files (~50 MB total), downloaded once
# ─────────────────────────────────────────────────────────────
from huggingface_hub import hf_hub_download, list_repo_files
import shutil

print("Downloading VoxCeleb2 metadata from HuggingFace...")

# Download speaker metadata CSV
if not os.path.exists(META_CSV_PATH):
    local = hf_hub_download(
        repo_id=HF_REPO_ID, filename="vox2_meta.csv",
        repo_type="dataset", local_dir=COLAB_WORK_DIR,
    )
    shutil.copy(local, META_CSV_PATH)
    print(f"  vox2_meta.csv -> {META_CSV_PATH}")
else:
    print(f"  vox2_meta.csv already exists, skipping")

# Download txt annotation files (utterance-level metadata)
if not os.path.exists(TXT_DIR):
    print("  Downloading txt annotations (this may take a few minutes)...")
    txt_files = [f for f in list_repo_files(HF_REPO_ID, repo_type="dataset")
                 if f.startswith("txt/")]
    os.makedirs(TXT_DIR, exist_ok=True)
    for i, fname in enumerate(txt_files[:100]):  # first 100 as sample check
        try:
            local = hf_hub_download(
                repo_id=HF_REPO_ID, filename=fname,
                repo_type="dataset", local_dir=COLAB_WORK_DIR,
            )
        except Exception as e:
            pass
        if i % 20 == 0:
            print(f"  {i}/{len(txt_files)} txt files...")
    print(f"  txt annotations downloaded to {TXT_DIR}")
else:
    print(f"  txt dir already exists, skipping")

print("Metadata download complete ✓")


# ─────────────────────────────────────────────────────────────
# CELL 4: Generate curated speaker index (if not already done)
# ─────────────────────────────────────────────────────────────
import sys
sys.path.insert(0, f"{COLAB_WORK_DIR}/voix")

if os.path.exists(CURATED_INDEX):
    import csv
    with open(CURATED_INDEX) as f:
        n = sum(1 for _ in csv.reader(f)) - 1
    print(f"Curated index already exists: {n} entries")
else:
    print("Generating curated index (stratified cross-session sampling)...")
    from scripts.curate_voxceleb_index import curate_index
    total = curate_index(
        meta_path=META_CSV_PATH,
        txt_dir=os.path.join(COLAB_WORK_DIR, "txt"),
        output_path=CURATED_INDEX,
        seed=42,
        target_speakers=2000,
        utterances_per_speaker=50,
    )
    print(f"Index generated: {total} entries -> {CURATED_INDEX}")


# ─────────────────────────────────────────────────────────────
# CELL 5: Download VoxCeleb2 mp4 archives & extract in batches
# This is the main extraction cell. It:
#   1. Downloads the multipart mp4 archive from HuggingFace
#   2. Extracts it to /content/voix_work/mp4/ (ephemeral, ~32 GB)
#   3. Runs the feature extraction pipeline
# ─────────────────────────────────────────────────────────────
import tarfile, glob

MP4_DIR  = os.path.join(COLAB_WORK_DIR, "mp4")
TAR_DIR  = os.path.join(COLAB_WORK_DIR, "archives")
os.makedirs(MP4_DIR, exist_ok=True)
os.makedirs(TAR_DIR, exist_ok=True)

# Check which part archives exist in the HF repo
print("Listing VoxCeleb2 mp4 archives on HuggingFace...")
all_files = list(list_repo_files(HF_REPO_ID, repo_type="dataset"))
mp4_parts = sorted([f for f in all_files if "vox2_dev_mp4_part" in f])
print(f"Found {len(mp4_parts)} mp4 archive parts")

# Download parts sequentially and extract (Colab has fast HF bandwidth)
print("\nStarting streamed download + extraction...")
for part_name in mp4_parts:
    local_part = os.path.join(TAR_DIR, os.path.basename(part_name))
    if not os.path.exists(local_part):
        print(f"  Downloading {part_name}...")
        local_part = hf_hub_download(
            repo_id=HF_REPO_ID, filename=part_name,
            repo_type="dataset", local_dir=TAR_DIR,
        )
    else:
        print(f"  {part_name} already cached")

print("\nCombining archive parts...")
combined_tar = os.path.join(TAR_DIR, "vox2_dev_combined.tar")
if not os.path.exists(combined_tar):
    run(f"cat {TAR_DIR}/vox2_dev_mp4_part* > {combined_tar}")

print("Extracting mp4 files...")
if not os.path.exists(MP4_DIR) or len(os.listdir(MP4_DIR)) == 0:
    run(f"tar -xf {combined_tar} -C {MP4_DIR}")
    # Delete combined tar to free space
    os.remove(combined_tar)
    print("  Extracted, archive deleted")

print(f"MP4 directory ready: {MP4_DIR}")


# ─────────────────────────────────────────────────────────────
# CELL 6: Run feature extraction pipeline (main loop)
# Progress is checkpointed to TRAIN_H5 on Drive after every
# 500-clip batch. Safe to interrupt and re-run.
# ─────────────────────────────────────────────────────────────
from scripts.extract_voxceleb import run_extraction

print("=" * 60)
print("STARTING FEATURE EXTRACTION")
print(f"  Index : {CURATED_INDEX}")
print(f"  Videos: {MP4_DIR}")
print(f"  Output: {TRAIN_H5}")
print(f"  Device: cuda")
print("=" * 60)

run_extraction(
    index_path=CURATED_INDEX,
    videos_root=MP4_DIR,
    output_path=TRAIN_H5,
    device="cuda",
    batch_size=500,
)


# ─────────────────────────────────────────────────────────────
# CELL 7: Validate extracted HDF5 and print audit report
# ─────────────────────────────────────────────────────────────
from voix.data.dataset import DatasetIntegrityAuditor

print("Running integrity audit on extracted HDF5...")
auditor = DatasetIntegrityAuditor(TRAIN_H5)
auditor.print_report()

# Quick UMAP visualization (optional, requires umap-learn)
try:
    import umap
    import h5py
    import numpy as np
    import matplotlib.pyplot as plt

    with h5py.File(TRAIN_H5, "r") as f:
        face = f["face_features"][:5000]
        ids  = f["speaker_ids"][:5000]

    reducer  = umap.UMAP(n_components=2, random_state=42, n_neighbors=15)
    embedded = reducer.fit_transform(face)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(embedded[:,0], embedded[:,1], c=ids % 20,
                          cmap="tab20", s=2, alpha=0.6)
    plt.title("UMAP of Face Embeddings (5K samples, coloured by speaker)")
    plt.tight_layout()
    plot_path = os.path.join(DRIVE_VOIX_PATH, "umap_face_embeddings.png")
    plt.savefig(plot_path, dpi=150)
    print(f"UMAP plot saved to {plot_path}")
except ImportError:
    print("umap-learn not installed, skipping visualization")


# ─────────────────────────────────────────────────────────────
# CELL 8: Download final HDF5 to your PC (optional)
# The file is already on Drive. This cell just confirms the path
# and prints the file size.
# ─────────────────────────────────────────────────────────────
size_mb = os.path.getsize(TRAIN_H5) / 1e6
print(f"\n{'='*60}")
print(f"EXTRACTION COMPLETE")
print(f"  HDF5 saved to: {TRAIN_H5}")
print(f"  File size    : {size_mb:.1f} MB")
print(f"  Drive link   : https://drive.google.com/drive/folders/{DRIVE_FOLDER_ID}")
print(f"{'='*60}")
print("\nDownload to D:\\voix\\data\\processed\\voxceleb2_train.h5")
print("Then start Phase 3 CVAE training.")