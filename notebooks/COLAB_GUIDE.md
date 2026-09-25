# VOIX — Colab Extraction Guide

Complete step-by-step instructions for running the VoxCeleb2 feature
extraction on Google Colab Pro using your Google Drive folder.

---

## Your Google Drive Folder

**Link:** https://drive.google.com/drive/folders/1Du9bWOYIR65ETGhjH1cgpTwRbMVyaBAM  
**Folder ID:** `1Du9bWOYIR65ETGhjH1cgpTwRbMVyaBAM`

The extraction script will write directly into this folder:
- `curated_index.csv` — the 100K speaker/utterance index (generated once)
- `voxceleb2_train.h5` — final 280 MB embedding file (what you need)
- `umap_face_embeddings.png` — validation plot

---

## Step 1: Open a New Colab Notebook

1. Go to https://colab.research.google.com
2. **File → New notebook**
3. **Runtime → Change runtime type → A100 GPU** (requires Colab Pro)
4. Click **Connect**

---

## Step 2: Run Each Cell

Copy each block below into a separate Colab cell and run them **in order**.
Each cell is independent — if a session times out, re-run from Cell 1 onwards.
The checkpoint in the HDF5 file on Drive ensures no work is repeated.

---

### Cell 1 — Mount Drive & Configure Paths

```python
from google.colab import drive
drive.mount('/content/drive', force_remount=False)

import os

DRIVE_FOLDER_ID = "1Du9bWOYIR65ETGhjH1cgpTwRbMVyaBAM"

# ⚠️  Find the exact path to your shared folder after mounting.
# Run: !find /content/drive -name "voix_extraction" -type d
# Then set DRIVE_VOIX_PATH to that path.
# If the folder does not exist yet, create it:
DRIVE_VOIX_PATH = "/content/drive/MyDrive/voix_extraction"
os.makedirs(DRIVE_VOIX_PATH, exist_ok=True)

COLAB_WORK_DIR = "/content/voix_work"
os.makedirs(COLAB_WORK_DIR, exist_ok=True)

print(f"Drive path  : {DRIVE_VOIX_PATH}")
print(f"Work dir    : {COLAB_WORK_DIR}")
!ls /content/drive/MyDrive/
```

> **If your folder appears at a different path**, update `DRIVE_VOIX_PATH` accordingly.
> Run `!find /content/drive/MyDrive -maxdepth 3 -type d` to locate it.

---

### Cell 2 — Install Dependencies

```python
import subprocess, sys

def sh(cmd):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print(r.stdout[-500:] if r.stdout else r.stderr[-500:])

# PyTorch with CUDA 12.1 (matches A100)
sh("pip install -q torch torchaudio --index-url https://download.pytorch.org/whl/cu121")

# Backbone libraries
sh("pip install -q insightface onnxruntime-gpu")
sh("pip install -q speechbrain mediapipe==1.0.1")
sh("pip install -q h5py soundfile huggingface_hub opencv-python pillow")

# ffmpeg binary
sh("apt-get install -qq ffmpeg")

# Clone VOIX repo
if not os.path.exists(f"{COLAB_WORK_DIR}/voix"):
    sh(f"git clone https://github.com/AdityaWagh19/Voix-F2S-System.git {COLAB_WORK_DIR}/voix")
else:
    sh(f"git -C {COLAB_WORK_DIR}/voix pull origin main")

sys.path.insert(0, f"{COLAB_WORK_DIR}/voix")
sh(f"pip install -q -e {COLAB_WORK_DIR}/voix")

import torch
print(f"\nGPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
print("✓ Dependencies ready")
```

Expected output: `GPU: NVIDIA A100-SXM4-40GB` and `VRAM: 40.0 GB`

---

### Cell 3 — Download VoxCeleb2 Metadata

```python
from huggingface_hub import hf_hub_download, list_repo_files
import shutil

HF_REPO_ID   = "Reverb/voxceleb2"
META_CSV_PATH = os.path.join(COLAB_WORK_DIR, "vox2_meta.csv")
TXT_DIR       = os.path.join(COLAB_WORK_DIR, "txt")

if not os.path.exists(META_CSV_PATH):
    print("Downloading vox2_meta.csv...")
    local = hf_hub_download(
        repo_id=HF_REPO_ID, filename="vox2_meta.csv",
        repo_type="dataset", local_dir=COLAB_WORK_DIR,
    )
    shutil.copy(local, META_CSV_PATH)
    print(f"  -> {META_CSV_PATH}")
else:
    print("vox2_meta.csv already cached ✓")

print(f"Meta CSV size: {os.path.getsize(META_CSV_PATH)/1024:.1f} KB")
```

---

### Cell 4 — Generate Curated Index (run once, saved to Drive)

```python
CURATED_INDEX = os.path.join(DRIVE_VOIX_PATH, "curated_index.csv")

if os.path.exists(CURATED_INDEX):
    import csv
    with open(CURATED_INDEX) as f:
        n = sum(1 for _ in f) - 1
    print(f"Index already exists: {n:,} entries ✓")
else:
    # Need txt annotations for cross-session sampling
    # If txt/ dir not present, download a subset
    if not os.path.exists(TXT_DIR):
        print("Downloading txt annotations...")
        txt_files = [f for f in list_repo_files(HF_REPO_ID, repo_type="dataset")
                     if f.endswith(".txt") and "/txt/" in f]
        for i, fname in enumerate(txt_files):
            try:
                hf_hub_download(repo_id=HF_REPO_ID, filename=fname,
                                repo_type="dataset", local_dir=COLAB_WORK_DIR)
            except: pass
            if (i+1) % 500 == 0: print(f"  {i+1}/{len(txt_files)}")
        print("txt annotations downloaded ✓")

    from scripts.curate_voxceleb_index import curate_index
    total = curate_index(
        meta_path=META_CSV_PATH,
        txt_dir=TXT_DIR,
        output_path=CURATED_INDEX,
        seed=42, target_speakers=2000, utterances_per_speaker=50,
    )
    print(f"Index written: {total:,} rows -> {CURATED_INDEX}")
```

---

### Cell 5 — Download VoxCeleb2 Videos (streaming, ~32 GB)

```python
MP4_DIR  = os.path.join(COLAB_WORK_DIR, "mp4")
TAR_DIR  = os.path.join(COLAB_WORK_DIR, "archives")
os.makedirs(MP4_DIR,  exist_ok=True)
os.makedirs(TAR_DIR,  exist_ok=True)

# List available mp4 archive parts
print("Listing archive parts on HuggingFace...")
all_files  = list(list_repo_files(HF_REPO_ID, repo_type="dataset"))
mp4_parts  = sorted([f for f in all_files if "vox2_dev_mp4_part" in f])
print(f"  Found {len(mp4_parts)} parts")

# Download all parts (each ~1-2 GB)
for i, part in enumerate(mp4_parts):
    dest = os.path.join(TAR_DIR, os.path.basename(part))
    if not os.path.exists(dest):
        print(f"  [{i+1}/{len(mp4_parts)}] Downloading {part}...")
        hf_hub_download(repo_id=HF_REPO_ID, filename=part,
                        repo_type="dataset", local_dir=TAR_DIR)
    else:
        print(f"  [{i+1}/{len(mp4_parts)}] {part} cached ✓")

# Combine and extract
combined = os.path.join(TAR_DIR, "combined.tar")
if not os.path.exists(combined):
    print("Combining parts...")
    sh(f"cat {TAR_DIR}/vox2_dev_mp4_part* > {combined}")

if len(os.listdir(MP4_DIR)) == 0:
    print("Extracting videos...")
    sh(f"tar -xf {combined} -C {MP4_DIR} --checkpoint=10000")
    os.remove(combined)
    print("Extracted, archive deleted ✓")
else:
    print(f"Videos already extracted: {len(os.listdir(MP4_DIR))} speaker dirs ✓")
```

> ⚠️ This cell takes ~20–30 min on A100 (Colab has ~500 Mbps HF bandwidth).

---

### Cell 6 — Run Extraction (main loop, checkpointed)

```python
TRAIN_H5 = os.path.join(DRIVE_VOIX_PATH, "voxceleb2_train.h5")

from scripts.extract_voxceleb import run_extraction

print("=" * 60)
print("STARTING FEATURE EXTRACTION")
print(f"  Index  : {CURATED_INDEX}")
print(f"  Videos : {MP4_DIR}")
print(f"  Output : {TRAIN_H5}")
print("=" * 60)

run_extraction(
    index_path=CURATED_INDEX,
    videos_root=MP4_DIR,
    output_path=TRAIN_H5,
    device="cuda",
    batch_size=500,
)
```

> ✅ **Checkpoint-safe**: if the session disconnects, re-run Cells 1–2 then this cell.
> Already-extracted batches are skipped automatically.

---

### Cell 7 — Validate & Plot

```python
from voix.data.dataset import DatasetIntegrityAuditor

auditor = DatasetIntegrityAuditor(TRAIN_H5)
auditor.print_report()

# Install umap for visualization
!pip install -q umap-learn matplotlib

import umap, h5py, numpy as np, matplotlib.pyplot as plt

with h5py.File(TRAIN_H5, "r") as f:
    face = f["face_features"][:5000]
    ids  = f["speaker_ids"][:5000]

reducer  = umap.UMAP(n_components=2, random_state=42)
embedded = reducer.fit_transform(face)

plt.figure(figsize=(10, 8))
plt.scatter(embedded[:,0], embedded[:,1], c=ids % 20, cmap="tab20", s=2, alpha=0.6)
plt.title("UMAP of Face Embeddings (5K samples)")
plt.tight_layout()
plt.savefig(os.path.join(DRIVE_VOIX_PATH, "umap_embeddings.png"), dpi=150)
plt.show()
print(f"Plot saved to Drive ✓")
```

---

### Cell 8 — Confirm Output & Download Link

```python
size_mb = os.path.getsize(TRAIN_H5) / 1e6
print(f"File: {TRAIN_H5}")
print(f"Size: {size_mb:.1f} MB")
print(f"\nDrive folder: https://drive.google.com/drive/folders/{DRIVE_FOLDER_ID}")
print("\nDownload to your PC:")
print("  D:\\voix\\data\\processed\\voxceleb2_train.h5")
print("\nPhase 3 CVAE training can begin after download.")
```

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `Drive folder not found` | Run `!find /content/drive/MyDrive -maxdepth 3 -type d` and update `DRIVE_VOIX_PATH` |
| `HF download fails` | Run `!huggingface-cli login` and enter your HF token (free at hf.co/settings/tokens) |
| `CUDA out of memory` | Reduce `batch_size` from 500 to 200 in Cell 6 |
| `Session disconnects` | Re-run Cells 1–2, then Cell 6 directly (checkpoint resumes) |
| `mp4 extraction very slow` | Normal — 32 GB takes ~15 min on A100 SSD |
| `solvePnP warnings in output` | Expected for some clips — they are skipped gracefully |

---

## Timeline on A100

| Step | Duration |
|---|---|
| Install deps | ~4 min |
| Download metadata | ~2 min |
| Generate curated index | ~3 min |
| Download + extract videos (~32 GB) | ~25 min |
| Feature extraction (100K clips) | ~6 hours |
| Validation + UMAP | ~5 min |
| **Total** | **~6.5 hours** |

Single uninterrupted A100 session. No manual steps needed after Cell 1.