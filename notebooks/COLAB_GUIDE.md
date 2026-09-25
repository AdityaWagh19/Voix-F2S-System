# VOIX — Colab T4 Extraction Guide

Complete guide for running the VoxCeleb2 extraction across **multiple T4 sessions**.
Designed for free Colab (T4 GPU, 12h session limit).

---

## Your Setup

| | |
|---|---|
| **GPU** | T4 (16 GB VRAM) |
| **Session limit** | ~12 hours per session |
| **Drive folder** | https://drive.google.com/drive/folders/1Du9bWOYIR65ETGhjH1cgpTwRbMVyaBAM |
| **Expected throughput** | ~1,000 clips/hour on T4 |
| **Total sessions needed** | **~8 sessions × 12h = ~4 days** (run one per day) |

---

## Checkpoint Architecture

```
Each session:
  startup  → reads Drive HDF5 → finds already-done clip keys
  running  → extracts 200 clips → flushes to Drive HDF5 → repeat
  shutdown → writes progress.json to Drive → session ends safely

Next session:
  startup  → reads same Drive HDF5 → skips done clips → continues
```

**Safe to interrupt at any time.** No work is lost. The Drive HDF5
grows by ~600 KB every 200 clips and is the single source of truth.

---

## One-Time vs Per-Session Cells

| Cell | What it does | Run |
|---|---|---|
| Cell 1 | Mount Drive, session counter | **Every session** |
| Cell 2 | Install deps, clone repo | **Every session** (~2 min) |
| Cell 3 | Download metadata csv + txt | **One-time only** |
| Cell 4 | Generate curated_index.csv | **One-time only** |
| Cell 5 | Download + extract videos | **Once per session disk** |
| **Cell 6** | **Run extraction (main loop)** | **Every session** |
| Cell 7 | Audit + UMAP | After completion |
| Cell 8 | Session summary | End of every session |

---

## CELL 1 — Mount Drive & session tracking

```python
from google.colab import drive
drive.mount('/content/drive', force_remount=False)

import os, json, time, torch

DRIVE_FOLDER_ID = "1Du9bWOYIR65ETGhjH1cgpTwRbMVyaBAM"
DRIVE_VOIX_PATH = "/content/drive/MyDrive/voix_extraction"
COLAB_WORK_DIR  = "/content/voix_work"

# Create folders
os.makedirs(DRIVE_VOIX_PATH, exist_ok=True)
os.makedirs(COLAB_WORK_DIR,  exist_ok=True)

# Persistent file paths (survive across sessions)
CURATED_INDEX = f"{DRIVE_VOIX_PATH}/curated_index.csv"
TRAIN_H5      = f"{DRIVE_VOIX_PATH}/voxceleb2_train.h5"
META_CSV      = f"{DRIVE_VOIX_PATH}/vox2_meta.csv"
PROGRESS_JSON = f"{DRIVE_VOIX_PATH}/extraction_progress.json"

# Load or init session counter
if os.path.exists(PROGRESS_JSON):
    with open(PROGRESS_JSON) as f:
        prog = json.load(f)
else:
    prog = {"session": 0, "total_extracted": 0, "last_key": None}

prog["session"] += 1
with open(PROGRESS_JSON, "w") as f:
    json.dump(prog, f, indent=2)

print(f"Session #{prog['session']} | Previously done: {prog['total_extracted']:,}")
print(f"GPU: {torch.cuda.get_device_name(0)}, VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.0f} GB")
```

> **If Drive folder path differs**, run `!find /content/drive/MyDrive -maxdepth 4 -type d | head -30` to locate it.

---

## CELL 2 — Install Dependencies (every session, ~2 min)

```python
import subprocess, sys, os

def sh(cmd):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if r.returncode != 0: print(r.stderr[-600:])

COLAB_WORK_DIR = "/content/voix_work"
os.makedirs(COLAB_WORK_DIR, exist_ok=True)

print("Installing... (cached after first session)")
sh("pip install -q torch torchaudio --index-url https://download.pytorch.org/whl/cu121")
sh("pip install -q insightface onnxruntime-gpu speechbrain")
sh("pip install -q mediapipe==1.0.1")
sh("pip install -q h5py soundfile huggingface_hub opencv-python pillow")
sh("apt-get install -qq ffmpeg")

if not os.path.exists(f"{COLAB_WORK_DIR}/voix"):
    sh(f"git clone https://github.com/AdityaWagh19/Voix-F2S-System.git {COLAB_WORK_DIR}/voix")
else:
    sh(f"git -C {COLAB_WORK_DIR}/voix pull origin main")

sys.path.insert(0, f"{COLAB_WORK_DIR}/voix")
sh(f"pip install -q -e {COLAB_WORK_DIR}/voix")
print("Done ✓")
```

---

## CELL 3 — Download Metadata (one-time)

```python
import os, shutil
from huggingface_hub import hf_hub_download, list_repo_files

HF_REPO = "Reverb/voxceleb2"
TXT_DIR = f"{COLAB_WORK_DIR}/txt"

if not os.path.exists(META_CSV):
    local = hf_hub_download(repo_id=HF_REPO, filename="vox2_meta.csv",
                            repo_type="dataset", local_dir=COLAB_WORK_DIR)
    shutil.copy(local, META_CSV)
    print(f"Saved vox2_meta.csv to Drive ✓")
else:
    print("vox2_meta.csv already on Drive ✓")

if not os.path.exists(TXT_DIR) or len(os.listdir(TXT_DIR)) < 50:
    os.makedirs(TXT_DIR, exist_ok=True)
    txt_files = [f for f in list_repo_files(HF_REPO, repo_type="dataset")
                 if f.startswith("txt/")]
    print(f"Downloading {len(txt_files)} txt files...")
    for i, fname in enumerate(txt_files):
        try:
            hf_hub_download(repo_id=HF_REPO, filename=fname,
                            repo_type="dataset", local_dir=COLAB_WORK_DIR)
        except: pass
        if (i+1) % 1000 == 0: print(f"  {i+1}/{len(txt_files)}")
    print("txt annotations done ✓")
else:
    print("txt annotations already cached ✓")
```

---

## CELL 4 — Generate Curated Index (one-time)

```python
import csv, sys
sys.path.insert(0, f"{COLAB_WORK_DIR}/voix")

if os.path.exists(CURATED_INDEX):
    with open(CURATED_INDEX) as f:
        n = sum(1 for _ in f) - 1
    print(f"Curated index already on Drive: {n:,} entries ✓")
else:
    from scripts.curate_voxceleb_index import curate_index
    total = curate_index(
        meta_path=META_CSV,
        txt_dir=f"{COLAB_WORK_DIR}/txt",
        output_path=CURATED_INDEX,
        seed=42, target_speakers=2000, utterances_per_speaker=50,
    )
    print(f"Index saved to Drive: {total:,} rows ✓")
```

---

## CELL 5 — Download Videos (once per fresh session disk, ~30 min)

> ⚠️ Colab ephemeral disk resets when your session ends. This cell re-runs if the videos aren't there.

```python
import os
from huggingface_hub import hf_hub_download, list_repo_files

MP4_DIR = f"{COLAB_WORK_DIR}/mp4"
TAR_DIR = f"{COLAB_WORK_DIR}/archives"
os.makedirs(MP4_DIR, exist_ok=True)
os.makedirs(TAR_DIR, exist_ok=True)

HF_REPO = "Reverb/voxceleb2"

# Count existing mp4s
import pathlib
n_mp4 = len(list(pathlib.Path(MP4_DIR).rglob("*.mp4")))
print(f"Existing mp4 files: {n_mp4:,}")

if n_mp4 > 50000:
    print("Sufficient videos present ✓")
else:
    all_files  = list(list_repo_files(HF_REPO, repo_type="dataset"))
    mp4_parts  = sorted([f for f in all_files if "vox2_dev_mp4_part" in f])
    print(f"Downloading {len(mp4_parts)} archive parts...")

    for i, part in enumerate(mp4_parts):
        dest = f"{TAR_DIR}/{os.path.basename(part)}"
        if not os.path.exists(dest):
            print(f"[{i+1}/{len(mp4_parts)}] {part}...", end=" ", flush=True)
            hf_hub_download(repo_id=HF_REPO, filename=part,
                            repo_type="dataset", local_dir=TAR_DIR)
            print(f"{os.path.getsize(dest)/1e9:.1f} GB ✓")
        else:
            print(f"[{i+1}/{len(mp4_parts)}] cached ✓")

    !cat {TAR_DIR}/vox2_dev_mp4_part* > {TAR_DIR}/combined.tar
    !tar -xf {TAR_DIR}/combined.tar -C {MP4_DIR}
    !rm {TAR_DIR}/combined.tar
    print("Extracted ✓")
```

---

## CELL 6 — ★ RUN EXTRACTION (run every session)

```python
import os, sys, json, time, csv, h5py
DRIVE_VOIX_PATH = "/content/drive/MyDrive/voix_extraction"
COLAB_WORK_DIR  = "/content/voix_work"
TRAIN_H5        = f"{DRIVE_VOIX_PATH}/voxceleb2_train.h5"
CURATED_INDEX   = f"{DRIVE_VOIX_PATH}/curated_index.csv"
PROGRESS_JSON   = f"{DRIVE_VOIX_PATH}/extraction_progress.json"
MP4_DIR         = f"{COLAB_WORK_DIR}/mp4"
sys.path.insert(0, f"{COLAB_WORK_DIR}/voix")

def save_progress(n_done, last_key):
    """Write progress.json to Drive after every batch."""
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

run_extraction(
    index_path=CURATED_INDEX,
    videos_root=MP4_DIR,
    output_path=TRAIN_H5,         # writes directly to Drive
    device="cuda",
    batch_size=200,               # ~12 min per checkpoint on T4
    session_max_hours=11.0,       # auto-stops 1h before 12h limit
    progress_callback=save_progress,
)

# Session summary
with h5py.File(TRAIN_H5, "r") as f:
    n = f["face_features"].shape[0] if "face_features" in f else 0
print(f"\nSession done. Total clips in HDF5: {n:,} / 100,000")
print(f"Progress: {n/100000*100:.1f}%")
if n < 100000:
    print("Re-run Cell 6 in a new T4 session to continue.")
else:
    print("EXTRACTION COMPLETE! Run Cell 7 for validation.")
```

---

## CELL 7 — Audit + Visualisation (after completion)

```python
import sys, os, h5py, numpy as np
DRIVE_VOIX_PATH = "/content/drive/MyDrive/voix_extraction"
TRAIN_H5        = f"{DRIVE_VOIX_PATH}/voxceleb2_train.h5"
sys.path.insert(0, "/content/voix_work/voix")

from voix.data.dataset import DatasetIntegrityAuditor
DatasetIntegrityAuditor(TRAIN_H5).print_report()

# Optional UMAP
!pip install -q umap-learn matplotlib
import umap, matplotlib.pyplot as plt
with h5py.File(TRAIN_H5, "r") as f:
    face = f["face_features"][:5000]
    ids  = f["speaker_ids"][:5000]

emb = umap.UMAP(n_components=2, random_state=42).fit_transform(face)
plt.figure(figsize=(12,9))
plt.scatter(emb[:,0], emb[:,1], c=ids%20, cmap="tab20", s=1.5, alpha=0.5)
plt.title("UMAP — Face Embeddings (5K samples)")
plt.axis("off")
plt.tight_layout()
plt.savefig(f"{DRIVE_VOIX_PATH}/umap_embeddings.png", dpi=150)
plt.show()
```

---

## Session Timeline (T4)

| Session | Clips done | Cumulative % | When to run |
|---|---|---|---|
| 1 | ~12,000 | 12% | Day 1 |
| 2 | ~24,000 | 24% | Day 2 |
| 3 | ~36,000 | 36% | Day 3 |
| 4 | ~48,000 | 48% | Day 4 |
| 5 | ~60,000 | 60% | Day 5 |
| 6 | ~72,000 | 72% | Day 6 |
| 7 | ~84,000 | 84% | Day 7 |
| 8 | ~100,000 | 100% | Day 8 |

> Actual throughput may vary. The script logs exact rate (clips/sec) each batch.
> A faster day = fewer sessions needed.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `Drive folder not found` | `!find /content/drive/MyDrive -maxdepth 4 -type d` then update `DRIVE_VOIX_PATH` |
| `Session timed out` | Normal — just re-run Cell 1 + Cell 2 + Cell 6 in new session |
| `HDF5 n=0 after restart` | Drive file exists but is empty — delete it and restart |
| `CUDA out of memory` | Change `batch_size=200` to `batch_size=50` in Cell 6 |
| `mp4 files missing` | Re-run Cell 5 (ephemeral disk reset) |
| `HF download fails` | `!huggingface-cli login` with free HF token |
| `No face detected warnings` | Expected for off-axis clips — they are skipped |