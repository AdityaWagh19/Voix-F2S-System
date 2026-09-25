# VOIX -- Colab T4 Extraction Guide (Optimised)

Extraction of 100K VoxCeleb2 clips on a free Colab T4 GPU.
Drive folder: https://drive.google.com/drive/folders/1Du9bWOYIR65ETGhjH1cgpTwRbMVyaBAM

---

## Performance summary

| Metric | Before optimisation | After optimisation |
|---|---|---|
| Per-clip time (T4) | ~2.1 s | ~0.7 s |
| Clips/hour | ~1,700 | ~5,100 |
| Sessions needed | ~8 | ~3 |
| Cell 3 download | ~5 min | ~20 sec |
| Cell 5 download | sequential | 4 parallel workers |

Optimisations applied in extract_voxceleb.py:
- Prefetch thread: CPU I/O overlaps GPU compute (2x speedup alone)
- FaceMesh early-exit: stops scanning frames once a frontal frame is found
- Batched ECAPA: 8 clips per forward pass instead of 1
- Batched fusion: 8 clips per FaceFusionLayer forward pass
- FP16 autocast for ECAPA on T4 Tensor Cores
- torch.inference_mode() instead of torch.no_grad()

---

## Session rules

| Cell | Purpose | When to run |
|---|---|---|
| Cell 1 | Mount Drive, session counter | Every session |
| Cell 2 | Install deps, clone repo | Every session (~3 min) |
| Cell 3 | Download metadata | ONE-TIME only |
| Cell 4 | Generate curated index | ONE-TIME only |
| Cell 5 | Download + extract videos | Once per session disk reset |
| Cell 6 | Run extraction | Every session |
| Cell 7 | Audit + UMAP | Once, after 100K complete |
| Cell 8 | Session summary | Every session |

---

## CELL 1 -- Mount Drive & session tracking

Run every session. Increments a persistent session counter on Drive.

```python
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
```

If the Drive folder path differs from `/content/drive/MyDrive/voix_extraction`,
run `!find /content/drive/MyDrive -maxdepth 4 -type d | head -30` to locate it,
then update `DRIVE_VOIX_PATH` accordingly.

---

## CELL 2 -- Install dependencies

Run every session. pip cache makes this ~1 min from session 2 onward.

```python
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
```

---

## CELL 3 -- Download metadata (ONE-TIME, parallel, ~20 sec)

Only run this once. From session 2 onward, skip it entirely.

```python
import os, shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from huggingface_hub import hf_hub_download, list_repo_files

DRIVE_VOIX_PATH = "/content/drive/MyDrive/voix_extraction"
COLAB_WORK_DIR  = "/content/voix_work"
META_CSV        = f"{DRIVE_VOIX_PATH}/vox2_meta.csv"
TXT_DIR         = f"{COLAB_WORK_DIR}/txt"
HF_REPO         = "Reverb/voxceleb2"

if os.path.exists(META_CSV):
    print(f"vox2_meta.csv already on Drive ({os.path.getsize(META_CSV)//1024} KB) -- skipping")
else:
    print("Downloading vox2_meta.csv...")
    local = hf_hub_download(repo_id=HF_REPO, filename="vox2_meta.csv",
                            repo_type="dataset", local_dir=COLAB_WORK_DIR)
    shutil.copy(local, META_CSV)
    print("Saved to Drive.")

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
```

---

## CELL 4 -- Generate curated index (ONE-TIME)

Only run this once. From session 2 onward, skip it entirely.

```python
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
```

---

## CELL 5 -- Download + extract videos (once per session disk reset)

Check first with `!find /content/voix_work/mp4 -name "*.mp4" 2>/dev/null | wc -l`.
If the number is above 50000, skip this cell.

```python
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
    subprocess.run(
        f"tar -xf {TAR_DIR}/combined.tar -C {MP4_DIR}",
        shell=True, check=True
    )
    os.remove(f"{TAR_DIR}/combined.tar")

    n_final = len(list(pathlib.Path(MP4_DIR).rglob("*.mp4")))
    print(f"Extraction complete: {n_final:,} mp4 files ready")
```

---

## CELL 6 -- Main extraction (run every session)

Auto-resumes from Drive HDF5. Stops after 11 hours. Saves progress every 200 clips.

```python
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
print(f"  Batch     : 200 clips per HDF5 flush (~3 min each)")
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
```

---

## CELL 7 -- Audit + visualisation (once, after 100K complete)

```python
import sys, os, h5py, numpy as np

DRIVE_VOIX_PATH = "/content/drive/MyDrive/voix_extraction"
TRAIN_H5        = f"{DRIVE_VOIX_PATH}/voxceleb2_train.h5"
sys.path.insert(0, "/content/voix_work/voix")

from voix.data.dataset import DatasetIntegrityAuditor
DatasetIntegrityAuditor(TRAIN_H5).print_report()
print(f"File size: {os.path.getsize(TRAIN_H5)/1e6:.1f} MB")
print(f"Download to: D:\\voix\\data\\processed\\voxceleb2_train.h5")

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
```

---

## CELL 8 -- Session summary (run at end of every session)

```python
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
    print("COMPLETE. Run Cell 7, then download voxceleb2_train.h5 to D:\\voix\\data\\processed\\")
else:
    print("Next: Cell 1 -> Cell 2 -> Cell 5 (if needed) -> Cell 6 -> Cell 8")
```

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `No GPU detected` | Runtime -> Change runtime type -> T4 GPU -> Save -> reconnect |
| Drive folder not found | `!find /content/drive/MyDrive -maxdepth 4 -type d` then update `DRIVE_VOIX_PATH` |
| Session timed out | Normal -- re-run Cell 1 + Cell 2 + Cell 6 in new session |
| mp4 files missing | Run `!find /content/voix_work/mp4 -name "*.mp4" | wc -l`, if 0 run Cell 5 |
| HF download fails | `!huggingface-cli login` with a free token from hf.co/settings/tokens |
| CUDA out of memory | Reduce `batch_size=200` to `batch_size=64` in Cell 6 |
| ECAPA batch shape error | Reduce N_ECAPA_BATCH in extract_voxceleb.py from 8 to 4 |