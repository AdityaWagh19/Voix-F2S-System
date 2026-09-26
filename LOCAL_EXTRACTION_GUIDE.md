# VOIX -- Local Extraction Guide (GTX 1650)

Run the full VoxCeleb2 feature extraction on your own machine.
No session limits, no restarts, checkpoint every 50 clips (~2.5 min).

---

## Hardware requirements

| | Minimum | Your GTX 1650 |
|---|---|---|
| GPU VRAM | 3 GB | 4 GB -- fine |
| Disk space | ~65 GB during download, ~35 GB final | Need this free on D:\ |
| RAM | 8 GB | Fine |
| OS | Windows 10/11 | Windows |

---

## Step 1 -- Install Python dependencies

Open a terminal in `D:\voix` and run:

```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install insightface onnxruntime-gpu
pip install speechbrain
pip install mediapipe==1.0.1
pip install h5py soundfile huggingface_hub opencv-python pillow
pip install -e .
```

Also install ffmpeg and make sure it is on your PATH:
- Download from https://ffmpeg.org/download.html
- Extract and add the `bin/` folder to System PATH
- Verify: open a terminal and run `ffmpeg -version`

---

## Step 2 -- Download VoxCeleb2 (~45 min at 100 Mbps)

This only runs once. Already-downloaded parts are skipped on re-run.

```
python scripts/download_voxceleb.py
```

What it downloads to `D:\voix\data\raw\voxceleb2\`:
- `vox2_meta.csv` -- speaker metadata (~200 KB)
- `txt/` -- utterance annotations (~50 MB)
- `dev/mp4/` -- all video clips (~32 GB after extraction)

Disk usage during download: ~65 GB (archives + extracted).
Archives are automatically deleted after extraction, leaving ~32 GB.

---

## Step 3 -- Generate curated index (run once, ~3 min)

```
python scripts/curate_voxceleb_index.py ^
  --meta D:\voix\data\raw\voxceleb2\vox2_meta.csv ^
  --txt  D:\voix\data\raw\voxceleb2\txt ^
  --out  D:\voix\data\processed\curated_index.csv
```

This creates a stratified 100K-clip index (1000 male + 1000 female speakers,
50 clips each), saved as `curated_index.csv` on your PC.

---

## Step 4 -- Run extraction (start this at night, leave it running)

```
python scripts/run_local.py
```

What it does:
- Disables Windows sleep for the duration
- Processes 50 clips, flushes to HDF5, saves progress.json, repeats
- Prints ETA at each checkpoint
- Logs everything to `D:\voix\data\processed\extraction.log`
- Stops cleanly on Ctrl+C (no data loss)
- Re-running resumes from exactly where it stopped

Expected runtime on GTX 1650:
- ~1,200 clips/hr
- 100K clips in ~83 hours
- Run overnight (8-10h) for 4-5 nights

---

## Step 5 -- Monitor progress

While the script is running, open a second terminal and run:

```
python scripts/check_progress.py
```

Or check the log file directly:
```
Get-Content D:\voix\data\processed\extraction.log -Tail 20
```

Or check the progress JSON:
```
Get-Content D:\voix\data\processed\extraction_progress.json
```

---

## Step 6 -- Validate when complete

Once `extraction_progress.json` shows `total_extracted: 100000`:

```
python scripts/validate_h5.py --h5 D:\voix\data\processed\voxceleb2_train.h5
```

This prints a full audit report and saves a UMAP plot.

---

## Files created by this pipeline

| File | Location | Size |
|---|---|---|
| `curated_index.csv` | `data/processed/` | ~8 MB |
| `voxceleb2_train.h5` | `data/processed/` | ~280 MB |
| `extraction_progress.json` | `data/processed/` | <1 KB |
| `extraction.log` | `data/processed/` | ~5 MB |

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `ffmpeg not found` | Add ffmpeg `bin/` folder to Windows PATH, restart terminal |
| `CUDA not available` | `python -c "import torch; print(torch.cuda.is_available())"` should print True. Reinstall PyTorch with CUDA. |
| `No mp4 files found` | Run Step 2 (download_voxceleb.py) first |
| `No index found` | Run Step 3 (curate_voxceleb_index.py) first |
| `CUDA out of memory` | Edit `BATCH_SIZE` in `scripts/run_local.py` from 50 to 20 |
| Script stops mid-run | Normal if PC slept or crashed -- just re-run, it resumes |
| `h5py file locked` | Another process has the HDF5 open. Close it and re-run. |