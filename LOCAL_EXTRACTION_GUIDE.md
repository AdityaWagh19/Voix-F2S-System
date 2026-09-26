# VOIX -- Local Extraction Guide (GTX 1650 / i5-11320H)

Verified hardware profile and extraction setup for your specific machine.

---

## Verified Hardware State

| Component | Spec | Notes |
|---|---|---|
| GPU | GTX 1650 Mobile (TU117), 14 SMs, 896 CUDA cores | CUDA 7.5, Turing |
| VRAM total | 4.29 GB | ~855 MB used by display driver |
| VRAM free for extraction | ~2.6 GB | After models load (~500 MB) |
| CPU | i5-11320H, 4 cores / 8 threads, 3.2-4.5 GHz | Tiger Lake H35 |
| RAM | 8 GB single-channel DDR4-3200 | Currently ~97% used |
| Storage | NVMe Samsung 512 GB | Fast, not a bottleneck |
| PyTorch | 2.4.1+cu121 | CUDA 12.1 |
| cuDNN | 9.1.0 | benchmark=True set by runner |

---

## RAM Warning -- READ THIS

Your machine has 8 GB RAM and it is currently at 97% usage.

When the extraction runs:
- SpeechBrain (ECAPA model): loads ~1.2 GB into RAM
- MediaPipe FaceMesh: loads ~400 MB into RAM
- Python + NumPy buffers: ~500 MB

**Close before starting each extraction run:**
- Chrome and all browser tabs
- Any video players or games
- Unnecessary background apps (check Task Manager)

Target: at least 3.5 GB free RAM before running.

The runner script will warn you and pause if RAM is below this threshold.

---

## Performance Tuning Applied

Changes made specifically for your hardware:

| Setting | Value | Reason |
|---|---|---|
| `PREFETCH_QUEUE` | 2 (was 4) | 8 GB RAM: 4 slots would use ~56 MB, 2 = ~28 MB safer |
| `N_ECAPA_BATCH` | 4 (was 8) | 3.46 GB free VRAM: batch 4 is safe, batch 8 is too tight |
| `N_FACE_BATCH` | 4 (was 8) | Same reason |
| `cudnn.benchmark` | True | Added: profiles fastest kernels per input shape (+10-20%) |
| GPU pre-warm | Yes | Added: dummy tensor eliminates slow first-batch JIT |
| ffmpeg stderr | DEVNULL | Was capture_output=True, now DEVNULL (no per-call buffer) |
| `batch_size` | 50 | HDF5 flush every ~2.5 min on GTX 1650 |

---

## Step 1 -- Install dependencies

Open terminal in D:\voix:

```
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install insightface onnxruntime-gpu
pip install speechbrain
pip install "mediapipe==1.0.1"
pip install h5py soundfile huggingface_hub opencv-python pillow psutil
pip install -e .
```

Install ffmpeg and add to PATH:
- Download: https://ffmpeg.org/download.html (Windows builds)
- Extract to C:\ffmpeg\
- Add C:\ffmpeg\bin to System PATH (search "Edit system environment variables")
- Verify: open new terminal, run `ffmpeg -version`

---

## Step 2 -- Download VoxCeleb2 (run once, ~45 min)

```
python scripts/download_voxceleb.py
```

Downloads to D:\voix\data\raw\voxceleb2\:
- vox2_meta.csv
- txt/ (utterance annotations)
- dev/mp4/ (all video clips, ~32 GB after extraction)

Disk used during download: ~65 GB peak, ~32 GB final.
Make sure D:\ has at least 65 GB free before starting.
Current free on D:\: ~155 GB -- you are fine.

---

## Step 3 -- Generate curated index (run once, ~3 min)

```
python scripts/curate_voxceleb_index.py ^
  --meta D:\voix\data\raw\voxceleb2\vox2_meta.csv ^
  --txt  D:\voix\data\raw\voxceleb2\txt ^
  --out  D:\voix\data\processed\curated_index.csv
```

---

## Step 4 -- Run extraction overnight

Before starting each run:
1. Close Chrome and all unnecessary apps
2. Open Task Manager and verify RAM free > 3.5 GB
3. Plug in charger (laptop will throttle on battery)
4. Run:

```
python scripts/run_local.py
```

The script:
- Checks RAM and warns if below 3.5 GB
- Enables cuDNN benchmark mode (faster kernels)
- Pre-warms GPU (eliminates slow first batch)
- Disables Windows sleep
- Checkpoints every 50 clips (~2.5 min intervals)
- Logs to data/processed/extraction.log
- Resumes from last checkpoint on re-run

---

## Expected Timeline

| Run | Duration | Clips done | Cumulative |
|---|---|---|---|
| Night 1 | 8 h | ~9,600 | 9,600 |
| Night 2 | 8 h | ~9,600 | 19,200 |
| Night 3 | 8 h | ~9,600 | 28,800 |
| ... | | | |
| Night 9 | 8 h | ~9,600 | ~86,400 |
| Night 10 | ~4 h | ~4,600 | 100,000 |

Estimated total: ~83 hours compute = ~10 overnight runs of 8 hours each.
You can also run it during the day if the machine is free.

---

## Monitor progress

Check live log:
```
Get-Content D:\voix\data\processed\extraction.log -Tail 20 -Wait
```

Check progress summary:
```
Get-Content D:\voix\data\processed\extraction_progress.json
```

---

## Troubleshooting

| Problem | Fix |
|---|---|
| RAM warning at startup | Close Chrome + apps. Check Task Manager. |
| VRAM out of memory | In run_local.py, change BATCH_SIZE from 50 to 20. In extract_voxceleb.py, change N_ECAPA_BATCH from 4 to 2. |
| Slow speed (well below 1200/hr) | Check Task Manager for RAM swap. Close more apps. |
| ffmpeg not found | Add ffmpeg bin/ to PATH, restart terminal |
| CUDA not available | `python -c "import torch; print(torch.cuda.is_available())"` must print True |
| Script stops mid-run | Just re-run -- checkpoint resumes automatically |
| h5py file locked | Another process has HDF5 open. Close it and re-run. |