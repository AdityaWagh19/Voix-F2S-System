"""Setup verification script for VOIX local extraction.

Checks all dependencies, CUDA, ffmpeg, and paths before you start
an overnight run. Run this BEFORE downloading VoxCeleb2.

Usage:
    python scripts/verify_setup.py

Expected output: all lines showing OK. Any FAIL means fix that first.
"""
import sys, os, time, subprocess, importlib
from pathlib import Path

VOIX_ROOT = Path(r"D:\voix")
PASS = "[  OK  ]"
FAIL = "[ FAIL ]"
WARN = "[ WARN ]"

errors = []
warnings = []

def check(label, fn):
    try:
        result = fn()
        msg = result if isinstance(result, str) else ""
        print(f"{PASS}  {label}{('  -- ' + msg) if msg else ''}")
        return True
    except Exception as e:
        print(f"{FAIL}  {label}  -- {e}")
        errors.append(label)
        return False

def warn(label, fn):
    try:
        result = fn()
        msg = result if isinstance(result, str) else ""
        print(f"{PASS}  {label}{('  -- ' + msg) if msg else ''}")
    except Exception as e:
        print(f"{WARN}  {label}  -- {e}")
        warnings.append(label)

print("=" * 60)
print("  VOIX Setup Verification")
print(f"  Python: {sys.version.split()[0]}")
print("=" * 60)

# ── Python version ────────────────────────────────────────────
print("\n-- Python & Core --")
check("Python >= 3.9", lambda: f"{sys.version_info.major}.{sys.version_info.minor}" if sys.version_info >= (3,9) else (_ for _ in ()).throw(Exception("Need >= 3.9")))

# ── PyTorch + CUDA ────────────────────────────────────────────
print("\n-- PyTorch / CUDA --")
def _check_torch():
    import torch
    v = torch.__version__
    assert torch.cuda.is_available(), "CUDA not available. Check driver + PyTorch install."
    return f"PyTorch {v}"
check("torch + CUDA", _check_torch)

def _check_gpu():
    import torch
    p = torch.cuda.get_device_properties(0)
    free, total = torch.cuda.mem_get_info(0)
    return f"{p.name}, VRAM {total/1e9:.1f} GB, {free/1e6:.0f} MB free, CUDA {p.major}.{p.minor}"
check("GPU accessible", _check_gpu)

def _check_cudnn():
    import torch
    assert torch.backends.cudnn.is_available(), "cuDNN not found"
    return f"cuDNN {torch.backends.cudnn.version()}"
check("cuDNN", _check_cudnn)

# ── ML Libraries ──────────────────────────────────────────────
print("\n-- ML Libraries --")

def _check_insightface():
    import insightface
    return f"insightface {insightface.__version__}"
check("insightface", _check_insightface)

def _check_onnxruntime():
    import onnxruntime as ort
    providers = ort.get_available_providers()
    gpu = "CUDAExecutionProvider" in providers
    return f"onnxruntime {ort.__version__}, GPU provider: {gpu}"
check("onnxruntime", _check_onnxruntime)

def _check_speechbrain():
    import speechbrain
    return f"speechbrain {speechbrain.__version__}"
check("speechbrain", _check_speechbrain)

def _check_mediapipe():
    import mediapipe as mp
    return f"mediapipe {mp.__version__}"
check("mediapipe", _check_mediapipe)

def _check_h5py():
    import h5py
    return f"h5py {h5py.__version__}"
check("h5py", _check_h5py)

def _check_soundfile():
    import soundfile as sf
    return f"soundfile {sf.__version__}"
check("soundfile", _check_soundfile)

def _check_cv2():
    import cv2
    return f"opencv {cv2.__version__}"
check("opencv-python", _check_cv2)

def _check_hf():
    import huggingface_hub
    return f"huggingface_hub {huggingface_hub.__version__}"
check("huggingface_hub", _check_hf)

# ── ffmpeg ───────────────────────────────────────────────────
print("\n-- ffmpeg --")
def _check_ffmpeg():
    r = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True)
    assert r.returncode == 0, "ffmpeg not found on PATH"
    line = r.stdout.split('\n')[0]
    return line[:60]
check("ffmpeg on PATH", _check_ffmpeg)

def _check_ffmpeg_encode():
    import tempfile, numpy as np
    # Write a tiny silent wav, check ffmpeg can read it
    import soundfile as sf
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        wav_in = f.name
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        wav_out = f.name
    sf.write(wav_in, np.zeros(16000, dtype=np.float32), 16000)
    r = subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "quiet", "-i", wav_in,
         "-ar", "16000", "-ac", "1", "-f", "wav", wav_out],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    ok = r.returncode == 0 and os.path.exists(wav_out)
    os.unlink(wav_in)
    if os.path.exists(wav_out): os.unlink(wav_out)
    assert ok, "ffmpeg failed to process a test wav file"
    return "encode test passed"
check("ffmpeg wav encode test", _check_ffmpeg_encode)

# ── VOIX package ─────────────────────────────────────────────
print("\n-- VOIX package --")
sys.path.insert(0, str(VOIX_ROOT))

def _check_voix():
    import voix
    return f"voix package importable from {VOIX_ROOT}"
check("voix importable", _check_voix)

def _check_voix_modules():
    from voix.data.morphology    import compute_craniofacial_ratios
    from voix.data.head_pose     import HeadPoseEstimator
    from voix.models.fusion      import FaceFusionLayer
    return "head_pose, morphology, fusion OK"
check("voix.data + voix.models", _check_voix_modules)

def _check_extract_script():
    from scripts.extract_voxceleb import run_extraction, _ffmpeg_extract_audio
    return "extract_voxceleb importable"
check("scripts.extract_voxceleb", _check_extract_script)

# ── Paths & Disk ─────────────────────────────────────────────
print("\n-- Paths & Disk --")

def _check_disk():
    import shutil
    total, used, free = shutil.disk_usage(VOIX_ROOT)
    assert free > 65e9, f"Need >65 GB free for download, have {free/1e9:.1f} GB"
    return f"{free/1e9:.1f} GB free on {VOIX_ROOT.drive}"
check("Disk space (>65 GB)", _check_disk)

def _check_processed_dir():
    d = VOIX_ROOT / "data" / "processed"
    d.mkdir(parents=True, exist_ok=True)
    return f"{d}"
check("data/processed/ writable", _check_processed_dir)

# ── RAM ───────────────────────────────────────────────────────
print("\n-- RAM --")
def _check_ram():
    import psutil
    m = psutil.virtual_memory()
    avail_gb = m.available / 1e9
    total_gb = m.total / 1e9
    msg = f"{avail_gb:.1f} GB free / {total_gb:.1f} GB total ({m.percent:.0f}% used)"
    if avail_gb < 3.5:
        raise Exception(f"Only {avail_gb:.1f} GB free. Close Chrome + other apps. Need >3.5 GB.")
    return msg
warn("RAM free (>3.5 GB)", _check_ram)

# ── Summary ───────────────────────────────────────────────────
print()
print("=" * 60)
if not errors:
    print(f"  ALL CHECKS PASSED ({len(warnings)} warnings)")
    print()
    if warnings:
        print(f"  Warnings to address: {', '.join(warnings)}")
        print()
    print("  Next steps:")
    print("  1. python scripts/download_voxceleb.py")
    print("  2. python scripts/curate_voxceleb_index.py ...")
    print("  3. python scripts/test_extraction.py   <-- 100-clip test run")
    print("  4. python scripts/run_local.py         <-- full overnight run")
else:
    print(f"  {len(errors)} CHECK(S) FAILED -- fix before proceeding:")
    for e in errors:
        print(f"    - {e}")
    print()
    print("  See LOCAL_EXTRACTION_GUIDE.md -> Troubleshooting section")
print("=" * 60)