"""Resumable downloader for torch+cu121 wheel.

Saves progress: if the connection drops, re-run this script and it
will resume from the last byte downloaded (HTTP Range request).

Usage:
    python scripts/download_torch_cuda.py

After download completes:
    pip install D:\torch_cu121\torch-2.4.1+cu121-cp312-cp312-win_amd64.whl
    pip install D:\torch_cu121\torchaudio-2.4.1+cu121-cp312-cp312-win_amd64.whl
"""

import os, sys, time, urllib.request
from pathlib import Path

DEST_DIR = Path(r"D:\torch_cu121")
DEST_DIR.mkdir(exist_ok=True)

WHEELS = [
    (
        "torch-2.4.1+cu121-cp312-cp312-win_amd64.whl",
        "https://download.pytorch.org/whl/cu121/torch-2.4.1%2Bcu121-cp312-cp312-win_amd64.whl",
    ),
    (
        "torchaudio-2.4.1+cu121-cp312-cp312-win_amd64.whl",
        "https://download.pytorch.org/whl/cu121/torchaudio-2.4.1%2Bcu121-cp312-cp312-win_amd64.whl",
    ),
]

def download_resumable(url: str, dest: Path) -> bool:
    """Download url to dest, resuming if partial file exists."""
    existing = dest.stat().st_size if dest.exists() else 0

    # Check remote file size
    req = urllib.request.Request(url, method="HEAD")
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            total = int(r.headers.get("Content-Length", 0))
    except Exception as e:
        print(f"  HEAD request failed: {e}")
        total = 0

    if total > 0 and existing == total:
        print(f"  Already complete: {dest.name} ({existing/1e9:.2f} GB)")
        return True

    if existing > 0:
        print(f"  Resuming from {existing/1e6:.0f} MB ({existing/total*100:.1f}% done)...")
    else:
        print(f"  Starting download: {total/1e9:.2f} GB")

    headers = {}
    if existing > 0:
        headers["Range"] = f"bytes={existing}-"

    req = urllib.request.Request(url, headers=headers)
    mode = "ab" if existing > 0 else "wb"

    try:
        t0 = time.time()
        last_print = time.time()
        downloaded = existing

        with urllib.request.urlopen(req, timeout=60) as r, open(dest, mode) as f:
            while True:
                chunk = r.read(524288)  # 512 KB chunks
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)

                now = time.time()
                if now - last_print >= 5:
                    elapsed = now - t0
                    speed = (downloaded - existing) / elapsed / 1e6
                    pct = downloaded / total * 100 if total else 0
                    eta = (total - downloaded) / ((downloaded - existing) / elapsed) if downloaded > existing else 0
                    print(f"  {downloaded/1e9:.2f}/{total/1e9:.2f} GB  ({pct:.1f}%)  {speed:.1f} MB/s  ETA: {eta/60:.0f} min")
                    last_print = now

        final_size = dest.stat().st_size
        if total > 0 and final_size != total:
            print(f"  Incomplete: got {final_size/1e9:.2f} GB of {total/1e9:.2f} GB -- re-run to resume")
            return False

        print(f"  Done: {dest.name} ({final_size/1e9:.2f} GB)")
        return True

    except Exception as e:
        print(f"  Connection dropped: {e}")
        print(f"  Re-run this script to resume from {dest.stat().st_size/1e6:.0f} MB")
        return False


print("=" * 58)
print("  Torch CUDA Wheel Downloader (resumable)")
print(f"  Saving to: {DEST_DIR}")
print("=" * 58)

all_done = True
for filename, url in WHEELS:
    dest = DEST_DIR / filename
    print(f"\n[{filename}]")
    ok = download_resumable(url, dest)
    if not ok:
        all_done = False

print()
if all_done:
    print("=" * 58)
    print("  All wheels downloaded. Run:")
    print(f"  pip install \"{DEST_DIR}\\torch-2.4.1+cu121-cp312-cp312-win_amd64.whl\"")
    print(f"  pip install \"{DEST_DIR}\\torchaudio-2.4.1+cu121-cp312-cp312-win_amd64.whl\"")
    print()
    print("  Then re-run: python scripts/verify_setup.py")
    print("=" * 58)
else:
    print("  Re-run this script to resume incomplete downloads.")