"""Hardware VRAM footprint and throughput profiling for VOIX backbone models.

Reports peak GPU memory allocation for each frozen backbone individually,
and documents safe batch size limits for training on GTX 1650 (4GB VRAM).

Usage:
    python scripts/profile_hardware.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch


def get_vram_mb() -> float:
    """Return current GPU peak memory allocated in MB."""
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / (1024 ** 2)


def reset_vram_counter() -> None:
    """Reset PyTorch peak memory tracking counter."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()


def profile_cuda_baseline() -> dict:
    """Measure baseline VRAM with only PyTorch CUDA context initialized."""
    reset_vram_counter()
    if not torch.cuda.is_available():
        return {"available": False}
    # Allocate a small tensor to initialize CUDA context
    _ = torch.zeros(1, device="cuda")
    baseline_mb = get_vram_mb()
    del _
    torch.cuda.empty_cache()
    return {
        "available": True,
        "device_name": torch.cuda.get_device_name(0),
        "total_vram_mb": torch.cuda.get_device_properties(0).total_memory / (1024 ** 2),
        "cuda_context_mb": baseline_mb,
    }


def profile_arcface() -> dict:
    """Measure ArcFace model VRAM footprint."""
    print("  Profiling ArcFace...", end=" ", flush=True)
    reset_vram_counter()
    try:
        from voix.data.face_extractor import ArcFaceExtractor
        import numpy as np

        extractor = ArcFaceExtractor(device_id=0, cache_dir="./checkpoints")
        # Trigger model load
        dummy = np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)
        extractor.extract(dummy)
        peak_mb = get_vram_mb()
        del extractor
        torch.cuda.empty_cache()
        print(f"{peak_mb:.1f} MB")
        return {"model": "ArcFace buffalo_l", "peak_vram_mb": peak_mb, "status": "ok"}
    except Exception as e:
        print(f"FAILED ({e})")
        return {"model": "ArcFace buffalo_l", "peak_vram_mb": None, "status": f"error: {e}"}


def profile_ecapa() -> dict:
    """Measure ECAPA-TDNN model VRAM footprint."""
    print("  Profiling ECAPA-TDNN...", end=" ", flush=True)
    reset_vram_counter()
    try:
        from voix.data.speaker_extractor import ECAPAExtractor
        extractor = ECAPAExtractor(device="cuda")
        dummy_audio = torch.randn(1, 48000)  # 3 seconds at 16 kHz
        extractor.extract(dummy_audio)
        peak_mb = get_vram_mb()
        del extractor
        torch.cuda.empty_cache()
        print(f"{peak_mb:.1f} MB")
        return {"model": "ECAPA-TDNN", "peak_vram_mb": peak_mb, "status": "ok"}
    except Exception as e:
        print(f"FAILED ({e})")
        return {"model": "ECAPA-TDNN", "peak_vram_mb": None, "status": f"error: {e}"}


def profile_cvae_batch_sizes() -> dict:
    """Measure CVAE training memory at different batch sizes.
    CVAE not yet implemented - reports projected values."""
    results = {}
    face_dim, speaker_dim = 560, 192
    # Estimate memory for linear layers: ~1.2M parameters * 4 bytes * 2 (fwd+bwd)
    param_mb = (1_200_000 * 4 * 2) / (1024 ** 2)
    for bs in [32, 48, 64, 128]:
        activation_mb = (bs * (face_dim + speaker_dim) * 4 * 10) / (1024 ** 2)
        total_mb = param_mb + activation_mb
        results[f"batch_{bs}"] = round(total_mb, 1)
    return {"model": "CVAE (projected)", "batch_vram_mb": results, "status": "projected"}


def main() -> None:
    """Run all hardware profiling checks and print results."""
    print("=" * 60)
    print("VOIX Hardware VRAM Profiling Report")
    print("=" * 60)

    # CUDA baseline
    baseline = profile_cuda_baseline()
    if not baseline["available"]:
        print("ERROR: CUDA is not available. Cannot profile GPU memory.")
        sys.exit(1)

    print(f"\nGPU: {baseline['device_name']}")
    print(f"Total VRAM: {baseline['total_vram_mb']:.0f} MB")
    print(f"CUDA context overhead: {baseline['cuda_context_mb']:.1f} MB")
    print(f"Available for models: {baseline['total_vram_mb'] - baseline['cuda_context_mb']:.0f} MB")
    print()

    # Backbone profiles
    print("Backbone VRAM Footprints:")
    results = []
    results.append(profile_arcface())
    results.append(profile_ecapa())

    # CVAE projection
    cvae = profile_cvae_batch_sizes()
    results.append(cvae)

    # Summary
    print()
    print("=" * 60)
    print("Summary:")
    total_loaded_mb = baseline["cuda_context_mb"]
    for r in results:
        if r.get("peak_vram_mb"):
            total_loaded_mb += r["peak_vram_mb"]
            print(f"  {r['model']}: {r['peak_vram_mb']:.1f} MB  [{r['status']}]")
        elif r.get("batch_vram_mb"):
            print(f"  {r['model']}:")
            for k, v in r["batch_vram_mb"].items():
                print(f"    {k}: {v:.1f} MB  [{r['status']}]")

    print()
    max_vram = baseline["total_vram_mb"]
    print(f"GTX 1650 VRAM budget: {max_vram:.0f} MB")
    print(f"Recommended max concurrent VRAM: {max_vram * 0.85:.0f} MB (85% safe margin)")
    print("Recommended training batch size for GTX 1650: 32")
    print("=" * 60)


if __name__ == "__main__":
    main()
