"""Extract paired (ECAPA 192-D, StyleTTS 2 128-D Timbre, StyleTTS 2 128-D Prosody) targets."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, r"D:\voix")

import torch
from styletts2 import tts
from voix.data.speaker_extractor import ECAPAExtractor


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    ecapa = ECAPAExtractor(device=device)
    m = tts.StyleTTS2()

    ref_root = r"D:\voix\data\raw\librispeech_ref"
    audio_files = []
    for root, dirs, files in os.walk(ref_root):
        for f in files:
            if f.endswith(".flac"):
                audio_files.append(os.path.join(root, f))

    print(f"Found {len(audio_files)} audio files.")
    # Use 500 diverse clips for comprehensive multi-speaker manifold coverage
    audio_files = audio_files[:500]

    ecapa_list = []
    timbre_list = []
    prosody_list = []

    t0 = time.time()
    for idx, path in enumerate(audio_files, 1):
        try:
            e_s = ecapa.extract_from_path(path).cpu().squeeze(0)  # (192,)
            ref_s = m.compute_style(path).cpu().squeeze(0)        # (256,)

            ecapa_list.append(e_s)
            timbre_list.append(ref_s[:128])
            prosody_list.append(ref_s[128:])
        except Exception as exc:
            print(f"Error processing {path}: {exc}")

        if idx % 50 == 0 or idx == len(audio_files):
            print(f"[{idx:3d}/{len(audio_files)}] Elapsed: {time.time() - t0:.1f}s")

    e_s_tensor = torch.stack(ecapa_list, dim=0)
    timbre_tensor = torch.stack(timbre_list, dim=0)
    prosody_tensor = torch.stack(prosody_list, dim=0)

    out_path = r"D:\voix\data\processed\style_pairs.pt"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save({
        "ecapa": e_s_tensor,
        "timbre": timbre_tensor,
        "prosody": prosody_tensor,
        "n_samples": len(ecapa_list),
    }, out_path)

    print(f"Saved {len(ecapa_list)} pairs to {out_path}!")
    print(f"ECAPA shape: {e_s_tensor.shape}, Timbre shape: {timbre_tensor.shape}")


if __name__ == "__main__":
    main()
