"""Batch & CLI Speech Waveform Generation Script for VOIX (Phase 4).

Generates K diverse candidate speech waveforms (.wav) for a given face from the test set.

Usage:
    python scripts/synthesize_samples.py --test-index 0 --num-voices 3 --text "Hello world"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from voix.data.dataset import VoxCelebPairedDataset
from voix.inference.pipeline import VoixPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="VOIX End-to-End Speech Waveform Generator")
    parser.add_argument("--test-index", type=int, default=0, help="Test set sample index")
    parser.add_argument(
        "--text",
        type=str,
        default="Welcome to VOIX. This speech waveform was synthesized directly from facial biometrics.",
        help="Text prompt for acoustic speech synthesis",
    )
    parser.add_argument("--num-voices", type=int, default=3, help="Number of candidate voices K")
    parser.add_argument("--temperature", type=float, default=1.0, help="Voice diversity temperature")
    parser.add_argument("--output-dir", type=str, default="D:/voix/artifacts/samples", help="Output directory")
    parser.add_argument("--prefix", type=str, default="voice", help="File prefix for saved wavs")
    parser.add_argument("--watermark", action="store_true", default=False, help="Enable AudioSeal watermarking")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("=" * 65)
    print("  VOIX END-TO-END WAVEFORM SYNTHESIS")
    print("=" * 65)
    print(f"Device:       {args.device}")
    print(f"Test Index:   {args.test_index}")
    print(f"Num Voices:   {args.num_voices}")
    print(f"Temperature:  {args.temperature}")
    print(f"Output Dir:   {args.output_dir}")
    print(f"Text Prompt:  \"{args.text}\"")
    print("=" * 65)

    # 1. Initialize Pipeline
    print("\n[1/3] Initializing VoixPipeline...")
    pipeline = VoixPipeline(device=args.device, enable_watermarking=args.watermark)

    # 2. Load Face Feature
    print("\n[2/3] Fetching face vector from test split...")
    _, _, test_ds = VoxCelebPairedDataset.from_index_csv(in_memory=True)
    if args.test_index >= len(test_ds):
        print(f"Error: test_index {args.test_index} exceeds test set size ({len(test_ds)})")
        return

    face, _, spk_id = test_ds[args.test_index]
    print(f"  Speaker ID: {spk_id}")

    # 3. Generate Waveforms
    print(f"\n[3/3] Synthesizing {args.num_voices} candidate speech waveforms...")
    results = pipeline.generate(
        face_input=face,
        text=args.text,
        num_voices=args.num_voices,
        temperature=args.temperature,
        watermark=args.watermark,
    )

    saved_files = pipeline.save_audio(
        results=results,
        output_dir=args.output_dir,
        prefix=f"{args.prefix}_spk{spk_id}_idx{args.test_index}",
    )

    print("\n" + "=" * 65)
    print("  SYNTHESIS COMPLETE")
    for idx, path in enumerate(saved_files, 1):
        dur = results[idx - 1]["waveform"].shape[1] / results[idx - 1]["sample_rate"]
        print(f"  Voice #{idx}: {path} ({dur:.2f}s, 24 kHz)")
    print("=" * 65)


if __name__ == "__main__":
    main()
