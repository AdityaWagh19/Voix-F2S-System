"""Inference CLI tool to generate multiple diverse voice candidates for a face embedding."""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn.functional as F

from voix.data.dataset import VoxCelebPairedDataset
from voix.models.cvae import CVAE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="VOIX Multi-Voice Generator (Phase 3)")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="D:/voix/checkpoints/cvae_best.pt",
        help="Path to trained CVAE checkpoint",
    )
    parser.add_argument(
        "--scaler",
        type=str,
        default="D:/voix/data/processed/feature_scaler.pt",
        help="Path to feature normalisation scaler",
    )
    parser.add_argument(
        "--test-index",
        type=int,
        default=0,
        help="Index of sample in the test set to synthesize voices for",
    )
    parser.add_argument(
        "--num-voices",
        type=int,
        default=5,
        help="Number of diverse candidate voices to sample (K)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature (higher = greater voice variance)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="D:/voix/checkpoints/sampled_voices.pt",
        help="Path to save generated candidate speaker vectors",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("=" * 65)
    print("  VOIX MULTI-VOICE INFERENCE GENERATOR")
    print("=" * 65)
    print(f"Checkpoint:   {args.checkpoint}")
    print(f"Num Voices:   {args.num_voices}")
    print(f"Temperature:  {args.temperature}")
    print("=" * 65)

    # 1. Load Model
    ckpt = torch.load(args.checkpoint, weights_only=False, map_location="cpu")
    model = CVAE()
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print("Loaded CVAE model successfully.")

    # 2. Load Scaler
    scaler = torch.load(args.scaler, weights_only=True)
    spk_mean = scaler["spk_mean"]
    spk_std  = scaler["spk_std"]

    # 3. Load Sample from Test Set
    _, _, test_ds = VoxCelebPairedDataset.from_index_csv()
    if args.test_index >= len(test_ds):
        print(f"Error: test_index {args.test_index} exceeds test set size ({len(test_ds)})")
        return

    face, spk_gt_norm, spk_id = test_ds[args.test_index]
    print(f"\nEvaluating Test Sample #{args.test_index} (Speaker ID: {spk_id})")

    # 4. Generate Deterministic Baseline
    with torch.no_grad():
        out_det = model(face.unsqueeze(0), deterministic=True)
        det_norm = out_det.e_s_hat.squeeze(0)

    # 5. Generate K Diverse Candidate Voices
    candidates_norm = model.sample_voices(
        face,
        num_samples=args.num_voices,
        temperature=args.temperature,
    )  # (K, 192)

    # Un-normalize to true ECAPA embedding space
    det_raw = det_norm * spk_std + spk_mean
    spk_gt_raw = spk_gt_norm * spk_std + spk_mean
    candidates_raw = candidates_norm * spk_std + spk_mean

    # Metrics
    cos_norm = F.cosine_similarity(det_norm.unsqueeze(0), spk_gt_norm.unsqueeze(0)).item()
    cos_raw  = F.cosine_similarity(det_raw.unsqueeze(0), spk_gt_raw.unsqueeze(0)).item()

    # Pairwise candidate diversity
    pairwise = torch.cdist(candidates_raw, candidates_raw)
    mask = torch.triu(torch.ones(args.num_voices, args.num_voices, dtype=torch.bool), diagonal=1)
    mean_dist = pairwise[mask].mean().item()

    print("\n--- Synthesis Results ---")
    print(f"Deterministic Voice vs Ground Truth:")
    print(f"  Normalized Cosine Sim : {cos_norm:.4f}")
    print(f"  Raw ECAPA Cosine Sim  : {cos_raw:.4f}")
    print(f"\n{args.num_voices}-Voice Candidates Diversity:")
    print(f"  Mean Pairwise Distance: {mean_dist:.4f}")

    print("\nPer-Candidate Cosine Similarity to Ground-Truth:")
    for k in range(args.num_voices):
        c_k_norm = candidates_norm[k].unsqueeze(0)
        c_k_raw  = candidates_raw[k].unsqueeze(0)
        sim_norm = F.cosine_similarity(c_k_norm, spk_gt_norm.unsqueeze(0)).item()
        sim_raw  = F.cosine_similarity(c_k_raw, spk_gt_raw.unsqueeze(0)).item()
        print(f"  Voice #{k+1}: Raw SECS = {sim_raw:.4f}  (Norm CosSim = {sim_norm:.4f})")

    # 6. Save Candidates
    save_dict = {
        "test_index": args.test_index,
        "speaker_id": spk_id,
        "deterministic_voice": det_raw,
        "candidate_voices": candidates_raw,
        "ground_truth_voice": spk_gt_raw,
        "temperature": args.temperature,
    }
    torch.save(save_dict, args.output)
    print(f"\nSaved generated voice embeddings to: {args.output}")
    print("=" * 65)


if __name__ == "__main__":
    main()
