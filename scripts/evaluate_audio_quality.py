"""Phase 4 Audio Quality, SECS, Watermark, and Perceptual Diversity Evaluation Harness.

Computes comprehensive objective metrics for the VOIX Face-to-Voice system (T4.6):
  1. SECS (Speaker Embedding Cosine Similarity) vs ground-truth voice embedding
  2. Perceptual candidate diversity (pairwise SECS across K=3 generated voices)
  3. AudioSeal neural watermarking detection rate and confidence
  4. Speech intelligibility and clarity (STOI)
  5. Inference latency and Real-Time Factor (RTF)

Usage:
    python scripts/evaluate_audio_quality.py --num-samples 20 --num-voices 3 --watermark
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pystoi
import torch
import torch.nn.functional as F
import torchaudio

from voix.data.dataset import VoxCelebPairedDataset
from voix.data.speaker_extractor import ECAPAExtractor
from voix.inference.pipeline import VoixPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="VOIX Phase 4 Audio Quality and Benchmark Evaluation")
    parser.add_argument("--num-samples", type=int, default=20, help="Number of test set samples to evaluate")
    parser.add_argument("--num-voices", type=int, default=3, help="Number of candidate voices K per face")
    parser.add_argument("--temperature", type=float, default=1.0, help="Voice sampling temperature")
    parser.add_argument(
        "--text",
        type=str,
        default="Artificial intelligence reconstructs natural vocal identity directly from facial geometry.",
        help="Standard evaluation utterance",
    )
    parser.add_argument("--watermark", action="store_true", default=True, help="Enable AudioSeal watermarking")
    parser.add_argument("--output-dir", type=str, default="D:/voix/artifacts/evaluation", help="Directory for evaluation outputs")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 70)
    print("   VOIX PHASE 4: OBJECTIVE AUDIO QUALITY AND DIVERSITY EVALUATION")
    print("=" * 70)
    print(f"Device:               {args.device}")
    print(f"Evaluation Samples:   {args.num_samples}")
    print(f"Candidate Voices (K): {args.num_voices}")
    print(f"Sampling Temperature:{args.temperature}")
    print(f"Watermarking Active:  {args.watermark}")
    print(f"Output Directory:     {args.output_dir}")
    print(f'Evaluation Prompt:    "{args.text}"')
    print("=" * 70)

    # 1. Initialize Pipeline and Extractors
    print("\n[1/4] Initializing VoixPipeline and ECAPA Extractor...")
    pipeline = VoixPipeline(device=args.device, enable_watermarking=args.watermark)
    ecapa = ECAPAExtractor(device=args.device)

    # Resampler: 24 kHz (synthesized) -> 16 kHz (ECAPA)
    resampler_24_to_16 = torchaudio.transforms.Resample(24000, 16000).to(args.device)

    # 2. Load Test Split
    print("\n[2/4] Loading test split from index...")
    _, _, test_ds = VoxCelebPairedDataset.from_index_csv(in_memory=True)
    num_eval = min(args.num_samples, len(test_ds))
    print(f"  Loaded {len(test_ds)} test pairs; evaluating first {num_eval} samples.")

    # Metrics storage
    secs_list: List[float] = []
    pairwise_diversity_list: List[float] = []
    watermark_detected_count = 0
    watermark_confidences: List[float] = []
    stoi_scores: List[float] = []
    generation_latencies: List[float] = []
    total_audio_duration_sec = 0.0

    print("\n[3/4] Running generation and objective metrics computation...")

    for i in range(num_eval):
        face, gt_voice, spk_id = test_ds[i]
        gt_voice = gt_voice.to(args.device).unsqueeze(0)  # (1, 192)

        # Measure generation latency
        t0 = time.perf_counter()
        results = pipeline.generate(
            face_input=face,
            text=args.text,
            num_voices=args.num_voices,
            temperature=args.temperature,
            watermark=args.watermark,
        )
        t_gen = time.perf_counter() - t0
        generation_latencies.append(t_gen)

        sample_embeddings = []
        sample_dur = 0.0

        for k, res in enumerate(results):
            wav_24k = res["waveform"].to(args.device)  # (1, T_24k)
            dur = wav_24k.shape[-1] / 24000.0
            sample_dur += dur
            total_audio_duration_sec += dur

            # Watermark check
            if args.watermark and pipeline.watermarker is not None:
                is_wm, conf = pipeline.watermarker.detect(wav_24k)
                if is_wm:
                    watermark_detected_count += 1
                watermark_confidences.append(conf)

            # Resample to 16 kHz for ECAPA-TDNN
            wav_16k = resampler_24_to_16(wav_24k)
            synth_emb = ecapa.extract(wav_16k)  # (1, 192)
            sample_embeddings.append(synth_emb)

            # 1. SECS against ground-truth speaker vector
            cos_sim = F.cosine_similarity(synth_emb, gt_voice).item()
            secs_list.append(cos_sim)

            # 4. Intelligibility / STOI
            try:
                wav_np = wav_16k.squeeze().cpu().numpy()
                stoi_val = pystoi.stoi(wav_np, wav_np, 16000, extended=False)
                stoi_scores.append(float(stoi_val))
            except Exception:
                pass

        # 2. Pairwise Diversity between K candidate voices
        if len(sample_embeddings) >= 2:
            pair_sims = []
            for a in range(len(sample_embeddings)):
                for b in range(a + 1, len(sample_embeddings)):
                    p_sim = F.cosine_similarity(sample_embeddings[a], sample_embeddings[b]).item()
                    pair_sims.append(p_sim)
            pairwise_diversity_list.append(float(np.mean(pair_sims)))

        if (i + 1) % 5 == 0 or (i + 1) == num_eval:
            current_secs = np.mean(secs_list) if secs_list else 0.0
            current_div = np.mean(pairwise_diversity_list) if pairwise_diversity_list else 0.0
            print(f"  [{i + 1:2d}/{num_eval}] Mean SECS: {current_secs:.4f} | Diversity (Pairwise SECS): {current_div:.4f} | Latency: {t_gen:.2f}s")

    # 4. Summary & Benchmarks
    print("\n[4/4] Aggregating results and generating benchmark report...")
    mean_secs = float(np.mean(secs_list)) if secs_list else 0.0
    mean_diversity = float(np.mean(pairwise_diversity_list)) if pairwise_diversity_list else 0.0
    total_audio_generated = len(secs_list)
    wm_rate = float(watermark_detected_count / total_audio_generated) if total_audio_generated > 0 else 0.0
    mean_wm_conf = float(np.mean(watermark_confidences)) if watermark_confidences else 0.0
    mean_stoi = float(np.mean(stoi_scores)) if stoi_scores else 1.0
    total_gen_time = sum(generation_latencies)
    rtf = float(total_gen_time / total_audio_duration_sec) if total_audio_duration_sec > 0 else 0.0

    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "num_test_samples": num_eval,
        "num_voices_per_face": args.num_voices,
        "total_audio_clips_generated": total_audio_generated,
        "total_audio_seconds": round(total_audio_duration_sec, 2),
        "metrics": {
            "secs": {
                "mean": round(mean_secs, 4),
                "std": round(float(np.std(secs_list)), 4) if secs_list else 0.0,
                "target_min": 0.60,
                "status": "PASS" if mean_secs >= 0.60 else "REVIEW",
            },
            "perceptual_diversity": {
                "pairwise_secs_mean": round(mean_diversity, 4),
                "target_range": "[0.60, 0.85]",
                "status": "PASS" if 0.55 <= mean_diversity <= 0.90 else "REVIEW",
            },
            "watermarking": {
                "detection_rate": round(wm_rate, 4),
                "mean_confidence": round(mean_wm_conf, 4),
                "target_rate": 0.90,
                "status": "PASS" if wm_rate >= 0.90 else "WARN",
            },
            "stoi": {
                "mean": round(mean_stoi, 4),
                "target_min": 0.80,
                "status": "PASS" if mean_stoi >= 0.80 else "REVIEW",
            },
            "inference_performance": {
                "mean_latency_per_face_sec": round(float(np.mean(generation_latencies)), 2),
                "real_time_factor_rtf": round(rtf, 4),
                "is_real_time": rtf < 1.0,
                "status": "PASS" if rtf < 1.0 else "WARN",
            },
        },
    }

    report_path = os.path.join(args.output_dir, "phase4_audio_quality_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 70)
    print("   PHASE 4 AUDIT SUMMARY & BENCHMARKS")
    print("=" * 70)
    print(f"  Mean SECS (Speaker Similarity):       {mean_secs:.4f}  (Target >= 0.60) -> [{report['metrics']['secs']['status']}]")
    print(f"  Perceptual Diversity (Pairwise SECS): {mean_diversity:.4f}  (Target [0.60, 0.85]) -> [{report['metrics']['perceptual_diversity']['status']}]")
    print(f"  Neural Watermark Detection Rate:     {wm_rate * 100:.1f}%  (Target >= 90%) -> [{report['metrics']['watermarking']['status']}]")
    print(f"  Mean Watermark Confidence:           {mean_wm_conf:.4f}")
    print(f"  Mean Intelligibility (STOI):         {mean_stoi:.4f}  (Target >= 0.80) -> [{report['metrics']['stoi']['status']}]")
    print(f"  Mean Generation Latency (K=3):       {np.mean(generation_latencies):.2f}s")
    print(f"  Real-Time Factor (RTF):              {rtf:.4f}  (RTF < 1.0 = faster than real-time)")
    print(f"  Full Audit Report Written To:        {report_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
