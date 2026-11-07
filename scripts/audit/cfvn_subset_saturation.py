"""Measure how CFVN retrieval scales with the number of training speakers.

The audit requires a subset-saturation check before committing to a full
corpus.  This script trains the same CFVN recipe on deterministic speaker
subsets of the training split and reports held-out face-to-voice retrieval on
the untouched validation split.

The subset is selected by speaker, never by clip, so no speaker is partially
represented across fractions.  The scaler is refit on each subset's train rows
to keep the comparison honest.

Usage:
    python scripts/audit/cfvn_subset_saturation.py --fractions 0.25,0.5,1.0 --epochs 12
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from voix.models.contrastive_net import ContrastiveFaceVoiceNet
from voix.training.data import (
    DEFAULT_FEATURES_V2,
    DEFAULT_INDEX_V2,
    SpeakerBatchSampler,
    build_face_scaler_v2,
    load_v2_dataset,
)
from voix.training.trainer_cfvn import (
    evaluate,
    normalize_faces,
    normalize_speaker_features,
    speaker_centroid_targets,
    supervised_contrastive_loss,
    variance_covariance_terms,
)
from voix.utils.seed import set_global_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CFVN training-speaker saturation curve"
    )
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES_V2)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX_V2)
    parser.add_argument("--face-key", default="face_features")
    parser.add_argument("--fractions", default="0.25,0.5,1.0")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-speakers", type=int, default=32)
    parser.add_argument("--batch-clips", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lambda-reg", type=float, default=5.0)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "results" / "cfvn_subset_saturation.json"
    )
    return parser.parse_args()


def _train_once(
    train,
    val,
    *,
    face_key: str,
    args: argparse.Namespace,
    device: str,
) -> Dict[str, float]:
    scaler = build_face_scaler_v2(
        train,
        output_path=ROOT / "results" / "saturation_scaler.pt",
        source="subset_saturation",
        face_key=face_key,
    )
    face_train = normalize_faces(train.face_raw, scaler)
    face_val = normalize_faces(val.face_raw, scaler)
    face_id_val = F.normalize(torch.from_numpy(val.face_raw).float(), dim=-1)
    spk_train = normalize_speaker_features(train.speaker_raw)
    spk_val = normalize_speaker_features(val.speaker_raw)
    spk_target = speaker_centroid_targets(spk_train, train.speaker_ids)

    model = ContrastiveFaceVoiceNet(
        face_dim=face_train.shape[1],
        out_dim=192,
        use_layernorm=False,
        residual=True,
    ).to(device)
    sampler = SpeakerBatchSampler(
        train.speaker_to_indices(),
        speakers_per_batch=args.batch_speakers,
        clips_per_speaker=args.batch_clips,
        seed=args.seed,
    )
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))
    labels = {
        speaker: index
        for index, speaker in enumerate(sorted(set(train.speaker_ids.tolist())))
    }

    best_score = -math.inf
    best_metrics: Dict[str, float] = {}
    for _epoch in range(1, args.epochs + 1):
        model.train()
        for batch in sampler:
            face = face_train[batch].to(device)
            target = spk_target[batch].to(device)
            label_tensor = torch.tensor(
                [labels[str(train.speaker_ids[i])] for i in batch],
                dtype=torch.long,
                device=device,
            )
            out = model(face)
            supcon = supervised_contrastive_loss(
                out, label_tensor, args.temperature
            )
            reg = (1.0 - F.cosine_similarity(out, target, dim=-1)).mean()
            var, cov = variance_covariance_terms(out)
            loss = supcon + args.lambda_reg * reg + 0.05 * var + 0.01 * cov
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        metrics = evaluate(model, val, face_val, spk_val, device, face_id_val)
        if metrics["selection_score"] > best_score:
            best_score = metrics["selection_score"]
            best_metrics = dict(metrics)
    return best_metrics


def main() -> None:
    args = parse_args()
    set_global_seed(args.seed)
    device = (
        args.device
        if (args.device != "cuda" or torch.cuda.is_available())
        else "cpu"
    )
    fractions = [float(value) for value in args.fractions.split(",") if value]

    dataset = load_v2_dataset(args.features, args.index, face_key=args.face_key)
    train = dataset.subset("train")
    val = dataset.subset("val")
    if len(val) == 0:
        raise ValueError("Saturation check requires a non-empty validation split")

    speakers = sorted(set(train.speaker_ids.tolist()))
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(speakers))
    results: List[Dict[str, float]] = []

    for fraction in fractions:
        if not 0.0 < fraction <= 1.0:
            raise ValueError(f"Invalid fraction: {fraction}")
        count = max(2, int(round(fraction * len(speakers))))
        chosen = {speakers[int(i)] for i in order[:count]}
        mask = np.asarray([str(s) in chosen for s in train.speaker_ids])
        subset = train.select(np.flatnonzero(mask))
        print("=" * 72)
        print(
            f"Fraction {fraction:.0%}: {len(chosen):,} speakers / "
            f"{len(subset):,} clips"
        )
        started = time.time()
        metrics = _train_once(
            subset,
            val,
            face_key=args.face_key,
            args=args,
            device=device,
        )
        row = {
            "fraction": fraction,
            "train_speakers": len(chosen),
            "train_clips": int(len(subset)),
            "val_r1": metrics["val_r1"],
            "val_r10": metrics["val_r10"],
            "val_chance_r1": metrics["val_chance_r1"],
            "val_chance_multiple": metrics["val_r1"]
            / max(metrics["val_chance_r1"], 1e-9),
            "val_gender_controlled_r1": metrics["val_gender_controlled_r1"],
            "val_gender_controlled_chance_r1": metrics[
                "val_gender_controlled_chance_r1"
            ],
            "cfvn_face_face_r1": metrics["cfvn_face_face_r1"],
            "raw_face_face_r1": metrics["raw_face_face_r1"],
            "within_speaker_cosine": metrics["within_speaker_cosine"],
            "between_speaker_cosine": metrics["between_speaker_cosine"],
            "seconds": round(time.time() - started, 1),
        }
        results.append(row)
        print(json.dumps(row, indent=2))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "features": str(args.features),
                "index": str(args.index),
                "face_key": args.face_key,
                "epochs": args.epochs,
                "seed": args.seed,
                "results": results,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
