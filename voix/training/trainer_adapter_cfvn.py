"""Train the StyleAdapter on FaceFeatureV2 CFVN embeddings.

The adapter maps the 192-D voice-space embedding produced by the V2 CFVN to the
256-D StyleTTS2 conditioning vector ``[timbre | prosody]``.  This version fixes
the legacy trainer's two evaluation defects:

* train/validation are the manifest's speaker-disjoint splits rather than a
  random clip split,
* the face representation is z-scored with the V2 scaler and passed through
  the V2 CFVN, exactly matching inference.

Style targets come from a ``StyleTargetV2`` artifact: one 256-D
``[timbre | prosody]`` vector per speaker, extracted from that speaker's own
VoxCeleb2 audio by the frozen StyleTTS2 encoders.  Because the targets carry
speaker labels and follow the manifest splits, both sides of the adapter are
speaker-disjoint.  The legacy 497-row ``style_pairs.pt`` artifact is still
loadable for reproducing old runs, but it has no speaker labels and its
nearest-neighbour matching is too noisy to train on.

Usage:
    python -m voix.training.trainer_adapter_cfvn \
        --features data/processed/voxceleb2_v2/features_v2.h5 \
        --index data/processed/voxceleb2_v2/index.csv \
        --scaler data/processed/voxceleb2_v2/face_scaler_v2.pt \
        --cfvn results/checkpoints/cfvn_v2_best.pt \
        --style-targets data/processed/style_targets_voxceleb2_5clip.pt \
        --output checkpoints/style_adapter_cfvn_best.pt
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset

from voix.data.feature_contract import load_scaler_payload
from voix.models.adapter import StyleAdapter
from voix.models.contrastive_net import ContrastiveFaceVoiceNet
from voix.models.cvae import CVAE
from voix.training.data import (
    DEFAULT_FEATURES_V2,
    DEFAULT_INDEX_V2,
    DEFAULT_SCALER_V2,
    load_v2_dataset,
    split_summary,
    validate_scaler_for_dataset,
)
from voix.training.v2_utils import (
    align_style_targets,
    embed_faces,
    load_style_targets,
    match_style_targets,
    normalize_face,
    pairwise_identity_stats,
    speaker_codes,
    variance_covariance_terms,
)
from voix.utils.seed import set_global_seed

ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_CFVN = ROOT / "results" / "checkpoints" / "cfvn_v2_best.pt"
DEFAULT_CVAE = ROOT / "checkpoints" / "cvae_cfvn_face192_latent128_diversity.pt"
DEFAULT_STYLE_TARGETS = (
    ROOT / "data" / "processed" / "style_targets_voxceleb2_5clip.pt"
)
DEFAULT_STYLE_PAIRS = ROOT / "data" / "processed" / "style_pairs.pt"
DEFAULT_SAVE = ROOT / "checkpoints" / "style_adapter_cfvn_best.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the dual-head StyleAdapter on speaker-disjoint V2 data"
    )
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES_V2)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX_V2)
    parser.add_argument("--scaler", type=Path, default=DEFAULT_SCALER_V2)
    parser.add_argument(
        "--face-key",
        default="face_features",
        help="HDF5 face dataset used to train the CFVN (face_features or arcface).",
    )
    parser.add_argument("--cfvn", type=Path, default=DEFAULT_CFVN)
    parser.add_argument("--cvae", type=Path, default=DEFAULT_CVAE)
    parser.add_argument(
        "--adapter-input",
        choices=("cvae", "cfvn"),
        default="cvae",
        help=(
            "'cvae' trains on the serving distribution (CVAE decoder samples); "
            "'cfvn' is the legacy diagnostic path and is not recommended."
        ),
    )
    parser.add_argument("--samples-per-clip", type=int, default=4)
    parser.add_argument("--cvae-temperature", type=float, default=1.0)
    parser.add_argument(
        "--style-targets",
        type=Path,
        default=DEFAULT_STYLE_TARGETS,
        help=(
            "StyleTargetV2 per-speaker artifact extracted from VoxCeleb2 audio "
            "with scripts/data/extract_style_targets_v2.py."
        ),
    )
    parser.add_argument(
        "--style-pairs",
        type=Path,
        default=None,
        help=(
            "Legacy 497-row style_pairs.pt artifact. Its rows are unlabeled "
            "and only weakly matched to V2 speakers; kept to reproduce old runs."
        ),
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_SAVE)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--lambda-cos", type=float, default=1.0)
    parser.add_argument(
        "--lambda-div",
        type=float,
        default=0.5,
        help=(
            "Weight on the different-speaker output-cosine penalty. Without "
            "this term the adapter learns the global style mean and collapses "
            "all identities toward the same 256-D vector."
        ),
    )
    parser.add_argument("--lambda-var", type=float, default=0.01)
    parser.add_argument(
        "--lambda-norm",
        type=float,
        default=1.0,
        help=(
            "Weight on matching the centered style residual norm. Diversity "
            "alone is scale-free and can shrink residuals toward zero."
        ),
    )
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_cvae(path: Path, face_dim: int, device: str) -> Tuple[CVAE, dict]:
    """Load the CVAE checkpoint whose decoder defines the serving input."""
    if not path.exists():
        raise FileNotFoundError(
            f"CVAE checkpoint not found: {path}\n"
            "Train it with voix.training.trainer_cvae_v3_diversity, or pass "
            "--adapter-input cfvn for a diagnostic run."
        )
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=True)
    except Exception:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    state = checkpoint.get("model_state_dict", checkpoint)
    latent_dim = int(checkpoint.get("latent_dim", 128))
    speaker_dim = int(checkpoint.get("speaker_dim", 192))
    checkpoint_face_dim = int(checkpoint.get("face_dim", face_dim))
    if checkpoint_face_dim != face_dim:
        raise ValueError(
            f"CVAE expects {checkpoint_face_dim}-D CFVN embeddings but the CFVN "
            f"produces {face_dim}-D; pass the matching --cfvn/--cvae pair"
        )
    model = CVAE(
        face_dim=face_dim,
        speaker_dim=speaker_dim,
        latent_dim=latent_dim,
    ).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, checkpoint


@torch.no_grad()
def cvae_sample_inputs(
    cvae: CVAE,
    cfvn_face: torch.Tensor,
    *,
    num_samples: int,
    temperature: float,
    device: str,
    batch_size: int = 512,
) -> torch.Tensor:
    """Return ``(N, K, 192)`` CVAE decoder samples for each face embedding."""
    outputs = []
    for start in range(0, len(cfvn_face), batch_size):
        batch = cfvn_face[start:start + batch_size].to(device)
        samples = cvae.sample_voices(
            batch,
            num_samples=num_samples,
            temperature=temperature,
        )
        # Inference L2-normalizes CVAE candidates before the adapter.  Train on
        # exactly that distribution, not on unnormalized decoder magnitudes.
        samples = F.normalize(samples, dim=-1)
        outputs.append(samples.cpu().view(batch.size(0), num_samples, -1))
    if not outputs:
        return cfvn_face.new_zeros((0, num_samples, 192))
    return torch.cat(outputs, dim=0)


@torch.no_grad()
def cvae_prior_mean_inputs(
    cvae: CVAE,
    cfvn_face: torch.Tensor,
    *,
    device: str,
    batch_size: int = 512,
) -> torch.Tensor:
    """Return the deterministic prior-mean decoder output used by the pipeline."""
    outputs = []
    for start in range(0, len(cfvn_face), batch_size):
        batch = cfvn_face[start:start + batch_size].to(device)
        prior_mu, _ = cvae.prior_net(batch)
        outputs.append(F.normalize(cvae.decoder(prior_mu), dim=-1).cpu())
    if not outputs:
        return cfvn_face.new_zeros((0, 192))
    return torch.cat(outputs, dim=0)


def _head_losses(
    prediction: torch.Tensor,
    target: torch.Tensor,
    lambda_cos: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """MSE plus directional cosine loss, reported separately per head."""
    mse_t = F.mse_loss(prediction[:, :128], target[:, :128])
    mse_p = F.mse_loss(prediction[:, 128:], target[:, 128:])
    cos_t = 1.0 - F.cosine_similarity(
        prediction[:, :128], target[:, :128], dim=-1
    ).mean()
    cos_p = 1.0 - F.cosine_similarity(
        prediction[:, 128:], target[:, 128:], dim=-1
    ).mean()
    loss = (mse_t + mse_p) + lambda_cos * (cos_t + cos_p)
    return loss, {
        "loss_timbre_mse": float(mse_t.item()),
        "loss_prosody_mse": float(mse_p.item()),
        "loss_timbre_cos": float(cos_t.item()),
        "loss_prosody_cos": float(cos_p.item()),
        "cosine_timbre": float(1.0 - cos_t.item()),
        "cosine_prosody": float(1.0 - cos_p.item()),
    }


def adapter_diversity_loss(
    predictions: torch.Tensor,
    codes: torch.Tensor,
    center: torch.Tensor | None = None,
) -> torch.Tensor:
    """Mean cosine between *different-speaker* adapter outputs.

    Minimising this term is what prevents the adapter from regressing to the
    target mean.  Same-speaker pairs are excluded because their outputs should
    be similar; the diagonal is excluded automatically by the same mask.

    When ``center`` is supplied, the cosine is computed on ``predictions -
    center``.  Style vectors have a large common component: on the V2 targets
    the mean norm is 2.72 and the centered norm is 1.30, while the centered
    across-speaker cosine is ~0.001.  Penalising raw cosine would therefore
    force the adapter to abandon the mean instead of learning the variation.
    Centering preserves the mean and makes the diversity term target the
    orthogonal style residual.

    This function must stay differentiable: an earlier version was wrapped in
    ``@torch.no_grad()``, so the penalty was logged but never trained on.
    """
    if center is not None:
        predictions = predictions - center.to(predictions.device)
    x = F.normalize(predictions.float(), dim=-1, eps=1e-6)
    codes = codes.view(-1).to(x.device)
    similarity = x @ x.T
    mask = codes[:, None] != codes[None, :]
    if not mask.any():
        return x.sum() * 0.0
    return similarity[mask].mean()


def adapter_centered_norm_loss(
    predictions: torch.Tensor,
    center: torch.Tensor,
    target_norm: float,
) -> torch.Tensor:
    """Match the magnitude of the style residual around the common mean.

    Centered cosine alone is scale-free: an adapter can make residuals
    orthogonal while shrinking them toward zero, which keeps the raw output
    cosine collapsed.  On the V2 targets the common mean has norm 2.72 and the
    residual has norm 1.30, so the residual magnitude has to be matched too.
    """
    centered = predictions - center.to(predictions.device)
    norms = centered.norm(dim=-1)
    return F.relu(float(target_norm) - norms).mean()


def evaluate(
    model: StyleAdapter,
    face: torch.Tensor,
    target: torch.Tensor,
    codes: torch.Tensor,
    device: str,
    batch_size: int,
    lambda_cos: float,
    lambda_div: float,
    diversity_center: torch.Tensor | None = None,
    target_centered_norm: float | None = None,
    lambda_norm: float = 1.0,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total = 0
    sums: Dict[str, float] = {}
    predictions = []
    for start in range(0, len(face), batch_size):
        face_batch = face[start:start + batch_size].to(device)
        target_batch = target[start:start + batch_size].to(device)
        prediction = model(face_batch)
        predictions.append(prediction.cpu())
        loss, metrics = _head_losses(prediction, target_batch, lambda_cos)
        count = face_batch.size(0)
        total_loss += float(loss.item()) * count
        total += count
        for key, value in metrics.items():
            sums[key] = sums.get(key, 0.0) + value * count
    predictions = torch.cat(predictions, dim=0)
    stats = pairwise_identity_stats(predictions, codes)
    raw_loss = total_loss / max(1, total)
    if diversity_center is None:
        centered_stats = stats
        centered_norm_mean = float("nan")
        centered_norm_loss = torch.tensor(0.0)
    else:
        centered_stats = pairwise_identity_stats(
            predictions - diversity_center.cpu(), codes
        )
        centered_norm_mean = float(
            (predictions - diversity_center.cpu()).norm(dim=-1).mean()
        )
        if target_centered_norm is None:
            centered_norm_loss = torch.tensor(0.0)
        else:
            centered_norm_loss = adapter_centered_norm_loss(
                predictions,
                diversity_center.cpu(),
                target_centered_norm,
            )
    diversity = centered_stats["across_cosine"]
    norm_loss = float(centered_norm_loss)
    return {
        "val_loss": raw_loss,
        "val_diversity_loss": diversity,
        "val_centered_norm_loss": norm_loss,
        "val_centered_norm_mean": centered_norm_mean,
        "val_selection_loss": (
            raw_loss
            + lambda_div * diversity
            + lambda_norm * norm_loss
        ),
        **{f"val_{key}": value / max(1, total) for key, value in sums.items()},
        **{f"val_output_{key}": value for key, value in stats.items()},
        **{
            f"val_output_centered_{key}": value
            for key, value in centered_stats.items()
        },
    }


def main() -> None:
    args = parse_args()
    set_global_seed(args.seed)
    device = args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"

    print("=" * 72)
    print("StyleAdapter training - FaceFeatureV2 / speaker-disjoint")
    print("=" * 72)

    dataset = load_v2_dataset(args.features, args.index, face_key=args.face_key)
    scaler = validate_scaler_for_dataset(
        load_scaler_payload(
            args.scaler,
            expected_dim=dataset.face_raw.shape[1],
            face_key=args.face_key,
        ),
        dataset,
        face_key=args.face_key,
    )
    summary = split_summary(dataset)
    print(f"Split summary: {json.dumps(summary)}")
    if "train" not in summary or "val" not in summary:
        raise ValueError("Adapter training requires both train and val splits")

    if args.style_pairs is not None:
        style_pairs = torch.load(
            args.style_pairs, map_location="cpu", weights_only=True
        )
        for key in ("ecapa", "timbre", "prosody"):
            if key not in style_pairs:
                raise ValueError(f"{args.style_pairs} is missing the {key!r} tensor")
        print(
            f"Style pairs (legacy): {args.style_pairs} "
            f"(ecapa={tuple(style_pairs['ecapa'].shape)}, "
            f"timbre={tuple(style_pairs['timbre'].shape)}, "
            f"prosody={tuple(style_pairs['prosody'].shape)})"
        )
        style_source = "style_pairs"
        style_target_path = args.style_pairs
        style_target_schema = "style_pairs"
        style_targets_speaker_disjoint = False
    else:
        style_targets = load_style_targets(args.style_targets)
        print(
            f"Style targets: {args.style_targets} "
            f"({len(style_targets['style'])} speakers, "
            f"{style_targets['style'].shape[1]}-D)"
        )
        style_source = "style_targets_v2"
        style_target_path = args.style_targets
        style_target_schema = style_targets["schema_version"]
        style_targets_speaker_disjoint = True

    face = normalize_face(dataset.face_raw, scaler)
    cfvn = ContrastiveFaceVoiceNet.from_checkpoint(str(args.cfvn), device=device).to(device)
    cfvn.eval()
    if cfvn.face_dim != face.shape[1]:
        raise ValueError(
            f"CFVN expects {cfvn.face_dim}-D faces but {args.face_key} provides "
            f"{face.shape[1]}-D; pass the matching --face-key/--cfvn pair"
        )
    if cfvn.out_dim != 192:
        raise ValueError(
            f"StyleAdapter/inference expect a 192-D CFVN output, got {cfvn.out_dim}"
        )
    print(f"CFVN: {args.cfvn} (face_dim={cfvn.face_dim}, out_dim={cfvn.out_dim})")

    print("Computing CFVN embeddings...")
    cfvn_face = embed_faces(cfvn, face, device)
    if style_source == "style_pairs":
        targets, match_stats = match_style_targets(
            dataset.speaker_raw,
            dataset.speaker_ids,
            style_pairs,
        )
    else:
        targets, match_stats = align_style_targets(
            dataset.speaker_ids,
            style_targets,
        )
    print(
        f"  CFVN embeddings: {tuple(cfvn_face.shape)} | "
        f"style targets: {tuple(targets.shape)}"
    )
    print(f"  Style alignment: {json.dumps(match_stats)}")

    train_mask = dataset.split == "train"
    val_mask = dataset.split == "val"
    train_cfvn = cfvn_face[train_mask]
    val_cfvn = cfvn_face[val_mask]
    train_target = targets[train_mask]
    val_target = targets[val_mask]
    target_center = train_target.mean(dim=0)
    target_centered_norm = float(
        (train_target - target_center).norm(dim=-1).mean()
    )
    print(
        f"  Target center norm={float(target_center.norm()):.4f} | "
        f"centered norm={target_centered_norm:.4f}"
    )
    train_speakers = len(set(dataset.speaker_ids[train_mask].tolist()))
    val_speakers = len(set(dataset.speaker_ids[val_mask].tolist()))
    train_codes_np, _ = speaker_codes(dataset.speaker_ids[train_mask])
    train_codes = torch.from_numpy(train_codes_np)
    val_codes_np, _ = speaker_codes(dataset.speaker_ids[val_mask])
    val_codes = torch.from_numpy(val_codes_np)

    cvae = None
    if args.adapter_input == "cvae":
        cvae, cvae_checkpoint = load_cvae(args.cvae, cfvn.out_dim, device)
        print(
            f"CVAE: {args.cvae} (latent_dim={cvae.latent_dim}, "
            f"trained_epoch={cvae_checkpoint.get('epoch')})"
        )
        print(
            f"Sampling {args.samples_per_clip} CVAE voices per training clip "
            f"(temperature={args.cvae_temperature})..."
        )
        train_samples = cvae_sample_inputs(
            cvae,
            train_cfvn,
            num_samples=args.samples_per_clip,
            temperature=args.cvae_temperature,
            device=device,
        )
        train_face = train_samples.reshape(-1, train_samples.shape[-1])
        train_target = (
            train_target.unsqueeze(1)
            .expand(-1, args.samples_per_clip, -1)
            .reshape(-1, train_target.shape[-1])
        )
        val_face = cvae_prior_mean_inputs(cvae, val_cfvn, device=device)
        val_input = "cvae_prior_mean"
        print(
            f"Train adapter inputs: {tuple(train_face.shape)} "
            f"({len(train_cfvn)} clips x {args.samples_per_clip} samples)"
        )
    else:
        train_face = train_cfvn
        val_face = val_cfvn
        val_input = "cfvn"
        print(
            "WARNING: --adapter-input cfvn trains on a distribution that is not "
            "the serving distribution; use only for diagnostics."
        )

    # ``train_face`` is face-major: all samples for one clip are contiguous.
    # Repeat the speaker code so the diversity term excludes same-speaker pairs.
    sample_multiplier = args.samples_per_clip if cvae is not None else 1
    train_codes = train_codes.repeat_interleave(sample_multiplier)
    if len(train_codes) != len(train_face):
        raise RuntimeError(
            f"speaker-code/input length mismatch: {len(train_codes)} vs "
            f"{len(train_face)}"
        )

    print(
        f"Train {len(train_cfvn)} clips / {train_speakers} speakers | "
        f"Val {len(val_cfvn)} / {val_speakers} (input={val_input})"
    )

    model = StyleAdapter(
        in_dim=cfvn.out_dim,
        hidden_dim=args.hidden_dim,
        out_dim=train_target.shape[1],
        dropout=args.dropout,
    ).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(
        optimizer, T_max=max(1, args.epochs), eta_min=args.lr * 0.01
    )

    train_loader = DataLoader(
        TensorDataset(train_face, train_target, train_codes),
        batch_size=args.batch_size,
        shuffle=True,
    )

    best_val_loss = math.inf
    best_selection_loss = math.inf
    best_epoch = 0
    stale_epochs = 0
    history: list[dict] = []
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_count = 0
        for face_batch, target_batch, code_batch in train_loader:
            face_batch = face_batch.to(device)
            target_batch = target_batch.to(device)
            code_batch = code_batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(face_batch)
            loss, _ = _head_losses(prediction, target_batch, args.lambda_cos)
            var_term, _ = variance_covariance_terms(prediction)
            diversity = adapter_diversity_loss(
                prediction,
                code_batch,
                center=target_center.to(device),
            )
            norm_term = adapter_centered_norm_loss(
                prediction,
                target_center.to(device),
                target_centered_norm,
            )
            loss = (
                loss
                + args.lambda_var * var_term
                + args.lambda_div * diversity
                + args.lambda_norm * norm_term
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            train_loss_sum += float(loss.item()) * face_batch.size(0)
            train_count += face_batch.size(0)
        scheduler.step()
        train_loss = train_loss_sum / max(1, train_count)

        metrics = evaluate(
            model,
            val_face,
            val_target,
            val_codes,
            device,
            args.batch_size,
            args.lambda_cos,
            args.lambda_div,
            diversity_center=target_center,
            target_centered_norm=target_centered_norm,
            lambda_norm=args.lambda_norm,
        )
        selection_loss = metrics["val_selection_loss"]
        improved = selection_loss < best_selection_loss
        if improved:
            best_selection_loss = selection_loss
            best_val_loss = metrics["val_loss"]
            best_epoch = epoch
            stale_epochs = 0
            args.output.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_loss": best_val_loss,
                    **metrics,
                    "train_loss": train_loss,
                    "hparams": {
                        "in_dim": cfvn.out_dim,
                        "hidden_dim": args.hidden_dim,
                        "out_dim": int(train_target.shape[1]),
                        "dropout": args.dropout,
                    },
                    "face_key": args.face_key,
                    "features": str(args.features),
                    "index": str(args.index),
                    "scaler": str(args.scaler),
                    "cfvn_checkpoint": str(args.cfvn),
                    "cvae_checkpoint": str(args.cvae) if cvae is not None else None,
                    "adapter_input": args.adapter_input,
                    "val_input": val_input,
                    "samples_per_clip": (
                        args.samples_per_clip if cvae is not None else 1
                    ),
                    "cvae_temperature": (
                        args.cvae_temperature if cvae is not None else None
                    ),
                    "lambda_div": args.lambda_div,
                    "lambda_norm": args.lambda_norm,
                    "val_selection_loss": selection_loss,
                    "lambda_var": args.lambda_var,
                    "style_target_source": style_source,
                    "style_targets": str(style_target_path),
                    "style_target_schema": style_target_schema,
                    "style_alignment": match_stats,
                    "target_center_norm": float(target_center.norm()),
                    "target_centered_norm": target_centered_norm,
                    "train_speakers": train_speakers,
                    "val_speakers": val_speakers,
                    "style_targets_speaker_disjoint": (
                        style_targets_speaker_disjoint
                    ),
                    "style_target_note": (
                        "per-speaker StyleTTS2 targets extracted from each "
                        "speaker's own VoxCeleb2 audio; both sides follow the "
                        "manifest speaker-disjoint split"
                        if style_targets_speaker_disjoint
                        else "legacy style_pairs.pt has no speaker labels; "
                        "face/voice splits are speaker-disjoint but "
                        "style-target provenance is not"
                    ),
                    "seed": args.seed,
                },
                args.output,
            )
        else:
            stale_epochs += 1

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            **metrics,
            "best": improved,
        }
        history.append(record)

        if epoch % 5 == 0 or epoch <= 3 or improved or epoch == args.epochs:
            print(
                f"  Ep {epoch:3d}: tr={train_loss:.5f} vl={metrics['val_loss']:.5f} "
                f"cos(t)={metrics['val_cosine_timbre']:.4f} "
                f"cos(p)={metrics['val_cosine_prosody']:.4f} "
                f"across={metrics['val_output_across_cosine']:.3f} "
                f"div={metrics['val_diversity_loss']:.3f} "
                f"margin={metrics['val_output_identity_margin']:.3f} "
                f"std={metrics['val_output_per_dim_std']:.3f}"
                f"{' *' if improved else ''}"
            )
        if stale_epochs >= args.patience:
            print(f"No improvement for {args.patience} epochs; stopping.")
            break

    elapsed = time.time() - start_time
    history_path = args.output.with_suffix(".history.json")
    history_path.write_text(
        json.dumps(
            {
                "history": history,
                "best_epoch": best_epoch,
                "best_val_loss": best_val_loss,
                "best_selection_loss": best_selection_loss,
                "lambda_div": args.lambda_div,
                "lambda_norm": args.lambda_norm,
                "style_target_source": style_source,
                "style_targets": str(style_target_path),
                "style_target_schema": style_target_schema,
                "style_alignment": match_stats,
                "target_center_norm": float(target_center.norm()),
                "target_centered_norm": target_centered_norm,
                "style_targets_speaker_disjoint": style_targets_speaker_disjoint,
                "elapsed_sec": elapsed,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print("-" * 72)
    print(
        f"Best epoch {best_epoch} | val loss={best_val_loss:.5f} | "
        f"selection loss={best_selection_loss:.5f}"
    )
    print(f"Saved: {args.output}")
    print(f"History: {history_path}")
    print(f"Elapsed: {elapsed / 60:.1f} min")


if __name__ == "__main__":
    main()
