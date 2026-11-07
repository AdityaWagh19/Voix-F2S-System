"""Train the conditional VAE on FaceFeatureV2 CFVN embeddings.

This is the V2 replacement for the legacy ``trainer_cvae_v3_diversity.py``.
The legacy trainer had three defects that made its checkpoints unusable for
honest evaluation:

* it read the old V1 fused corpus (whose face representation was nearly
  constant across identities),
* it split clips randomly instead of by speaker, leaking identities between
  train and validation,
* its "diversity" loss returned ``-mean(off_diagonal_similarity)`` and was
  added with a positive weight, so minimising the total loss *encouraged* the
  decoder to collapse every face onto the same voice.

The V2 trainer fixes all three:

* face inputs are the canonical FaceFeatureV2 representation, z-scored with the
  speaker-disjoint train scaler and mapped through the V2 CFVN checkpoint,
* train/validation are the manifest's speaker-disjoint splits,
* diversity/separation losses only use *different-speaker* pairs and are
  minimised, so they push identities apart without fighting reconstruction.

A prior-reconstruction term is also included.  The CVAE decoder is trained on
the recognition posterior ``q(z | e_s, e_f)`` but at inference it only sees the
conditional prior ``p(z | e_f)``; optimising the prior path directly closes
that train/serve gap and is what makes held-out face-to-voice retrieval a
meaningful model-selection metric.

Usage:
    python -m voix.training.trainer_cvae_v3_diversity \
        --features data/processed/voxceleb2_v2/features_v2.h5 \
        --index data/processed/voxceleb2_v2/index.csv \
        --scaler data/processed/voxceleb2_v2/face_scaler_v2.pt \
        --cfvn results/checkpoints/cfvn_v2_best.pt \
        --output checkpoints/cvae_cfvn_face192_latent128_diversity.pt
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset

from voix.data.feature_contract import load_scaler_payload
from voix.evaluation.retrieval import speaker_retrieval
from voix.models.contrastive_net import ContrastiveFaceVoiceNet
from voix.models.cvae import CVAE
from voix.training.data import (
    DEFAULT_FEATURES_V2,
    DEFAULT_INDEX_V2,
    DEFAULT_SCALER_V2,
    SpeakerBatchSampler,
    load_v2_dataset,
    normalize_speaker_features,
    split_summary,
    validate_scaler_for_dataset,
)
from voix.training.losses import CVAELoss
from voix.training.v2_utils import (
    embed_faces,
    normalize_face,
    pairwise_identity_stats,
    query_gallery_indices,
    speaker_codes,
    variance_covariance_terms,
)
from voix.utils.seed import set_global_seed

ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_CFVN = ROOT / "results" / "checkpoints" / "cfvn_v2_best.pt"
DEFAULT_SAVE = ROOT / "checkpoints" / "cvae_cfvn_face192_latent128_diversity.pt"

# Kept as module-level defaults so existing callers can still import them.
FACE_DIM = 192          # CFVN output dimension
SPEAKER_DIM = 192       # ECAPA dimension
LATENT_DIM = 128


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------

class InfoNCELoss(nn.Module):
    """Supervised contrastive loss between predicted and target embeddings.

    Same-speaker rows are positives; all other rows in the batch are negatives.
    This is the correct in-batch replacement for the legacy random external
    gallery, which mixed speakers from an unrelated corpus into every batch.
    """

    def __init__(self, temperature: float = 0.07) -> None:
        super().__init__()
        self.T = float(temperature)

    def forward(
        self,
        anchors: torch.Tensor,
        targets: torch.Tensor,
        speaker_codes: torch.Tensor,
    ) -> torch.Tensor:
        anchors = F.normalize(anchors.float(), dim=-1)
        targets = F.normalize(targets.float(), dim=-1)
        logits = anchors @ targets.T / self.T
        same = speaker_codes[:, None] == speaker_codes[None, :]
        log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
        positive_count = same.sum(dim=1)
        valid = positive_count > 0
        if not valid.any():
            return anchors.sum() * 0.0
        mean_log_prob = (log_prob * same).sum(dim=1)[valid] / positive_count[valid]
        return -mean_log_prob.mean()


def _different_speaker_mask(speaker_codes: torch.Tensor) -> torch.Tensor:
    """Boolean mask selecting pairs that belong to different speakers."""
    codes = speaker_codes.view(-1)
    return codes[:, None] != codes[None, :]


def face_separation_loss(
    prior_mu: torch.Tensor,
    speaker_codes: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Mean cosine similarity between different-speaker prior means.

    Minimising this separates identities in the latent prior.  Same-speaker
    pairs are excluded because their prior means *should* be similar.
    """
    mu = F.normalize(prior_mu.float(), dim=-1)
    similarity = mu @ mu.T
    if speaker_codes is None:
        mask = ~torch.eye(len(mu), dtype=torch.bool, device=mu.device)
    else:
        mask = _different_speaker_mask(speaker_codes.to(mu.device))
    if not mask.any():
        return mu.sum() * 0.0
    return similarity[mask].mean()


def decoder_diversity_loss(
    e_s_hat: torch.Tensor,
    speaker_codes: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Mean cosine similarity between different-speaker decoder outputs.

    The legacy implementation returned the *negative* of this quantity, so the
    positive training weight drove the decoder toward a single collapsed
    output.  Returning the similarity itself makes minimisation push different
    identities apart.
    """
    e_s_hat = F.normalize(e_s_hat.float(), dim=-1)
    similarity = e_s_hat @ e_s_hat.T
    if speaker_codes is None:
        mask = ~torch.eye(len(e_s_hat), dtype=torch.bool, device=e_s_hat.device)
    else:
        mask = _different_speaker_mask(speaker_codes.to(e_s_hat.device))
    if not mask.any():
        return e_s_hat.sum() * 0.0
    return similarity[mask].mean()


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_prior_path(
    model: CVAE,
    face: torch.Tensor,
    voice: torch.Tensor,
    speakers: np.ndarray,
    device: str,
    batch_size: int = 512,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Evaluate the inference path: prior mean -> decoder -> retrieval.

    The posterior is deliberately not used here.  At serving time the CVAE
    never sees the target voice, so selecting a checkpoint on posterior
    reconstruction would overstate quality.
    """
    model.eval()
    predictions = []
    priors = []
    for start in range(0, len(face), batch_size):
        batch = face[start:start + batch_size].to(device)
        output = model(batch, e_s=None, deterministic=True)
        predictions.append(output.e_s_hat.cpu())
        priors.append(output.prior_mu.cpu())
    predicted = torch.cat(predictions, dim=0) if predictions else face.new_zeros((0, 192))
    prior_mu = torch.cat(priors, dim=0) if priors else face.new_zeros((0, 128))

    query, gallery = query_gallery_indices(speakers)
    retrieval = speaker_retrieval(
        predicted[query],
        speakers[query],
        voice[gallery],
        speakers[gallery],
    )
    cosine = F.cosine_similarity(predicted, voice, dim=-1)
    codes, _ = speaker_codes(speakers)
    code_tensor = torch.from_numpy(codes)
    prior_stats = pairwise_identity_stats(prior_mu, code_tensor)
    decoder_stats = pairwise_identity_stats(predicted, code_tensor)
    details = {
        "prediction_target_cosine_mean": float(cosine.mean()),
        "prediction_target_cosine_p10": float(torch.quantile(cosine, 0.10)),
        "prediction_target_cosine_p50": float(torch.quantile(cosine, 0.50)),
        "prediction_target_cosine_p90": float(torch.quantile(cosine, 0.90)),
        **{f"prior_latent_{key}": value for key, value in prior_stats.items()},
        **{f"decoder_{key}": value for key, value in decoder_stats.items()},
    }
    return {f"val_{key}": float(value) for key, value in retrieval.items()}, details


# ---------------------------------------------------------------------------
# CLI / training
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the CVAE on speaker-disjoint FaceFeatureV2 data"
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
    parser.add_argument("--output", type=Path, default=DEFAULT_SAVE)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--batch-speakers", type=int, default=32)
    parser.add_argument("--batch-clips", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--lambda-fb", type=float, default=0.15)
    parser.add_argument("--beta-max", type=float, default=1.0)
    parser.add_argument("--lambda-nce", type=float, default=1.0)
    parser.add_argument("--lambda-fs", type=float, default=0.5)
    parser.add_argument("--lambda-div", type=float, default=0.5)
    parser.add_argument("--lambda-prior", type=float, default=0.5)
    parser.add_argument("--lambda-var", type=float, default=0.05)
    parser.add_argument("--lambda-cov", type=float, default=0.01)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--latent-dim", type=int, default=LATENT_DIM)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_global_seed(args.seed)
    device = args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"

    print("=" * 72)
    print("CVAE training - FaceFeatureV2 / speaker-disjoint")
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
    print(
        f"Scaler: {args.scaler} "
        f"(schema={scaler.get('schema_version')}, n_train={scaler.get('n_train')})"
    )

    if "train" not in summary or "val" not in summary:
        raise ValueError("CVAE training requires both train and val splits")

    face = normalize_face(dataset.face_raw, scaler)
    voice = normalize_speaker_features(dataset.speaker_raw)

    cfvn = ContrastiveFaceVoiceNet.from_checkpoint(str(args.cfvn), device=device).to(device)
    cfvn.eval()
    if cfvn.face_dim != face.shape[1]:
        raise ValueError(
            f"CFVN expects {cfvn.face_dim}-D faces but {args.face_key} provides "
            f"{face.shape[1]}-D; pass the matching --face-key/--cfvn pair"
        )
    print(f"CFVN: {args.cfvn} (face_dim={cfvn.face_dim}, out_dim={cfvn.out_dim})")

    print("Computing CFVN embeddings...")
    cfvn_face = embed_faces(cfvn, face, device)
    print(f"  CFVN embeddings: {tuple(cfvn_face.shape)}")

    train_mask = dataset.split == "train"
    val_mask = dataset.split == "val"
    train_face = cfvn_face[train_mask]
    train_voice = voice[train_mask]
    train_speakers = dataset.speaker_ids[train_mask]
    val_face = cfvn_face[val_mask]
    val_voice = voice[val_mask]
    val_speakers = dataset.speaker_ids[val_mask]

    train_codes, train_speaker_list = speaker_codes(train_speakers)
    val_codes, val_speaker_list = speaker_codes(val_speakers)

    mapping: Dict[str, list[int]] = {}
    for index, speaker in enumerate(train_speakers):
        mapping.setdefault(str(speaker), []).append(index)
    mapping = {speaker: np.asarray(indices, dtype=np.int64) for speaker, indices in mapping.items()}

    sampler = SpeakerBatchSampler(
        mapping,
        speakers_per_batch=args.batch_speakers,
        clips_per_speaker=args.batch_clips,
        seed=args.seed,
    )
    train_ds = TensorDataset(
        train_face,
        train_voice,
        torch.from_numpy(train_codes),
    )
    train_loader = DataLoader(train_ds, batch_sampler=sampler)
    val_loader = DataLoader(
        TensorDataset(val_face, val_voice, torch.from_numpy(val_codes)),
        batch_size=args.batch_size,
        shuffle=False,
    )
    print(
        f"Train {len(train_face)} clips / {len(train_speaker_list)} speakers | "
        f"Val {len(val_face)} / {len(val_speaker_list)}"
    )

    model = CVAE(
        face_dim=cfvn.out_dim,
        speaker_dim=SPEAKER_DIM,
        latent_dim=args.latent_dim,
        dropout=args.dropout,
    ).to(device)
    loss_fn = CVAELoss(
        lambda_fb=args.lambda_fb,
        beta_max=args.beta_max,
        warmup_steps=max(1, len(train_loader) * 30),
    )
    nce_loss = InfoNCELoss(temperature=args.temperature)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, args.epochs), eta_min=args.lr * 0.01)

    best_score: Tuple[float, float, float] = (-math.inf, -math.inf, -math.inf)
    best_val_loss = math.inf
    best_epoch = 0
    stale_epochs = 0
    global_step = 0
    history: list[dict] = []
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_recon_sum = 0.0
        train_prior_sum = 0.0
        train_count = 0

        for face_batch, voice_batch, codes_batch in train_loader:
            face_batch = face_batch.to(device)
            voice_batch = voice_batch.to(device)
            codes_batch = codes_batch.to(device)

            optimizer.zero_grad(set_to_none=True)
            output = model(face_batch, voice_batch)

            total_loss, metrics = loss_fn(
                e_s_hat=output.e_s_hat,
                e_s=voice_batch,
                mu=output.mu,
                logvar=output.logvar,
                prior_mu=output.prior_mu,
                prior_logvar=output.prior_logvar,
                step=global_step,
            )

            nce = nce_loss(output.e_s_hat, voice_batch, codes_batch)
            separation = face_separation_loss(output.prior_mu, codes_batch)
            diversity = decoder_diversity_loss(output.e_s_hat, codes_batch)
            prior_recon = 1.0 - F.cosine_similarity(
                model.decoder(output.prior_mu), voice_batch, dim=-1
            ).mean()
            var_term, cov_term = variance_covariance_terms(output.e_s_hat)

            loss = (
                total_loss
                + args.lambda_nce * nce
                + args.lambda_fs * separation
                + args.lambda_div * diversity
                + args.lambda_prior * prior_recon
                + args.lambda_var * var_term
                + args.lambda_cov * cov_term
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            global_step += 1

            batch_size = face_batch.size(0)
            train_loss_sum += float(loss.item()) * batch_size
            train_recon_sum += float(metrics["loss_recon"]) * batch_size
            train_prior_sum += float(prior_recon.item()) * batch_size
            train_count += batch_size

            if args.max_steps and global_step >= args.max_steps:
                break

        scheduler.step()
        train_loss = train_loss_sum / max(1, train_count)
        train_recon = train_recon_sum / max(1, train_count)
        train_prior = train_prior_sum / max(1, train_count)

        model.eval()
        val_loss_sum = 0.0
        val_recon_sum = 0.0
        val_prior_sum = 0.0
        val_nce_sum = 0.0
        val_separation_sum = 0.0
        val_diversity_sum = 0.0
        val_var_sum = 0.0
        val_cov_sum = 0.0
        val_active_units_sum = 0.0
        val_count = 0
        with torch.no_grad():
            for face_batch, voice_batch, codes_batch in val_loader:
                face_batch = face_batch.to(device)
                voice_batch = voice_batch.to(device)
                codes_batch = codes_batch.to(device)
                output = model(face_batch, voice_batch)
                base_loss, metrics = loss_fn(
                    e_s_hat=output.e_s_hat,
                    e_s=voice_batch,
                    mu=output.mu,
                    logvar=output.logvar,
                    prior_mu=output.prior_mu,
                    prior_logvar=output.prior_logvar,
                    step=global_step,
                )
                nce = nce_loss(output.e_s_hat, voice_batch, codes_batch)
                separation = face_separation_loss(output.prior_mu, codes_batch)
                diversity = decoder_diversity_loss(output.e_s_hat, codes_batch)
                prior_recon = 1.0 - F.cosine_similarity(
                    model.decoder(output.prior_mu), voice_batch, dim=-1
                ).mean()
                var_term, cov_term = variance_covariance_terms(output.e_s_hat)
                total = (
                    base_loss
                    + args.lambda_nce * nce
                    + args.lambda_fs * separation
                    + args.lambda_div * diversity
                    + args.lambda_prior * prior_recon
                    + args.lambda_var * var_term
                    + args.lambda_cov * cov_term
                )
                batch_size = face_batch.size(0)
                val_loss_sum += float(total.item()) * batch_size
                val_recon_sum += float(metrics["loss_recon"]) * batch_size
                val_prior_sum += float(prior_recon.item()) * batch_size
                val_nce_sum += float(nce.item()) * batch_size
                val_separation_sum += float(separation.item()) * batch_size
                val_diversity_sum += float(diversity.item()) * batch_size
                val_var_sum += float(var_term.item()) * batch_size
                val_cov_sum += float(cov_term.item()) * batch_size
                val_active_units_sum += float(metrics["active_units"]) * batch_size
                val_count += batch_size

        val_loss = val_loss_sum / max(1, val_count)
        val_active_units = int(round(val_active_units_sum / max(1, val_count)))
        val_recon = val_recon_sum / max(1, val_count)
        val_prior = val_prior_sum / max(1, val_count)
        val_nce = val_nce_sum / max(1, val_count)
        val_separation = val_separation_sum / max(1, val_count)
        val_diversity = val_diversity_sum / max(1, val_count)
        val_var = val_var_sum / max(1, val_count)
        val_cov = val_cov_sum / max(1, val_count)
        retrieval, cosine = evaluate_prior_path(
            model, val_face, val_voice, val_speakers, device
        )

        score = (
            retrieval["val_recall_at_1"],
            retrieval["val_recall_at_10"],
            -val_loss,
        )
        improved = score > best_score
        if improved:
            best_score = score
            best_val_loss = val_loss
            best_epoch = epoch
            stale_epochs = 0
            args.output.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_loss": val_loss,
                    "val_recon_loss": val_recon,
                    "val_prior_recon_loss": val_prior,
                    "val_nce": val_nce,
                    "val_separation": val_separation,
                    "val_diversity": val_diversity,
                    "val_variance": val_var,
                    "val_covariance": val_cov,
                    "val_active_units": val_active_units,
                    "val_r1": retrieval["val_recall_at_1"],
                    "val_r5": retrieval["val_recall_at_5"],
                    "val_r10": retrieval["val_recall_at_10"],
                    "val_chance_r1": retrieval["val_chance_at_1"],
                    **cosine,
                    "face_dim": cfvn.out_dim,
                    "speaker_dim": SPEAKER_DIM,
                    "latent_dim": args.latent_dim,
                    "face_key": args.face_key,
                    "features": str(args.features),
                    "index": str(args.index),
                    "scaler": str(args.scaler),
                    "cfvn_checkpoint": str(args.cfvn),
                    "train_speakers": len(train_speaker_list),
                    "val_speakers": len(val_speaker_list),
                    "lambda_nce": args.lambda_nce,
                    "lambda_fs": args.lambda_fs,
                    "lambda_div": args.lambda_div,
                    "lambda_prior": args.lambda_prior,
                    "lambda_var": args.lambda_var,
                    "lambda_cov": args.lambda_cov,
                    "seed": args.seed,
                },
                args.output,
            )
        else:
            stale_epochs += 1

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_recon": train_recon,
            "train_prior_recon": train_prior,
            "val_loss": val_loss,
            "val_recon": val_recon,
            "val_prior_recon": val_prior,
            "val_nce": val_nce,
            "val_separation": val_separation,
            "val_diversity": val_diversity,
            "val_variance": val_var,
            "val_covariance": val_cov,
            "val_active_units": val_active_units,
            **retrieval,
            **cosine,
            "best": improved,
        }
        history.append(record)

        if epoch % 10 == 0 or epoch <= 3 or improved or epoch == args.epochs:
            print(
                f"  Ep {epoch:3d}: tr={train_loss:.5f} vl={val_loss:.5f} "
                f"prior={val_prior:.4f} R@1={retrieval['val_recall_at_1']:.4f} "
                f"(chance {retrieval['val_chance_at_1']:.4f}) "
                f"R@10={retrieval['val_recall_at_10']:.4f} "
                f"margin={cosine['decoder_identity_margin']:.3f} "
                f"au={val_active_units}/{args.latent_dim}"
                f"{' *' if improved else ''}"
            )

        if args.max_steps and global_step >= args.max_steps:
            print(f"Reached --max-steps={args.max_steps}; stopping.")
            break
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
                "best_val_r1": best_score[0],
                "best_val_r10": best_score[1],
                "elapsed_sec": elapsed,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print("-" * 72)
    print(
        f"Best epoch {best_epoch} | val R@1={best_score[0]:.4f} "
        f"R@10={best_score[1]:.4f} | val loss={best_val_loss:.5f}"
    )
    print(f"Saved: {args.output}")
    print(f"History: {history_path}")
    print(f"Elapsed: {elapsed / 60:.1f} min")


if __name__ == "__main__":
    main()
