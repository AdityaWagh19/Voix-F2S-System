"""CVAE Trainer v3 for VOIX -- Face-Discriminative Contrastive Conditioning.

Extends v2 with a strong face-discrimination objective on the prior_net output:
  L_total = L_CVAE + lambda_nce * InfoNCE(prior_mu, voice) + lambda_fs * FaceSepLoss(prior_mu)

The InfoNCE term forces prior_mu(face_i) to align with voice(face_i).
The FaceSepLoss term forces prior_mu(face_i) to be DISTINCT from prior_mu(face_j) for i != j.
This directly optimizes face_separation: different faces must map to different latents.

Training-serving parity: The CVAE prior is trained on CFVN-produced face
embeddings (560-D fused face -> 192-D ECAPA space), matching the inference
input distribution. This eliminates the train/serve skew where the prior was
trained on speaker-average ECAPA but inference used face-mapped ECAPA.

Expected improvement: FaceSep 0.99 -> < 0.35 with lambda_nce=1.0 + lambda_fs=0.5.
"""
from __future__ import annotations

import os, sys, time
sys.path.insert(0, r"D:\voix")

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset, random_split

from voix.models.cvae import CVAE
from voix.models.contrastive_net import ContrastiveFaceVoiceNet
from voix.training.losses import CVAELoss

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
H5_PATH      = r"D:\voix\data\processed\features.h5"
CFVN_PATH    = r"D:\voix\results\checkpoints\D_joint_infonce_best.pt"
GALLERY_PATH = r"D:\voix\data\processed\retrieval_gallery.pt"
SAVE_PATH    = r"D:\voix\checkpoints\cvae_cfvn_face192_latent128.pt"

EPOCHS       = 150
BATCH_SIZE   = 512
LR           = 2e-4
WEIGHT_DECAY = 1e-4
VAL_FRAC     = 0.1
SEED         = 42
LAMBDA_FB    = 0.15   # free-bits: prevent posterior collapse without over-suppressing KL
BETA_MAX     = 1.0
LAMBDA_NCE   = 1.0   # InfoNCE: align prior_mu with target voice (was 0.15 -- too weak)
LAMBDA_FS    = 0.5   # FaceSepLoss: push different faces' priors apart (NEW -- critical)
TEMPERATURE  = 0.07  # InfoNCE temperature (standard SimCLR / MoCo value)
FACE_DIM     = 192   # CFVN output dim (matches inference input)
SPEAKER_DIM  = 192
LATENT_DIM   = 128

N_HARD_NEG   = 128   # hard negatives per batch for InfoNCE


def set_seed(seed: int) -> None:
    import random; random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


class InfoNCELoss(nn.Module):
    """InfoNCE / NT-Xent contrastive loss.

    Pulls prior_mu(face_i) toward voice(face_i) and pushes away from all others.
    Temperature=0.07 follows MoCo/SimCLR best practices.
    """
    def __init__(self, temperature: float = 0.07) -> None:
        super().__init__()
        self.T = temperature

    def forward(
        self,
        anchors: torch.Tensor,    # (B, D) -- prior_mu projections, L2-normed
        positives: torch.Tensor,  # (B, D) -- corresponding voice embeddings, L2-normed
        negatives: torch.Tensor,  # (N, D) -- hard negative voices, L2-normed
    ) -> torch.Tensor:
        anchors   = F.normalize(anchors,   dim=-1)
        positives = F.normalize(positives, dim=-1)
        negatives = F.normalize(negatives, dim=-1)

        # Positive similarity: (B,)
        pos_sim = (anchors * positives).sum(dim=-1, keepdim=True) / self.T   # (B, 1)

        # Negative similarities: (B, N)
        neg_sim = anchors @ negatives.T / self.T  # (B, N)

        # Concatenate: logits (B, 1+N), labels = 0 for all (positive is first)
        logits = torch.cat([pos_sim, neg_sim], dim=1)  # (B, 1+N)
        labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
        return F.cross_entropy(logits, labels)


def face_separation_loss(prior_mu: torch.Tensor) -> torch.Tensor:
    """Force different faces in the batch to map to DISTINCT latents.

    Computes pairwise cosine similarity between prior_mu vectors in the batch,
    then minimizes the mean OFF-DIAGONAL similarity. This directly optimizes
    face_separation: if all faces map to the same latent, loss is high (~1.0);
    if faces are well-separated, loss is low (~0.0).

    Args:
        prior_mu: (B, latent_dim) prior means from the CVAE prior_net.

    Returns:
        Scalar loss: mean off-diagonal cosine similarity (lower = better separation).
    """
    mu = F.normalize(prior_mu, dim=-1)  # (B, D)
    sim = mu @ mu.T  # (B, B) pairwise cosine similarity
    B = sim.size(0)
    # Mask out diagonal (self-similarity = 1.0)
    mask = ~torch.eye(B, dtype=torch.bool, device=sim.device)
    off_diag = sim[mask]  # (B*(B-1),)
    return off_diag.mean()


@torch.no_grad()
def compute_face_separation(model: CVAE, e_f_val: torch.Tensor, n: int = 512) -> float:
    model.eval()
    n = min(n, len(e_f_val) // 2)
    idx = torch.randperm(len(e_f_val))[:n * 2]
    mu_a, _ = model.prior_net(e_f_val[idx[:n]])
    mu_b, _ = model.prior_net(e_f_val[idx[n:]])
    return F.cosine_similarity(mu_a, mu_b, dim=1).mean().item()


def main() -> None:
    set_seed(SEED)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print("=" * 72)
    print("  VOIX -- CVAE TRAINER v3  (Face-Discriminative Contrastive Conditioning)")
    print("=" * 72)
    print(f"  Device: {device}  |  Epochs: {EPOCHS}  |  lambda_nce: {LAMBDA_NCE}  |  lambda_fs: {LAMBDA_FS}")

    # Load ContrastiveFaceVoiceNet (560-D fused face -> 192-D ECAPA space)
    # This is the same model used at inference -- training-serving parity.
    print("\nLoading ContrastiveFaceVoiceNet ...")
    cfvn = ContrastiveFaceVoiceNet.from_checkpoint(CFVN_PATH, device=str(device)).to(device)
    cfvn.eval()
    print(f"  CFVN loaded from {CFVN_PATH}")

    # Load retrieval gallery (hard negatives)
    gallery = torch.load(GALLERY_PATH, map_location="cpu", weights_only=False)
    gal_voices = F.normalize(gallery["voice_embeddings"].float(), dim=-1).to(device)  # (3029, 192)
    print(f"Gallery: {gal_voices.shape} L2-normed voice embeddings (hard negatives)")

    # Build dataset: pass face_features through CFVN to get CFVN embeddings
    # This matches inference: face_image -> 560-D fused face -> CFVN -> 192-D ECAPA-space embedding
    print("Building dataset from features.h5 (computing CFVN embeddings) ...")
    with h5py.File(H5_PATH, "r") as f:
        meta   = [m.decode() if isinstance(m, bytes) else str(m) for f in [f] for m in f["metadata"][:]]
        E_f_raw = torch.tensor(f["face_features"][:], dtype=torch.float32)
        E_s_raw = torch.tensor(f["speaker_features"][:], dtype=torch.float32)

    N = len(E_f_raw)
    print(f"  {N:,} total samples in features.h5")

    # Compute CFVN embeddings in batches to avoid OOM
    cfvn_embeddings = []
    batch_size = 512
    with torch.no_grad():
        for i in range(0, N, batch_size):
            batch = E_f_raw[i:i+batch_size].to(device)
            cfvn_batch = cfvn(batch, normalize=True)  # (B, 192) L2-normalized
            cfvn_embeddings.append(cfvn_batch.cpu())
    E_f = torch.cat(cfvn_embeddings, dim=0)  # (N, 192) CFVN-produced face-voice embeddings
    E_s = F.normalize(E_s_raw, p=2, dim=-1)   # (N, 192) L2-normalized speaker embeddings

    print(f"  {N:,} samples  |  E_f (CFVN) norm={E_f.norm(dim=1).mean():.5f}  E_s norm={E_s.norm(dim=1).mean():.5f}")

    n_val   = int(N * VAL_FRAC)
    n_train = N - n_val
    gen     = torch.Generator().manual_seed(SEED)
    train_ds, val_ds = random_split(TensorDataset(E_f, E_s), [n_train, n_val], generator=gen)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0, pin_memory=(device.type=="cuda"))
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=(device.type=="cuda"))
    val_e_f = E_f[val_ds.indices].to(device)

    steps_per_ep = len(train_loader)
    warmup_steps = steps_per_ep * max(1, int(EPOCHS * 0.20))
    print(f"  Train: {n_train:,}  Val: {n_val:,}  Steps/ep: {steps_per_ep}  Warmup: {warmup_steps:,}")

    model = CVAE(face_dim=FACE_DIM, speaker_dim=SPEAKER_DIM, latent_dim=LATENT_DIM, dropout=0.05).to(device)
    # Projection head: maps latent_dim -> speaker_dim for InfoNCE alignment
    nce_proj = torch.nn.Linear(LATENT_DIM, SPEAKER_DIM, bias=False).to(device)
    all_params = list(model.parameters()) + list(nce_proj.parameters())
    n_params = sum(p.numel() for p in all_params if p.requires_grad)
    print(f"  CVAE + NCE-proj params: {n_params:,}")

    loss_fn  = CVAELoss(lambda_fb=LAMBDA_FB, beta_max=BETA_MAX, warmup_steps=warmup_steps)
    nce_loss = InfoNCELoss(temperature=TEMPERATURE)
    optimizer = AdamW(all_params, lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=LR * 0.01)

    best_val_cos = -1.0
    best_face_sep = 1.0  # Track best face separation (lower is better)
    global_step  = 0
    t0 = time.time()

    print("\n" + "-" * 95)
    print(f"{'Ep':>4}  {'TrLoss':>8}  {'TrCos':>7}  {'VlLoss':>8}  {'VlCos':>7}  {'NCE':>7}  {'FaceSep':>8}  {'FaceSepL':>8}  {'Beta':>6}")
    print("-" * 95)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        tr_loss_sum = tr_cos_sum = tr_nce_sum = tr_fs_sum = 0.0
        n_tr = 0

        for e_f_b, e_s_b in train_loader:
            e_f_b = e_f_b.to(device, non_blocking=True)
            e_s_b = e_s_b.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            out = model(e_f_b, e_s_b)

            total_loss, metrics = loss_fn(
                e_s_hat=out.e_s_hat, e_s=e_s_b,
                mu=out.mu, logvar=out.logvar,
                prior_mu=out.prior_mu, prior_logvar=out.prior_logvar,
                step=global_step,
            )

            # InfoNCE: align prior_mu with target voice
            prior_mu_proj = nce_proj(out.prior_mu)  # (B, 192)
            neg_idx  = torch.randperm(len(gal_voices), device=device)[:N_HARD_NEG]
            hard_neg = gal_voices[neg_idx]  # (N_HARD_NEG, 192)
            nce = nce_loss(prior_mu_proj, e_s_b, hard_neg)

            # FaceSepLoss: push different faces' priors apart
            fs_loss = face_separation_loss(out.prior_mu)

            loss = total_loss + LAMBDA_NCE * nce + LAMBDA_FS * fs_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            loss_fn.step()
            global_step += 1

            b = e_f_b.size(0)
            tr_loss_sum += metrics["loss_total"] * b
            tr_cos_sum  += F.cosine_similarity(out.e_s_hat.detach(), e_s_b, dim=1).mean().item() * b
            tr_nce_sum  += nce.item() * b
            tr_fs_sum   += fs_loss.item() * b
            n_tr        += b

        scheduler.step()
        tr_loss = tr_loss_sum / n_tr
        tr_cos  = tr_cos_sum  / n_tr
        tr_nce  = tr_nce_sum  / n_tr
        tr_fs   = tr_fs_sum   / n_tr

        model.eval()
        vl_loss_sum = vl_cos_sum = 0.0
        last_au = 0; n_vl = 0
        with torch.no_grad():
            for e_f_b, e_s_b in val_loader:
                e_f_b = e_f_b.to(device, non_blocking=True)
                e_s_b = e_s_b.to(device, non_blocking=True)
                out = model(e_f_b, e_s_b)
                _, vl_m = loss_fn(
                    e_s_hat=out.e_s_hat, e_s=e_s_b,
                    mu=out.mu, logvar=out.logvar,
                    prior_mu=out.prior_mu, prior_logvar=out.prior_logvar,
                    step=global_step,
                )
                b = e_f_b.size(0)
                vl_loss_sum += vl_m["loss_total"] * b
                vl_cos_sum  += F.cosine_similarity(out.e_s_hat, e_s_b, dim=1).mean().item() * b
                n_vl += b; last_au = vl_m["active_units"]

        vl_loss  = vl_loss_sum / n_vl
        vl_cos   = vl_cos_sum  / n_vl
        face_sep = compute_face_separation(model, val_e_f)
        cur_beta = loss_fn.beta(global_step)
        is_best  = vl_cos > best_val_cos and face_sep < 0.5
        marker   = " [BEST]" if is_best else ""

        if epoch % 10 == 0 or is_best or epoch <= 3 or epoch == EPOCHS:
            print(f"{epoch:>4}  {tr_loss:>8.4f}  {tr_cos:>7.4f}  {vl_loss:>8.4f}  {vl_cos:>7.4f}  {tr_nce:>7.4f}  {face_sep:>8.4f}  {tr_fs:>8.4f}  {cur_beta:>6.3f}{marker}")
            import sys as _sys; _sys.stdout.flush()

        if is_best:
            best_val_cos = vl_cos
            best_face_sep = face_sep
            torch.save({
                "epoch": epoch, "model_state_dict": model.state_dict(),
                "val_loss": vl_loss, "val_cosine_sim": vl_cos,
                "face_separation": face_sep, "active_units": last_au,
                "global_step": global_step,
                "face_dim": FACE_DIM, "latent_dim": LATENT_DIM,
                "lambda_nce": LAMBDA_NCE, "lambda_fs": LAMBDA_FS,
            }, SAVE_PATH)

    elapsed = time.time() - t0
    print("\n" + "=" * 72)
    print(f"  COMPLETE in {elapsed/60:.1f} min  |  Best val_cos: {best_val_cos:.4f}  |  Best face_sep: {best_face_sep:.4f}")
    print(f"  Saved: {SAVE_PATH}")
    ckpt = torch.load(SAVE_PATH, map_location="cpu", weights_only=False)
    print(f"  [OK] face_separation = {ckpt['face_separation']:.4f}  (target: < 0.35)")
    print(f"  [OK] active_units    = {ckpt['active_units']} / {LATENT_DIM}")
    print("=" * 72)


if __name__ == "__main__":
    main()