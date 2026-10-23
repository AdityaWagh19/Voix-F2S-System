"""CVAE Trainer v3 with Decoder Diversity Loss - Phase 7.

Retrains CVAE with stronger face-discrimination loss to make the decoder
produce distinct ECAPA embeddings for different faces.

Key changes from trainer_cvae_v3.py:
- Added decoder diversity loss: pushes decoder outputs for different faces apart
- Increased lambda_nce to 1.0 for stronger contrastive signal
- Added lambda_diversity for decoder diversity loss
"""

from __future__ import annotations

import sys
from pathlib import Path

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
CFVN_PATH    = r"D:\voix\results/checkpoints/D_joint_infonce_best.pt"
GALLERY_PATH = r"D:\voix\data/processed/retrieval_gallery.pt"
SAVE_PATH    = r"D:\voix/checkpoints/cvae_cfvn_face192_latent128_diversity.pt"

EPOCHS       = 150
BATCH_SIZE   = 512
LR           = 2e-4
WEIGHT_DECAY = 1e-4
VAL_FRAC     = 0.1
SEED         = 42
LAMBDA_FB    = 0.15
BETA_MAX     = 1.0
LAMBDA_NCE   = 1.0
LAMBDA_FS    = 0.5
LAMBDA_DIV   = 0.5  # NEW: decoder diversity loss weight
TEMPERATURE  = 0.07
FACE_DIM     = 192
SPEAKER_DIM  = 192
LATENT_DIM   = 128

N_HARD_NEG   = 128


def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class InfoNCELoss(nn.Module):
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.T = temperature

    def forward(self, anchors, positives, negatives):
        anchors = F.normalize(anchors, dim=-1)
        positives = F.normalize(positives, dim=-1)
        negatives = F.normalize(negatives, dim=-1)

        pos_sim = (anchors * positives).sum(dim=-1, keepdim=True) / self.T
        neg_sim = anchors @ negatives.T / self.T
        logits = torch.cat([pos_sim, neg_sim], dim=1)
        labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
        return F.cross_entropy(logits, labels)


def face_separation_loss(prior_mu: torch.Tensor) -> torch.Tensor:
    """Force different faces to have different prior_mu."""
    mu = F.normalize(prior_mu, dim=-1)
    sim = mu @ mu.T
    B = sim.size(0)
    mask = ~torch.eye(B, dtype=torch.bool, device=sim.device)
    off_diag = sim[mask]
    return off_diag.mean()


def decoder_diversity_loss(e_s_hat: torch.Tensor) -> torch.Tensor:
    """Force decoder outputs for different faces to be different."""
    e_s_hat = F.normalize(e_s_hat, dim=-1)
    sim = e_s_hat @ e_s_hat.T
    B = sim.size(0)
    mask = ~torch.eye(B, dtype=torch.bool, device=sim.device)
    off_diag = sim[mask]
    return -off_diag.mean()  # Negative because we want to MINIMIZE similarity


@torch.no_grad()
def compute_face_separation(model, e_f_val, n=512):
    model.eval()
    n = min(n, len(e_f_val) // 2)
    idx = torch.randperm(len(e_f_val))[:n * 2]
    mu_a, _ = model.prior_net(e_f_val[idx[:n]])
    mu_b, _ = model.prior_net(e_f_val[idx[n:]])
    return F.cosine_similarity(mu_a, mu_b, dim=1).mean().item()


def main():
    set_seed(SEED)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load CFVN
    print("Loading ContrastiveFaceVoiceNet...")
    cfvn = ContrastiveFaceVoiceNet.from_checkpoint(CFVN_PATH, device=str(device)).to(device)
    cfvn.eval()

    # Load gallery
    gallery = torch.load(GALLERY_PATH, map_location="cpu", weights_only=False)
    gal_voices = F.normalize(gallery["voice_embeddings"].float(), dim=-1).to(device)

    # Build dataset from features.h5
    print("Building dataset from features.h5...")
    with h5py.File(H5_PATH, "r") as f:
        E_f_raw = torch.tensor(f["face_features"][:], dtype=torch.float32)
        E_s_raw = torch.tensor(f["speaker_features"][:], dtype=torch.float32)

    N = len(E_f_raw)
    print(f"  {N:,} samples")

    # Compute CFVN embeddings
    cfvn_embeddings = []
    batch_size = 512
    with torch.no_grad():
        for i in range(0, N, batch_size):
            batch = E_f_raw[i:i+batch_size].to(device)
            cfvn_batch = cfvn(batch, normalize=True)
            cfvn_embeddings.append(cfvn_batch.cpu())
    E_f = torch.cat(cfvn_embeddings, dim=0)
    E_s = F.normalize(E_s_raw, dim=-1)

    print(f"  CFVN embeddings: {E_f.shape}")
    print(f"  ECAPA embeddings: {E_s.shape}")

    # Split
    n_val = max(100, int(N * VAL_FRAC))
    n_train = N - n_val
    gen = torch.Generator().manual_seed(SEED)
    train_ds, val_ds = random_split(TensorDataset(E_f, E_s), [n_train, n_val], generator=gen)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    val_e_f = E_f[val_ds.indices].to(device)

    model = CVAE(face_dim=FACE_DIM, speaker_dim=SPEAKER_DIM, latent_dim=LATENT_DIM, dropout=0.05).to(device)
    nce_proj = torch.nn.Linear(LATENT_DIM, SPEAKER_DIM, bias=False).to(device)
    all_params = list(model.parameters()) + list(nce_proj.parameters())

    loss_fn = CVAELoss(lambda_fb=LAMBDA_FB, beta_max=BETA_MAX, warmup_steps=len(train_loader) * 30)
    nce_loss = InfoNCELoss(temperature=TEMPERATURE)
    optimizer = AdamW(all_params, lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=LR * 0.01)

    best_val_loss = float('inf')
    global_step = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        tr_loss_sum = 0.0
        n_tr = 0

        for e_f_b, e_s_b in train_loader:
            e_f_b = e_f_b.to(device)
            e_s_b = e_s_b.to(device)

            optimizer.zero_grad()
            out = model(e_f_b, e_s_b)

            total_loss, metrics = loss_fn(
                e_s_hat=out.e_s_hat, e_s=e_s_b,
                mu=out.mu, logvar=out.logvar,
                prior_mu=out.prior_mu, prior_logvar=out.prior_logvar,
                step=global_step,
            )

            # InfoNCE
            prior_mu_proj = nce_proj(out.prior_mu)
            neg_idx = torch.randperm(len(gal_voices), device=device)[:N_HARD_NEG]
            hard_neg = gal_voices[neg_idx]
            nce = nce_loss(prior_mu_proj, e_s_b, hard_neg)

            # Face separation loss
            fs_loss = face_separation_loss(out.prior_mu)

            # Decoder diversity loss
            div_loss = decoder_diversity_loss(out.e_s_hat)

            loss = total_loss + LAMBDA_NCE * nce + LAMBDA_FS * fs_loss + LAMBDA_DIV * div_loss
            loss.backward()
            optimizer.step()
            global_step += 1

            tr_loss_sum += metrics["loss_total"] * e_f_b.size(0)
            n_tr += e_f_b.size(0)

        tr_loss = tr_loss_sum / n_tr

        # Validation
        model.eval()
        vl_loss_sum = 0.0
        n_vl = 0
        with torch.no_grad():
            for e_f_b, e_s_b in val_loader:
                e_f_b = e_f_b.to(device)
                e_s_b = e_s_b.to(device)
                out = model(e_f_b, e_s_b)
                _, vl_m = loss_fn(
                    e_s_hat=out.e_s_hat, e_s=e_s_b,
                    mu=out.mu, logvar=out.logvar,
                    prior_mu=out.prior_mu, prior_logvar=out.prior_logvar,
                )
                vl_loss_sum += vl_m["loss_total"] * e_f_b.size(0)
                n_vl += e_f_b.size(0)

        vl_loss = vl_loss_sum / n_vl

        if epoch % 10 == 0 or epoch <= 3 or epoch == EPOCHS:
            face_sep = compute_face_separation(model, val_e_f)
            print(f"  Ep {epoch:3d}: tr_loss={tr_loss:.6f} vl_loss={vl_loss:.6f} face_sep={face_sep:.4f}")

        if vl_loss < best_val_loss:
            best_val_loss = vl_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_loss": vl_loss,
                "face_dim": FACE_DIM,
                "latent_dim": LATENT_DIM,
                "lambda_nce": LAMBDA_NCE,
                "lambda_fs": LAMBDA_FS,
                "lambda_div": LAMBDA_DIV,
            }, SAVE_PATH)

    print(f"\nTraining complete! Best val_loss: {best_val_loss:.6f}")
    print(f"Saved to: {SAVE_PATH}")


if __name__ == "__main__":
    main()
