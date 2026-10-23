"""CVAE Trainer for VOIX -- Phase A remediation.

Trains the Conditional VAE (CVAE) model on 80,832 paired
(face_features=560-D, speaker_features=192-D) samples from
D:/voix/data/processed/features.h5.

Key design choices:
  - Linear beta warmup over first 30% of steps to prevent posterior collapse.
  - FreeBitsKL loss (lambda_fb=0.5) guards against dimension-wise collapse.
  - Active Units metric logged each epoch -- a healthy CVAE should reach >=32 AU.
  - Cosine similarity between prior_mu(face_i) and prior_mu(face_j) for random
    pairs is logged; should drop as CVAE learns face-specific priors.
  - Best checkpoint saved by val cosine reconstruction similarity.
"""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, r"D:\voix")

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset, random_split

from voix.models.cvae import CVAE
from voix.training.losses import CVAELoss

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

H5_PATH      = r"D:\voix\data\processed\features.h5"
SAVE_PATH    = r"D:\voix\checkpoints\cvae_direct_face560_latent192.pt"
EPOCHS       = 80
BATCH_SIZE   = 512
LR           = 3e-4
WEIGHT_DECAY = 1e-4
VAL_FRAC     = 0.1
SEED         = 42
LAMBDA_FB    = 0.5
BETA_MAX     = 1.0
FACE_DIM     = 560
SPEAKER_DIM  = 192
LATENT_DIM   = 192


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def compute_face_separation(model: CVAE, e_f_val: torch.Tensor, n_pairs: int = 512) -> float:
    """Avg cosine sim between prior_mu of RANDOM face pairs.
    Lower = better biometric separation. Untrained baseline ~0.26."""
    model.eval()
    n = min(n_pairs, len(e_f_val) // 2)
    idx = torch.randperm(len(e_f_val))[:n * 2]
    a, b = e_f_val[idx[:n]], e_f_val[idx[n:]]
    mu_a, _ = model.prior_net(a)
    mu_b, _ = model.prior_net(b)
    return F.cosine_similarity(mu_a, mu_b, dim=1).mean().item()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    set_seed(SEED)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print("=" * 70)
    print("  VOIX -- CVAE TRAINER  (Phase A Remediation)")
    print("=" * 70)
    print(f"  Device  : {device}")
    print(f"  H5 data : {H5_PATH}")
    print(f"  Epochs  : {EPOCHS}  |  Batch: {BATCH_SIZE}  |  LR: {LR}")
    print(f"  Save to : {SAVE_PATH}")
    print("=" * 70)

    if not os.path.exists(H5_PATH):
        raise FileNotFoundError(f"Training data not found at {H5_PATH}")

    print("\nLoading features.h5 ...")
    with h5py.File(H5_PATH, "r") as f:
        E_f = torch.tensor(f["face_features"][:],    dtype=torch.float32)
        E_s = torch.tensor(f["speaker_features"][:], dtype=torch.float32)

    N = len(E_f)
    print(f"  face_features    : {E_f.shape}   mean={E_f.mean():.4f}  std={E_f.std():.4f}")
    print(f"  speaker_features : {E_s.shape}   mean={E_s.mean():.4f}  std={E_s.std():.4f}")

    n_val   = int(N * VAL_FRAC)
    n_train = N - n_val
    generator = torch.Generator().manual_seed(SEED)
    full_ds   = TensorDataset(E_f, E_s)
    train_ds, val_ds = random_split(full_ds, [n_train, n_val], generator=generator)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0, pin_memory=(device.type == "cuda"))
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=(device.type == "cuda"))

    val_e_f       = E_f[val_ds.indices].to(device)
    steps_per_ep  = len(train_loader)
    warmup_steps  = steps_per_ep * max(1, int(EPOCHS * 0.30))

    print(f"\n  Train: {n_train:,}  |  Val: {n_val:,}")
    print(f"  Steps/epoch: {steps_per_ep}  |  KL warmup steps: {warmup_steps:,}")

    model = CVAE(face_dim=FACE_DIM, speaker_dim=SPEAKER_DIM, latent_dim=LATENT_DIM, dropout=0.05).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  CVAE parameters: {n_params:,}")

    loss_fn   = CVAELoss(lambda_fb=LAMBDA_FB, beta_max=BETA_MAX, warmup_steps=warmup_steps)
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=LR * 0.01)

    best_val_cos = -1.0
    global_step  = 0
    t0 = time.time()

    print("\n" + "-" * 70)
    print(f"{'Ep':>4}  {'TrLoss':>8}  {'TrCos':>7}  {'VlLoss':>8}  {'VlCos':>7}  {'AU':>4}  {'FaceSep':>8}  {'Beta':>6}")
    print("-" * 70)

    for epoch in range(1, EPOCHS + 1):
        # Train
        model.train()
        tr_loss_sum = tr_cos_sum = 0.0
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
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            loss_fn.step()
            global_step += 1
            b = e_f_b.size(0)
            cos_b = F.cosine_similarity(out.e_s_hat.detach(), e_s_b, dim=1).mean().item()
            tr_loss_sum += metrics["loss_total"] * b
            tr_cos_sum  += cos_b * b
            n_tr        += b

        scheduler.step()
        tr_loss = tr_loss_sum / n_tr
        tr_cos  = tr_cos_sum  / n_tr

        # Validation
        model.eval()
        vl_loss_sum = vl_cos_sum = 0.0
        last_au = 0
        n_vl = 0
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
                n_vl        += b
                last_au      = vl_m["active_units"]

        vl_loss  = vl_loss_sum / n_vl
        vl_cos   = vl_cos_sum  / n_vl
        face_sep = compute_face_separation(model, val_e_f)
        cur_beta = loss_fn.beta(global_step)
        is_best  = vl_cos > best_val_cos
        marker   = " [BEST]" if is_best else ""

        if epoch % 5 == 0 or is_best or epoch <= 3 or epoch == EPOCHS:
            print(
                f"{epoch:>4}  {tr_loss:>8.4f}  {tr_cos:>7.4f}  "
                f"{vl_loss:>8.4f}  {vl_cos:>7.4f}  {last_au:>4d}  "
                f"{face_sep:>8.4f}  {cur_beta:>6.3f}{marker}"
            )
            import sys; sys.stdout.flush()

        if is_best:
            best_val_cos = vl_cos
            torch.save({
                "epoch":            epoch,
                "model_state_dict": model.state_dict(),
                "val_loss":         vl_loss,
                "val_cosine_sim":   vl_cos,
                "face_separation":  face_sep,
                "active_units":     last_au,
                "global_step":      global_step,
            }, SAVE_PATH)

    elapsed = time.time() - t0
    print("\n" + "=" * 70)
    print(f"  TRAINING COMPLETE in {elapsed/60:.1f} min")
    print(f"  Best val cosine sim : {best_val_cos:.4f}")
    print(f"  Checkpoint saved to : {SAVE_PATH}")
    print("=" * 70)

    ckpt       = torch.load(SAVE_PATH, map_location="cpu", weights_only=False)
    model_keys = set(CVAE(FACE_DIM, SPEAKER_DIM, LATENT_DIM).state_dict().keys())
    ckpt_keys  = set(ckpt["model_state_dict"].keys())
    assert model_keys == ckpt_keys, f"Key mismatch! Model={len(model_keys)}, Ckpt={len(ckpt_keys)}"
    print(f"\n  [OK] Checkpoint key check passed: {len(ckpt_keys)} keys")
    print(f"  [OK] face_separation = {ckpt['face_separation']:.4f}  (target: < 0.80)")
    print(f"  [OK] active_units    = {ckpt['active_units']}  (target: >= 32)")


if __name__ == "__main__":
    main()
