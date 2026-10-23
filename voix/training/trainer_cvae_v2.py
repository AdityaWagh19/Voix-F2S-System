"""CVAE Trainer v2 for VOIX -- Properly conditioned on speaker identity.

Key insight from v1 failure analysis:
  - features.h5 face_features have ZERO identity signal (built with old random FusionLayer)
  - speaker_features are unnormalized ECAPA (norm=302) -- prior cannot learn structure

This v2 trainer uses:
  - face_proxy: L2-normalized speaker_avg_ecapa[speaker_id] (192-D, delta=0.966)
  - target:     L2-normalized individual utterance ECAPA (192-D, delta=0.513)

At inference, ContrastiveFaceVoiceNet(face_image) outputs 192-D in the same ECAPA space,
making it a drop-in replacement for the face_proxy at test time.
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
SPK_AVG_PATH = r"D:\voix\data\processed\speaker_avg_ecapa.pt"
SAVE_PATH    = r"D:\voix\checkpoints\cvae_best.pt"

EPOCHS       = 100
BATCH_SIZE   = 512
LR           = 3e-4
WEIGHT_DECAY = 1e-4
VAL_FRAC     = 0.1
SEED         = 42
LAMBDA_FB    = 0.2    # lower than v1 -- allow more KL pressure to flow
BETA_MAX     = 1.0
FACE_DIM     = 192    # ContrastiveFaceVoiceNet output dim
SPEAKER_DIM  = 192
LATENT_DIM   = 128    # smaller latent -- easier to condition


def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def compute_face_separation(model: CVAE, e_f_val: torch.Tensor, n_pairs: int = 512) -> float:
    model.eval()
    n = min(n_pairs, len(e_f_val) // 2)
    idx = torch.randperm(len(e_f_val))[:n * 2]
    a, b = e_f_val[idx[:n]], e_f_val[idx[n:]]
    mu_a, _ = model.prior_net(a)
    mu_b, _ = model.prior_net(b)
    return F.cosine_similarity(mu_a, mu_b, dim=1).mean().item()


def main() -> None:
    set_seed(SEED)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print("=" * 70)
    print("  VOIX -- CVAE TRAINER v2  (Speaker-Identity Conditioned)")
    print("=" * 70)
    print(f"  Device: {device} | Epochs: {EPOCHS} | Batch: {BATCH_SIZE}")
    print(f"  face_dim={FACE_DIM} (ContrastiveFaceVoiceNet output space)")
    print(f"  latent_dim={LATENT_DIM}")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 1. Load speaker average ECAPA embeddings (identity signal)
    # ------------------------------------------------------------------
    spk_avg_raw = torch.load(SPK_AVG_PATH, map_location="cpu", weights_only=False)
    # Normalise all speaker averages to unit sphere
    spk_avg = {k: F.normalize(torch.tensor(v, dtype=torch.float32).unsqueeze(0), dim=-1).squeeze()
               for k, v in spk_avg_raw.items()}
    print(f"\nLoaded {len(spk_avg)} speaker average embeddings (L2-normalised)")

    # ------------------------------------------------------------------
    # 2. Load h5 and build aligned (face_proxy, speaker_target) pairs
    # ------------------------------------------------------------------
    print("Building aligned dataset from features.h5 ...")
    with h5py.File(H5_PATH, "r") as f:
        meta = [m.decode() if isinstance(m, bytes) else str(m) for m in f["metadata"][:]]
        E_s_raw = torch.tensor(f["speaker_features"][:], dtype=torch.float32)

    spk_ids = [m.split("/")[0] for m in meta]
    covered_idx = [i for i, s in enumerate(spk_ids) if s in spk_avg]
    N = len(covered_idx)
    print(f"  Using {N}/{len(meta)} samples ({100*N/len(meta):.1f}% coverage)")

    # Build tensors
    E_f_list, E_s_list = [], []
    for i in covered_idx:
        E_f_list.append(spk_avg[spk_ids[i]])
    E_f = torch.stack(E_f_list)                               # (N, 192) L2-normalised face proxy
    E_s = F.normalize(E_s_raw[covered_idx], p=2, dim=-1)      # (N, 192) L2-normalised speaker target
    print(f"  E_f norm: mean={E_f.norm(dim=1).mean():.5f}  (face proxy, 192-D)")
    print(f"  E_s norm: mean={E_s.norm(dim=1).mean():.5f}  (speaker target, 192-D)")

    # ------------------------------------------------------------------
    # 3. Train / val split
    # ------------------------------------------------------------------
    n_val   = int(N * VAL_FRAC)
    n_train = N - n_val
    gen     = torch.Generator().manual_seed(SEED)
    full_ds = TensorDataset(E_f, E_s)
    train_ds, val_ds = random_split(full_ds, [n_train, n_val], generator=gen)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0, pin_memory=(device.type=="cuda"))
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=(device.type=="cuda"))
    val_e_f = E_f[val_ds.indices].to(device)

    steps_per_ep = len(train_loader)
    warmup_steps = steps_per_ep * max(1, int(EPOCHS * 0.25))
    print(f"\n  Train: {n_train:,}  Val: {n_val:,}  Steps/ep: {steps_per_ep}  Warmup: {warmup_steps:,}")

    # ------------------------------------------------------------------
    # 4. Model, loss, optimiser
    # ------------------------------------------------------------------
    model = CVAE(face_dim=FACE_DIM, speaker_dim=SPEAKER_DIM, latent_dim=LATENT_DIM, dropout=0.05).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  CVAE parameters: {n_params:,}")

    loss_fn   = CVAELoss(lambda_fb=LAMBDA_FB, beta_max=BETA_MAX, warmup_steps=warmup_steps)
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=LR * 0.01)

    # ------------------------------------------------------------------
    # 5. Training loop
    # ------------------------------------------------------------------
    best_val_cos = -1.0
    global_step  = 0
    t0 = time.time()

    print("\n" + "-" * 74)
    print(f"{'Ep':>4}  {'TrLoss':>8}  {'TrCos':>7}  {'VlLoss':>8}  {'VlCos':>7}  {'AU':>4}  {'FaceSep':>8}  {'Beta':>6}")
    print("-" * 74)

    for epoch in range(1, EPOCHS + 1):
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
            tr_loss_sum += metrics["loss_total"] * b
            tr_cos_sum  += F.cosine_similarity(out.e_s_hat.detach(), e_s_b, dim=1).mean().item() * b
            n_tr        += b

        scheduler.step()
        tr_loss = tr_loss_sum / n_tr
        tr_cos  = tr_cos_sum  / n_tr

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
        is_best  = vl_cos > best_val_cos
        marker   = " [BEST]" if is_best else ""

        if epoch % 5 == 0 or is_best or epoch <= 3 or epoch == EPOCHS:
            print(f"{epoch:>4}  {tr_loss:>8.4f}  {tr_cos:>7.4f}  {vl_loss:>8.4f}  {vl_cos:>7.4f}  {last_au:>4d}  {face_sep:>8.4f}  {cur_beta:>6.3f}{marker}")
            import sys as _sys; _sys.stdout.flush()

        if is_best:
            best_val_cos = vl_cos
            torch.save({
                "epoch": epoch, "model_state_dict": model.state_dict(),
                "val_loss": vl_loss, "val_cosine_sim": vl_cos,
                "face_separation": face_sep, "active_units": last_au,
                "global_step": global_step,
                "face_dim": FACE_DIM, "latent_dim": LATENT_DIM,
            }, SAVE_PATH)

    elapsed = time.time() - t0
    print("\n" + "=" * 70)
    print(f"  TRAINING COMPLETE in {elapsed/60:.1f} min")
    print(f"  Best val cosine sim: {best_val_cos:.4f}")
    print(f"  Saved to: {SAVE_PATH}")
    print("=" * 70)

    ckpt = torch.load(SAVE_PATH, map_location="cpu", weights_only=False)
    print(f"\n  [OK] face_separation = {ckpt['face_separation']:.4f}  (target: < 0.60)")
    print(f"  [OK] active_units    = {ckpt['active_units']}  (target: >= 16)")


if __name__ == "__main__":
    main()
