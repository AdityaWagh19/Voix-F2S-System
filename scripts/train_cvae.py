"""CVAE Training Script for VOIX (Phase 3 - Ground-Up Reconstruction).

Trains the True Conditional Variational Autoencoder (CVAE v2) to map 560-D fused face
representations into 192-D ECAPA-TDNN speaker embedding space:
  - Recognition posterior q_phi(z | e_s, e_f)
  - Conditional prior p_psi(z | e_f)
  - Conditional decoder p_theta(e_s | z, e_f)

Usage:
    python scripts/train_cvae.py --epochs 35 --batch-size 256 --lr 3e-4
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from voix.data.dataset import VoxCelebPairedDataset, WeightedSpeakerSampler, build_dataloader
from voix.models.cvae import CVAE
from voix.training.losses import CVAELoss, CosineReconstructionLoss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train VOIX True CVAE v2")
    parser.add_argument("--h5-path", type=str, default="D:/voix/data/processed/features.h5")
    parser.add_argument("--index-csv", type=str, default="D:/voix/data/processed/clean_index.csv")
    parser.add_argument("--scaler-path", type=str, default="D:/voix/data/processed/feature_scaler.pt")
    parser.add_argument("--ecapa-scaler-path", type=str, default="D:/voix/data/processed/ecapa_scaler.pt",
                        help="L2-norm only ECAPA scaler (do not use z-score for ECAPA)")
    parser.add_argument("--checkpoint-dir", type=str, default="D:/voix/checkpoints")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--lambda-fb", type=float, default=0.01, help="Free bits floor in nats")
    parser.add_argument("--beta-max", type=float, default=1.0, help="Maximum KL weight (1.0 = standard CVAE ELBO)")
    parser.add_argument("--warmup-ratio", type=float, default=0.30, help="Fraction of steps for beta warmup")
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()



def _apply_config_defaults(args):
    """Fill argparse defaults from environment.yaml (CLI flags take priority)."""
    try:
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from voix.utils.config import load_config
        cfg = load_config()
        tcfg = cfg.get("training", {}).get("cvae", {})
        # Only touch defaults that were not explicitly overridden by the user.
        if args.beta_max == 1.0 and "beta_max" in tcfg:
            args.beta_max = float(tcfg["beta_max"])
        if args.warmup_ratio == 0.30 and "warmup_ratio" in tcfg:
            args.warmup_ratio = float(tcfg["warmup_ratio"])
        if args.epochs == 50 and "epochs" in tcfg:
            args.epochs = int(tcfg["epochs"])
        if args.batch_size == 256 and "batch_size" in tcfg:
            args.batch_size = int(tcfg["batch_size"])
        if args.lr == 3e-4 and "lr" in tcfg:
            args.lr = float(tcfg["lr"])
        if args.lambda_fb == 0.01 and "lambda_fb" in tcfg:
            args.lambda_fb = float(tcfg["lambda_fb"])
        if args.dropout == 0.05 and "dropout" in tcfg:
            args.dropout = float(tcfg["dropout"])
    except Exception as exc:
        print(f"[train_cvae] Warning: could not load environment.yaml: {exc}")
    return args

def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.05,
) -> LambdaLR:
    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return LambdaLR(optimizer, lr_lambda)


def evaluate(
    model: CVAE,
    val_loader: torch.utils.data.DataLoader,
    loss_fn: CVAELoss,
    device: torch.device,
    current_step: int,
    face_mean: torch.Tensor | None = None,
    face_std: torch.Tensor | None = None,
) -> dict:
    """Evaluate True CVAE using the conditional prior p_psi(z | e_f)."""
    model.eval()
    total_loss = 0.0
    recon_loss_sum = 0.0
    kl_loss_sum = 0.0
    cosine_sim_sum = 0.0
    active_units_list: list[int] = []
    n_samples = 0

    with torch.no_grad():
        for face, spk, _ in val_loader:
            face = face.to(device, non_blocking=True)
            spk = spk.to(device, non_blocking=True)
            # Phase 2.3: Z-score face features, L2-normalize ECAPA targets
            if face_mean is not None:
                face = (face - face_mean) / face_std
            spk = torch.nn.functional.normalize(spk, p=2, dim=-1)
            b = face.size(0)

            # Val: use prior MEAN (deterministic) for cleaner signal.
            # Stochastic noise (σ≈1) drowns learned prior_mu when σ not yet small.
            # A rising val_cos_det indicates the prior IS learning face→voice.
            out = model(face, e_s=None, deterministic=True)
            loss, metrics = loss_fn(
                e_s_hat=out.e_s_hat,
                e_s=spk,
                mu=out.mu,
                logvar=out.logvar,
                prior_mu=out.prior_mu,
                prior_logvar=out.prior_logvar,
                step=current_step,
            )

            cos_sim = F.cosine_similarity(out.e_s_hat, spk, dim=-1).mean().item()

            total_loss += loss.item() * b
            recon_loss_sum += metrics["loss_recon"] * b
            kl_loss_sum += metrics["loss_kl"] * b
            cosine_sim_sum += cos_sim * b
            active_units_list.append(metrics["active_units"])
            n_samples += b

    return {
        "val_loss": total_loss / n_samples,
        "val_recon_loss": recon_loss_sum / n_samples,
        "val_kl_loss": kl_loss_sum / n_samples,
        "val_cosine_sim": cosine_sim_sum / n_samples,
        "val_active_units": int(np.mean(active_units_list)),
    }


def main() -> None:
    args = parse_args()
    args = _apply_config_defaults(args)
    set_seed(args.seed)

    print("=" * 65)
    print("  VOIX TRUE CVAE v2 MAPPER TRAINING")
    print("=" * 65)
    print(f"Device:           {args.device}")
    if args.device.startswith("cuda"):
        print(f"GPU:              {torch.cuda.get_device_name(0)}")
    print(f"Epochs:           {args.epochs}")
    print(f"Batch Size:       {args.batch_size}")
    print(f"Learning Rate:    {args.lr}")
    print(f"Free Bits Floor:  {args.lambda_fb} nats/dim")
    print(f"Beta Max:         {args.beta_max}")
    print(f"Warmup Ratio:     {args.warmup_ratio}")
    print("=" * 65)

    device = torch.device(args.device)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # 1. Load Datasets
    print("\n[1/4] Loading pre-normalized datasets into RAM...")
    t0 = time.time()
    # Phase 2.3 fix: do NOT apply scaler inside dataset.
    # ECAPA is L2-normalized in the training loop (not z-scored).
    # Face features are z-scored using face_scaler.pt loaded below.
    train_ds, val_ds, test_ds = VoxCelebPairedDataset.from_index_csv(
        h5_path=args.h5_path,
        index_csv=args.index_csv,
        scaler_path=None,  # Do NOT apply z-score inside dataset
        in_memory=True,
    )
    # Load face scaler and apply ONLY to face features in the training loop
    _face_scaler = torch.load(
        r'D:/voix/data/processed/face_scaler.pt', weights_only=True, map_location='cpu'
    )
    _face_mean = _face_scaler['face_mean'].to(device)
    _face_std  = _face_scaler['face_std'].to(device)
    print(f"  Train samples: {len(train_ds):,}")
    print(f"  Val samples:   {len(val_ds):,}")
    print(f"  Test samples:  {len(test_ds):,} (held out)")
    print(f"  Dataset load time: {time.time() - t0:.2f}s")

    # 2. Build DataLoaders — Phase 2.2: WeightedSpeakerSampler for balanced training
    _sampler = WeightedSpeakerSampler(
        train_ds,
        max_per_speaker=20,
        min_per_speaker=5,
        seed=args.seed,
    )
    print(f'  Effective samples/epoch (balanced sampler): {len(_sampler):,}')
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=_sampler,
        num_workers=0,
        pin_memory=False,
        drop_last=True,
    )
    val_loader = build_dataloader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        seed=args.seed,
    )

    # 3. Model & Loss Setup
    print("\n[2/4] Initializing True CVAE Model and Loss Function...")
    model = CVAE(
        face_dim=560,
        speaker_dim=192,
        latent_dim=192,
        dropout=args.dropout,
    ).to(device)

    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(args.warmup_ratio * total_steps)

    loss_fn = CVAELoss(
        lambda_fb=args.lambda_fb,
        beta_max=args.beta_max,
        warmup_steps=warmup_steps,
        reduction="mean",
    )
    aux_recon = CosineReconstructionLoss()

    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
    )

    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.05 * total_steps),
        num_training_steps=total_steps,
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total trainable parameters: {total_params:,}")
    print(f"  Total training steps:       {total_steps:,}")
    print(f"  Beta annealing warmup steps: {warmup_steps:,}")

    # 4. Training Loop
    print("\n[3/4] Starting Training...")
    best_val_cosine = -1.0
    history: list[dict] = []
    global_step = 0

    best_checkpoint_path = os.path.join(args.checkpoint_dir, "cvae_best.pt")
    latest_checkpoint_path = os.path.join(args.checkpoint_dir, "cvae_latest.pt")
    history_path = os.path.join(args.checkpoint_dir, "training_history.json")

    start_train_time = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_t0 = time.time()
        train_loss_sum = 0.0
        train_recon_sum = 0.0
        train_kl_sum = 0.0
        train_samples = 0

        for face, spk, _ in train_loader:
            face = face.to(device, non_blocking=True)
            spk = spk.to(device, non_blocking=True)
            # Phase 2.3: Z-score face features, L2-normalize ECAPA targets
            face = (face - _face_mean) / _face_std
            spk = torch.nn.functional.normalize(spk, p=2, dim=-1)
            b = face.size(0)

            optimizer.zero_grad(set_to_none=True)

            # 1. Forward through joint posterior q(z | e_s, e_f)
            out_q = model(face, e_s=spk, deterministic=False)

            # 2. CVAE ELBO loss with conditional KL
            loss_cvae, metrics = loss_fn(
                e_s_hat=out_q.e_s_hat,
                e_s=spk,
                mu=out_q.mu,
                logvar=out_q.logvar,
                prior_mu=out_q.prior_mu,
                prior_logvar=out_q.prior_logvar,
                step=global_step,
            )

            # 3. Auxiliary prior reconstruction to ensure p(z|e_f) decoder alignment
            # Decoder takes z only (no e_f skip) — pass prior_mu as z
            e_s_hat_prior = model.decoder(out_q.prior_mu)
            loss_prior_recon = aux_recon(e_s_hat_prior, spk)

            # 4. Prior mean-matching: force p_psi to track the recognition posterior.
            # This provides direct gradient signal to the prior network beyond KL.
            # Reference: InfoVAE (Zhao et al. 2017), CVAE-CA (Shen et al. 2018)
            loss_prior_match = torch.nn.functional.mse_loss(
                out_q.prior_mu, out_q.mu.detach()  # prior_mu → recognition_mu
            )

            loss = loss_cvae + 0.5 * loss_prior_recon + 1.0 * loss_prior_match

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            lr_scheduler.step()

            train_loss_sum += loss.item() * b
            train_recon_sum += metrics["loss_recon"] * b
            train_kl_sum += metrics["loss_kl"] * b
            train_samples += b
            global_step += 1

        train_loss = train_loss_sum / train_samples
        train_recon = train_recon_sum / train_samples
        train_kl = train_kl_sum / train_samples
        epoch_time = time.time() - epoch_t0

        # Validation (prior only)
        val_metrics = evaluate(model, val_loader, loss_fn, device, global_step, _face_mean, _face_std)
        current_lr = optimizer.param_groups[0]["lr"]
        current_beta = loss_fn.beta(global_step)

        log_entry = {
            "epoch": epoch,
            "global_step": global_step,
            "train_loss": round(train_loss, 4),
            "train_recon": round(train_recon, 4),
            "train_kl": round(train_kl, 4),
            "val_loss": round(val_metrics["val_loss"], 4),
            "val_recon": round(val_metrics["val_recon_loss"], 4),
            "val_kl": round(val_metrics["val_kl_loss"], 4),
            "val_cosine_sim": round(val_metrics["val_cosine_sim"], 4),
            "val_active_units": val_metrics["val_active_units"],
            "beta": round(current_beta, 4),
            "lr": f"{current_lr:.2e}",
            "epoch_sec": round(epoch_time, 2),
        }
        history.append(log_entry)

        is_best = val_metrics["val_cosine_sim"] > best_val_cosine
        best_marker = " [*BEST]" if is_best else ""
        # Phase 3.2: Prior network sensitivity diagnostic (first 3 epochs)
        if epoch <= 3:
            with torch.no_grad():
                _e_f_sample = next(iter(val_loader))[0][:4].to(device)
                _e_f_zero = torch.zeros_like(_e_f_sample)
                _mu_real, _ = model.prior_net(_e_f_sample)
                _mu_zero, _ = model.prior_net(_e_f_zero)
                _sensitivity = (_mu_real - _mu_zero).norm(dim=-1).mean().item()
            if _sensitivity < 0.01:
                raise RuntimeError(f"STOP: Prior network sensitivity={_sensitivity:.4f} — net ignoring face input. Check architecture.")
            print(f"  [Diag] Prior net sensitivity to face: {_sensitivity:.4f} (should be > 0.01)")

        print(
            f"Epoch {epoch:02d}/{args.epochs:02d} [{epoch_time:.1f}s] | "
            f"Train Loss: {train_loss:.4f} (Recon: {train_recon:.4f}, KL/dim: {train_kl:.3f}) | "
            f"Val Prior CosSim: {val_metrics['val_cosine_sim']:.4f}{best_marker} | "
            f"AU: {val_metrics['val_active_units']}/192 | Beta: {current_beta:.3f}"
        )

        # Save latest checkpoint
        torch.save({
            "epoch": epoch,
            "global_step": global_step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_cosine_sim": val_metrics["val_cosine_sim"],
            "args": vars(args),
        }, latest_checkpoint_path)

        # Save best checkpoint
        if is_best:
            best_val_cosine = val_metrics["val_cosine_sim"]
            torch.save({
                "epoch": epoch,
                "global_step": global_step,
                "model_state_dict": model.state_dict(),
                "val_cosine_sim": best_val_cosine,
                "val_metrics": val_metrics,
                "args": vars(args),
            }, best_checkpoint_path)

        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

    total_train_sec = time.time() - start_train_time
    print("\n" + "=" * 65)
    print("  TRAINING COMPLETE")
    print(f"  Total time:          {total_train_sec / 60:.2f} minutes")
    print(f"  Best Val Prior Cos:  {best_val_cosine:.4f}")
    print(f"  Best Checkpoint:     {best_checkpoint_path}")
    print(f"  History Log:         {history_path}")
    print("=" * 65)


if __name__ == "__main__":
    main()
