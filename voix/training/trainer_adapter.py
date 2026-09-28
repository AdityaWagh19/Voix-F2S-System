"""Style Projection Adapter Training Engine (Phase 4).

Trains the StyleAdapter to project 192-D ECAPA-TDNN speaker embeddings
into the 128-D style conditioning manifold for StyleTTS 2.

Usage:
    python -m voix.training.trainer_adapter --epochs 20 --batch-size 64
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from voix.data.dataset import VoxCelebPairedDataset, build_dataloader
from voix.models.adapter import StyleAdapter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train VOIX Style Projection Adapter")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--checkpoint-dir", type=str, default="D:/voix/checkpoints")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


class AdapterLoss(nn.Module):
    """Multi-objective loss for style projection adapter.

    Combines:
      1. Manifold preserve loss (preserves pairwise speaker geometry in style space)
      2. Regularization to prevent representation explosion
      3. Unit-sphere alignment
    """

    def __init__(self, lambda_geom: float = 1.0, lambda_reg: float = 0.01) -> None:
        super().__init__()
        self.lambda_geom = lambda_geom
        self.lambda_reg = lambda_reg

    def forward(self, s_hat: torch.Tensor, e_s: torch.Tensor) -> tuple[torch.Tensor, dict]:
        # Normalize both spaces for geometric comparison
        s_norm = F.normalize(s_hat, p=2, dim=-1)   # (B, 128)
        e_norm = F.normalize(e_s,   p=2, dim=-1)   # (B, 192)

        # Pairwise Gram matrices (cosine similarity between batch items)
        # s_norm @ s_norm.T vs e_norm @ e_norm.T
        gram_s = torch.mm(s_norm, s_norm.t())      # (B, B)
        gram_e = torch.mm(e_norm, e_norm.t())      # (B, B)

        # Preserves speaker relational geometry across modalities
        loss_geom = F.mse_loss(gram_s, gram_e)

        # Norm regularization: prevent style vectors from drifting to extreme values
        loss_reg = (torch.norm(s_hat, p=2, dim=-1) - 1.0).pow(2).mean()

        total = self.lambda_geom * loss_geom + self.lambda_reg * loss_reg

        metrics = {
            "loss_total": total.item(),
            "loss_geom": loss_geom.item(),
            "loss_reg": loss_reg.item(),
        }
        return total, metrics


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    print("=" * 65)
    print("  VOIX STYLE PROJECTION ADAPTER TRAINING (Phase 4)")
    print("=" * 65)
    print(f"Device:        {args.device}")
    print(f"Epochs:        {args.epochs}")
    print(f"Batch Size:    {args.batch_size}")
    print(f"Learning Rate: {args.lr}")
    print("=" * 65)

    # 1. Load Data
    print("\n[1/3] Loading dataset...")
    train_ds, val_ds, _ = VoxCelebPairedDataset.from_index_csv(in_memory=True)
    train_loader = build_dataloader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader   = build_dataloader(val_ds,   batch_size=args.batch_size, shuffle=False)
    print(f"  Train samples: {len(train_ds):,}")
    print(f"  Val samples:   {len(val_ds):,}")

    # 2. Model & Optimizer
    model = StyleAdapter(in_dim=192, hidden_dim=256, out_dim=128).to(device)
    loss_fn = AdapterLoss()
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    best_val_loss = float("inf")
    best_path = os.path.join(args.checkpoint_dir, "style_adapter_best.pt")

    # 3. Training Loop
    print("\n[2/3] Training StyleAdapter...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_sum = 0.0
        n_samples = 0

        for _, spk, _ in train_loader:
            spk = spk.to(device, non_blocking=True)
            b = spk.size(0)

            optimizer.zero_grad(set_to_none=True)
            s_hat = model(spk)
            loss, _ = loss_fn(s_hat, spk)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss_sum += loss.item() * b
            n_samples += b

        scheduler.step()
        train_loss = train_loss_sum / n_samples

        # Validation
        model.eval()
        val_loss_sum = 0.0
        val_samples = 0
        with torch.no_grad():
            for _, spk, _ in val_loader:
                spk = spk.to(device, non_blocking=True)
                b = spk.size(0)
                s_hat = model(spk)
                v_loss, _ = loss_fn(s_hat, spk)
                val_loss_sum += v_loss.item() * b
                val_samples += b

        val_loss = val_loss_sum / val_samples
        is_best = val_loss < best_val_loss
        marker = " [*BEST]" if is_best else ""

        print(
            f"Epoch {epoch:02d}/{args.epochs:02d} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f}{marker}"
        )

        if is_best:
            best_val_loss = val_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_loss": val_loss,
            }, best_path)

    print("\n" + "=" * 65)
    print("  STYLE ADAPTER TRAINING COMPLETE")
    print(f"  Best Val Loss:   {best_val_loss:.6f}")
    print(f"  Best Checkpoint: {best_path}")
    print("=" * 65)


if __name__ == "__main__":
    main()
