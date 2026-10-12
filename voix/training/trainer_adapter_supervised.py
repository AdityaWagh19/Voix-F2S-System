"""Supervised Style Projection Adapter Training Engine (Ground-Up Phase 4 Fix).

Trains StyleAdapter to project 192-D ECAPA-TDNN speaker embeddings
directly onto StyleTTS 2's genuine 128-D acoustic timbre manifold
extracted from reference speech.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, r"D:\voix")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset

from voix.models.adapter import StyleAdapter


class SupervisedAdapterLoss(nn.Module):
    """Supervised acoustic manifold alignment loss."""

    def __init__(self, lambda_cos: float = 1.0) -> None:
        super().__init__()
        self.lambda_cos = lambda_cos
        self.mse = nn.MSELoss()

    def forward(self, s_hat: torch.Tensor, s_target: torch.Tensor) -> tuple[torch.Tensor, dict]:
        # 1. Exact coordinate reconstruction in style space
        l_mse = self.mse(s_hat, s_target)

        # 2. Directional angular alignment
        cos_sim = F.cosine_similarity(s_hat, s_target, dim=-1)
        l_cos = (1.0 - cos_sim).mean()

        total = l_mse + self.lambda_cos * l_cos
        metrics = {
            "loss_total": total.item(),
            "loss_mse": l_mse.item(),
            "loss_cos": l_cos.item(),
            "mean_cos_sim": cos_sim.mean().item(),
        }
        return total, metrics


def main() -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("=" * 65)
    print("  SUPERVISED STYLE ADAPTER TRAINING (Ground-Up Fix)")
    print("=" * 65)
    print(f"Device: {device}")

    pairs_path = r"D:\voix\data\processed\style_pairs.pt"
    if not os.path.exists(pairs_path):
        raise FileNotFoundError(f"Pairs file not found at {pairs_path}")

    data = torch.load(pairs_path, map_location=device, weights_only=True)
    ecapa = data["ecapa"].float()   # (N, 192)
    timbre = data["timbre"].float() # (N, 128)
    n_total = len(ecapa)
    print(f"Loaded {n_total} ground-truth pairs from {pairs_path}")

    # 80/20 train/val split
    n_train = int(n_total * 0.8)
    indices = torch.randperm(n_total, generator=torch.Generator().manual_seed(42))
    train_idx = indices[:n_train]
    val_idx = indices[n_train:]

    train_ds = TensorDataset(ecapa[train_idx], timbre[train_idx])
    val_ds = TensorDataset(ecapa[val_idx], timbre[val_idx])

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

    print(f"Train samples: {len(train_ds)} | Val samples: {len(val_ds)}")

    model = StyleAdapter(in_dim=192, hidden_dim=256, out_dim=128).to(device)
    loss_fn = SupervisedAdapterLoss(lambda_cos=1.0)
    optimizer = AdamW(model.parameters(), lr=5e-4, weight_decay=1e-3)
    epochs = 40
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    best_val_loss = float("inf")
    best_ckpt_path = r"D:\voix\checkpoints\style_adapter_best.pt"

    t0 = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_cos_sum = 0.0
        n_samples = 0

        for batch_e, batch_s in train_loader:
            optimizer.zero_grad(set_to_none=True)
            s_hat = model(batch_e)
            loss, metrics = loss_fn(s_hat, batch_s)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            b = batch_e.size(0)
            train_loss_sum += loss.item() * b
            train_cos_sum += metrics["mean_cos_sim"] * b
            n_samples += b

        scheduler.step()
        train_loss = train_loss_sum / n_samples
        train_cos = train_cos_sum / n_samples

        # Validation
        model.eval()
        val_loss_sum = 0.0
        val_cos_sum = 0.0
        val_samples = 0
        with torch.no_grad():
            for batch_e, batch_s in val_loader:
                s_hat = model(batch_e)
                v_loss, v_metrics = loss_fn(s_hat, batch_s)
                b = batch_e.size(0)
                val_loss_sum += v_loss.item() * b
                val_cos_sum += v_metrics["mean_cos_sim"] * b
                val_samples += b

        val_loss = val_loss_sum / val_samples
        val_cos = val_cos_sum / val_samples

        is_best = val_loss < best_val_loss
        marker = " [*BEST]" if is_best else ""

        if epoch % 5 == 0 or is_best or epoch == epochs:
            print(
                f"Epoch {epoch:02d}/{epochs} | "
                f"Train Loss: {train_loss:.4f} (Cos: {train_cos:.3f}) | "
                f"Val Loss: {val_loss:.4f} (Cos: {val_cos:.3f}){marker}"
            )

        if is_best:
            best_val_loss = val_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_loss": val_loss,
                "val_cosine_sim": val_cos,
            }, best_ckpt_path)

    print("=" * 65)
    print(f"TRAINING COMPLETE in {time.time() - t0:.1f}s!")
    print(f"Best Val Loss: {best_val_loss:.4f} | Saved to: {best_ckpt_path}")
    print("=" * 65)


if __name__ == "__main__":
    main()
