"""StyleAdapter retraining on CFVN-produced ECAPA embeddings.

Trains the StyleAdapter to map CFVN-produced face embeddings (192-D ECAPA space)
to StyleTTS2 style targets (128-D timbre).

Training-serving parity: The adapter is trained on CFVN-produced ECAPA embeddings
(same as inference input), not raw ECAPA. This eliminates the distribution
mismatch that caused all voices to sound similar.

Usage:
    python -m voix.training.trainer_adapter_cfvn
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
from torch.utils.data import DataLoader, TensorDataset, random_split

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from voix.models.adapter import StyleAdapter
from voix.models.contrastive_net import ContrastiveFaceVoiceNet

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
H5_PATH      = r"D:\voix\data\processed\features.h5"
STYLE_PAIRS  = r"D:\voix\data\processed\style_pairs.pt"
CFVN_PATH    = r"D:\voix\results/checkpoints/D_joint_infonce_best.pt"
SAVE_PATH    = r"D:\voix\checkpoints/style_adapter_cfvn_best.pt"

EPOCHS       = 100
BATCH_SIZE   = 64
LR           = 5e-4
WEIGHT_DECAY = 1e-3
VAL_FRAC     = 0.2
SEED         = 42


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 1. Load ContrastiveFaceVoiceNet
    print("\nLoading ContrastiveFaceVoiceNet...")
    cfvn = ContrastiveFaceVoiceNet.from_checkpoint(CFVN_PATH, device=str(device)).to(device)
    cfvn.eval()

    # 2. Load features.h5
    print("Loading features.h5...")
    with h5py.File(H5_PATH, "r") as f:
        face_features = torch.tensor(f["face_features"][:], dtype=torch.float32)
        speaker_features = torch.tensor(f["speaker_features"][:], dtype=torch.float32)

    print(f"  face_features: {face_features.shape}")
    print(f"  speaker_features: {speaker_features.shape}")

    # 3. Compute CFVN embeddings for all face_features
    print("Computing CFVN embeddings for all face features...")
    cfvn_embeddings = []
    batch_size = 512
    with torch.no_grad():
        for i in range(0, len(face_features), batch_size):
            batch = face_features[i:i+batch_size].to(device)
            cfvn_batch = cfvn(batch, normalize=True)
            cfvn_embeddings.append(cfvn_batch.cpu())
    cfvn_embeddings = torch.cat(cfvn_embeddings, dim=0)
    print(f"  CFVN embeddings: {cfvn_embeddings.shape}")
    print(f"  CFVN embedding norm: {torch.norm(cfvn_embeddings, dim=-1).mean().item():.4f}")

    # 4. Load style_pairs.pt
    print("Loading style_pairs.pt...")
    data = torch.load(STYLE_PAIRS, weights_only=True)
    ecapa_pairs = data["ecapa"].float()
    timbre = data["timbre"].float()
    prosody = data["prosody"].float()
    print(f"  ecapa_pairs: {ecapa_pairs.shape}")
    print(f"  timbre: {timbre.shape}")
    print(f"  prosody: {prosody.shape}")

    # 5. Match speaker_features (ECAPA from features.h5) with ecapa_pairs
    #    by nearest neighbor (cosine similarity)
    print("Matching speaker_features with ecapa_pairs by nearest neighbor...")
    style_targets = []
    match_sims = []
    for i in range(len(speaker_features)):
        spk_emb = speaker_features[i:i+1]
        sims = torch.nn.functional.cosine_similarity(spk_emb, ecapa_pairs, dim=1)
        best_idx = torch.argmax(sims).item()
        best_sim = sims[best_idx].item()
        match_sims.append(best_sim)
        style_targets.append(timbre[best_idx:best_idx+1])  # Use timbre as target

    style_targets = torch.cat(style_targets, dim=0)
    match_sims = torch.tensor(match_sims)
    print(f"  style_targets: {style_targets.shape}")
    print(f"  Match similarity: mean={match_sims.mean().item():.4f}, "
          f"min={match_sims.min().item():.4f}, max={match_sims.max().item():.4f}")

    # 6. Train StyleAdapter on (CFVN_embedding, timbre) pairs
    print(f"\nTraining StyleAdapter on {len(cfvn_embeddings)} pairs...")

    # Split train/val
    n_val = max(100, int(len(cfvn_embeddings) * VAL_FRAC))
    n_train = len(cfvn_embeddings) - n_val

    perm = torch.randperm(len(cfvn_embeddings))
    train_idx = perm[:n_train]
    val_idx = perm[n_train:]

    train_cfvn = cfvn_embeddings[train_idx]
    train_targets = style_targets[train_idx]
    val_cfvn = cfvn_embeddings[val_idx]
    val_targets = style_targets[val_idx]

    print(f"  Train: {len(train_cfvn)}, Val: {len(val_cfvn)}")

    # 7. Initialize StyleAdapter (outputs 128-D)
    model = StyleAdapter(in_dim=192, hidden_dim=256, out_dim=128).to(device)
    loss_fn = nn.MSELoss()
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    train_ds = TensorDataset(train_cfvn, train_targets)
    val_ds = TensorDataset(val_cfvn, val_targets)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    # 8. Training loop
    best_val_loss = float('inf')
    for epoch in range(EPOCHS):
        model.train()
        tr_loss = 0.0
        n_tr = 0
        for be, bs in train_loader:
            be, bs = be.to(device), bs.to(device)
            optimizer.zero_grad()
            pred = model(be)
            loss = loss_fn(pred, bs)
            loss.backward()
            optimizer.step()
            tr_loss += loss.item() * be.size(0)
            n_tr += be.size(0)
        tr_loss /= n_tr

        model.eval()
        va_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for be, bs in val_loader:
                be, bs = be.to(device), bs.to(device)
                pred = model(be)
                loss = loss_fn(pred, bs)
                va_loss += loss.item() * be.size(0)
                n_val += be.size(0)
        va_loss /= n_val

        is_best = va_loss < best_val_loss
        if epoch % 10 == 0 or is_best or epoch < 5:
            print(f"  Ep {epoch:3d}: tr_loss={tr_loss:.6f} va_loss={va_loss:.6f}"
                  f"{' *' if is_best else ''}")

        if is_best:
            best_val_loss = va_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_loss": va_loss,
                "train_loss": tr_loss,
                "hparams": {"in_dim": 192, "hidden_dim": 256, "out_dim": 128},
            }, SAVE_PATH)

    print(f"\nTraining complete! Best val_loss: {best_val_loss:.6f}")
    print(f"Saved to: {SAVE_PATH}")


if __name__ == "__main__":
    main()
