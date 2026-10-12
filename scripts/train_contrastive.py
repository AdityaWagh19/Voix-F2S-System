"""
Voix Phase 3: Contrastive InfoNCE / SupCon Face-to-Voice Training.

Learns a joint metric embedding space between facial biometrics and speaker voices:
  1. Contrastive InfoNCE / SupCon loss with temperature tau=0.07
  2. Multi-speaker negative discrimination (pushes different speakers apart)
  3. Quality clip filtering (Task 1.3: excludes noisy outlier clips)
  4. Cross-modal retrieval evaluation (Recall@1, Recall@3, Recall@10) on unseen speakers

Usage:
  python scripts/train_contrastive.py --name C_infonce --epochs 40 --device cuda
"""

import argparse, json, time, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import h5py
from torch.utils.data import Dataset, DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def parse_args():
    p = argparse.ArgumentParser(description="Voix Contrastive InfoNCE Trainer")
    p.add_argument("--name",        default="C_infonce", help="Run name")
    p.add_argument("--epochs",      type=int,   default=40)
    p.add_argument("--batch",       type=int,   default=512)
    p.add_argument("--lr",          type=float, default=1e-3)
    p.add_argument("--tau",         type=float, default=0.07, help="InfoNCE temperature")
    p.add_argument("--embed-dim",   type=int,   default=192, help="Joint embedding dimension")
    p.add_argument("--joint-loss", action="store_true", default=True, help="Combine cosine + InfoNCE")
    p.add_argument("--joint-weight", type=float, default=0.1, help="Weight of InfoNCE vs cosine")
    p.add_argument("--filter-outliers", action="store_true", default=True, help="Filter noisy clips (<2 sigma)")
    p.add_argument("--device",      default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


class ContrastiveFaceVoiceNet(nn.Module):
    """Deep projection network mapping face features into normalized speaker space."""
    def __init__(self, in_face_dim=560, out_dim=192, hidden=(512, 384, 256)):
        super().__init__()
        layers, prev = [], in_face_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(0.1)]
            prev = h
        self.backbone = nn.Sequential(*layers)
        self.voice_head = nn.Linear(prev, out_dim)

    def forward(self, face_features):
        feat = self.backbone(face_features)
        voice = self.voice_head(feat)
        return F.normalize(voice, dim=-1)


def info_nce_loss(face_embs, spk_embs, spk_labels, tau=0.07):
    """
    Supervised Contrastive / InfoNCE loss between face and speaker embeddings.
    Handles multiple clips per speaker in the batch.
    """
    sim_matrix = (face_embs @ spk_embs.T) / tau  # (B, B)
    
    # Mask of identical speaker pairs
    spk_labels = spk_labels.contiguous().view(-1, 1)
    pos_mask = torch.eq(spk_labels, spk_labels.T).float()  # (B, B)
    
    # Row-wise log_softmax
    log_prob_f2s = F.log_softmax(sim_matrix, dim=1)
    # Col-wise log_softmax
    log_prob_s2f = F.log_softmax(sim_matrix.T, dim=1)
    
    # Mean of positive log-probabilities (SupCon)
    pos_counts = pos_mask.sum(1).clamp(min=1.0)
    loss_f2s = -(pos_mask * log_prob_f2s).sum(1) / pos_counts
    loss_s2f = -(pos_mask * log_prob_s2f).sum(1) / pos_counts
    
    return 0.5 * (loss_f2s.mean() + loss_s2f.mean())


class PairedDataset(Dataset):
    def __init__(self, faces, spks, spk_ids):
        self.faces = faces
        self.spks = spks
        self.spk_ids = spk_ids

    def __len__(self):
        return len(self.faces)

    def __getitem__(self, idx):
        return self.faces[idx], self.spks[idx], self.spk_ids[idx]


def load_datasets(filter_outliers=True):
    h5_path  = ROOT / "data" / "processed" / "features.h5"
    csv_path = ROOT / "data" / "processed" / "clean_index.csv"
    df = pd.read_csv(csv_path)

    with h5py.File(h5_path, "r") as f:
        all_face = f["face_features"][:]
        all_spk  = f["speaker_features"][:]

    # Scaler
    sc_path = ROOT / "data" / "processed" / "face_scaler.pt"
    if sc_path.exists():
        sc   = torch.load(sc_path, weights_only=True, map_location="cpu")
        face_mean = sc["face_mean"].numpy()
        face_std  = sc["face_std"].numpy() + 1e-8
    else:
        face_mean, face_std = 0.0, 1.0

    # Speaker integer encoding
    unique_spks = sorted(df["speaker_id"].unique())
    spk2int = {s: i for i, s in enumerate(unique_spks)}
    df["spk_code"] = df["speaker_id"].map(spk2int)

    # Optional Task 1.3 Outlier filtering on train set
    train_df = df[df["split"] == "train"].copy()
    val_df   = df[df["split"] == "val"].copy()

    if filter_outliers:
        spk_avg = torch.load(ROOT / "data" / "processed" / "speaker_avg_ecapa.pt", weights_only=False)
        sims = []
        for _, row in train_df.iterrows():
            idx = row["hdf5_index"]
            spk = row["speaker_id"]
            avg = spk_avg[spk]
            feat = all_spk[idx]
            cos = np.dot(feat, avg) / (np.linalg.norm(feat) * np.linalg.norm(avg) + 1e-8)
            sims.append(cos)
        sims = np.array(sims)
        thresh = sims.mean() - 2 * sims.std()
        keep_mask = sims >= thresh
        n_dropped = np.sum(~keep_mask)
        train_df = train_df.iloc[keep_mask].reset_index(drop=True)
        print(f"  [Task 1.3 Filter] Dropped {n_dropped} outlier clips (kept {len(train_df)})")

    train_idx = train_df["hdf5_index"].values
    val_idx   = val_df["hdf5_index"].values

    train_face = (all_face[train_idx].astype(np.float32) - face_mean) / face_std
    val_face   = (all_face[val_idx].astype(np.float32) - face_mean) / face_std

    train_spk = all_spk[train_idx].astype(np.float32)
    val_spk   = all_spk[val_idx].astype(np.float32)

    # Normalize ground truth ECAPA embeddings
    train_spk /= np.linalg.norm(train_spk, axis=1, keepdims=True) + 1e-8
    val_spk   /= np.linalg.norm(val_spk, axis=1, keepdims=True) + 1e-8

    train_ds = PairedDataset(torch.tensor(train_face), torch.tensor(train_spk), torch.tensor(train_df["spk_code"].values))
    val_ds   = PairedDataset(torch.tensor(val_face), torch.tensor(val_spk), torch.tensor(val_df["spk_code"].values))

    print(f"  Train: {len(train_ds)} clips | Val: {len(val_ds)} clips")
    return train_ds, val_ds, val_face.shape[1]


def evaluate_metrics(model, val_loader, device):
    model.eval()
    all_preds, all_gts, all_spkids = [], [], []
    with torch.no_grad():
        for face, spk, spk_id in val_loader:
            face = face.to(device)
            p = model(face).cpu()
            all_preds.append(p)
            all_gts.append(spk)
            all_spkids.append(spk_id)

    pred = torch.cat(all_preds)
    gt   = torch.cat(all_gts)
    spk_ids = torch.cat(all_spkids)

    val_cos = F.cosine_similarity(pred, gt, dim=-1).mean().item()

    # Cross-modal retrieval accuracy
    n = min(2000, len(pred))
    idx = torch.randperm(len(pred))[:n]
    pq = pred[idx]
    gq = gt[idx]
    sq = spk_ids[idx]

    sim = pq @ gq.T  # (n, n)
    match_mask = (sq.unsqueeze(1) == sq.unsqueeze(0))  # (n, n) same speaker

    recalls = {}
    for k in [1, 3, 10]:
        topk = torch.topk(sim, k=min(k, n), dim=1).indices  # (n, k)
        correct = 0
        for i in range(n):
            if any(match_mask[i, topk[i]].tolist()):
                correct += 1
        recalls[f"recall_at_{k}"] = correct / n

    return val_cos, recalls


def main():
    args = parse_args()
    device = args.device

    print("\n" + "=" * 65)
    print(f"  VOIX CONTRASTIVE InfoNCE TRAINING: {args.name}")
    print(f"  epochs={args.epochs} | batch={args.batch} | lr={args.lr} | tau={args.tau}")
    print("=" * 65 + "\n")

    train_ds, val_ds, face_dim = load_datasets(filter_outliers=args.filter_outliers)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=args.batch, shuffle=False, pin_memory=True)

    model = ContrastiveFaceVoiceNet(in_face_dim=face_dim, out_dim=args.embed_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    ckpt_dir = ROOT / "results" / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_val_cos, history = -1.0, []
    print(f"\n  {'Ep':>3}  {'Tr_loss':>8}  {'Val_cos':>8}  {'R@1':>7}  {'R@3':>7}  {'R@10':>7}  {'Time':>5}")
    print(f"  {'-'*55}")

    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        total_loss, n_batches = 0.0, 0

        for face, spk, spk_id in train_loader:
            face, spk, spk_id = face.to(device), spk.to(device), spk_id.to(device)
            face_emb = model(face)
            nce_loss = info_nce_loss(face_emb, spk, spk_id, tau=args.tau)
            if args.joint_loss:
                cos_loss = (1 - F.cosine_similarity(face_emb, spk, dim=-1)).mean()
                loss = cos_loss + args.joint_weight * nce_loss
            else:
                loss = nce_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        scheduler.step()
        tr_loss = total_loss / max(1, n_batches)

        val_cos, recalls = evaluate_metrics(model, val_loader, device)
        elapsed = time.time() - t0
        mark = " *" if val_cos > best_val_cos else ""

        if val_cos > best_val_cos:
            best_val_cos = val_cos
            torch.save(model.state_dict(), ckpt_dir / f"{args.name}_best.pt")

        print(f"  {ep:>3}  {tr_loss:>8.4f}  {val_cos:>8.4f}  {recalls['recall_at_1']:>7.4f}  {recalls['recall_at_3']:>7.4f}  {recalls['recall_at_10']:>7.4f}  {elapsed:>4.1f}s{mark}")
        history.append({
            "epoch": ep,
            "train_loss": round(tr_loss, 5),
            "val_cos": round(val_cos, 5),
            **{k: round(v, 5) for k, v in recalls.items()}
        })

    torch.save(model.state_dict(), ckpt_dir / f"{args.name}_final.pt")
    with open(ckpt_dir / f"{args.name}_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nBest val_cos: {best_val_cos:.4f}")
    print(f"Checkpoint saved: results/checkpoints/{args.name}_best.pt")


if __name__ == "__main__":
    main()
