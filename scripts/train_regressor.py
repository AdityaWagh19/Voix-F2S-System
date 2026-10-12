"""
Voix Phase 1-3 Regressor Trainer.
Trains a direct face->ECAPA cosine regressor with optional:
  --speaker-avg   : use per-speaker averaged ECAPA targets
  --gender-loss   : add gender auxiliary BCE loss
  --face-type     : morphology (560D) or arcface (512D)

Usage:
  python scripts/train_regressor.py --name A_baseline
  python scripts/train_regressor.py --name B_spkavg_gender --speaker-avg --gender-loss
  python scripts/train_regressor.py --name C_arcface --face-type arcface --speaker-avg --gender-loss
"""

import argparse, json, time, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import h5py
from torch.utils.data import TensorDataset, DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--name",         required=True, help="Run name (used for checkpoint filename)")
    p.add_argument("--face-type",    default="morphology", choices=["morphology", "arcface"])
    p.add_argument("--speaker-avg",  action="store_true", help="Use speaker-averaged ECAPA targets")
    p.add_argument("--gender-loss",  action="store_true", help="Add gender aux BCE loss")
    p.add_argument("--gender-weight",type=float, default=0.3)
    p.add_argument("--epochs",       type=int,   default=50)
    p.add_argument("--batch",        type=int,   default=512)
    p.add_argument("--lr",           type=float, default=1e-3)
    p.add_argument("--hidden",       nargs="+",  type=int, default=[512, 384, 256])
    p.add_argument("--device",       default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


class DirectRegressor(nn.Module):
    def __init__(self, in_d, out_d=192, hidden=(512, 384, 256), gender_head=False):
        super().__init__()
        layers, prev = [], in_d
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(0.1)]
            prev = h
        self.backbone = nn.Sequential(*layers)
        self.voice_head  = nn.Linear(prev, out_d)
        self.gender_head  = nn.Linear(prev, 1) if gender_head else None

    def forward(self, x):
        feat   = self.backbone(x)
        voice  = self.voice_head(feat)
        gender = self.gender_head(feat) if self.gender_head is not None else None
        return voice, gender


def load_data(face_type, speaker_avg, device):
    h5_path  = ROOT / "data" / "processed" / "features.h5"
    csv_path = ROOT / "data" / "processed" / "clean_index.csv"
    df = pd.read_csv(csv_path)

    with h5py.File(h5_path, "r") as f:
        face_key = "arcface_features" if face_type == "arcface" else "face_features"
        if face_key not in f:
            raise KeyError(f"No key '{face_key}' in H5. Run ArcFace extraction first.")
        all_face = f[face_key][:]
        all_spk  = f["speaker_features"][:]

    # Speaker-averaged ECAPA targets
    if speaker_avg:
        spkavg_path = ROOT / "data" / "processed" / "speaker_avg_ecapa.pt"
        if spkavg_path.exists():
            spk_avg = torch.load(spkavg_path, weights_only=True)
            print(f"  Loaded speaker-avg targets for {len(spk_avg)} speakers")
        else:
            print("  Computing speaker-avg targets...")
            from collections import defaultdict
            spk_embs = defaultdict(list)
            for _, row in df[df["split"] == "train"].iterrows():
                spk_embs[row["speaker_id"]].append(all_spk[row["hdf5_index"]])
            spk_avg = {k: np.mean(v, axis=0) for k, v in spk_embs.items()}
            torch.save(spk_avg, spkavg_path)
            print(f"  Saved speaker-avg targets for {len(spk_avg)} speakers")
    else:
        spk_avg = None

    # Load face scaler
    sc_name = "arcface_scaler.pt" if face_type == "arcface" else "face_scaler.pt"
    sc_path = ROOT / "data" / "processed" / sc_name
    if sc_path.exists():
        sc   = torch.load(sc_path, weights_only=True, map_location="cpu")
        mean = sc["face_mean"].numpy()
        std  = sc["face_std"].numpy() + 1e-8
    else:
        mean, std = 0.0, 1.0

    splits = {}
    for split in ["train", "val"]:
        rows     = df[df["split"] == split].reset_index(drop=True)
        idx      = rows["hdf5_index"].values
        face     = (all_face[idx].astype(np.float32) - mean) / std

        if spk_avg is not None and split == "train":
            # Use speaker-averaged targets for training
            targets = np.array([spk_avg.get(r, all_spk[i]) for r, i in
                                 zip(rows["speaker_id"], idx)], dtype=np.float32)
        else:
            targets = all_spk[idx].astype(np.float32)

        norms   = np.linalg.norm(targets, axis=1, keepdims=True) + 1e-8
        targets /= norms

        gender_labels = None
        if "gender" in rows.columns:
            gender_labels = torch.tensor(
                (rows["gender"].str.lower().str.strip() == "m").values.astype(np.float32)
            )

        splits[split] = (
            torch.tensor(face),
            torch.tensor(targets),
            gender_labels,
        )
        print(f"  {split}: {len(face)} samples | face_dim={face.shape[1]}")
    return splits


def main():
    args   = parse_args()
    device = args.device
    print(f"\n{'='*60}")
    print(f"  VOIX REGRESSOR TRAINING: {args.name}")
    print(f"  face_type={args.face_type} | speaker_avg={args.speaker_avg} | gender_loss={args.gender_loss}")
    print(f"  epochs={args.epochs} | batch={args.batch} | lr={args.lr}")
    print(f"{'='*60}\n")

    splits   = load_data(args.face_type, args.speaker_avg, device)
    train_F, train_S, train_G = splits["train"]
    val_F,   val_S,   _       = splits["val"]

    face_dim = train_F.shape[1]
    use_gender = args.gender_loss and train_G is not None

    # DataLoader
    if use_gender:
        train_ds = TensorDataset(train_F, train_S, train_G)
    else:
        train_ds = TensorDataset(train_F, train_S)
    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True, pin_memory=True)
    val_dl   = DataLoader(TensorDataset(val_F, val_S), batch_size=args.batch, shuffle=False, pin_memory=True)

    model     = DirectRegressor(face_dim, hidden=tuple(args.hidden), gender_head=use_gender).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
    n_params  = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,}\n")

    print(f"  {'Ep':>3}  {'Tr_cos':>8}  {'Val_cos':>8}  {'G_acc':>7}  {'s':>5}")
    print(f"  {'-'*42}")

    best_val, history = -1.0, []
    ckpt_dir = ROOT / "results" / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        model.train(); tr_cos, n = 0.0, 0
        for batch in train_dl:
            if use_gender:
                face, spk, gender = [x.to(device) for x in batch]
            else:
                face, spk = batch[0].to(device), batch[1].to(device)
                gender = None

            voice_pred, gender_pred = model(face)
            loss = (1 - F.cosine_similarity(voice_pred, spk, dim=-1)).mean()
            if use_gender and gender_pred is not None:
                loss = loss + args.gender_weight * F.binary_cross_entropy_with_logits(
                    gender_pred.squeeze(1), gender)
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            with torch.no_grad():
                tr_cos += F.cosine_similarity(voice_pred, spk, dim=-1).mean().item() * face.size(0)
            n += face.size(0)
        scheduler.step()

        model.eval(); val_cos, g_correct, nv = 0.0, 0.0, 0
        with torch.no_grad():
            for face, spk in val_dl:
                face, spk = face.to(device), spk.to(device)
                vp, gp    = model(face)
                val_cos  += F.cosine_similarity(vp, spk, dim=-1).mean().item() * face.size(0)
                nv += face.size(0)
        tr_cos /= n; val_cos /= nv
        elapsed = time.time() - t0
        mark    = " *" if val_cos > best_val else ""
        if val_cos > best_val:
            best_val = val_cos
            torch.save(model.state_dict(), ckpt_dir / f"{args.name}_best.pt")
        print(f"  {ep:>3}  {tr_cos:>8.4f}  {val_cos:>8.4f}  {'---':>7}  {elapsed:>5.1f}s{mark}")
        history.append({"ep": ep, "train_cos": round(tr_cos,5), "val_cos": round(val_cos,5)})

    torch.save(model.state_dict(), ckpt_dir / f"{args.name}_final.pt")
    json.dump(history, open(ckpt_dir / f"{args.name}_history.json","w"), indent=2)

    print(f"\nBest val_cos: {best_val:.4f}")
    print(f"Checkpoint:   results/checkpoints/{args.name}_best.pt")
    print(f"\nRun benchmark:")
    print(f"  python scripts/benchmark.py --checkpoint results/checkpoints/{args.name}_best.pt --name {args.name} --face-type {args.face_type} --skip-audio")


if __name__ == "__main__":
    main()
