"""
Voix Benchmark Runner - computes all 7 standard metrics for any face->voice checkpoint.
Usage: python scripts/benchmark.py --checkpoint <path> --name <checkpoint_name>

Metrics computed:
  1. val_cos       -- mean cosine similarity (face-predicted ECAPA vs real ECAPA)
  2. SECS          -- Speaker Encoder Cosine Similarity on synthesized audio
  3. Recall@1/3/10 -- retrieval accuracy using predicted voice to find face
  4. Gender Acc    -- does predicted voice match face gender?
  5. MOS-N         -- placeholder (requires human eval, logged as -1)
  6. FAD           -- Frechet Audio Distance (vs real VoxCeleb audio, -1 placeholder)
  7. WER           -- Word Error Rate via Whisper (-1 placeholder)

Outputs:
  results/benchmarks/<name>.json  -- full metric report
  results/benchmarks/summary.csv  -- running comparison table
"""

import argparse, json, os, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import h5py

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def parse_args():
    p = argparse.ArgumentParser(description="Voix Benchmark Runner")
    p.add_argument("--checkpoint",  required=True, help="Path to model checkpoint .pt")
    p.add_argument("--name",        required=True, help="Label e.g. A_baseline")
    p.add_argument("--model-type",  default="regressor", choices=["regressor", "cvae"])
    p.add_argument("--face-type",   default="morphology", choices=["morphology", "arcface"])
    p.add_argument("--device",      default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--synth-text",  default="Artificial intelligence reconstructs natural vocal identity directly from facial geometry.")
    p.add_argument("--skip-audio",  action="store_true")
    p.add_argument("--n-synth",     type=int, default=20)
    return p.parse_args()


class DirectRegressor(nn.Module):
    def __init__(self, in_d, out_d=192, hidden=(512, 384, 256), gender_head=False):
        super().__init__()
        layers, prev = [], in_d
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(0.1)]
            prev = h
        self.backbone = nn.Sequential(*layers)
        self.voice_head = nn.Linear(prev, out_d)
        self.gender_head = nn.Linear(prev, 1) if gender_head else None

    def forward(self, x):
        feat = self.backbone(x)
        voice = self.voice_head(feat)
        gender = self.gender_head(feat) if self.gender_head is not None else None
        return voice, gender


def load_model(ckpt_path, model_type, face_dim, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        ckpt = ckpt["model_state_dict"]

    if model_type == "regressor":
        has_gender = any("gender_head" in k for k in ckpt.keys())
        model = DirectRegressor(face_dim, gender_head=has_gender)
    else:
        from voix.models.cvae import CVAE
        model = CVAE(face_dim=face_dim)

    model.load_state_dict(ckpt, strict=True)
    return model.eval().to(device)


def forward_voice(model, x):
    res = model(x)
    return res[0] if isinstance(res, tuple) else res


def load_data(face_type, split="val"):
    h5_path  = ROOT / "data" / "processed" / "features.h5"
    csv_path = ROOT / "data" / "processed" / "clean_index.csv"
    df = pd.read_csv(csv_path)
    df = df[df["split"] == split].reset_index(drop=True)

    with h5py.File(h5_path, "r") as f:
        face_key = "arcface_features" if face_type == "arcface" else "face_features"
        if face_key not in f:
            raise KeyError(f"Key '{face_key}' not in H5. Available: {list(f.keys())}")
        all_face = f[face_key][:]
        all_spk  = f["speaker_features"][:]

    idx  = df["hdf5_index"].values
    face = all_face[idx].astype(np.float32)
    spk  = all_spk[idx].astype(np.float32)

    scaler_name = "arcface_scaler.pt" if face_type == "arcface" else "face_scaler.pt"
    sc_path = ROOT / "data" / "processed" / scaler_name
    if sc_path.exists():
        sc   = torch.load(sc_path, weights_only=True, map_location="cpu")
        face = (face - sc["face_mean"].numpy()) / (sc["face_std"].numpy() + 1e-8)

    spk /= np.linalg.norm(spk, axis=1, keepdims=True) + 1e-8
    return torch.tensor(face), torch.tensor(spk), df


def compute_embedding_metrics(model, face_t, spk_t, device, batch=512):
    preds = []
    for i in range(0, len(face_t), batch):
        with torch.no_grad():
            p = forward_voice(model, face_t[i:i+batch].to(device)).cpu()
            preds.append(p)
    pred_all = torch.cat(preds)
    val_cos  = F.cosine_similarity(pred_all, spk_t, dim=-1).mean().item()

    n   = min(2000, len(pred_all))
    idx = torch.randperm(len(pred_all))[:n]
    pq  = F.normalize(pred_all[idx], dim=-1)
    gt  = F.normalize(spk_t[idx], dim=-1)
    sim = pq @ gt.T
    recalls = {}
    for k in [1, 3, 10]:
        topk = torch.topk(sim, k=min(k, n), dim=1).indices
        recalls[f"recall_at_{k}"] = (topk == torch.arange(n).unsqueeze(1)).any(1).float().mean().item()
    return val_cos, recalls


def _compute_gender_centroids():
    csv_path = ROOT / "data" / "processed" / "clean_index.csv"
    h5_path  = ROOT / "data" / "processed" / "features.h5"
    df_train = pd.read_csv(csv_path)
    df_train = df_train[df_train["split"] == "train"]
    with h5py.File(h5_path, "r") as f:
        all_spk = f["speaker_features"][:]
    m_idx = df_train[df_train["gender"].str.lower() == "m"]["hdf5_index"].values
    f_idx = df_train[df_train["gender"].str.lower() == "f"]["hdf5_index"].values
    centroids = {
        "m": torch.tensor(all_spk[m_idx].astype(np.float32)).mean(0),
        "f": torch.tensor(all_spk[f_idx].astype(np.float32)).mean(0),
    }
    save = ROOT / "data" / "processed" / "gender_centroids.pt"
    torch.save(centroids, save)
    print(f"  [gender] Centroids saved (m={len(m_idx)}, f={len(f_idx)})")


def compute_gender_accuracy(model, face_t, df, device, batch=512):
    if "gender" not in df.columns: return -1.0
    cp = ROOT / "data" / "processed" / "gender_centroids.pt"
    if not cp.exists(): _compute_gender_centroids()
    c   = torch.load(cp, weights_only=True)
    m_c = F.normalize(c["m"], dim=0).to(device)
    f_c = F.normalize(c["f"], dim=0).to(device)
    preds = []
    gender_preds = []
    for i in range(0, len(face_t), batch):
        with torch.no_grad():
            res = model(face_t[i:i+batch].to(device))
            if isinstance(res, tuple):
                preds.append(res[0].cpu())
                if res[1] is not None:
                    gender_preds.append(res[1].cpu())
            else:
                preds.append(res.cpu())
    gt_m = torch.tensor(df["gender"].str.lower().str.strip() == "m").bool()
    if gender_preds and len(gender_preds) == len(preds):
        g_logits = torch.cat(gender_preds).squeeze(-1)
        pred_m = g_logits > 0
        return (pred_m == gt_m).float().mean().item()
    pred_all = F.normalize(torch.cat(preds), dim=-1).to(device)
    sim_m    = (pred_all @ m_c).cpu()
    sim_f    = (pred_all @ f_c).cpu()
    pred_m   = sim_m > sim_f
    return (pred_m == gt_m).float().mean().item()


def compute_audio_metrics(model, face_t, spk_t, df, device, synth_text, n_synth, name):
    try:
        from voix.inference.pipeline import VoixPipeline
        from voix.data.speaker_extractor import ECAPAExtractor
        import torchaudio
    except ImportError as e:
        print(f"  [audio] Skipping audio metrics: {e}")
        return {"secs": -1.0, "secs_std": -1.0, "n_synthesized": 0, "wer": -1.0, "fad": -1.0}

    audio_dir = ROOT / "results" / "audio_samples" / name
    audio_dir.mkdir(parents=True, exist_ok=True)
    print("  Loading VoixPipeline & ECAPAExtractor...")
    pipeline = VoixPipeline(device=device, enable_watermarking=False)
    ecapa = ECAPAExtractor(device=device)
    resampler = torchaudio.transforms.Resample(24000, 16000).to(device)

    speakers   = df["speaker_id"].unique()
    np.random.seed(42)
    eval_spks  = np.random.choice(speakers, min(n_synth, len(speakers)), replace=False)
    secs_list  = []

    for idx_num, spk_id in enumerate(eval_spks):
        rows    = df[df["speaker_id"] == spk_id]
        row_idx = rows.index[0]
        real_spk = spk_t[row_idx]
        face_emb = face_t[row_idx].unsqueeze(0).to(device)
        with torch.no_grad():
            pred_voice = forward_voice(model, face_emb)
            pred_ecapa = F.normalize(pred_voice, dim=-1)
        try:
            style = pipeline.adapter(pred_ecapa)
            gender_val = 0 if str(df.loc[row_idx, "gender"]).strip().lower() == "m" else 1
            wav = pipeline.tts.synthesize(text=synth_text, style=style, gender=gender_val)
            wav_16k = resampler(wav.to(device))
            gen_emb = ecapa.extract(wav_16k).squeeze(0).cpu()
            secs = F.cosine_similarity(F.normalize(gen_emb, dim=0).unsqueeze(0),
                                       F.normalize(real_spk, dim=0).unsqueeze(0)).item()
            secs_list.append(secs)
            out_file = audio_dir / f"{spk_id.replace('/', '_')}.wav"
            torchaudio.save(str(out_file), wav.cpu(), 24000)
            print(f"  [{idx_num+1}/{len(eval_spks)}] {spk_id} SECS={secs:.4f}")
        except Exception as ex:
            print(f"  [audio] Failed for {spk_id}: {ex}")

    return {
        "secs": float(np.mean(secs_list)) if secs_list else -1.0,
        "secs_std": float(np.std(secs_list)) if secs_list else -1.0,
        "n_synthesized": len(secs_list),
        "wer": -1.0, "fad": -1.0,
    }


def main():
    args   = parse_args()
    device = args.device
    print(f"\n{'='*65}")
    print(f"  VOIX BENCHMARK: {args.name}")
    print(f"  Checkpoint:  {args.checkpoint}")
    print(f"  Face type:   {args.face_type}  |  Device: {device}")
    print(f"{'='*65}\n")
    t0 = time.time()

    print("[1/4] Loading validation data...")
    face_t, spk_t, df_val = load_data(args.face_type, "val")
    face_dim = face_t.shape[1]
    print(f"  {len(face_t)} samples | face_dim={face_dim}")

    print("[2/4] Loading model...")
    model   = load_model(args.checkpoint, args.model_type, face_dim, device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  params={n_params:,}")

    print("[3/4] Computing embedding metrics...")
    val_cos, recalls = compute_embedding_metrics(model, face_t, spk_t, device)
    print(f"  val_cos={val_cos:.4f}  R@1={recalls['recall_at_1']:.4f}  R@3={recalls['recall_at_3']:.4f}  R@10={recalls['recall_at_10']:.4f}")

    print("[3b] Gender accuracy...")
    g_acc = compute_gender_accuracy(model, face_t, df_val, device)
    print(f"  gender_acc={g_acc:.4f}" if g_acc >= 0 else "  gender_acc=N/A")

    audio = {"secs": -1.0, "secs_std": -1.0, "n_synthesized": 0, "wer": -1.0, "fad": -1.0}
    if not args.skip_audio:
        print(f"[4/4] Audio metrics ({args.n_synth} speakers)...")
        audio = compute_audio_metrics(model, face_t, spk_t, df_val,
                                      device, args.synth_text, args.n_synth, args.name)
    else:
        print("[4/4] Audio metrics skipped.")

    report = {
        "name": args.name, "checkpoint": str(args.checkpoint),
        "model_type": args.model_type, "face_type": args.face_type,
        "n_params": n_params, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_s": round(time.time() - t0, 1),
        "metrics": {
            "val_cos":         round(val_cos, 5),
            "recall_at_1":     round(recalls["recall_at_1"], 5),
            "recall_at_3":     round(recalls["recall_at_3"], 5),
            "recall_at_10":    round(recalls["recall_at_10"], 5),
            "gender_accuracy": round(g_acc, 5) if g_acc >= 0 else -1,
            "secs":            round(audio["secs"], 5),
            "secs_std":        round(audio["secs_std"], 5),
            "n_synthesized":   audio["n_synthesized"],
            "wer":             audio["wer"],
            "fad":             audio["fad"],
            "mos_n":           -1.0,
        }
    }

    m = report["metrics"]
    print(f"\n{'='*65}")
    print(f"  RESULTS: {args.name}")
    print(f"  val_cos={m['val_cos']:.4f}  R@1={m['recall_at_1']:.4f}  R@3={m['recall_at_3']:.4f}  R@10={m['recall_at_10']:.4f}")
    print(f"  GenderAcc={m['gender_accuracy']:.4f}  SECS={m['secs']:.4f}  WER={m['wer']}  FAD={m['fad']}")
    print(f"{'='*65}\n")

    out_dir = ROOT / "results" / "benchmarks"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"{args.name}.json", "w") as f:
        json.dump(report, f, indent=2)

    summary_path = out_dir / "summary.csv"
    row = {"name": args.name, **m}
    if summary_path.exists():
        sdf = pd.read_csv(summary_path)
        sdf = sdf[sdf["name"] != args.name]
        sdf = pd.concat([sdf, pd.DataFrame([row])], ignore_index=True)
    else:
        sdf = pd.DataFrame([row])
    sdf.to_csv(summary_path, index=False)
    print(f"Saved: results/benchmarks/{args.name}.json")
    print(f"       results/benchmarks/summary.csv")


if __name__ == "__main__":
    main()
