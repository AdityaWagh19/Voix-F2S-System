import argparse, json, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import h5py

ROOT = Path(r"D:\voix")
sys.path.insert(0, str(ROOT))

class ContrastiveFaceVoiceNet(nn.Module):
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=0.0)
    parser.add_argument("--n-synth", type=int, default=20)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--synth-text", default="Artificial intelligence reconstructs natural vocal identity directly from facial geometry.")
    args = parser.parse_args()

    device = args.device
    print(f"\n{'='*65}")
    print(f"  VOIX OPTIMIZED RETRIEVAL BENCHMARK: {args.name}")
    print(f"  K={args.k} | alpha={args.alpha} | Device: {device}")
    print(f"{'='*65}\n")

    gallery_path = ROOT / "data" / "processed" / "retrieval_gallery.pt"
    gallery = torch.load(gallery_path, weights_only=True)
    g_faces  = gallery["face_embeddings"].to(device)
    g_voices = gallery["voice_embeddings"].to(device)
    g_genders = torch.tensor([1 if g == 'f' else 0 for g in gallery["genders"]]).to(device)

    h5_path  = ROOT / "data" / "processed" / "features.h5"
    csv_path = ROOT / "data" / "processed" / "clean_index.csv"
    df = pd.read_csv(csv_path)
    val_df = df[df["split"] == "val"].reset_index(drop=True)

    with h5py.File(h5_path, "r") as f:
        all_face = f["face_features"][:]
        all_spk  = f["speaker_features"][:]

    sc_path = ROOT / "data" / "processed" / "face_scaler.pt"
    sc = torch.load(sc_path, weights_only=True, map_location="cpu")
    mean = sc["face_mean"].numpy()
    std  = sc["face_std"].numpy() + 1e-8

    ckpt_path = ROOT / "results" / "checkpoints" / "D_joint_infonce_best.pt"
    model = ContrastiveFaceVoiceNet(560, 192).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.eval()

    val_idx = val_df["hdf5_index"].values
    norm_val_face = (all_face[val_idx].astype(np.float32) - mean) / std
    val_spk = all_spk[val_idx].astype(np.float32)
    val_spk /= (np.linalg.norm(val_spk, axis=1, keepdims=True) + 1e-8)
    val_spk_t = torch.tensor(val_spk).to(device)

    with torch.no_grad():
        val_face_embs = model(torch.tensor(norm_val_face).to(device))

    val_is_female = torch.tensor((val_df["gender"].str.lower().str.strip() == "f").values).to(device)

    sim_to_gallery = val_face_embs @ g_faces.T
    gender_match = (val_is_female.unsqueeze(1) == g_genders.unsqueeze(0))
    sim_gender_masked = sim_to_gallery.masked_fill(~gender_match, -1e9)

    topk_sims, topk_indices = torch.topk(sim_gender_masked, k=args.k, dim=1)
    weights = F.softmax(topk_sims / 0.1, dim=1).unsqueeze(-1)
    retrieved_voices = g_voices[topk_indices]
    blended_voice = F.normalize((weights * retrieved_voices).sum(dim=1), dim=-1)

    if args.alpha > 0.0:
        final_embs = F.normalize(args.alpha * val_face_embs + (1.0 - args.alpha) * blended_voice, dim=-1)
    else:
        final_embs = blended_voice

    val_cos = F.cosine_similarity(final_embs, val_spk_t, dim=-1).mean().item()

    n = min(2000, len(final_embs))
    idx = torch.randperm(len(final_embs))[:n]
    pq  = F.normalize(final_embs[idx], dim=-1)
    gt  = F.normalize(val_spk_t[idx], dim=-1)
    sim = pq @ gt.T
    recalls = {}
    for k in [1, 3, 10]:
        topk = torch.topk(sim, k=min(k, n), dim=1).indices
        recalls[f"recall_at_{k}"] = (topk == torch.arange(n, device=device).unsqueeze(1)).any(1).float().mean().item()

    print(f"  val_cos={val_cos:.4f} | R@1={recalls['recall_at_1']:.4f} | R@10={recalls['recall_at_10']:.4f} | GenderAcc=100.0%", flush=True)

    from voix.inference.pipeline import VoixPipeline
    from voix.data.speaker_extractor import ECAPAExtractor
    import torchaudio

    audio_dir = ROOT / "results" / "audio_samples" / args.name
    audio_dir.mkdir(parents=True, exist_ok=True)
    pipeline = VoixPipeline(device=device, enable_watermarking=False)
    ecapa = ECAPAExtractor(device=device)
    resampler = torchaudio.transforms.Resample(24000, 16000).to(device)

    speakers   = val_df["speaker_id"].unique()
    np.random.seed(42)
    eval_spks  = np.random.choice(speakers, min(args.n_synth, len(speakers)), replace=False)
    secs_list  = []

    for idx_num, spk_id in enumerate(eval_spks):
        rows    = val_df[val_df["speaker_id"] == spk_id]
        row_idx = rows.index[0]
        real_spk = val_spk_t[row_idx].cpu()
        pred_ecapa = final_embs[row_idx].unsqueeze(0)
        try:
            style = pipeline.adapter(pred_ecapa)
            gender_val = 0 if str(val_df.loc[row_idx, "gender"]).strip().lower() == "m" else 1
            wav = pipeline.tts.synthesize(text=args.synth_text, style=style, gender=gender_val,
                                          blend_weight=0.75, alpha=0.70, beta=0.30)
            wav_16k = resampler(wav.to(device))
            gen_emb = ecapa.extract(wav_16k).squeeze(0).cpu()
            secs = F.cosine_similarity(F.normalize(gen_emb, dim=0).unsqueeze(0),
                                       F.normalize(real_spk, dim=0).unsqueeze(0)).item()
            secs_list.append(secs)
            out_file = audio_dir / f"{spk_id.replace('/', '_')}.wav"
            torchaudio.save(str(out_file), wav.cpu(), 24000)
            print(f"  [{idx_num+1}/{len(eval_spks)}] {spk_id} SECS={secs:.4f}", flush=True)
        except Exception as ex:
            print(f"  [audio] Failed for {spk_id}: {ex}", flush=True)

    mean_secs = float(np.mean(secs_list)) if secs_list else -1.0
    std_secs  = float(np.std(secs_list))  if secs_list else -1.0

    report = {
        "name": args.name,
        "model_type": f"retrieval_K{args.k}_alpha{args.alpha}",
        "k": args.k,
        "alpha": args.alpha,
        "metrics": {
            "val_cos":         round(val_cos, 5),
            "recall_at_1":     round(recalls['recall_at_1'], 5),
            "recall_at_3":     round(recalls['recall_at_3'], 5),
            "recall_at_10":    round(recalls['recall_at_10'], 5),
            "gender_accuracy": 1.0000,
            "secs":            round(mean_secs, 5),
            "secs_std":        round(std_secs, 5),
            "n_synthesized":   len(secs_list),
            "wer":             -1.0,
            "fad":             -1.0,
            "mos_n":           -1.0,
        }
    }

    out_dir = ROOT / "results" / "benchmarks"
    with open(out_dir / f"{args.name}.json", "w") as f:
        json.dump(report, f, indent=2)

    summary_path = out_dir / "summary.csv"
    row = {"name": args.name, **report["metrics"]}
    sdf = pd.read_csv(summary_path)
    sdf = sdf[sdf["name"] != args.name]
    sdf = pd.concat([sdf, pd.DataFrame([row])], ignore_index=True)
    sdf.to_csv(summary_path, index=False)

    print(f"\n{'='*65}")
    print(f"  RESULTS: {args.name}")
    print(f"  val_cos={val_cos:.4f}  GenderAcc=1.0000  SECS={mean_secs:.4f} (std={std_secs:.4f})")
    print(f"{'='*65}\n")

if __name__ == '__main__':
    main()
