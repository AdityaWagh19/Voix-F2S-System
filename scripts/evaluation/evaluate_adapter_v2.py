"""Evaluate the V2 StyleAdapter on speaker-disjoint validation and test splits.

Reports both the deterministic prior-mean serving input and the sampled CVAE
serving distribution, plus raw and centered pairwise cosine.  The centered
statistics matter because StyleTTS2 style vectors have a large common component;
the adapter must preserve that mean while matching the orthogonal residual.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from voix.data.feature_contract import load_scaler_payload
from voix.models.adapter import StyleAdapter
from voix.models.contrastive_net import ContrastiveFaceVoiceNet
from voix.training.data import load_v2_dataset
from voix.training.trainer_adapter_cfvn import (
    cvae_prior_mean_inputs,
    cvae_sample_inputs,
    load_cvae,
)
from voix.training.v2_utils import (
    align_style_targets,
    embed_faces,
    load_style_targets,
    normalize_face,
    pairwise_identity_stats,
    speaker_codes,
)


def _load_adapter(path: Path, device: str) -> Tuple[StyleAdapter, dict]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    hparams = checkpoint.get("hparams", {})
    model = StyleAdapter(
        in_dim=int(hparams.get("in_dim", 192)),
        hidden_dim=int(hparams.get("hidden_dim", 256)),
        out_dim=int(hparams.get("out_dim", 256)),
        dropout=float(hparams.get("dropout", 0.0)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model, checkpoint


def _stats(
    predictions: torch.Tensor,
    target: torch.Tensor,
    codes: torch.Tensor,
    center: torch.Tensor,
) -> Dict[str, float]:
    raw = pairwise_identity_stats(predictions, codes)
    centered = pairwise_identity_stats(predictions - center, codes)
    output = predictions.float()
    target = target.float()
    centered_output = output - center
    return {
        "n": int(len(output)),
        "raw_within_cosine": raw["within_cosine"],
        "raw_across_cosine": raw["across_cosine"],
        "raw_identity_margin": raw["identity_margin"],
        "centered_within_cosine": centered["within_cosine"],
        "centered_across_cosine": centered["across_cosine"],
        "centered_identity_margin": centered["identity_margin"],
        "per_dim_std": float(F.normalize(output, dim=-1).std(dim=0).mean()),
        "norm_mean": float(output.norm(dim=-1).mean()),
        "centered_norm_mean": float(centered_output.norm(dim=-1).mean()),
        "mse": float(F.mse_loss(output, target)),
        "target_cosine": float(
            F.cosine_similarity(output, target, dim=-1).mean()
        ),
        "target_cosine_timbre": float(
            F.cosine_similarity(output[:, :128], target[:, :128], dim=-1).mean()
        ),
        "target_cosine_prosody": float(
            F.cosine_similarity(output[:, 128:], target[:, 128:], dim=-1).mean()
        ),
        "target_mean_cosine": float(
            F.cosine_similarity(
                center.expand_as(target), target, dim=-1
            ).mean()
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--features",
        type=Path,
        default=ROOT / "data" / "processed" / "voxceleb2_v2" / "features_v2.h5",
    )
    parser.add_argument(
        "--index",
        type=Path,
        default=ROOT / "data" / "processed" / "voxceleb2_v2" / "index.csv",
    )
    parser.add_argument(
        "--scaler",
        type=Path,
        default=ROOT / "data" / "processed" / "voxceleb2_v2" / "face_scaler_v2.pt",
    )
    parser.add_argument(
        "--cfvn",
        type=Path,
        default=ROOT / "results" / "checkpoints" / "cfvn_v2_best.pt",
    )
    parser.add_argument(
        "--cvae",
        type=Path,
        default=ROOT / "checkpoints" / "cvae_cfvn_face192_latent128_diversity.pt",
    )
    parser.add_argument(
        "--adapter",
        type=Path,
        default=ROOT / "checkpoints" / "style_adapter_cfvn_best.pt",
    )
    parser.add_argument(
        "--style-targets",
        type=Path,
        default=ROOT / "data" / "processed" / "style_targets_voxceleb2_5clip.pt",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "adapter_v2_evaluation.json",
    )
    parser.add_argument("--samples-per-face", type=int, default=4)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu"

    dataset = load_v2_dataset(args.features, args.index, face_key="face_features")
    style_payload = load_style_targets(args.style_targets)
    targets, style_stats = align_style_targets(dataset.speaker_ids, style_payload)
    scaler = load_scaler_payload(
        args.scaler,
        expected_dim=dataset.face_raw.shape[1],
        face_key="face_features",
    )
    face = normalize_face(dataset.face_raw, scaler)
    cfvn = ContrastiveFaceVoiceNet.from_checkpoint(
        str(args.cfvn), device=device
    ).to(device)
    cfvn.eval()
    cfvn_face = embed_faces(cfvn, face, device)
    cvae, _ = load_cvae(args.cvae, cfvn.out_dim, device)
    adapter, checkpoint = _load_adapter(args.adapter, device)

    train_mask = dataset.split == "train"
    center = targets[train_mask].mean(dim=0).cpu()
    report: Dict = {
        "artifacts": {
            "features": str(args.features),
            "index": str(args.index),
            "scaler": str(args.scaler),
            "cfvn": str(args.cfvn),
            "cvae": str(args.cvae),
            "adapter": str(args.adapter),
            "style_targets": str(args.style_targets),
        },
        "adapter_epoch": int(checkpoint.get("epoch", -1)),
        "adapter_hparams": checkpoint.get("hparams", {}),
        "adapter_training": {
            key: checkpoint.get(key)
            for key in (
                "adapter_input",
                "val_input",
                "samples_per_clip",
                "lambda_div",
                "lambda_norm",
                "lambda_var",
                "val_loss",
                "val_output_across_cosine",
                "val_output_centered_across_cosine",
                "target_center_norm",
                "target_centered_norm",
            )
        },
        "style_targets": style_stats,
        "target_center_norm": float(center.norm()),
        "splits": {},
    }

    for split in ("val", "test"):
        mask = dataset.split == split
        if not mask.any():
            continue
        codes_np, _ = speaker_codes(dataset.speaker_ids[mask])
        codes = torch.from_numpy(codes_np)
        target = targets[mask].cpu()
        split_cfvn = cfvn_face[mask]
        prior = cvae_prior_mean_inputs(cvae, split_cfvn, device=device)
        samples = cvae_sample_inputs(
            cvae,
            split_cfvn,
            num_samples=args.samples_per_face,
            temperature=1.0,
            device=device,
        )
        with torch.no_grad():
            prior_out = adapter(prior.to(device)).cpu()
            sample_out = adapter(
                samples.reshape(-1, samples.shape[-1]).to(device)
            ).cpu()
        sample_codes = codes.repeat_interleave(args.samples_per_face)
        sample_target = target.repeat_interleave(args.samples_per_face, dim=0)
        report["splits"][split] = {
            "clips": int(mask.sum()),
            "speakers": int(len(set(dataset.speaker_ids[mask].tolist()))),
            "prior_mean": _stats(prior_out, target, codes, center),
            "cvae_samples": _stats(
                sample_out, sample_target, sample_codes, center
            ),
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
