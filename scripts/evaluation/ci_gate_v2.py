"""CI gate for the retrained FaceFeatureV2 stack.

Loads the held-out speaker split, computes cross-modal retrieval with negative
controls, embedding health, CVAE decoder diversity, and adapter output
diversity, then applies ``voix.evaluation.gate`` thresholds.

Exit code is non-zero when any check fails or a required artifact is missing.

Example:
    python scripts/evaluation/ci_gate_v2.py \
        --features data/processed/voxceleb2_v2/features_v2.h5 \
        --index data/processed/voxceleb2_v2/index.csv \
        --output results/ci_gate_v2.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import h5py
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from voix.data.feature_contract import load_scaler_payload
from voix.evaluation.gate import GateThresholds, evaluate_gate
from voix.evaluation.retrieval import (
    constant_control,
    embedding_health,
    gender_controlled_retrieval,
    shuffled_control,
    speaker_retrieval,
)
from voix.models.adapter import StyleAdapter
from voix.models.contrastive_net import ContrastiveFaceVoiceNet
from voix.models.cvae import CVAE
from voix.training.v2_utils import (
    align_style_targets,
    embed_faces,
    load_style_targets,
    normalize_face,
    query_gallery_indices,
    speaker_codes,
)

DEFAULTS = {
    "features": ROOT / "data" / "processed" / "voxceleb2_v2" / "features_v2.h5",
    "index": ROOT / "data" / "processed" / "voxceleb2_v2" / "index.csv",
    "scaler": ROOT / "data" / "processed" / "voxceleb2_v2" / "face_scaler_v2.pt",
    "cfvn": ROOT / "results" / "checkpoints" / "cfvn_v2_best.pt",
    "cvae": ROOT / "checkpoints" / "cvae_cfvn_face192_latent128_diversity.pt",
    "adapter": ROOT / "checkpoints" / "style_adapter_cfvn_best.pt",
    "style_targets": ROOT
    / "data"
    / "processed"
    / "style_targets_voxceleb2_5clip.pt",
}


def _decode(values) -> np.ndarray:
    return np.asarray(
        [
            value.decode("utf-8")
            if isinstance(value, (bytes, np.bytes_))
            else str(value)
            for value in values
        ],
        dtype=object,
    )


def _l2(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-8)


def _pairwise_cosine(
    embeddings: torch.Tensor, codes: torch.Tensor
) -> Tuple[float, float]:
    """Mean within- and across-speaker cosine for one embedding per row."""
    x = F.normalize(embeddings.float().cpu(), dim=-1)
    codes = codes.view(-1).cpu()
    similarity = x @ x.T
    n = len(x)
    off_diagonal = ~torch.eye(n, dtype=torch.bool)
    same = codes[:, None] == codes[None, :]
    within_mask = same & off_diagonal
    across_mask = ~same
    within = (
        float(similarity[within_mask].mean())
        if within_mask.any()
        else float("nan")
    )
    across = (
        float(similarity[across_mask].mean())
        if across_mask.any()
        else float("nan")
    )
    return within, across


def _prior_active_units(prior_mu: torch.Tensor, prior_logvar: torch.Tensor) -> int:
    """Latent dimensions where the conditional prior departs from N(0, 1)."""
    kl = 0.5 * (
        prior_logvar.exp() + prior_mu.pow(2) - 1.0 - prior_logvar
    )
    return int((kl.mean(dim=0) > 0.1).sum().item())


def _posterior_active_units(
    cvae: CVAE,
    face: torch.Tensor,
    voice: torch.Tensor,
) -> int:
    """Latent dimensions with mean posterior-to-prior KL > 0.1 nats.

    This is the same definition the CVAE trainer uses for validation.  The
    prior-vs-standard-normal count is not a posterior-collapse diagnostic: a
    collapsed posterior can still have a non-trivial conditional prior.
    """
    prior_mu, prior_logvar = cvae.prior_net(face)
    mu, logvar = cvae.recognition_net(torch.cat([voice, face], dim=-1))
    kl = 0.5 * (
        prior_logvar
        - logvar
        - 1.0
        + torch.exp(logvar - prior_logvar)
        + (mu - prior_mu).pow(2) * torch.exp(-prior_logvar)
    )
    return int((kl.mean(dim=0) > 0.1).sum().item())


def _load_cvae(path: Path, device: str) -> CVAE:
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=True)
    except Exception:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    state = checkpoint.get("model_state_dict", checkpoint)
    model = CVAE(
        face_dim=int(checkpoint.get("face_dim", 192)),
        speaker_dim=int(checkpoint.get("speaker_dim", 192)),
        latent_dim=int(checkpoint.get("latent_dim", 128)),
    ).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def _load_adapter(path: Path, device: str) -> StyleAdapter:
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=True)
    except Exception:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    state = checkpoint.get("model_state_dict", checkpoint)
    hparams = checkpoint.get("hparams", {})
    model = StyleAdapter(
        in_dim=int(hparams.get("in_dim", 192)),
        hidden_dim=int(hparams.get("hidden_dim", 256)),
        out_dim=int(hparams.get("out_dim", 256)),
        dropout=float(hparams.get("dropout", 0.1)),
    ).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


@torch.no_grad()
def _cvae_metrics(
    cvae: CVAE,
    face: torch.Tensor,
    voice: torch.Tensor,
    codes: torch.Tensor,
    *,
    device: str,
    samples_per_face: int = 4,
) -> Dict[str, float]:
    face = face.to(device)
    voice = voice.to(device)
    samples = cvae.sample_voices(
        face, num_samples=samples_per_face, temperature=1.0
    )
    samples = samples.view(face.size(0), samples_per_face, -1)
    flat = samples.reshape(-1, samples.shape[-1]).cpu()
    flat_codes = codes.repeat_interleave(samples_per_face)
    within, across = _pairwise_cosine(flat, flat_codes)

    prior_mu, prior_logvar = cvae.prior_net(face)
    prior_within, prior_across = _pairwise_cosine(
        prior_mu.cpu(), codes
    )
    return {
        "decoder_within_face_cosine": within,
        "decoder_across_face_cosine": across,
        "decoder_pairwise_cosine": across,
        "face_margin": within - across,
        "prior_pairwise_cosine": prior_across,
        "active_units": _posterior_active_units(cvae, face, voice),
        "prior_active_units": _prior_active_units(prior_mu, prior_logvar),
    }


@torch.no_grad()
def _adapter_metrics(
    adapter: StyleAdapter,
    cvae: CVAE,
    face: torch.Tensor,
    codes: torch.Tensor,
    target: torch.Tensor,
    center: torch.Tensor,
    *,
    device: str,
    samples_per_face: int = 4,
) -> Dict[str, float]:
    face = face.to(device)
    samples = cvae.sample_voices(
        face, num_samples=samples_per_face, temperature=1.0
    )
    samples = F.normalize(
        samples.view(face.size(0), samples_per_face, -1), dim=-1
    )
    style = adapter(samples.reshape(-1, samples.shape[-1]))
    flat_codes = codes.repeat_interleave(samples_per_face)
    within, across = _pairwise_cosine(style, flat_codes)
    per_dim_std = float(F.normalize(style, dim=-1).std(dim=0).mean().item())

    # Fidelity to the target style distribution.  Diversity-only selection can
    # shrink the common style mean while making residuals orthogonal; the raw
    # cosine check catches that, but an explicit target-cosine check makes the
    # failure mode impossible to miss.
    target = target.to(device).repeat_interleave(samples_per_face, dim=0)
    center = center.to(device)
    target_cosine = F.cosine_similarity(style, target, dim=-1).mean()
    target_cosine_timbre = F.cosine_similarity(
        style[:, :128], target[:, :128], dim=-1
    ).mean()
    target_cosine_prosody = F.cosine_similarity(
        style[:, 128:], target[:, 128:], dim=-1
    ).mean()
    return {
        "output_within_face_cosine": within,
        "output_across_face_cosine": across,
        "output_pairwise_cosine": across,
        "output_per_dim_std": per_dim_std,
        "output_target_cosine": float(target_cosine.item()),
        "output_target_cosine_timbre": float(target_cosine_timbre.item()),
        "output_target_cosine_prosody": float(target_cosine_prosody.item()),
        "output_norm_mean": float(style.norm(dim=-1).mean().item()),
        "output_centered_norm_mean": float(
            (style - center).norm(dim=-1).mean().item()
        ),
        "target_norm_mean": float(target.norm(dim=-1).mean().item()),
        "target_centered_norm_mean": float(
            (target - center).norm(dim=-1).mean().item()
        ),
        "target_mean_cosine": float(
            F.cosine_similarity(
                center.expand_as(target), target, dim=-1
            ).mean().item()
        ),
    }


def build_report(args: argparse.Namespace) -> Dict:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    artifacts = {
        "features": Path(args.features),
        "index": Path(args.index),
        "scaler": Path(args.scaler),
        "cfvn": Path(args.cfvn),
        "cvae": Path(args.cvae),
        "adapter": Path(args.adapter),
        "style_targets": Path(args.style_targets),
    }
    missing = [name for name, path in artifacts.items() if not path.exists()]
    report: Dict = {
        "artifacts": {
            **{name: str(path) for name, path in artifacts.items()},
            "missing": missing,
        },
        "split": args.split,
        "device": device,
        "retrieval": {},
        "health": {},
        "cvae": {},
        "adapter": {},
    }
    if missing:
        return report

    with h5py.File(artifacts["features"], "r") as store:
        clip_ids = _decode(store["clip_ids"][:])
        speakers = _decode(store["speaker_ids"][:])
        splits = _decode(store["split"][:])
        genders = _decode(store["gender"][:])
        arcface = np.asarray(store["arcface"][:], dtype=np.float32)
        fused = np.asarray(store["face_features"][:], dtype=np.float32)
        voice = np.asarray(store["speaker_features"][:], dtype=np.float32)

    style_payload = load_style_targets(artifacts["style_targets"])
    targets_all, style_stats = align_style_targets(speakers, style_payload)
    report["style_targets"] = style_stats

    split = args.split
    if split == "test" and not (splits == "test").any():
        split = "val"
    mask = splits == split
    if not mask.any():
        raise ValueError(f"No clips found for split {args.split!r}")
    report["split"] = split
    report["clips"] = int(mask.sum())
    report["speakers"] = int(len(set(speakers[mask].tolist())))

    if args.max_faces and mask.sum() > args.max_faces:
        # Deterministic, speaker-stratified subsample.
        selected = []
        rng = np.random.default_rng(42)
        for speaker in sorted(set(speakers[mask].tolist())):
            indices = np.flatnonzero(mask & (speakers == speaker))
            rng.shuffle(indices)
            selected.extend(indices[: max(1, args.max_faces // report["speakers"])])
        mask = np.zeros_like(mask)
        mask[np.asarray(selected[: args.max_faces], dtype=np.int64)] = True
        report["clips"] = int(mask.sum())

    eval_speakers = speakers[mask]
    query_idx, gallery_idx = query_gallery_indices(eval_speakers)
    if len(query_idx) < 2:
        raise ValueError("CI gate needs at least two speakers with 2+ clips")
    query_speakers = eval_speakers[query_idx]
    gallery_speakers = eval_speakers[gallery_idx]
    report["query_speakers"] = int(len(set(query_speakers.tolist())))

    # The serving scaler is an artifact, not something the gate may silently
    # re-derive.  Rebuilding it here would hide a stale or mismatched scaler.
    train_mask = splits == "train"
    if not train_mask.any():
        raise ValueError("CI gate requires a train split in the corpus")

    cfvn = ContrastiveFaceVoiceNet.from_checkpoint(
        str(artifacts["cfvn"]), device=device
    ).to(device)
    face_key = "arcface" if cfvn.face_dim == 512 else "face_features"
    face_raw = arcface if face_key == "arcface" else fused

    if face_key == "face_features":
        scaler_path = Path(args.scaler)
    else:
        scaler_path = Path(args.scaler).with_name("face_scaler_arcface_v2.pt")
    if not scaler_path.exists():
        raise FileNotFoundError(
            f"Missing {face_key} scaler: {scaler_path}. Train the matching CFVN "
            "with trainer_cfvn before running the gate."
        )
    scaler = load_scaler_payload(
        scaler_path, expected_dim=face_raw.shape[1], face_key=face_key
    )
    expected_train = int(train_mask.sum())
    expected_train_speakers = int(len(set(speakers[train_mask].tolist())))
    if int(scaler.get("n_train", -1)) != expected_train:
        raise ValueError(
            f"Scaler {scaler_path} was fit on {scaler.get('n_train')} train rows "
            f"but the corpus has {expected_train}; rebuild it with "
            "scripts/data/build_face_scaler_v2.py"
        )
    if int(scaler.get("n_train_speakers", -1)) != expected_train_speakers:
        raise ValueError(
            f"Scaler {scaler_path} was fit on {scaler.get('n_train_speakers')} "
            f"train speakers but the corpus has {expected_train_speakers}; "
            "rebuild it with scripts/data/build_face_scaler_v2.py"
        )
    report["scaler"] = str(scaler_path)
    report["face_key"] = face_key
    report["face_dim"] = int(face_raw.shape[1])
    face_norm = normalize_face(face_raw[mask], scaler)
    cfvn_embedding = embed_faces(cfvn, face_norm, device)

    # The fused diagnostic always uses the canonical 560-D scaler, even when
    # the CFVN itself is the raw-ArcFace baseline.
    fused_scaler = load_scaler_payload(
        Path(args.scaler), expected_dim=fused.shape[1], face_key="face_features"
    )
    if int(fused_scaler.get("n_train", -1)) != expected_train:
        raise ValueError(
            f"Canonical fused scaler {args.scaler} is stale: n_train="
            f"{fused_scaler.get('n_train')}, corpus has {expected_train}"
        )

    def retrieval_block(
        query_features: torch.Tensor,
        gallery_features: torch.Tensor,
        *,
        controls: bool,
    ) -> Dict[str, float]:
        result = speaker_retrieval(
            query_features, query_speakers, gallery_features, gallery_speakers
        )
        block = {key: float(value) for key, value in result.items()}
        if controls:
            shuffled = shuffled_control(
                query_features,
                query_speakers,
                gallery_features,
                gallery_speakers,
            )
            constant = constant_control(
                gallery_features, gallery_speakers
            )
            block["shuffled_recall_at_1"] = float(shuffled["recall_at_1"])
            block["constant_recall_at_1"] = float(constant["recall_at_1"])
        return block

    # The primary contract is cross-modal face-to-voice retrieval.  An earlier
    # version compared face embeddings to face embeddings here, so the gate
    # reported identity-preservation R@1 (0.85) while never checking the actual
    # face-to-voice task (0.09).  Keep the face-to-face block as a diagnostic.
    voice_t = torch.from_numpy(voice[mask])
    report["retrieval"]["cfvn"] = retrieval_block(
        cfvn_embedding[query_idx],
        voice_t[gallery_idx],
        controls=True,
    )
    report["retrieval"]["cfvn_face_face"] = retrieval_block(
        cfvn_embedding[query_idx],
        cfvn_embedding[gallery_idx],
        controls=False,
    )
    report["retrieval"]["cfvn_gender_controlled"] = {
        key: float(value)
        for key, value in gender_controlled_retrieval(
            cfvn_embedding[query_idx],
            query_speakers,
            genders[mask][query_idx],
            voice_t[gallery_idx],
            gallery_speakers,
            genders[mask][gallery_idx],
        ).items()
    }
    report["retrieval"]["raw_arcface"] = retrieval_block(
        torch.from_numpy(_l2(arcface[mask])[query_idx]),
        torch.from_numpy(_l2(arcface[mask])[gallery_idx]),
        controls=False,
    )
    report["retrieval"]["raw_arcface_gender_controlled"] = {
        key: float(value)
        for key, value in gender_controlled_retrieval(
            torch.from_numpy(_l2(arcface[mask])[query_idx]),
            query_speakers,
            genders[mask][query_idx],
            torch.from_numpy(_l2(arcface[mask])[gallery_idx]),
            gallery_speakers,
            genders[mask][gallery_idx],
        ).items()
    }
    fused_z = (
        fused[mask] - fused_scaler["face_mean"].numpy()
    ) / np.maximum(fused_scaler["face_std"].numpy(), 1e-6)
    report["retrieval"]["fused_z"] = retrieval_block(
        torch.from_numpy(_l2(fused_z)[query_idx]),
        torch.from_numpy(_l2(fused_z)[gallery_idx]),
        controls=False,
    )
    report["health"]["cfvn"] = embedding_health(cfvn_embedding, eval_speakers)

    # CVAE and adapter are evaluated on the same CFVN embeddings.
    cvae = _load_cvae(artifacts["cvae"], device)
    adapter = _load_adapter(artifacts["adapter"], device)
    codes, _ = speaker_codes(eval_speakers)
    codes_t = torch.from_numpy(codes)
    report["cvae"] = _cvae_metrics(
        cvae,
        cfvn_embedding,
        voice_t,
        codes_t,
        device=device,
    )
    target_center = targets_all[train_mask].mean(dim=0)
    report["adapter"] = _adapter_metrics(
        adapter,
        cvae,
        cfvn_embedding,
        codes_t,
        targets_all[mask],
        target_center,
        device=device,
    )
    report["voice_health"] = embedding_health(
        torch.from_numpy(voice[mask]), eval_speakers
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FaceFeatureV2 CI gate")
    parser.add_argument("--features", type=Path, default=DEFAULTS["features"])
    parser.add_argument("--index", type=Path, default=DEFAULTS["index"])
    parser.add_argument("--scaler", type=Path, default=DEFAULTS["scaler"])
    parser.add_argument("--cfvn", type=Path, default=DEFAULTS["cfvn"])
    parser.add_argument("--cvae", type=Path, default=DEFAULTS["cvae"])
    parser.add_argument("--adapter", type=Path, default=DEFAULTS["adapter"])
    parser.add_argument(
        "--style-targets", type=Path, default=DEFAULTS["style_targets"]
    )
    parser.add_argument("--split", default="test", choices=("test", "val"))
    parser.add_argument(
        "--max-faces",
        type=int,
        default=0,
        help=(
            "Deterministic speaker-stratified subset size. 0 (default) uses "
            "the full split; small subsets can be underpowered for the 10x "
            "chance-multiple check."
        ),
    )
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "ci_gate_v2.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = build_report(args)
    result = evaluate_gate(report)
    report["gate"] = result.to_dict()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    for check in result.checks:
        status = "PASS" if check.passed else "FAIL"
        print(f"[{status}] {check.name}: {check.value} (threshold {check.threshold})")
    print(f"Gate {'PASSED' if result.passed else 'FAILED'}: {args.output}")
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
