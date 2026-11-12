#!/usr/bin/env python3
"""Compare face-encoder ceilings for the StyleAdapter identity path.

Phase 1 of ``docs/IDENTITY_NEXT_PHASE.md``.  The script fits train-only ridge
maps from several face representations to the centered per-speaker StyleTTS2
style targets, then reports validation/test residual cosine, the centered
different-speaker pair tail, and the demo worst-pair separability.

The ridge is a linear predictability ceiling, not a proposed serving model.  It
does not synthesize audio and does not use ECAPA, the TTS renderer, or any
evaluation recognizer to select a representation.  The candidate is selected by
validation residual cosine only; test and demo metrics are reported after that
selection.

Example:
    python scripts/audit/compare_face_encoder_ceiling.py \
        --output results/audit/face_encoder_ceiling/report.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from voix.data.feature_contract import (  # noqa: E402
    FaceFeatureBuilder,
    load_scaler_payload,
)
from voix.models.contrastive_net import ContrastiveFaceVoiceNet  # noqa: E402
from voix.models.cvae import CVAE  # noqa: E402
from voix.training.data import (  # noqa: E402
    load_v2_dataset,
    validate_scaler_for_dataset,
)
from voix.training.v2_utils import (  # noqa: E402
    align_style_targets,
    load_style_targets,
    normalize_face,
)

DEFAULT_FEATURES = (
    ROOT / "data" / "processed" / "voxceleb2_v2" / "features_v2.h5"
)
DEFAULT_INDEX = ROOT / "data" / "processed" / "voxceleb2_v2" / "index.csv"
DEFAULT_STYLE_TARGETS = (
    ROOT / "data" / "processed" / "style_targets_voxceleb2_5clip.pt"
)
DEFAULT_FUSED_SCALER = (
    ROOT / "data" / "processed" / "voxceleb2_v2" / "face_scaler_v2.pt"
)
DEFAULT_ARCFACE_SCALER = (
    ROOT / "data" / "processed" / "voxceleb2_v2" / "face_scaler_arcface_v2.pt"
)
DEFAULT_CFVN = ROOT / "results" / "checkpoints" / "cfvn_v2_best.pt"
DEFAULT_CFVN_ARCFACE = ROOT / "results" / "checkpoints" / "cfvn_v2_arcface.pt"
DEFAULT_CVAE = (
    ROOT / "checkpoints" / "cvae_cfvn_face192_latent128_diversity.pt"
)
DEFAULT_FACE_REPORT = (
    ROOT / "results" / "audit" / "face_voice_diversity" / "face_voice_diversity.json"
)
DEFAULT_OUTPUT = (
    ROOT / "results" / "audit" / "face_encoder_ceiling" / "report.json"
)
DEFAULT_DEMO_CACHE = (
    ROOT / "results" / "audit" / "face_encoder_ceiling" / "demo_face_features.json"
)

# The three selection faces and eight held-out faces used by the identity
# audit.  These are the only demo faces that enter the Phase 1 decision.
GATE_FACE_IDS = [
    "C1_02",
    "C2_09",
    "C2_12",
    "C1_01",
    "C1_03",
    "C1_05",
    "C2_08",
    "C1_04",
    "C2_07",
    "C2_13",
    "C2_15",
]

WATCH_PAIRS = [
    ("C1_03", "C1_05"),
    ("C1_04", "C2_15"),
    ("C1_01", "C2_08"),
    ("C1_04", "C2_07"),
]

REPRESENTATIONS: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("cfvn_es", ("cfvn_es",)),
    ("cfvn_es_arcface", ("cfvn_es_arcface",)),
    ("arcface", ("arcface",)),
    ("fused", ("fused",)),
    ("cfvn_es+fused", ("cfvn_es", "fused")),
    ("cfvn_es+arcface", ("cfvn_es", "arcface")),
    ("fused+arcface", ("fused", "arcface")),
    ("cfvn_es+fused+arcface", ("cfvn_es", "fused", "arcface")),
)

DEFAULT_LAMBDAS = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]

# Pre-registered Phase 1 thresholds.  Do not change these after seeing results.
THRESHOLDS = {
    "linear_residual_gain": 0.05,
    "linear_pair_p90_gain": 0.05,
    "linear_target_cosine_tolerance": 0.01,
    "demo_watch_pair_improvement": 0.10,
    "demo_watch_pair_max_centered": 0.50,
    "demo_gate_max_centered": 0.60,
    "conditional_gate_max_centered": 0.70,
}

LIMITATIONS = [
    "The ridge is a linear predictability ceiling, not a serving model.",
    "No audio is synthesized; a style-cosine screen is not an audio-gate pass.",
    "ECAPA, the TTS renderer, and evaluation recognizers are not used to select a representation.",
    "The official test split has 118 speakers; speaker-level pair tails are noisy.",
    "The demo set has 19 faces, of which 11 are the gate set; it is not a speaker-generalization benchmark.",
    "VoxCeleb2-only supervision may not cover the demo face/TTS domain.",
    "No perceptual, MOS, or identity-accuracy claim follows from these numbers.",
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_args(args: argparse.Namespace) -> None:
    for name in ("features", "index", "style_targets", "fused_scaler", "arcface_scaler"):
        path = getattr(args, name)
        if not path.is_file():
            raise FileNotFoundError(f"--{name.replace('_', '-')} not found: {path}")
    for name in ("cfvn", "cfvn_arcface", "cvae"):
        path = getattr(args, name)
        if not path.is_file():
            raise FileNotFoundError(f"--{name.replace('_', '-')} not found: {path}")
    if not args.face_report.is_file():
        raise FileNotFoundError(f"--face-report not found: {args.face_report}")
    if not args.ridge_lambdas or any(value <= 0 for value in args.ridge_lambdas):
        raise ValueError("--ridge-lambdas must be positive")


def _check_alignment(fused_dataset, arcface_dataset) -> None:
    """Fail if the two HDF5 views are not row-aligned."""
    for field in ("speaker_ids", "split", "clip_ids", "gender"):
        left = getattr(fused_dataset, field)
        right = getattr(arcface_dataset, field)
        if len(left) != len(right) or any(str(a) != str(b) for a, b in zip(left, right)):
            raise ValueError(f"fused and ArcFace datasets are not aligned on {field}")


def _load_cvae(path: Path, device: str) -> CVAE:
    """Load the frozen CVAE with the architecture recorded in its checkpoint."""
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state_dict = (
        checkpoint["model_state_dict"]
        if "model_state_dict" in checkpoint
        else checkpoint
    )
    model = CVAE(
        face_dim=int(checkpoint.get("face_dim", 192)),
        speaker_dim=int(checkpoint.get("speaker_dim", 192)),
        latent_dim=int(checkpoint.get("latent_dim", 128)),
    )
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def _serving_es(
    cfvn: ContrastiveFaceVoiceNet,
    cvae: CVAE,
    face: torch.Tensor,
    device: str,
    batch_size: int = 1024,
) -> torch.Tensor:
    """Reproduce the frozen serving e_s: CFVN -> CVAE prior mean -> decoder."""
    cfvn.eval()
    cvae.eval()
    outputs = []
    for start in range(0, len(face), batch_size):
        batch = face[start : start + batch_size].to(device)
        e_f_voice = cfvn(batch, normalize=True)
        mu_prior, _ = cvae.prior_net(e_f_voice)
        e_s = F.normalize(cvae.decoder(mu_prior), dim=-1, eps=1e-6)
        outputs.append(e_s.cpu())
    if not outputs:
        return face.new_zeros((0, cvae.speaker_dim))
    return torch.cat(outputs, dim=0)


def _load_blocks(
    args: argparse.Namespace,
    device: str,
) -> Tuple[Any, Dict[str, np.ndarray], Dict[str, Any]]:
    """Load the face blocks and the frozen CFVN/CVAE embeddings."""
    fused_dataset = load_v2_dataset(
        args.features, args.index, face_key="face_features"
    )
    arcface_dataset = load_v2_dataset(args.features, args.index, face_key="arcface")
    _check_alignment(fused_dataset, arcface_dataset)

    fused_scaler = validate_scaler_for_dataset(
        load_scaler_payload(
            args.fused_scaler,
            expected_dim=fused_dataset.face_raw.shape[1],
            face_key="face_features",
        ),
        fused_dataset,
        face_key="face_features",
    )
    arcface_scaler = validate_scaler_for_dataset(
        load_scaler_payload(
            args.arcface_scaler,
            expected_dim=arcface_dataset.face_raw.shape[1],
            face_key="arcface",
        ),
        arcface_dataset,
        face_key="arcface",
    )

    fused_norm = normalize_face(fused_dataset.face_raw, fused_scaler)
    arcface_norm = normalize_face(arcface_dataset.face_raw, arcface_scaler)

    cfvn = ContrastiveFaceVoiceNet.from_checkpoint(str(args.cfvn), device=device)
    cfvn_arcface = ContrastiveFaceVoiceNet.from_checkpoint(
        str(args.cfvn_arcface), device=device
    )
    if cfvn.face_dim != fused_dataset.face_raw.shape[1]:
        raise ValueError(
            f"CFVN expects {cfvn.face_dim}-D fused faces, dataset has "
            f"{fused_dataset.face_raw.shape[1]}"
        )
    if cfvn_arcface.face_dim != arcface_dataset.face_raw.shape[1]:
        raise ValueError(
            f"ArcFace CFVN expects {cfvn_arcface.face_dim}-D faces, dataset has "
            f"{arcface_dataset.face_raw.shape[1]}"
        )
    cvae = _load_cvae(args.cvae, device)
    if cvae.face_dim != cfvn.out_dim:
        raise ValueError(
            f"CVAE expects {cvae.face_dim}-D CFVN embeddings, canonical CFVN "
            f"produces {cfvn.out_dim}-D"
        )
    if cvae.face_dim != cfvn_arcface.out_dim:
        raise ValueError(
            f"CVAE expects {cvae.face_dim}-D CFVN embeddings, ArcFace CFVN "
            f"produces {cfvn_arcface.out_dim}-D"
        )

    print("[ceiling] embedding fused faces through CFVN -> CVAE", flush=True)
    serving_es = _serving_es(cfvn, cvae, fused_norm, device)
    print("[ceiling] embedding raw ArcFace through ArcFace CFVN -> CVAE", flush=True)
    serving_es_arcface = _serving_es(cfvn_arcface, cvae, arcface_norm, device)

    blocks = {
        "cfvn_es": serving_es.numpy(),
        "cfvn_es_arcface": serving_es_arcface.numpy(),
        "arcface": arcface_dataset.face_raw,
        "fused": fused_dataset.face_raw,
    }
    artifacts = {
        "fused_scaler": fused_scaler,
        "arcface_scaler": arcface_scaler,
        "cfvn": cfvn,
        "cfvn_arcface": cfvn_arcface,
        "cvae": cvae,
    }
    return fused_dataset, blocks, artifacts


def _speaker_centroids(
    values: np.ndarray,
    speaker_ids: np.ndarray,
) -> Tuple[np.ndarray, List[str]]:
    """Average each block to one row per speaker, in stable speaker order."""
    ids = [str(value) for value in speaker_ids]
    unique = sorted(set(ids))
    lookup = {speaker: index for index, speaker in enumerate(unique)}
    codes = np.asarray([lookup[speaker] for speaker in ids], dtype=np.int64)
    sums = np.zeros((len(unique), values.shape[1]), dtype=np.float64)
    for dim in range(values.shape[1]):
        sums[:, dim] = np.bincount(
            codes,
            weights=values[:, dim].astype(np.float64),
            minlength=len(unique),
        )
    counts = np.bincount(codes, minlength=len(unique)).astype(np.float64)
    return (sums / counts[:, None]).astype(np.float32), unique


def _normalize_centroids(values: np.ndarray, block: str) -> np.ndarray:
    """Renormalize embedding blocks after speaker averaging."""
    if block not in {"cfvn_es", "cfvn_es_arcface", "arcface"}:
        return values
    norm = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.clip(norm, 1e-8, None)


def _speaker_matrix(
    blocks: Mapping[str, np.ndarray],
    block_names: Sequence[str],
    speaker_ids: np.ndarray,
) -> Tuple[np.ndarray, List[str], Dict[str, np.ndarray]]:
    centroids: Dict[str, np.ndarray] = {}
    names: Optional[List[str]] = None
    for block in block_names:
        values, current_names = _speaker_centroids(blocks[block], speaker_ids)
        values = _normalize_centroids(values, block)
        centroids[block] = values
        if names is None:
            names = current_names
        elif names != current_names:
            raise ValueError("speaker order mismatch across blocks")
    if names is None:
        raise ValueError("no blocks requested")
    return np.concatenate([centroids[block] for block in block_names], axis=1), names, centroids


def _speaker_targets(
    targets: torch.Tensor,
    speaker_ids: np.ndarray,
    unique_speakers: Sequence[str],
) -> torch.Tensor:
    """Return one target row per speaker, verifying within-speaker constancy."""
    ids = [str(value) for value in speaker_ids]
    groups: Dict[str, List[int]] = {}
    for index, speaker in enumerate(ids):
        groups.setdefault(speaker, []).append(index)
    if set(groups) != set(unique_speakers):
        missing = sorted(set(unique_speakers) - set(groups))
        extra = sorted(set(groups) - set(unique_speakers))
        raise ValueError(
            f"speaker order mismatch: missing={missing[:5]}, extra={extra[:5]}"
        )
    indices = [groups[speaker][0] for speaker in unique_speakers]
    values = targets[indices].float()
    for speaker, members in groups.items():
        if len(members) > 1:
            first = targets[members[0]]
            repeated = first.unsqueeze(0).expand(len(members), -1)
            if not torch.allclose(targets[members], repeated, atol=1e-6):
                raise ValueError(f"style target varies within speaker {speaker!r}")
    return values


def _residual_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    center: torch.Tensor,
) -> Dict[str, float]:
    prediction = prediction.float()
    target = target.float()
    center = center.float()
    pred_residual = prediction - center
    target_residual = target - center
    cosine = F.cosine_similarity(pred_residual, target_residual, dim=-1)
    target_cosine = F.cosine_similarity(prediction, target, dim=-1)
    return {
        "n": int(len(prediction)),
        "target_cosine": float(target_cosine.mean()),
        "target_cosine_timbre": float(
            F.cosine_similarity(
                prediction[:, :128], target[:, :128], dim=-1
            ).mean()
        ),
        "target_cosine_prosody": float(
            F.cosine_similarity(
                prediction[:, 128:], target[:, 128:], dim=-1
            ).mean()
        ),
        "residual_cosine": float(cosine.mean()),
        "residual_cosine_timbre": float(
            F.cosine_similarity(
                pred_residual[:, :128], target_residual[:, :128], dim=-1
            ).mean()
        ),
        "residual_cosine_prosody": float(
            F.cosine_similarity(
                pred_residual[:, 128:], target_residual[:, 128:], dim=-1
            ).mean()
        ),
        "residual_mse": float(F.mse_loss(pred_residual, target_residual)),
        "pred_residual_norm": float(pred_residual.norm(dim=-1).mean()),
        "target_residual_norm": float(target_residual.norm(dim=-1).mean()),
        "residual_norm_ratio": float(
            pred_residual.norm(dim=-1).mean()
            / target_residual.norm(dim=-1).mean().clamp_min(1e-8)
        ),
    }


def _centered_pair_tail(
    prediction: torch.Tensor,
    center: torch.Tensor,
    codes: torch.Tensor,
) -> Dict[str, float]:
    """Mean/p90/p99/max of centered different-speaker cosine."""
    x = F.normalize(prediction.float() - center.float(), dim=-1, eps=1e-8)
    codes = codes.view(-1).to(x.device)
    similarity = x @ x.T
    mask = codes[:, None] != codes[None, :]
    if not bool(mask.any()):
        return {
            "n_pairs": 0,
            "mean": float("nan"),
            "p90": float("nan"),
            "p99": float("nan"),
            "max": float("nan"),
            "frac_gt_0.5": float("nan"),
            "frac_gt_0.8": float("nan"),
        }
    values = similarity[mask]
    return {
        "n_pairs": int(values.numel()),
        "mean": float(values.mean()),
        "p90": float(values.quantile(0.90)),
        "p99": float(values.quantile(0.99)),
        "max": float(values.max()),
        "frac_gt_0.5": float((values > 0.5).float().mean()),
        "frac_gt_0.8": float((values > 0.8).float().mean()),
    }


def _fit_standardized_ridge(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    center: torch.Tensor,
    lambdas: Sequence[float],
) -> Dict[str, Any]:
    """Fit ridge on standardized train rows; select lambda by val residual cosine."""
    x_train = x_train.float().double()
    y_train = y_train.float().double()
    x_val = x_val.float().double()
    y_val = y_val.float().double()
    center = center.float().double()

    x_mean = x_train.mean(dim=0)
    x_std = x_train.std(dim=0).clamp_min(1e-6)
    xz = (x_train - x_mean) / x_std
    xtx = xz.T @ xz
    xty = xz.T @ y_train
    eye = torch.eye(xtx.shape[0], dtype=xtx.dtype)

    rows: List[Dict[str, Any]] = []
    best: Optional[Dict[str, Any]] = None
    for lam in lambdas:
        regularized = xtx + float(lam) * eye
        try:
            chol = torch.linalg.cholesky(regularized)
            weights = torch.cholesky_solve(xty, chol)
        except Exception:
            weights = torch.linalg.solve(regularized, xty)
        prediction = center + ((x_val - x_mean) / x_std) @ weights
        metrics = _residual_metrics(prediction, y_val, center)
        row = {"lambda": float(lam), "val": metrics}
        rows.append(row)
        if best is None or metrics["residual_cosine"] > best["val"]["residual_cosine"]:
            best = {"lambda": float(lam), "val": metrics, "weights": weights}
    if best is None:
        raise ValueError("no ridge lambdas were provided")
    return {
        "x_mean": x_mean.float(),
        "x_std": x_std.float(),
        "rows": rows,
        "best": best,
    }


def _apply_ridge(
    x: torch.Tensor,
    model: Mapping[str, Any],
    center: torch.Tensor,
) -> torch.Tensor:
    x = x.float()
    xz = (x - model["x_mean"]) / model["x_std"]
    return center.float() + xz @ model["best"]["weights"].float()


def _split_codes(names: Sequence[str]) -> torch.Tensor:
    return torch.arange(len(names), dtype=torch.long)


def _evaluate_representation(
    blocks: Mapping[str, np.ndarray],
    block_names: Sequence[str],
    dataset,
    targets: torch.Tensor,
    center: torch.Tensor,
    split_speakers: Mapping[str, Sequence[str]],
    lambdas: Sequence[float],
) -> Dict[str, Any]:
    """Fit one representation and return val/test metrics and selected weights."""
    split_matrices: Dict[str, np.ndarray] = {}
    split_names: Dict[str, List[str]] = {}
    for split in ("train", "val", "test"):
        mask = dataset.split == split
        split_blocks = {
            block: blocks[block][mask] for block in block_names
        }
        matrix, names, _ = _speaker_matrix(
            split_blocks, block_names, dataset.speaker_ids[mask]
        )
        split_matrices[split] = matrix
        split_names[split] = names
        if list(names) != list(split_speakers[split]):
            raise ValueError(f"speaker order mismatch for split {split}")

    split_targets: Dict[str, torch.Tensor] = {}
    for split in ("train", "val", "test"):
        mask = dataset.split == split
        split_targets[split] = _speaker_targets(
            targets[mask], dataset.speaker_ids[mask], split_names[split]
        )

    train_x = torch.from_numpy(split_matrices["train"])
    val_x = torch.from_numpy(split_matrices["val"])
    test_x = torch.from_numpy(split_matrices["test"])
    train_y = split_targets["train"]
    val_y = split_targets["val"]
    test_y = split_targets["test"]

    ridge = _fit_standardized_ridge(
        train_x, train_y, val_x, val_y, center, lambdas
    )
    output: Dict[str, Any] = {
        "input_dim": int(train_x.shape[1]),
        "blocks": list(block_names),
        "val_rows": ridge["rows"],
        "selected_lambda": float(ridge["best"]["lambda"]),
        "selected_by": "val_residual_cosine",
        "splits": {},
    }
    for split, x, y in (
        ("val", val_x, val_y),
        ("test", test_x, test_y),
    ):
        prediction = _apply_ridge(x, ridge, center)
        metrics = _residual_metrics(prediction, y, center)
        metrics["pair_tail"] = _centered_pair_tail(
            prediction, center, _split_codes(split_names[split])
        )
        metrics["target_pair_tail"] = _centered_pair_tail(
            y, center, _split_codes(split_names[split])
        )
        output["splits"][split] = metrics
    output["_ridge"] = ridge
    return output


def _parse_pair(pair: str) -> Tuple[str, str]:
    parts = [part.strip() for part in pair.split(":")]
    if len(parts) != 2 or not all(parts):
        raise ValueError(f"watch pair must look like C1_03:C1_05, got {pair!r}")
    return parts[0], parts[1]


def _label_for_id(label_by_id: Mapping[str, str], face_id: str) -> str:
    if face_id not in label_by_id:
        raise KeyError(f"face id {face_id!r} is not in the face report")
    return label_by_id[face_id]


def _demo_rows_from_cache(cache: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {str(row["label"]): dict(row) for row in cache["rows"]}


def _extract_demo_rows(
    args: argparse.Namespace,
    labels: Sequence[str],
    entries: Mapping[str, Mapping[str, Any]],
    artifacts: Mapping[str, Any],
    device: str,
) -> Dict[str, Dict[str, Any]]:
    """Extract raw ArcFace/fused features and verify cached canonical e_s."""
    import cv2

    from voix.data.demographics import (
        SoftDemographicEstimator,
        aspect_ratio_from_bbox,
    )
    from voix.data.face_extractor import (
        ArcFaceExtractor,
        FaceMeshExtractor,
        crop_face_bbox,
    )
    from voix.data.morphology import compute_craniofacial_ratios

    arcface = ArcFaceExtractor(
        device_id=0 if args.arcface_device == "cuda" else -1,
        cache_dir=str(ROOT / "checkpoints"),
    )
    facemesh = FaceMeshExtractor(
        model_path=str(ROOT / "checkpoints" / "mediapipe" / "face_landmarker.task")
    )
    demographics = SoftDemographicEstimator(device=device)
    builder = FaceFeatureBuilder(device=device)
    cfvn = artifacts["cfvn"]
    cvae = artifacts["cvae"]
    fused_scaler = artifacts["fused_scaler"]

    rows: Dict[str, Dict[str, Any]] = {}
    for label in labels:
        entry = entries[label]
        face_path = Path(str(entry["face"]))
        if not face_path.is_file():
            raise FileNotFoundError(f"demo face not found: {face_path}")
        face_sha = _sha256(face_path)
        if face_sha != str(entry["face_sha256"]):
            raise RuntimeError(
                f"face hash mismatch for {label}: {face_sha} != {entry['face_sha256']}"
            )
        image = cv2.imread(str(face_path))
        if image is None:
            raise ValueError(f"could not read demo face: {face_path}")
        e_id, detected_gender, detected_age, bbox = arcface.extract_full_with_bbox(
            image
        )
        if e_id is None:
            raise ValueError(f"no face detected in demo image: {face_path}")
        # The frozen report stores the exact extracted gender/age used to build
        # the cached e_s.  Reuse those values so the fused representation and
        # the parity check follow the same contract.
        gender = int(entry["gender"])
        age = float(entry["age"])
        face_crop = crop_face_bbox(image, bbox) if bbox is not None else image
        landmarks = facemesh.extract(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB))
        if landmarks is not None:
            e_geo = compute_craniofacial_ratios(landmarks)
        else:
            e_geo = np.zeros(32, dtype=np.float32)
        aspect_ratio = (
            aspect_ratio_from_bbox(bbox) if bbox is not None else 1.35
        )
        e_demo = demographics.estimate(
            e_id,
            gender=gender,
            age=age,
            aspect_ratio=aspect_ratio,
        )
        fused = builder.fuse(e_id, e_geo, e_demo).cpu().numpy().reshape(-1)
        fused_norm = normalize_face(fused[None, :], fused_scaler)
        with torch.no_grad():
            e_s = (
                _serving_es(cfvn, cvae, fused_norm, device)
                .numpy()
                .reshape(-1)
            )
        cached_e_s = np.asarray(entry["e_s_vector"], dtype=np.float32).reshape(-1)
        max_abs_diff = float(np.max(np.abs(e_s - cached_e_s)))
        cosine = float(
            F.cosine_similarity(
                torch.from_numpy(e_s).float(),
                torch.from_numpy(cached_e_s).float(),
                dim=0,
            )
        )
        if max_abs_diff > 1e-4:
            raise RuntimeError(
                f"canonical e_s parity failed for {label}: max_abs_diff="
                f"{max_abs_diff:.3e} cosine={cosine:.6f}. The cached demo "
                "features and the current pipeline do not match."
            )
        rows[label] = {
            "label": label,
            "face": str(face_path),
            "face_sha256": face_sha,
            "gender": gender,
            "age": age,
            "detected_gender": int(detected_gender) if detected_gender is not None else None,
            "detected_age": float(detected_age) if detected_age is not None else None,
            "arcface": np.asarray(e_id, dtype=np.float32).reshape(-1).tolist(),
            "fused": fused.astype(np.float32).tolist(),
            "cfvn_es": e_s.astype(np.float32).tolist(),
            "parity_max_abs_diff": max_abs_diff,
            "parity_cosine": cosine,
        }
        print(
            f"[ceiling] demo face {label} parity max_abs_diff={max_abs_diff:.2e}",
            flush=True,
        )
    return rows


def _load_demo_rows(
    args: argparse.Namespace,
    labels: Sequence[str],
    entries: Mapping[str, Mapping[str, Any]],
    artifacts: Mapping[str, Any],
    device: str,
) -> Dict[str, Dict[str, Any]]:
    """Load or extract demo features, with face-hash and report-hash checks."""
    report_sha = _sha256(args.face_report)
    artifact_hashes = {
        "cfvn_sha256": _sha256(args.cfvn),
        "cfvn_arcface_sha256": _sha256(args.cfvn_arcface),
        "cvae_sha256": _sha256(args.cvae),
    }
    if args.reuse_demo_features and args.demo_cache.is_file():
        cache = json.loads(args.demo_cache.read_text(encoding="utf-8"))
        if (
            cache.get("face_report_sha256") == report_sha
            and all(cache.get(key) == value for key, value in artifact_hashes.items())
        ):
            cached_rows = _demo_rows_from_cache(cache)
            if all(label in cached_rows for label in labels):
                for label in labels:
                    row = cached_rows[label]
                    face_path = Path(str(row["face"]))
                    if not face_path.is_file():
                        raise FileNotFoundError(f"demo face not found: {face_path}")
                    if _sha256(face_path) != str(row["face_sha256"]):
                        raise RuntimeError(
                            f"cached demo face hash mismatch for {label}"
                        )
                print("[ceiling] reusing cached demo features", flush=True)
                return {label: cached_rows[label] for label in labels}

    print("[ceiling] extracting demo face features", flush=True)
    rows = _extract_demo_rows(args, labels, entries, artifacts, device)
    cache = {
        "schema_version": "voix.face_encoder_demo_features.v2",
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "face_report": str(args.face_report),
        "face_report_sha256": report_sha,
        "cfvn_sha256": _sha256(args.cfvn),
        "cfvn_arcface_sha256": _sha256(args.cfvn_arcface),
        "cvae_sha256": _sha256(args.cvae),
        "arcface_device": args.arcface_device,
        "rows": [rows[label] for label in labels],
    }
    args.demo_cache.parent.mkdir(parents=True, exist_ok=True)
    args.demo_cache.write_text(json.dumps(cache, indent=2), encoding="utf-8")
    return rows


def _demo_blocks(
    demo_rows: Mapping[str, Mapping[str, Any]],
    artifacts: Mapping[str, Any],
    device: str,
) -> Dict[str, Dict[str, np.ndarray]]:
    """Build the same block names for each demo face."""
    cfvn_arcface = artifacts["cfvn_arcface"]
    cvae = artifacts["cvae"]
    arcface_scaler = artifacts["arcface_scaler"]
    blocks: Dict[str, Dict[str, np.ndarray]] = {}
    for label, row in demo_rows.items():
        arcface = np.asarray(row["arcface"], dtype=np.float32).reshape(1, -1)
        arcface_norm = normalize_face(arcface, arcface_scaler)
        e_s_arcface = (
            _serving_es(cfvn_arcface, cvae, arcface_norm, device)
            .numpy()
            .reshape(-1)
        )
        blocks[label] = {
            "cfvn_es": np.asarray(row["cfvn_es"], dtype=np.float32),
            "cfvn_es_arcface": e_s_arcface.astype(np.float32),
            "arcface": arcface.reshape(-1),
            "fused": np.asarray(row["fused"], dtype=np.float32),
        }
    return blocks


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(
        F.cosine_similarity(
            torch.from_numpy(np.asarray(a, dtype=np.float32)).reshape(1, -1),
            torch.from_numpy(np.asarray(b, dtype=np.float32)).reshape(1, -1),
            dim=-1,
        )
    )


def _pair_key(a: str, b: str) -> str:
    return f"{a}/{b}"


def _demo_pair_metrics(
    label_by_id: Mapping[str, str],
    demo_blocks: Mapping[str, Mapping[str, np.ndarray]],
    demo_rows: Mapping[str, Mapping[str, Any]],
    entries: Mapping[str, Mapping[str, Any]],
    variant_models: Mapping[str, Mapping[str, Any]],
    center: torch.Tensor,
    gate_ids: Sequence[str],
) -> Dict[str, Any]:
    """Predict demo styles and compute same-gender pair separability."""
    labels_by_id = {
        face_id: _label_for_id(label_by_id, face_id) for face_id in label_by_id
    }
    demo_labels = list(demo_blocks)
    predictions: Dict[str, Dict[str, torch.Tensor]] = {}
    for variant, blocks in REPRESENTATIONS:
        model = variant_models[variant]
        x = np.stack(
            [
                np.concatenate(
                    [demo_blocks[label][block] for block in blocks], axis=0
                )
                for label in demo_labels
            ],
            axis=0,
        )
        predictions[variant] = {
            label: _apply_ridge(
                torch.from_numpy(x[index]).unsqueeze(0), model, center
            )
            .reshape(-1)
            .cpu()
            for index, label in enumerate(demo_labels)
        }

    gate_labels = [labels_by_id[face_id] for face_id in gate_ids]
    canonical_styles = {
        label: np.asarray(
            (
                entries[label]["style_without_geo_vector"]
                if "style_without_geo_vector" in entries[label]
                else entries[label]["style_vector"]
            ),
            dtype=np.float32,
        ).reshape(-1)
        for label in demo_labels
    }

    pairs: List[Dict[str, Any]] = []
    for index, label_a in enumerate(demo_labels):
        for label_b in demo_labels[index + 1 :]:
            gender_a = int(demo_rows[label_a]["gender"])
            gender_b = int(demo_rows[label_b]["gender"])
            if gender_a != gender_b:
                continue
            id_a = "_".join(label_a.split("_")[:2])
            id_b = "_".join(label_b.split("_")[:2])
            pair: Dict[str, Any] = {
                "a": id_a,
                "b": id_b,
                "a_label": label_a,
                "b_label": label_b,
                "gender": "female" if gender_a == 0 else "male",
                "input_cosines": {
                    block: _cosine(
                        demo_blocks[label_a][block], demo_blocks[label_b][block]
                    )
                    for block in ("cfvn_es", "cfvn_es_arcface", "arcface", "fused")
                },
                "canonical_adapter_style_cosine": _cosine(
                    canonical_styles[label_a], canonical_styles[label_b]
                ),
                "canonical_adapter_centered_cosine": float(
                    F.cosine_similarity(
                        torch.from_numpy(canonical_styles[label_a])
                        - center.float(),
                        torch.from_numpy(canonical_styles[label_b])
                        - center.float(),
                        dim=0,
                    )
                ),
                "variants": {},
            }
            for variant, _blocks in REPRESENTATIONS:
                style_a = predictions[variant][label_a]
                style_b = predictions[variant][label_b]
                pair["variants"][variant] = {
                    "raw_cosine": float(
                        F.cosine_similarity(style_a, style_b, dim=0)
                    ),
                    "centered_cosine": float(
                        F.cosine_similarity(
                            style_a - center.float(), style_b - center.float(), dim=0
                        )
                    ),
                }
            pairs.append(pair)

    watch_pairs = {
        _pair_key(a, b): next(
            (
                pair
                for pair in pairs
                if {pair["a"], pair["b"]} == {a, b}
            ),
            None,
        )
        for a, b in WATCH_PAIRS
    }
    missing_watch = [
        key for key, pair in watch_pairs.items() if pair is None
    ]
    if missing_watch:
        raise KeyError(f"watch pairs missing from demo set: {missing_watch}")

    def summarize(subset: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
        rows: Dict[str, Any] = {}
        for variant, _blocks in REPRESENTATIONS:
            values = [
                float(pair["variants"][variant]["centered_cosine"])
                for pair in subset
            ]
            raw_values = [
                float(pair["variants"][variant]["raw_cosine"]) for pair in subset
            ]
            if not values:
                continue
            rows[variant] = {
                "n_pairs": len(values),
                "centered_mean": float(np.mean(values)),
                "centered_median": float(np.median(values)),
                "centered_min": float(np.min(values)),
                "centered_max": float(np.max(values)),
                "centered_frac_gt_0.5": float(np.mean(np.asarray(values) > 0.5)),
                "centered_frac_gt_0.65": float(
                    np.mean(np.asarray(values) > 0.65)
                ),
                "raw_mean": float(np.mean(raw_values)),
                "raw_median": float(np.median(raw_values)),
                "raw_max": float(np.max(raw_values)),
                "n_centered_le_0.50": int(np.sum(np.asarray(values) <= 0.50)),
                "n_centered_le_0.60": int(np.sum(np.asarray(values) <= 0.60)),
            }
        return rows

    gate_pair_set = [
        pair
        for pair in pairs
        if pair["a_label"] in gate_labels and pair["b_label"] in gate_labels
    ]
    top_pairs = {}
    for variant, _blocks in REPRESENTATIONS:
        top_pairs[variant] = sorted(
            [
                {
                    "pair": _pair_key(pair["a"], pair["b"]),
                    "centered_cosine": pair["variants"][variant][
                        "centered_cosine"
                    ],
                    "raw_cosine": pair["variants"][variant]["raw_cosine"],
                }
                for pair in gate_pair_set
            ],
            key=lambda row: float(row["centered_cosine"]),
            reverse=True,
        )[:10]

    watch_table = {}
    for key, pair in watch_pairs.items():
        variant_values = pair["variants"]
        watch_table[key] = {
            "input_cosines": pair["input_cosines"],
            "canonical_adapter_style_cosine": pair[
                "canonical_adapter_style_cosine"
            ],
            "canonical_adapter_centered_cosine": pair[
                "canonical_adapter_centered_cosine"
            ],
            "variants": variant_values,
        }

    return {
        "n_faces": len(demo_labels),
        "n_gate_faces": len(gate_labels),
        "same_gender_pairs": pairs,
        "gate_summary": summarize(gate_pair_set),
        "all_summary": summarize(pairs),
        "top_gate_pairs": top_pairs,
        "watch_pairs": watch_table,
    }


def _identity_overlap(
    demo_rows: Mapping[str, Mapping[str, Any]],
    train_arcface: np.ndarray,
    train_speakers: np.ndarray,
) -> Dict[str, Any]:
    """Nearest VoxCeleb2 train face for each demo face, as a leakage diagnostic."""
    train = torch.from_numpy(np.asarray(train_arcface, dtype=np.float32))
    train = F.normalize(train, dim=-1)
    speakers = [str(value) for value in train_speakers]
    output: Dict[str, Any] = {}
    for label, row in demo_rows.items():
        query = F.normalize(
            torch.from_numpy(np.asarray(row["arcface"], dtype=np.float32)), dim=0
        )
        similarity = train @ query
        top_values, top_indices = torch.topk(similarity, k=min(5, len(similarity)))
        top = [
            {
                "speaker": speakers[int(index)],
                "cosine": float(value),
            }
            for value, index in zip(top_values.tolist(), top_indices.tolist())
        ]
        output[label] = {
            "top1": top[0],
            "top5": top,
            "n_train_faces_ge_0.5": int((similarity >= 0.5).sum()),
            "n_train_faces_ge_0.6": int((similarity >= 0.6).sum()),
            "n_train_faces_ge_0.7": int((similarity >= 0.7).sum()),
            "possible_overlap": bool(top[0]["cosine"] >= 0.6),
        }
    return output


def _decision(
    baseline: Mapping[str, Any],
    candidate_name: str,
    candidate: Mapping[str, Any],
    demo: Mapping[str, Any],
) -> Dict[str, Any]:
    """Apply the pre-registered Phase 1 decision matrix."""
    base_val = baseline["splits"]["val"]
    base_test = baseline["splits"]["test"]
    cand_val = candidate["splits"]["val"]
    cand_test = candidate["splits"]["test"]

    linear_checks = {
        "L1_val_residual_gain": {
            "value": cand_val["residual_cosine"] - base_val["residual_cosine"],
            "threshold": THRESHOLDS["linear_residual_gain"],
            "passed": cand_val["residual_cosine"] - base_val["residual_cosine"]
            >= THRESHOLDS["linear_residual_gain"],
        },
        "L2_test_residual_gain": {
            "value": cand_test["residual_cosine"] - base_test["residual_cosine"],
            "threshold": THRESHOLDS["linear_residual_gain"],
            "passed": cand_test["residual_cosine"] - base_test["residual_cosine"]
            >= THRESHOLDS["linear_residual_gain"],
        },
        "L3_val_pair_p90_gain": {
            "value": base_val["pair_tail"]["p90"] - cand_val["pair_tail"]["p90"],
            "threshold": THRESHOLDS["linear_pair_p90_gain"],
            "passed": base_val["pair_tail"]["p90"] - cand_val["pair_tail"]["p90"]
            >= THRESHOLDS["linear_pair_p90_gain"],
        },
        "L4_test_pair_p90_gain": {
            "value": base_test["pair_tail"]["p90"] - cand_test["pair_tail"]["p90"],
            "threshold": THRESHOLDS["linear_pair_p90_gain"],
            "passed": base_test["pair_tail"]["p90"] - cand_test["pair_tail"]["p90"]
            >= THRESHOLDS["linear_pair_p90_gain"],
        },
        "L5_test_frac_gt_0.8": {
            "value": cand_test["pair_tail"]["frac_gt_0.8"],
            "baseline": base_test["pair_tail"]["frac_gt_0.8"],
            "passed": cand_test["pair_tail"]["frac_gt_0.8"]
            <= base_test["pair_tail"]["frac_gt_0.8"],
        },
        "L6_test_target_cosine": {
            "value": cand_test["target_cosine"],
            "baseline": base_test["target_cosine"],
            "tolerance": THRESHOLDS["linear_target_cosine_tolerance"],
            "passed": cand_test["target_cosine"]
            >= base_test["target_cosine"]
            - THRESHOLDS["linear_target_cosine_tolerance"],
        },
    }
    linear_pass = all(check["passed"] for check in linear_checks.values())

    base_watch = demo["baseline_watch_pairs"]
    cand_watch = demo["candidate_watch_pairs"]
    watch_checks = {}
    for key in base_watch:
        base_value = float(base_watch[key]["variants"]["cfvn_es"]["centered_cosine"])
        cand_value = float(cand_watch[key]["variants"][candidate_name]["centered_cosine"])
        watch_checks[key] = {
            "baseline_centered_cosine": base_value,
            "candidate_centered_cosine": cand_value,
            "improvement": base_value - cand_value,
            "improvement_threshold": THRESHOLDS["demo_watch_pair_improvement"],
            "max_centered": THRESHOLDS["demo_watch_pair_max_centered"],
            "improved": base_value - cand_value
            >= THRESHOLDS["demo_watch_pair_improvement"],
            "below_max": cand_value
            <= THRESHOLDS["demo_watch_pair_max_centered"],
        }
    gate_max = float(
        demo["candidate_gate_summary"][candidate_name]["centered_max"]
    )
    demo_checks = {
        "D1_D2_watch_pairs": {
            "pairs": watch_checks,
            "passed": all(
                check["improved"] and check["below_max"]
                for check in watch_checks.values()
            ),
        },
        "D3_gate_max": {
            "value": gate_max,
            "threshold": THRESHOLDS["demo_gate_max_centered"],
            "passed": gate_max <= THRESHOLDS["demo_gate_max_centered"],
        },
        "D4_watch_improvement": {
            "passed": all(check["improved"] for check in watch_checks.values()),
        },
    }
    demo_pass = all(check["passed"] for check in demo_checks.values())
    conditional_checks = {
        "linear_pass": linear_pass,
        "watch_improvement": all(
            check["improved"] for check in watch_checks.values()
        ),
        "gate_max": gate_max <= THRESHOLDS["conditional_gate_max_centered"],
        "any_watch_worse": any(
            check["improvement"] < 0 for check in watch_checks.values()
        ),
    }
    if linear_pass and demo_pass:
        status = "go"
        rationale = (
            "The selected input clears the pre-registered linear and demo "
            "screens. Pre-register one Phase 2 adapter side-channel candidate."
        )
    elif (
        linear_pass
        and conditional_checks["watch_improvement"]
        and conditional_checks["gate_max"]
        and not conditional_checks["any_watch_worse"]
    ):
        status = "conditional"
        rationale = (
            "The selected input clears the linear screen and improves both "
            "watch pairs, but the demo style screen is not fully met. Allow "
            "one Phase 2 retrain; if it fails either fixed gate, move to "
            "paired data."
        )
    else:
        status = "no_go"
        rationale = (
            "The selected input does not clear the pre-registered linear/demo "
            "screen, or a watch pair regresses. Do not retrain the adapter on "
            "the same supervised data; move to paired face/voice data."
        )
    return {
        "status": status,
        "selected_representation": candidate_name,
        "linear_checks": linear_checks,
        "linear_pass": bool(linear_pass),
        "demo_checks": demo_checks,
        "demo_pass": bool(demo_pass),
        "conditional_checks": conditional_checks,
        "rationale": rationale,
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument(
        "--style-targets", type=Path, default=DEFAULT_STYLE_TARGETS
    )
    parser.add_argument("--fused-scaler", type=Path, default=DEFAULT_FUSED_SCALER)
    parser.add_argument(
        "--arcface-scaler", type=Path, default=DEFAULT_ARCFACE_SCALER
    )
    parser.add_argument("--cfvn", type=Path, default=DEFAULT_CFVN)
    parser.add_argument(
        "--cfvn-arcface", type=Path, default=DEFAULT_CFVN_ARCFACE
    )
    parser.add_argument("--cvae", type=Path, default=DEFAULT_CVAE)
    parser.add_argument("--face-report", type=Path, default=DEFAULT_FACE_REPORT)
    parser.add_argument(
        "--demo-labels",
        nargs="+",
        default=None,
        help="Demo labels to include; default is every face in the report.",
    )
    parser.add_argument(
        "--ridge-lambdas",
        type=float,
        nargs="+",
        default=DEFAULT_LAMBDAS,
    )
    parser.add_argument("--demo-cache", type=Path, default=DEFAULT_DEMO_CACHE)
    parser.add_argument(
        "--reuse-demo-features",
        action="store_true",
        help="Reuse the demo feature cache when its hashes still match.",
    )
    parser.add_argument(
        "--arcface-device",
        choices=("cpu", "cuda"),
        default="cpu",
        help="Execution provider for demo ArcFace extraction.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    _validate_args(args)
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    device = (
        args.device
        if args.device != "cuda" or torch.cuda.is_available()
        else "cpu"
    )

    started = datetime.now(timezone.utc)
    print(f"[ceiling] device={device}", flush=True)
    dataset, blocks, artifacts = _load_blocks(args, device)
    style_payload = load_style_targets(args.style_targets)
    targets, style_stats = align_style_targets(
        dataset.speaker_ids, style_payload
    )

    split_speakers: Dict[str, List[str]] = {}
    for split in ("train", "val", "test"):
        mask = dataset.split == split
        split_speakers[split] = sorted(
            {str(value) for value in dataset.speaker_ids[mask]}
        )

    train_mask = dataset.split == "train"
    train_targets = _speaker_targets(
        targets[train_mask],
        dataset.speaker_ids[train_mask],
        split_speakers["train"],
    )
    center = train_targets.mean(dim=0).float()
    target_centered_norm = float(
        (train_targets - center).norm(dim=-1).mean()
    )

    report: Dict[str, Any] = {
        "schema_version": "voix.face_encoder_ceiling.v1",
        "created_at": started.strftime("%Y-%m-%d %H:%M:%S UTC"),
        "level": "diagnostic",
        "pre_registration": "docs/IDENTITY_NEXT_PHASE.md",
        "protocol": {
            "unit": "one row per speaker",
            "selection": "validation residual cosine only",
            "test_used_for_selection": False,
            "ridge_lambdas": [float(value) for value in args.ridge_lambdas],
            "target_center": "train speaker-level target mean",
            "standardization": "train speaker-centroid mean/std",
            "thresholds": dict(THRESHOLDS),
            "representations": {
                name: list(blocks_requested)
                for name, blocks_requested in REPRESENTATIONS
            },
        },
        "artifacts": {
            "features": str(args.features),
            "index": str(args.index),
            "style_targets": str(args.style_targets),
            "fused_scaler": str(args.fused_scaler),
            "arcface_scaler": str(args.arcface_scaler),
            "cfvn": str(args.cfvn),
            "cfvn_arcface": str(args.cfvn_arcface),
            "cvae": str(args.cvae),
            "face_report": str(args.face_report),
            "demo_cache": str(args.demo_cache),
            "features_sha256": _sha256(args.features),
            "style_targets_sha256": _sha256(args.style_targets),
            "cfvn_sha256": _sha256(args.cfvn),
            "cfvn_arcface_sha256": _sha256(args.cfvn_arcface),
            "cvae_sha256": _sha256(args.cvae),
            "face_report_sha256": _sha256(args.face_report),
        },
        "style_targets": style_stats,
        "target_center_norm": float(center.norm()),
        "target_centered_norm": target_centered_norm,
        "splits": {
            split: {
                "clips": int((dataset.split == split).sum()),
                "speakers": len(split_speakers[split]),
            }
            for split in ("train", "val", "test")
        },
        "representations": {},
        "limitations": list(LIMITATIONS),
    }

    for name, block_names in REPRESENTATIONS:
        print(f"[ceiling] fitting {name} ({block_names})", flush=True)
        result = _evaluate_representation(
            blocks,
            block_names,
            dataset,
            targets,
            center,
            split_speakers,
            args.ridge_lambdas,
        )
        report["representations"][name] = result

    # The candidate is chosen by validation residual cosine only.
    selected_name = max(
        report["representations"],
        key=lambda name: report["representations"][name]["splits"]["val"][
            "residual_cosine"
        ],
    )
    print(f"[ceiling] selected by validation residual cosine: {selected_name}", flush=True)

    face_report = json.loads(args.face_report.read_text(encoding="utf-8"))
    entries = {str(entry["label"]): entry for entry in face_report["entries"]}
    label_by_id = {
        "_".join(label.split("_")[:2]): label for label in entries
    }
    if args.demo_labels is None:
        demo_labels = list(entries)
    else:
        missing = [label for label in args.demo_labels if label not in entries]
        if missing:
            raise KeyError(f"demo labels missing from report: {missing}")
        demo_labels = list(dict.fromkeys(args.demo_labels))
    missing_gate = [
        face_id for face_id in GATE_FACE_IDS if face_id not in label_by_id
    ]
    if missing_gate:
        raise KeyError(f"gate faces missing from report: {missing_gate}")
    missing_gate_labels = [
        label_by_id[face_id]
        for face_id in GATE_FACE_IDS
        if label_by_id[face_id] not in demo_labels
    ]
    if missing_gate_labels:
        raise ValueError(
            "--demo-labels must include every gate face; missing "
            f"{missing_gate_labels}"
        )

    demo_rows = _load_demo_rows(
        args, demo_labels, entries, artifacts, device
    )
    demo_blocks = _demo_blocks(demo_rows, artifacts, device)
    variant_models = {
        name: report["representations"][name]["_ridge"]
        for name, _blocks in REPRESENTATIONS
    }
    demo_metrics = _demo_pair_metrics(
        label_by_id,
        demo_blocks,
        demo_rows,
        entries,
        variant_models,
        center,
        GATE_FACE_IDS,
    )

    train_arcface = blocks["arcface"][train_mask]
    train_speakers = dataset.speaker_ids[train_mask]
    overlap = _identity_overlap(demo_rows, train_arcface, train_speakers)

    baseline = report["representations"]["cfvn_es"]
    candidate = report["representations"][selected_name]
    demo_for_decision = {
        "baseline_watch_pairs": demo_metrics["watch_pairs"],
        "candidate_watch_pairs": demo_metrics["watch_pairs"],
        "candidate_gate_summary": demo_metrics["gate_summary"],
    }
    decision = _decision(
        baseline, selected_name, candidate, demo_for_decision
    )

    # Descriptive only: what the pre-registered matrix would say for every
    # screened input.  The selected candidate remains the one chosen by
    # validation residual cosine; this block does not authorize a Phase 2 run.
    selection_sensitivity: Dict[str, Any] = {}
    for name, representation in report["representations"].items():
        variant_demo = {
            "baseline_watch_pairs": demo_metrics["watch_pairs"],
            "candidate_watch_pairs": demo_metrics["watch_pairs"],
            "candidate_gate_summary": demo_metrics["gate_summary"],
        }
        selection_sensitivity[name] = _decision(
            baseline, name, representation, variant_demo
        )

    # The raw ridge weights are large and are not needed in the report.
    for name in report["representations"]:
        report["representations"][name].pop("_ridge", None)

    report["demo"] = {
        "labels": demo_labels,
        "gate_face_ids": list(GATE_FACE_IDS),
        "watch_pairs": [_pair_key(a, b) for a, b in WATCH_PAIRS],
        "features": demo_metrics,
        "identity_overlap": overlap,
    }
    report["decision"] = decision
    report["selection_sensitivity"] = selection_sensitivity
    report["selected_representation"] = selected_name
    report["completed_at"] = datetime.now(timezone.utc).strftime(
        "%Y-%m-%d %H:%M:%S UTC"
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        f"[ceiling] decision={decision['status']} selected={selected_name}",
        flush=True,
    )
    print(f"Wrote {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
