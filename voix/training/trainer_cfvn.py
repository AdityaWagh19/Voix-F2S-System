"""Train ContrastiveFaceVoiceNet on the speaker-disjoint FaceFeatureV2 corpus.

The default face input is the 560-D fused representation.  Pass
``--face-key arcface`` to train the raw-ArcFace baseline; it uses a separate
scaler artifact so the canonical 560-D serving scaler is never overwritten.

Usage:
    python -m voix.training.trainer_cfvn --epochs 40 --batch-speakers 32 --batch-clips 4
    python -m voix.training.trainer_cfvn --face-key arcface \
        --output results/checkpoints/cfvn_v2_arcface_best.pt

The optional ``--lambda-struct`` term distills the raw-face similarity
distribution into the 192-D output, and ``--face-face-weight`` controls how
strongly checkpoint selection rewards identity preservation.  Both default to
the original behavior (0.0 and 0.10); experimental runs should write to a
separate ``--output`` so the canonical checkpoint is never overwritten.

Selection is based on held-out-speaker retrieval (R@1/R@10), never on mean
cosine. The script refuses to train if train/val/test speakers overlap.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from voix.data.feature_contract import load_face_scaler, load_scaler_payload
from voix.evaluation.retrieval import (
    embedding_health,
    gender_controlled_retrieval,
    shuffled_control,
    speaker_retrieval,
)
from voix.models.contrastive_net import ContrastiveFaceVoiceNet
from voix.training.data import (
    DEFAULT_FEATURES,
    DEFAULT_FEATURES_V2,
    DEFAULT_INDEX,
    DEFAULT_INDEX_V2,
    DEFAULT_SCALER,
    DEFAULT_SCALER_V2,
    CleanFeatureDataset,
    SpeakerBatchSampler,
    build_face_scaler_v2,
    load_clean_dataset,
    load_v2_dataset,
    normalize_speaker_features,
    split_summary,
    validate_scaler_for_dataset,
)
from voix.training.v2_utils import query_gallery_indices
from voix.utils.seed import set_global_seed

ROOT = Path(__file__).resolve().parent.parent.parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train CFVN on speaker-disjoint data")
    p.add_argument("--features", type=Path, default=DEFAULT_FEATURES_V2)
    p.add_argument("--index", type=Path, default=DEFAULT_INDEX_V2)
    p.add_argument("--scaler", type=Path, default=DEFAULT_SCALER_V2)
    p.add_argument(
        "--schema",
        choices=("auto", "v1", "v2"),
        default="auto",
        help="Corpus schema. 'auto' detects FaceFeatureV2 by the clip_ids dataset.",
    )
    p.add_argument(
        "--face-key",
        default="face_features",
        help="HDF5 face dataset for FaceFeatureV2 (face_features or arcface).",
    )
    p.add_argument("--output", type=Path, default=ROOT / "results" / "checkpoints" / "cfvn_v2_best.pt")
    p.add_argument(
        "--init",
        type=Path,
        default=None,
        help="Optional checkpoint to initialize the model for fine-tuning.",
    )
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-speakers", type=int, default=32)
    p.add_argument("--batch-clips", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--temperature", type=float, default=0.07)
    p.add_argument("--lambda-reg", type=float, default=5.0)
    p.add_argument(
        "--reg-target",
        choices=["centroid", "clip"],
        default="centroid",
        help=(
            "Regression target for voice alignment. 'centroid' averages each "
            "speaker's ECAPA embeddings; 'clip' uses the paired clip embedding."
        ),
    )
    p.add_argument("--lambda-var", type=float, default=0.05)
    p.add_argument("--lambda-cov", type=float, default=0.01)
    p.add_argument(
        "--lambda-struct",
        type=float,
        default=0.0,
        help=(
            "Weight for raw-face structure distillation. 0 preserves the "
            "original trainer; experimental runs should use a separate output."
        ),
    )
    p.add_argument(
        "--struct-temperature",
        type=float,
        default=0.1,
        help="Temperature for the raw-face similarity distillation.",
    )
    p.add_argument(
        "--face-face-weight",
        type=float,
        default=0.10,
        help="Selection-score weight for CFVN face-face R@1.",
    )
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--residual", action="store_true", default=True)
    p.add_argument("--no-residual", dest="residual", action="store_false")
    p.add_argument("--layernorm", action="store_true", default=False)
    return p.parse_args()


def resolve_scaler_path(args: argparse.Namespace) -> Path:
    """Return the scaler that belongs to the selected face representation.

    The canonical ``face_scaler_v2.pt`` is 560-D and is consumed by the serving
    pipeline.  Training on raw ArcFace must not rebuild it as a 512-D scaler.
    """
    scaler = Path(args.scaler)
    if args.face_key == "face_features":
        return scaler
    canonical = Path(DEFAULT_SCALER_V2)
    same_as_canonical = scaler.name == canonical.name
    try:
        same_as_canonical = same_as_canonical or (
            scaler.resolve() == canonical.resolve()
        )
    except OSError:
        pass
    if same_as_canonical:
        return scaler.with_name("face_scaler_arcface_v2.pt")
    return scaler


def resolve_or_build_scaler(
    args: argparse.Namespace,
    dataset: CleanFeatureDataset,
    face_dim: int,
) -> Tuple[Path, dict]:
    """Load the scaler for this corpus, rebuilding it when it is stale.

    A scaler that merely has the right dimension and face key can still have
    been fit on an earlier shard.  ``validate_scaler_for_dataset`` compares the
    recorded train row/speaker counts, so growing the corpus forces a rebuild
    instead of silently reusing the old normalization.
    """
    scaler_path = resolve_scaler_path(args)
    scaler = None
    if scaler_path.exists():
        try:
            payload = load_scaler_payload(
                scaler_path,
                expected_dim=face_dim,
                face_key=args.face_key,
            )
            scaler = validate_scaler_for_dataset(
                payload, dataset, face_key=args.face_key
            )
        except (ValueError, KeyError) as exc:
            print(f"Rebuilding incompatible scaler {scaler_path}: {exc}")
    if scaler is None:
        scaler = build_face_scaler_v2(
            dataset,
            scaler_path,
            source=str(args.features),
            face_key=args.face_key,
        )
    return scaler_path, scaler


def normalize_faces(face_raw: np.ndarray, scaler: dict) -> torch.Tensor:
    mean = scaler["face_mean"].float()
    std = torch.clamp(scaler["face_std"].float(), min=1e-6)
    return (torch.from_numpy(face_raw).float() - mean) / std


def speaker_centroid_targets(
    embeddings: torch.Tensor,
    speaker_ids: np.ndarray,
) -> torch.Tensor:
    """Replace each row with its L2-normalized speaker centroid.

    Cross-modal retrieval is scored at speaker level, so regressing to a noisy
    per-clip ECAPA embedding teaches the model to reproduce channel variation
    that the face cannot see.  The centroid is the correct target.
    """
    embeddings = embeddings.float()
    ids = [str(s) for s in speaker_ids]
    unique = sorted(set(ids))
    index = {speaker: i for i, speaker in enumerate(unique)}
    codes = torch.tensor([index[s] for s in ids], dtype=torch.long)
    sums = torch.zeros(len(unique), embeddings.shape[1], dtype=torch.float32)
    sums.index_add_(0, codes, embeddings)
    counts = torch.zeros(len(unique), dtype=torch.float32)
    counts.index_add_(0, codes, torch.ones(len(ids), dtype=torch.float32))
    centroids = F.normalize(sums / counts.clamp(min=1.0).unsqueeze(1), dim=-1)
    return centroids[codes]


def supervised_contrastive_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """SupCon loss with all same-speaker pairs as positives."""
    if embeddings.shape[0] < 2:
        return embeddings.sum() * 0.0
    sim = (embeddings @ embeddings.T) / temperature
    n = sim.shape[0]
    self_mask = torch.eye(n, dtype=torch.bool, device=sim.device)
    pos_mask = (labels[:, None] == labels[None, :]) & ~self_mask
    logits_mask = ~self_mask
    sim = sim.masked_fill(~logits_mask, -1e9)
    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    pos_count = pos_mask.sum(dim=1)
    valid = pos_count > 0
    if not valid.any():
        return embeddings.sum() * 0.0
    mean_log_prob = (log_prob * pos_mask).sum(dim=1)[valid] / pos_count[valid]
    return -mean_log_prob.mean()


def structure_preservation_loss(
    embeddings: torch.Tensor,
    reference: torch.Tensor,
    temperature: float = 0.1,
) -> torch.Tensor:
    """Match the CFVN output's neighbor structure to the raw face space.

    The CFVN is trained to regress voice centroids, which improves cross-modal
    retrieval but slowly collapses face-face identity (validation face-face R@1
    falls from 0.882 to 0.813 over 28 epochs).  This term distills the raw face
    similarity distribution into the 192-D output, so the output keeps the
    identity neighbors that the raw 560-D face representation already has.

    Both inputs are L2-normalized internally; the diagonal is excluded.
    """
    if embeddings.shape[0] < 2:
        return embeddings.sum() * 0.0
    out = F.normalize(embeddings, dim=-1)
    ref = F.normalize(reference, dim=-1)
    sim_out = out @ out.T
    sim_ref = ref @ ref.T
    n = sim_out.shape[0]
    mask = ~torch.eye(n, dtype=torch.bool, device=sim_out.device)
    log_q = torch.log_softmax(
        sim_out[mask].view(n, n - 1) / temperature, dim=1
    )
    p = torch.softmax(sim_ref[mask].view(n, n - 1) / temperature, dim=1)
    return F.kl_div(log_q, p, reduction="batchmean")


def variance_covariance_terms(embeddings: torch.Tensor, target_std: float = 0.05) -> Tuple[torch.Tensor, torch.Tensor]:
    std = embeddings.std(dim=0)
    var_loss = F.relu(target_std - std).mean()
    centered = embeddings - embeddings.mean(dim=0, keepdim=True)
    cov = centered.T @ centered / max(1, embeddings.shape[0] - 1)
    off_diag = cov - torch.diag(torch.diag(cov))
    cov_loss = off_diag.pow(2).sum() / embeddings.shape[1]
    return var_loss, cov_loss


@torch.no_grad()
def embed_all(model: ContrastiveFaceVoiceNet, features: torch.Tensor, device: str, batch: int = 1024) -> torch.Tensor:
    if len(features) == 0:
        return torch.zeros((0, model.out_dim), dtype=torch.float32)
    outs = []
    model.eval()
    for i in range(0, len(features), batch):
        outs.append(model(features[i:i + batch].to(device)).cpu())
    return torch.cat(outs, dim=0)


def fit_linear_probe(
    face_train: torch.Tensor,
    speaker_train: torch.Tensor,
    ridge: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Closed-form ridge regression baseline from face features to ECAPA space."""
    x = face_train.float()
    y = speaker_train.float()
    ones = torch.ones(len(x), 1)
    x_aug = torch.cat([x, ones], dim=1)
    gram = x_aug.T @ x_aug
    reg = torch.eye(gram.shape[0]) * ridge
    reg[-1, -1] = 0.0  # do not regularize the bias
    weights = torch.linalg.solve(gram + reg, x_aug.T @ y)
    return weights[:-1], weights[-1]


def apply_linear_probe(face: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    return F.normalize(face.float() @ weight + bias, dim=-1)


def query_gallery_split(speakers: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Per speaker: first clip as query, remaining clips as gallery.

    Singleton speakers are excluded so a clip can never match itself.
    """
    return query_gallery_indices(speakers)


@torch.no_grad()
def evaluate(
    model: ContrastiveFaceVoiceNet,
    dataset: CleanFeatureDataset,
    face_norm: torch.Tensor,
    speaker_norm: torch.Tensor,
    device: str,
    face_identity: torch.Tensor | None = None,
    face_face_weight: float = 0.10,
) -> Dict[str, float]:
    speakers = dataset.speaker_ids
    q_idx, g_idx = query_gallery_split(speakers)
    if len(q_idx) == 0 or len(g_idx) == 0:
        return {
            "val_r1": float("nan"),
            "val_r5": float("nan"),
            "val_r10": float("nan"),
            "val_chance_r1": float("nan"),
            "val_shuffled_r1": float("nan"),
            "raw_face_face_r1": float("nan"),
            "raw_face_face_r10": float("nan"),
            "cfvn_face_face_r1": float("nan"),
            "cfvn_face_face_r10": float("nan"),
            "val_gender_controlled_r1": float("nan"),
            "val_gender_controlled_r10": float("nan"),
            "val_gender_controlled_chance_r1": float("nan"),
            "cfvn_face_face_gender_controlled_r1": float("nan"),
            "raw_face_face_gender_controlled_r1": float("nan"),
            "within_speaker_cosine": float("nan"),
            "between_speaker_cosine": float("nan"),
            "per_dim_std": float("nan"),
            "collapse_ratio": float("nan"),
            "selection_score": float("nan"),
            "n_eval_speakers": 0.0,
        }

    # Query and gallery must be different clips.  Passing the same rows as both
    # makes every speaker centroid match itself and reports R@1 = 1.0.
    out = embed_all(model, face_norm, device)
    voice_retrieval = speaker_retrieval(
        out[q_idx], speakers[q_idx], speaker_norm[g_idx], speakers[g_idx]
    )
    health = embedding_health(out, speakers)
    shuffled = shuffled_control(
        out[q_idx], speakers[q_idx], speaker_norm[g_idx], speakers[g_idx]
    )

    identity = face_identity if face_identity is not None else face_norm
    raw_face_face = speaker_retrieval(
        identity[q_idx], speakers[q_idx], identity[g_idx], speakers[g_idx]
    )
    # Does the CFVN preserve identity across different faces of one speaker?
    # This is the metric that can actually detect a CFVN-induced collapse.
    cfvn_face_face = speaker_retrieval(
        out[q_idx], speakers[q_idx], out[g_idx], speakers[g_idx]
    )
    # Demographic-shortcut control: the fused representation carries gender
    # and age channels, so restrict the gallery to same-gender speakers.  A
    # model that only learned demographics drops to chance here.
    gender_voice = gender_controlled_retrieval(
        out[q_idx],
        speakers[q_idx],
        dataset.gender[q_idx],
        speaker_norm[g_idx],
        speakers[g_idx],
        dataset.gender[g_idx],
    )
    gender_cfvn_face_face = gender_controlled_retrieval(
        out[q_idx],
        speakers[q_idx],
        dataset.gender[q_idx],
        out[g_idx],
        speakers[g_idx],
        dataset.gender[g_idx],
    )
    gender_raw_face_face = gender_controlled_retrieval(
        identity[q_idx],
        speakers[q_idx],
        dataset.gender[q_idx],
        identity[g_idx],
        speakers[g_idx],
        dataset.gender[g_idx],
    )

    metrics = {
        "val_r1": voice_retrieval["recall_at_1"],
        "val_r5": voice_retrieval["recall_at_5"],
        "val_r10": voice_retrieval["recall_at_10"],
        "val_chance_r1": voice_retrieval["chance_at_1"],
        "val_shuffled_r1": shuffled["recall_at_1"],
        "raw_face_face_r1": raw_face_face["recall_at_1"],
        "raw_face_face_r10": raw_face_face["recall_at_10"],
        "cfvn_face_face_r1": cfvn_face_face["recall_at_1"],
        "cfvn_face_face_r10": cfvn_face_face["recall_at_10"],
        "val_gender_controlled_r1": gender_voice["recall_at_1"],
        "val_gender_controlled_r10": gender_voice["recall_at_10"],
        "val_gender_controlled_chance_r1": gender_voice["chance_at_1"],
        "cfvn_face_face_gender_controlled_r1": gender_cfvn_face_face[
            "recall_at_1"
        ],
        "raw_face_face_gender_controlled_r1": gender_raw_face_face[
            "recall_at_1"
        ],
        "n_eval_speakers": voice_retrieval["n_speakers"],
        **health,
    }
    metrics["selection_score"] = (
        metrics["val_r1"]
        + 0.25 * metrics["val_r10"]
        + face_face_weight * metrics["cfvn_face_face_r1"]
    )
    return metrics


def main() -> None:
    args = parse_args()
    set_global_seed(args.seed)
    device = args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"

    import h5py

    schema = args.schema
    if schema == "auto":
        with h5py.File(args.features, "r") as store:
            schema = "v2" if "clip_ids" in store else "v1"

    print("=" * 72)
    print(f"CFVN training — {schema.upper()} / speaker-disjoint")
    print("=" * 72)
    if schema == "v2":
        dataset = load_v2_dataset(
            args.features,
            args.index,
            face_key=args.face_key,
        )
        face_dim = int(dataset.face_raw.shape[1])
        scaler_path, scaler = resolve_or_build_scaler(args, dataset, face_dim)
    else:
        dataset = load_clean_dataset(args.features, args.index)
        face_dim = int(dataset.face_raw.shape[1])
        scaler_path = Path(args.scaler)
        scaler = load_face_scaler(scaler_path)
    print("Split summary:", json.dumps(split_summary(dataset), indent=None))
    print(
        f"Scaler: {scaler_path} (schema={scaler.get('schema_version')}, "
        f"key={scaler.get('face_key', args.face_key)}, "
        f"n_train={scaler.get('n_train')}, speakers={scaler.get('n_train_speakers')})"
    )
    print(f"Face representation: {args.face_key} ({face_dim}-D)")

    train = dataset.subset("train")
    val = dataset.subset("val")
    # A per-shard smoke corpus can legitimately have no official-test speakers.
    # Validation is still required for checkpoint selection.
    test = dataset.subset("test", allow_empty=True)
    if len(val) == 0:
        raise ValueError("CFVN training requires a non-empty validation split")
    print(
        f"Train {len(train)} clips / {len(set(train.speaker_ids))} speakers | "
        f"Val {len(val)} / {len(set(val.speaker_ids))} | "
        f"Test {len(test)} / {len(set(test.speaker_ids))}"
    )

    face_train = normalize_faces(train.face_raw, scaler)
    face_val = normalize_faces(val.face_raw, scaler)
    face_test = normalize_faces(test.face_raw, scaler)
    # Identity-preservation baseline: L2-normalized raw face vectors, not the
    # z-scored model input.  This makes face-face R@1 comparable to the
    # extraction-time probe and avoids penalizing a healthy corpus for a
    # monotonic per-dimension transform.
    face_id_train = F.normalize(torch.from_numpy(train.face_raw).float(), dim=-1)
    face_id_val = F.normalize(torch.from_numpy(val.face_raw).float(), dim=-1)
    face_id_test = F.normalize(torch.from_numpy(test.face_raw).float(), dim=-1)
    spk_train = normalize_speaker_features(train.speaker_raw)
    spk_val = normalize_speaker_features(val.speaker_raw)
    spk_test = normalize_speaker_features(test.speaker_raw)
    spk_train_target = (
        speaker_centroid_targets(spk_train, train.speaker_ids)
        if args.reg_target == "centroid"
        else spk_train
    )
    print(f"Regression target: {args.reg_target}")

    # Baselines: linear face -> voice probe and raw face -> face. These make it
    # possible to tell whether CFVN adds anything over a simple linear map.
    probe_w, probe_b = fit_linear_probe(face_train, spk_train)
    probe_val = apply_linear_probe(face_val, probe_w, probe_b)
    q_idx, g_idx = query_gallery_split(val.speaker_ids)
    if len(q_idx) and len(g_idx):
        raw_cross = speaker_retrieval(
            probe_val[q_idx], val.speaker_ids[q_idx],
            spk_val[g_idx], val.speaker_ids[g_idx],
        )
        raw_face_face = speaker_retrieval(
            face_id_val[q_idx], val.speaker_ids[q_idx],
            face_id_val[g_idx], val.speaker_ids[g_idx],
        )
    else:
        raw_cross = {"recall_at_1": float("nan"), "chance_at_1": float("nan")}
        raw_face_face = {"recall_at_1": float("nan")}
    print(
        f"Linear-probe baseline: face->voice R@1={raw_cross['recall_at_1']:.4f} "
        f"(chance {raw_cross['chance_at_1']:.4f}) | "
        f"raw face->face R@1={raw_face_face['recall_at_1']:.4f}"
    )

    model = ContrastiveFaceVoiceNet(
        face_dim=face_dim,
        out_dim=192,
        use_layernorm=args.layernorm,
        residual=args.residual,
    ).to(device)
    if args.init is not None:
        payload = torch.load(args.init, map_location="cpu", weights_only=False)
        state = payload.get("model_state_dict", payload)
        model.load_state_dict(state)
        print(f"Initialized model from {args.init}")
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    sampler = SpeakerBatchSampler(
        train.speaker_to_indices(),
        speakers_per_batch=args.batch_speakers,
        clips_per_speaker=args.batch_clips,
        seed=args.seed,
    )
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))

    label_map = {spk: i for i, spk in enumerate(sorted(set(train.speaker_ids.tolist())))}
    best_score = -math.inf
    best_epoch = -1
    best_state = None
    history = []
    stale = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        losses = []
        parts = {"supcon": [], "reg": [], "var": [], "cov": [], "struct": []}
        for batch_idx in sampler:
            face = face_train[batch_idx].to(device)
            target = spk_train_target[batch_idx].to(device)
            labels = torch.tensor(
                [label_map[str(train.speaker_ids[i])] for i in batch_idx],
                dtype=torch.long,
                device=device,
            )
            out = model(face)
            supcon = supervised_contrastive_loss(out, labels, args.temperature)
            reg = (1.0 - F.cosine_similarity(out, target, dim=-1)).mean()
            var, cov = variance_covariance_terms(out)
            if args.lambda_struct > 0.0:
                struct = structure_preservation_loss(
                    out,
                    face_id_train[batch_idx].to(device),
                    args.struct_temperature,
                )
            else:
                struct = out.sum() * 0.0
            loss = (
                supcon
                + args.lambda_reg * reg
                + args.lambda_var * var
                + args.lambda_cov * cov
                + args.lambda_struct * struct
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            parts["supcon"].append(float(supcon.detach().cpu()))
            parts["reg"].append(float(reg.detach().cpu()))
            parts["var"].append(float(var.detach().cpu()))
            parts["cov"].append(float(cov.detach().cpu()))
            parts["struct"].append(float(struct.detach().cpu()))
        scheduler.step()

        metrics = evaluate(
            model,
            val,
            face_val,
            spk_val,
            device,
            face_id_val,
            face_face_weight=args.face_face_weight,
        )
        metrics.update({
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "train_supcon": float(np.mean(parts["supcon"])),
            "train_reg": float(np.mean(parts["reg"])),
            "train_var": float(np.mean(parts["var"])),
            "train_cov": float(np.mean(parts["cov"])),
            "train_struct": float(np.mean(parts["struct"])),
            "lr": float(scheduler.get_last_lr()[0]),
            "seconds": round(time.time() - t0, 1),
        })
        history.append(metrics)
        print(
            f"Epoch {epoch:3d}/{args.epochs} | loss {metrics['train_loss']:.4f} "
            f"(sup {metrics['train_supcon']:.3f} reg {metrics['train_reg']:.3f} "
            f"struct {metrics['train_struct']:.3f}) | "
            f"R@1 {metrics['val_r1']:.4f} (chance {metrics['val_chance_r1']:.4f}) "
            f"R@10 {metrics['val_r10']:.4f} | "
            f"gender-ctrl R@1 {metrics['val_gender_controlled_r1']:.4f} "
            f"(chance {metrics['val_gender_controlled_chance_r1']:.4f}) | "
            f"CFVN face-face R@1 {metrics['cfvn_face_face_r1']:.4f} | "
            f"raw face-face R@1 {metrics['raw_face_face_r1']:.4f} | "
            f"between {metrics['between_speaker_cosine']:.3f} | "
            f"std {metrics['per_dim_std']:.4f} | score {metrics['selection_score']:.4f}"
        )

        if metrics["selection_score"] > best_score + 1e-6:
            best_score = metrics["selection_score"]
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                print(f"Early stopping at epoch {epoch} (patience {args.patience})")
                break

    if best_state is None:
        raise RuntimeError("Training produced no checkpoint")

    model.load_state_dict(best_state)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": best_state,
            "architecture": model.architecture(),
            "epoch": best_epoch,
            "selection_score": best_score,
            "schema": schema,
            "face_key": args.face_key if schema == "v2" else "face_features",
            "face_dim": face_dim,
            "face_normalization": "zscore_train_split",
            "scaler_schema": scaler.get("schema_version", "FaceFeatureV2"),
            "scaler_path": str(scaler_path),
            "scaler_n_train": scaler.get("n_train"),
            "scaler_n_train_speakers": scaler.get("n_train_speakers"),
            "training": {
                "epochs": int(args.epochs),
                "batch_speakers": int(args.batch_speakers),
                "batch_clips": int(args.batch_clips),
                "lr": float(args.lr),
                "weight_decay": float(args.weight_decay),
                "temperature": float(args.temperature),
                "lambda_reg": float(args.lambda_reg),
                "lambda_var": float(args.lambda_var),
                "lambda_cov": float(args.lambda_cov),
                "lambda_struct": float(args.lambda_struct),
                "struct_temperature": float(args.struct_temperature),
                "face_face_weight": float(args.face_face_weight),
                "init": str(args.init) if args.init is not None else None,
                "reg_target": str(args.reg_target),
                "seed": int(args.seed),
            },
            "history": history,
        },
        args.output,
    )

    test_metrics = evaluate(
        model,
        test,
        face_test,
        spk_test,
        device,
        face_id_test,
        face_face_weight=args.face_face_weight,
    )
    test_metrics.update({
        "val_linear_face_to_voice_r1": raw_cross["recall_at_1"],
        "val_linear_face_to_voice_chance_r1": raw_cross["chance_at_1"],
        "val_raw_face_face_r1": raw_face_face["recall_at_1"],
        "best_epoch": best_epoch,
        "best_selection_score": best_score,
        "checkpoint": str(args.output),
    })
    eval_path = args.output.with_suffix(".eval.json")
    eval_path.write_text(json.dumps({"test": test_metrics, "history": history}, indent=2), encoding="utf-8")
    history_path = args.output.with_suffix(".history.json")
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")

    print("=" * 72)
    print(f"Best epoch {best_epoch} | val score {best_score:.4f}")
    if len(test) and math.isfinite(test_metrics.get("val_r1", float("nan"))):
        multiple = (
            test_metrics["val_r1"]
            / max(test_metrics.get("val_chance_r1", 0.0), 1e-9)
        )
        print(
            f"TEST: face->voice R@1 {test_metrics['val_r1']:.4f} "
            f"(chance {test_metrics['val_chance_r1']:.4f}, {multiple:.1f}x) | "
            f"R@10 {test_metrics['val_r10']:.4f} | "
            f"gender-ctrl R@1 {test_metrics['val_gender_controlled_r1']:.4f} "
            f"(chance {test_metrics['val_gender_controlled_chance_r1']:.4f}) | "
            f"CFVN face->face R@1 {test_metrics['cfvn_face_face_r1']:.4f} | "
            f"raw face->face R@1 {test_metrics['raw_face_face_r1']:.4f} | "
            f"shuffled R@1 {test_metrics['val_shuffled_r1']:.4f}"
        )
    else:
        print("TEST: unavailable (no held-out test clips in this corpus)")
    print(f"Saved: {args.output}")
    print(f"Eval:  {eval_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
