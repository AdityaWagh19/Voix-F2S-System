"""Shared helpers for FaceFeatureV2 trainers.

Keeping representation, batching, and style-target matching in one place makes
the CFVN/CVAE/adapter pipeline use exactly the same face contract.  These
helpers are deliberately small and side-effect free so they can be unit-tested
without a GPU or a real corpus.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from voix.models.contrastive_net import ContrastiveFaceVoiceNet


def normalize_face(face_raw: np.ndarray, scaler: dict) -> torch.Tensor:
    """Apply the canonical FaceFeatureV2 z-score to raw face features."""
    mean = scaler["face_mean"].float()
    std = torch.clamp(scaler["face_std"].float(), min=1e-6)
    face = torch.as_tensor(face_raw, dtype=torch.float32)
    if face.ndim != 2 or face.shape[1] != mean.shape[0]:
        raise ValueError(
            f"face features have shape {tuple(face.shape)} but scaler expects "
            f"(*, {mean.shape[0]})"
        )
    return (face - mean) / std


@torch.no_grad()
def embed_faces(
    cfvn: ContrastiveFaceVoiceNet,
    face: torch.Tensor,
    device: str,
    batch_size: int = 1024,
) -> torch.Tensor:
    """Map normalized face features to CFVN voice-space embeddings."""
    cfvn.eval()
    outputs = []
    for start in range(0, len(face), batch_size):
        batch = face[start:start + batch_size].to(device)
        outputs.append(cfvn(batch, normalize=True).cpu())
    return torch.cat(outputs, dim=0) if outputs else face.new_zeros((0, cfvn.out_dim))


def speaker_codes(speaker_ids: np.ndarray) -> Tuple[np.ndarray, list[str]]:
    """Encode string speaker ids as stable integers for batch masks."""
    speakers = sorted({str(s) for s in speaker_ids})
    mapping = {speaker: index for index, speaker in enumerate(speakers)}
    codes = np.asarray([mapping[str(s)] for s in speaker_ids], dtype=np.int64)
    return codes, speakers


def query_gallery_indices(speakers: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Per speaker: first clip is the query, the rest form the gallery.

    Speakers with a single clip are skipped.  Putting a singleton in both the
    query and gallery would create a guaranteed self-match and inflate recall.
    """
    groups: Dict[str, list[int]] = {}
    for index, speaker in enumerate(speakers):
        groups.setdefault(str(speaker), []).append(index)
    query: list[int] = []
    gallery: list[int] = []
    for indices in groups.values():
        if len(indices) < 2:
            continue
        query.append(indices[0])
        gallery.extend(indices[1:])
    return np.asarray(query, dtype=np.int64), np.asarray(gallery, dtype=np.int64)


def different_speaker_mask(codes: torch.Tensor) -> torch.Tensor:
    """Boolean mask selecting pairs that belong to different speakers."""
    codes = codes.view(-1)
    return codes[:, None] != codes[None, :]


def variance_covariance_terms(
    embeddings: torch.Tensor,
    target_std: float = 0.05,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Variance floor and off-diagonal covariance penalty.

    These are the same terms used by the CFVN trainer.  A collapsed embedding
    has near-zero per-dimension variance and highly correlated dimensions, so
    both quantities are useful collapse diagnostics in addition to losses.
    """
    if embeddings.shape[0] < 2:
        zero = embeddings.sum() * 0.0
        return zero, zero
    std = embeddings.std(dim=0)
    var_loss = F.relu(target_std - std).mean()
    centered = embeddings - embeddings.mean(dim=0, keepdim=True)
    covariance = centered.T @ centered / max(1, embeddings.shape[0] - 1)
    off_diagonal = covariance - torch.diag(torch.diag(covariance))
    cov_loss = off_diagonal.pow(2).sum() / embeddings.shape[1]
    return var_loss, cov_loss


def pairwise_identity_stats(
    embeddings: torch.Tensor,
    codes: torch.Tensor,
) -> Dict[str, float]:
    """Within-speaker, across-speaker, and margin cosine statistics."""
    x = F.normalize(embeddings.float(), dim=-1)
    codes = codes.view(-1).to(x.device)
    similarity = x @ x.T
    n = len(x)
    off_diagonal = ~torch.eye(n, dtype=torch.bool, device=x.device)
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
    return {
        "within_cosine": within,
        "across_cosine": across,
        "identity_margin": within - across,
        "per_dim_std": float(x.std(dim=0).mean()),
    }


def load_style_targets(path: Path | str) -> dict:
    """Load a ``StyleTargetV2`` per-speaker style-target artifact.

    The artifact stores one 256-D ``[timbre | prosody]`` vector per speaker,
    extracted from that speaker's own VoxCeleb2 audio by the frozen StyleTTS2
    style and predictor encoders.  Unlike the legacy ``style_pairs.pt``
    artifact, the targets are aligned with the voice that produced them and
    carry speaker labels, so the train/val/test split can be enforced.
    """
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} is not a style-target dictionary")
    schema = str(payload.get("schema_version", ""))
    if schema != "StyleTargetV2":
        raise ValueError(
            f"{path} has schema_version={schema!r}; expected 'StyleTargetV2'. "
            "Re-extract it with scripts/data/extract_style_targets_v2.py"
        )
    for key in ("style", "speaker_ids"):
        if key not in payload:
            raise ValueError(f"{path} is missing the {key!r} entry")
    style = torch.as_tensor(payload["style"], dtype=torch.float32)
    speaker_ids = [str(speaker) for speaker in payload["speaker_ids"]]
    if style.ndim != 2 or style.shape[1] != 256:
        raise ValueError(
            f"{path} stores style targets with shape {tuple(style.shape)}; "
            "expected (*, 256)"
        )
    if len(style) != len(speaker_ids):
        raise ValueError(
            f"{path} has {len(style)} style rows but {len(speaker_ids)} speaker ids"
        )
    if not torch.isfinite(style).all():
        raise ValueError(f"{path} contains non-finite style targets")
    if len(set(speaker_ids)) != len(speaker_ids):
        raise ValueError(f"{path} contains duplicate speaker ids")
    n_clips = [int(value) for value in payload.get("n_clips", [1] * len(style))]
    if len(n_clips) != len(style):
        raise ValueError(
            f"{path} has {len(n_clips)} n_clips entries but {len(style)} style rows"
        )
    if any(value < 1 for value in n_clips):
        raise ValueError(f"{path} contains a speaker with n_clips < 1")
    return {
        "style": style,
        "speaker_ids": speaker_ids,
        "clip_ids": [str(value) for value in payload.get("clip_ids", [])],
        "split": [str(value) for value in payload.get("split", [])],
        "gender": [str(value) for value in payload.get("gender", [])],
        "n_clips": n_clips,
        "clips_per_speaker": int(payload.get("clips_per_speaker", 1)),
        "source": str(payload.get("source", "")),
        "schema_version": schema,
        "path": str(path),
    }


def align_style_targets(
    speaker_ids: np.ndarray,
    payload: dict,
    *,
    require_all: bool = True,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Expand per-speaker style targets to every dataset row.

    ``require_all`` is the training contract: a missing train/validation
    speaker would silently receive a zero target, so the default is to fail
    loudly instead of training on an incomplete map.
    """
    style = payload["style"]
    target_speakers = payload["speaker_ids"]
    lookup = {speaker: index for index, speaker in enumerate(target_speakers)}
    rows = [str(speaker) for speaker in speaker_ids]
    missing = sorted({speaker for speaker in rows if speaker not in lookup})
    if missing and require_all:
        preview = ", ".join(missing[:5])
        raise ValueError(
            f"{len(missing)} dataset speakers have no style target ({preview}"
            f"{', ...' if len(missing) > 5 else ''}); re-run "
            "scripts/data/extract_style_targets_v2.py for the missing speakers"
        )
    if missing:
        return torch.zeros((len(rows), style.shape[1])), {
            "style_target_speakers": 0,
            "style_target_missing_speakers": float(len(missing)),
        }

    index = torch.as_tensor([lookup[speaker] for speaker in rows], dtype=torch.long)
    targets = style.index_select(0, index)
    unique = torch.unique(index, sorted=True)
    unique_targets = style.index_select(0, unique)
    normalized = F.normalize(unique_targets, dim=-1)
    similarity = normalized @ normalized.T
    n = len(unique_targets)
    off_diagonal = ~torch.eye(n, dtype=torch.bool)
    across = (
        float(similarity[off_diagonal].mean()) if n > 1 else float("nan")
    )
    centered = unique_targets - unique_targets.mean(dim=0, keepdim=True)
    centered_normalized = F.normalize(centered, dim=-1, eps=1e-6)
    centered_similarity = centered_normalized @ centered_normalized.T
    centered_across = (
        float(centered_similarity[off_diagonal].mean())
        if n > 1
        else float("nan")
    )
    stats = {
        "style_target_speakers": float(n),
        "style_target_clips": float(len(rows)),
        "style_target_missing_speakers": 0.0,
        "style_target_across_cosine": across,
        "style_target_centered_cosine": centered_across,
        "style_target_mean_norm": float(unique_targets.mean(dim=0).norm()),
        "style_target_centered_norm_mean": float(
            centered.norm(dim=-1).mean()
        ),
        "style_target_per_dim_std": float(unique_targets.std(dim=0).mean()),
        "style_target_norm_mean": float(unique_targets.norm(dim=-1).mean()),
        "style_target_mean_clips": float(
            np.mean(payload.get("n_clips", [1]))
        ),
        "style_target_clips_per_speaker": float(
            payload.get("clips_per_speaker", 1)
        ),
    }
    return targets, stats


@torch.no_grad()
def match_style_targets(
    speaker_features: np.ndarray,
    speaker_ids: np.ndarray,
    style_pairs: dict,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Match each V2 speaker to a frozen StyleTTS2 style target.

    Matching is performed on the *speaker centroid* rather than per clip.  A
    centroid is a more stable ECAPA estimate than a single utterance, and it
    guarantees every clip from one speaker receives the same style target.

    The style-pair artifact has no speaker labels, so this function cannot prove
    that its source recordings are disjoint from the V2 train/val speakers.
    Callers must record that limitation in the training report; the face/voice
    split itself remains speaker-disjoint.
    """
    voice = F.normalize(torch.as_tensor(speaker_features, dtype=torch.float32), dim=-1)
    style_ecapa = F.normalize(torch.as_tensor(style_pairs["ecapa"], dtype=torch.float32), dim=-1)
    timbre = torch.as_tensor(style_pairs["timbre"], dtype=torch.float32)
    prosody = torch.as_tensor(style_pairs["prosody"], dtype=torch.float32)

    groups: Dict[str, list[int]] = {}
    for index, speaker in enumerate(speaker_ids):
        groups.setdefault(str(speaker), []).append(index)

    targets = torch.empty(len(voice), timbre.shape[1] + prosody.shape[1], dtype=torch.float32)
    similarities = []
    for speaker, indices in groups.items():
        centroid = F.normalize(voice[indices].mean(dim=0), dim=-1)
        similarity = centroid @ style_ecapa.T
        best = int(torch.argmax(similarity).item())
        similarities.append(float(similarity[best].item()))
        target = torch.cat([timbre[best], prosody[best]], dim=-1)
        targets[indices] = target

    stats = {
        "style_match_cosine_mean": float(np.mean(similarities)),
        "style_match_cosine_min": float(np.min(similarities)),
        "style_match_cosine_max": float(np.max(similarities)),
        "style_match_speakers": len(similarities),
    }
    return targets, stats
