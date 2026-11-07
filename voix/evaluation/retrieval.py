"""Speaker-disjoint retrieval and embedding-health metrics.

These metrics are the quality gate for the face-to-voice stack. They are used by
the CFVN/CVAE/adapter trainers and by the evaluation harness. All functions are
pure PyTorch/NumPy and do not load models implicitly.
"""
from __future__ import annotations

from typing import Dict, Iterable, Tuple

import numpy as np
import torch
import torch.nn.functional as F


def l2_normalize(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return F.normalize(x.float(), p=2, dim=-1, eps=eps)


def speaker_centroids(
    features: torch.Tensor,
    speaker_ids: Iterable,
) -> Tuple[torch.Tensor, list]:
    """Return L2-normalized mean embedding per speaker.

    Args:
        features: (N, D) embeddings.
        speaker_ids: length-N speaker labels.

    Returns:
        (S, D) centroids and the sorted speaker label list.
    """
    features = features.float().cpu()
    ids = list(speaker_ids)
    if len(ids) != len(features):
        raise ValueError("speaker_ids length does not match features")
    unique = sorted(set(ids))
    index = {spk: i for i, spk in enumerate(unique)}
    sums = torch.zeros(len(unique), features.shape[-1], dtype=torch.float32)
    counts = torch.zeros(len(unique), 1, dtype=torch.float32)
    for feat, spk in zip(features, ids):
        row = index[spk]
        sums[row] += feat
        counts[row] += 1.0
    return l2_normalize(sums / counts.clamp(min=1.0)), unique


def speaker_retrieval(
    query_features: torch.Tensor,
    query_speakers: Iterable,
    gallery_features: torch.Tensor,
    gallery_speakers: Iterable,
    ks: Tuple[int, ...] = (1, 5, 10),
) -> Dict[str, float]:
    """Speaker-level cross-modal retrieval (mean query vs mean gallery)."""
    q_cent, q_ids = speaker_centroids(query_features, query_speakers)
    g_cent, g_ids = speaker_centroids(gallery_features, gallery_speakers)
    gallery_index = {spk: i for i, spk in enumerate(g_ids)}
    valid = [i for i, spk in enumerate(q_ids) if spk in gallery_index]
    if not valid:
        raise ValueError("No overlapping speakers between query and gallery")
    q = q_cent[valid]
    target = torch.tensor([gallery_index[q_ids[i]] for i in valid], dtype=torch.long)
    sim = q @ g_cent.T
    order = torch.argsort(sim, dim=1, descending=True)
    result: Dict[str, float] = {"n_speakers": float(len(valid))}
    for k in ks:
        topk = order[:, : min(k, order.shape[1])]
        result[f"recall_at_{k}"] = float((topk == target.unsqueeze(1)).any(1).float().mean())
    result["chance_at_1"] = 1.0 / max(1, len(g_ids))
    return result


def _speaker_gender_map(
    speakers: Iterable, genders: Iterable
) -> Dict[str, str]:
    speakers = list(speakers)
    genders = list(genders)
    if len(speakers) != len(genders):
        raise ValueError("speakers and genders must have the same length")
    mapping: Dict[str, str] = {}
    for speaker, gender in zip(speakers, genders):
        label = str(gender).strip().lower() or "unknown"
        previous = mapping.get(str(speaker))
        if previous is not None and previous != label:
            raise ValueError(
                f"Speaker {speaker!r} has inconsistent genders: "
                f"{previous!r} vs {label!r}"
            )
        mapping[str(speaker)] = label
    return mapping


def gender_controlled_retrieval(
    query_features: torch.Tensor,
    query_speakers: Iterable,
    query_genders: Iterable,
    gallery_features: torch.Tensor,
    gallery_speakers: Iterable,
    gallery_genders: Iterable,
    ks: Tuple[int, ...] = (1, 5, 10),
) -> Dict[str, float]:
    """Speaker retrieval with the gallery restricted to the query's gender.

    Fused face features contain gender/age channels, so an unrestricted
    face-to-voice score can be inflated by demographic matching rather than
    identity.  Restricting the gallery to same-gender speakers makes that
    shortcut uninformative; chance rises to roughly ``1 / n_same_gender`` and a
    model that only predicts demographics collapses to chance.
    """
    q_cent, q_ids = speaker_centroids(query_features, query_speakers)
    g_cent, g_ids = speaker_centroids(gallery_features, gallery_speakers)
    q_gender = _speaker_gender_map(query_speakers, query_genders)
    g_gender = _speaker_gender_map(gallery_speakers, gallery_genders)
    gallery_index = {spk: i for i, spk in enumerate(g_ids)}
    by_gender: Dict[str, list] = {}
    for index, speaker in enumerate(g_ids):
        by_gender.setdefault(g_gender[speaker], []).append(index)

    hits = {k: [] for k in ks}
    ranks = []
    chances = []
    for query_index, speaker in enumerate(q_ids):
        if speaker not in gallery_index:
            continue
        candidates = by_gender.get(q_gender[speaker], [])
        if len(candidates) < 2:
            continue
        sims = q_cent[query_index] @ g_cent[candidates].T
        order = torch.argsort(sims, descending=True)
        target_local = candidates.index(gallery_index[speaker])
        rank = int((order == target_local).nonzero(as_tuple=False)[0, 0])
        ranks.append(rank)
        chances.append(1.0 / len(candidates))
        for k in ks:
            hits[k].append(rank < k)

    if not ranks:
        result = {"n_speakers": 0.0, "chance_at_1": float("nan")}
        for k in ks:
            result[f"recall_at_{k}"] = float("nan")
        return result

    result = {
        "n_speakers": float(len(ranks)),
        "chance_at_1": float(np.mean(chances)),
        "mean_rank": float(np.mean(ranks)),
    }
    for k in ks:
        result[f"recall_at_{k}"] = float(np.mean(hits[k]))
    return result



def clip_retrieval(
    query_features: torch.Tensor,
    gallery_features: torch.Tensor,
    ks: Tuple[int, ...] = (1, 10),
) -> Dict[str, float]:
    """Clip-level retrieval with one query per gallery item."""
    q = l2_normalize(query_features)
    g = l2_normalize(gallery_features)
    if q.shape[0] != g.shape[0]:
        raise ValueError("Clip retrieval expects aligned query/gallery rows")
    sim = q @ g.T
    target = torch.arange(len(q))
    order = torch.argsort(sim, dim=1, descending=True)
    result = {"n": float(len(q)), "chance_at_1": 1.0 / max(1, len(q))}
    for k in ks:
        topk = order[:, : min(k, order.shape[1])]
        result[f"recall_at_{k}"] = float((topk == target.unsqueeze(1)).any(1).float().mean())
    return result


def embedding_health(features: torch.Tensor, speaker_ids: Iterable) -> Dict[str, float]:
    """Within/between speaker cosine, per-dim std, and collapse ratio."""
    features = features.float().cpu()
    ids = list(speaker_ids)
    if len(ids) != len(features):
        raise ValueError("speaker_ids length does not match features")
    normed = l2_normalize(features)
    sim = normed @ normed.T
    ids_arr = np.asarray(ids)
    same = ids_arr[:, None] == ids_arr[None, :]
    n = len(ids_arr)
    off_diag = ~np.eye(n, dtype=bool)
    within_mask = same & off_diag
    between_mask = (~same) & off_diag
    within = float(sim[within_mask].mean()) if within_mask.any() else float("nan")
    between = float(sim[between_mask].mean()) if between_mask.any() else float("nan")
    per_dim_std = float(normed.std(dim=0).mean())
    # A healthy 192-D unit sphere has per-dim std ~ 1/sqrt(192) = 0.072.
    collapse_ratio = per_dim_std / (1.0 / np.sqrt(features.shape[-1]))
    return {
        "within_speaker_cosine": within,
        "between_speaker_cosine": between,
        "per_dim_std": per_dim_std,
        "collapse_ratio": float(collapse_ratio),
    }


def shuffled_control(
    query_features: torch.Tensor,
    query_speakers: Iterable,
    gallery_features: torch.Tensor,
    gallery_speakers: Iterable,
    seed: int = 42,
) -> Dict[str, float]:
    """Negative control: shuffle query speakers, destroying the pairing."""
    q = list(query_speakers)
    rng = np.random.default_rng(seed)
    rng.shuffle(q)
    return speaker_retrieval(query_features, q, gallery_features, gallery_speakers)


def constant_control(
    gallery_features: torch.Tensor,
    gallery_speakers: Iterable,
    ks: Tuple[int, ...] = (1, 5, 10),
) -> Dict[str, float]:
    """Negative control: every query is the same constant vector.

    A retrieval metric that scores well here is measuring gallery order, not
    identity.  The expected R@1 is approximately ``1 / n_gallery_speakers``.
    """
    gallery_features = gallery_features.float().cpu()
    gallery_speakers = list(gallery_speakers)
    unique = sorted(set(gallery_speakers))
    if not unique:
        raise ValueError("constant_control needs at least one gallery speaker")
    queries = torch.ones(
        len(unique), gallery_features.shape[-1], dtype=torch.float32
    )
    return speaker_retrieval(queries, unique, gallery_features, gallery_speakers, ks=ks)
