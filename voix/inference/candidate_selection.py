"""Deterministic candidate selection for one-to-many face-to-voice generation.

Sampling ``K`` voices independently from the conditional prior can return
near-duplicates: in the 15-asset demo, candidates 1 and 3 had ECAPA cosine
0.939.  This module selects a primary candidate closest to the conditional
mean and then greedily adds the candidates that are most dissimilar to the
already-selected set.

The selection operates in the same normalized speaker-embedding space the
pipeline uses, so it does not require an extra model or audio pass.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def select_diverse_embeddings(
    embeddings: torch.Tensor,
    k: int,
    reference: torch.Tensor | None = None,
) -> torch.Tensor:
    """Select ``k`` candidates: one central mode plus diverse alternatives.

    Args:
        embeddings: ``(N, D)`` candidate speaker embeddings, ``N >= k``.
        k: Number of candidates to return, ``k >= 1``.
        reference: Optional ``(D,)`` embedding that defines the primary mode
            (for example the decoded conditional-prior mean).  When omitted,
            the normalized mean of the candidate pool is used.

    Returns:
        ``(k, D)`` tensor.  Row 0 is the candidate closest to the reference
        mode.  Later rows maximize the minimum cosine distance to the
        already-selected rows.

    Raises:
        ValueError: if the input is not 2-D, ``k`` is out of range, the
            candidate count is smaller than ``k``, or ``reference`` has the
            wrong shape.
    """
    if embeddings.ndim != 2:
        raise ValueError(
            f"embeddings must be 2-D (N, D), got shape {tuple(embeddings.shape)}"
        )
    n, dim = int(embeddings.shape[0]), int(embeddings.shape[1])
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    if k > n:
        raise ValueError(f"k={k} exceeds the {n} available candidates")

    unit = F.normalize(embeddings.float(), dim=-1)
    if reference is None:
        mode = F.normalize(unit.mean(dim=0, keepdim=True), dim=-1)
    else:
        if reference.ndim == 2 and reference.shape[0] == 1:
            reference = reference.squeeze(0)
        if reference.ndim != 1 or int(reference.shape[0]) != dim:
            raise ValueError(
                f"reference must have shape ({dim},), got {tuple(reference.shape)}"
            )
        mode = F.normalize(reference.float().unsqueeze(0), dim=-1)
    centrality = unit @ mode.squeeze(0)

    # Deterministic tie-break: argmax/argmin return the first occurrence.
    primary = int(torch.argmax(centrality).item())
    selected = [primary]
    remaining = [index for index in range(n) if index != primary]

    while len(selected) < k:
        selected_unit = unit[selected]  # (m, D)
        similarities = unit[remaining] @ selected_unit.T  # (r, m)
        max_similarity = similarities.max(dim=1).values
        best_position = int(torch.argmin(max_similarity).item())
        selected.append(remaining.pop(best_position))

    return embeddings[selected]

