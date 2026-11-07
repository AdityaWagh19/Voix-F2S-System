"""Apply a class-contrast accent residual in StyleTTS2 style space.

The residual is a 256-D vector whose first half controls timbre and whose
second half controls prosody.  Accent identity is carried mostly by prosody,
so the timbre half can be down-weighted to protect the speaker identity that
the face-to-voice path produced.  The residual is a class contrast, not a
speaker-disentangled direction: GLOBE has no within-speaker cross-accent
pairs, so timbre down-weighting and the ECAPA donor gate are the safeguards.
This is a research transform; it is not part of the serving adapter.
"""
from __future__ import annotations

import torch

STYLE_DIM = 256
PROSODY_START = 128


def apply_accent_residual(
    style: torch.Tensor,
    residual: torch.Tensor,
    *,
    scale: float = 0.2,
    timbre_weight: float = 0.25,
) -> torch.Tensor:
    """Return ``style + scale * [timbre_weight * timbre | prosody]``."""
    if style.shape[-1] != STYLE_DIM or residual.numel() != STYLE_DIM:
        raise ValueError("expected 256-D style and residual vectors")
    adjusted = residual.reshape(-1).clone().float()
    adjusted[:PROSODY_START] *= float(timbre_weight)
    return style + float(scale) * adjusted.to(
        device=style.device, dtype=style.dtype
    )
