"""Learned accent residuals on the frozen StyleTTS2 style space (A4).

The adapter input is a 192-D ECAPA speaker embedding, a 5-way accent one-hot,
and an intensity scalar.  The output is a bounded 256-D style residual.  It is
trained on speaker-centered GLOBE style vectors, so the target removes the
speaker mean but cannot remove every identity cue; a post-hoc speaker probe is
part of the evaluation.

This is a research artifact, not a serving checkpoint.  Irish is never a
training class.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

STYLE_DIM = 256
ECAPA_DIM = 192
PROSODY_START = 128
ACCENT_SCHEMA = "voix.accent_adapter.v1"
ACCENTS: Tuple[str, ...] = (
    "us",
    "england",
    "australian",
    "canadian",
    "india_south_asia",
)
ACCENT_TO_INDEX: Dict[str, int] = {
    accent: index for index, accent in enumerate(ACCENTS)
}


class AccentResidualAdapter(nn.Module):
    """Small MLP that predicts a bounded style residual from ECAPA + accent."""

    def __init__(
        self,
        feature_dim: int = ECAPA_DIM,
        style_dim: int = STYLE_DIM,
        hidden_dim: int = 256,
        n_accents: int = len(ACCENTS),
        max_norm: float = 0.5,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if feature_dim <= 0 or style_dim <= 0 or hidden_dim <= 0:
            raise ValueError("feature_dim, style_dim, and hidden_dim must be positive")
        if n_accents < 2:
            raise ValueError("n_accents must be at least 2")
        if max_norm <= 0.0:
            raise ValueError("max_norm must be positive")
        self.feature_dim = int(feature_dim)
        self.style_dim = int(style_dim)
        self.hidden_dim = int(hidden_dim)
        self.n_accents = int(n_accents)
        self.max_norm = float(max_norm)
        input_dim = self.feature_dim + self.n_accents + 1
        self.net = nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.hidden_dim, self.style_dim),
        )

    def forward(
        self,
        ecapa: torch.Tensor,
        accent_onehot: torch.Tensor,
        intensity: torch.Tensor,
    ) -> torch.Tensor:
        if ecapa.dim() == 1:
            ecapa = ecapa.unsqueeze(0)
        if accent_onehot.dim() == 1:
            accent_onehot = accent_onehot.unsqueeze(0)
        if intensity.dim() == 0:
            intensity = intensity.unsqueeze(0)
        if ecapa.shape[-1] != self.feature_dim:
            raise ValueError(
                f"ecapa must have width {self.feature_dim}, got {ecapa.shape[-1]}"
            )
        if accent_onehot.shape[-1] != self.n_accents:
            raise ValueError(
                f"accent_onehot must have width {self.n_accents}, "
                f"got {accent_onehot.shape[-1]}"
            )
        if intensity.shape[-1] != 1:
            intensity = intensity.reshape(-1, 1)
        batch = ecapa.shape[0]
        if accent_onehot.shape[0] != batch or intensity.shape[0] != batch:
            raise ValueError("ecapa, accent_onehot, and intensity must agree")

        inputs = torch.cat(
            [
                ecapa,
                accent_onehot.to(ecapa.dtype),
                intensity.to(ecapa.dtype).reshape(batch, 1),
            ],
            dim=-1,
        )
        raw = self.net(inputs)
        delta = torch.tanh(raw)
        norm = delta.norm(dim=-1, keepdim=True)
        delta = delta * torch.clamp(self.max_norm / (norm + 1e-8), max=1.0)
        return delta * (intensity.reshape(-1) > 0.0).unsqueeze(-1).to(delta.dtype)

    @property
    def n_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


def accent_one_hot(
    accent: str,
    *,
    accents: Sequence[str] = ACCENTS,
) -> torch.Tensor:
    key = str(accent).strip().lower()
    if key not in accents:
        raise ValueError(f"unknown accent {accent!r}; expected one of {list(accents)}")
    index = list(accents).index(key)
    vector = torch.zeros(len(accents), dtype=torch.float32)
    vector[index] = 1.0
    return vector


def apply_accent_residual(style: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
    """Add a learned accent residual without an extra scale factor."""
    if delta.shape[-1] != style.shape[-1]:
        raise ValueError("style and residual widths differ")
    if not torch.any(delta != 0):
        return style
    return style + delta.to(device=style.device, dtype=style.dtype)


def predict_accent_residual(
    model: AccentResidualAdapter,
    ecapa: Any,
    accent: str,
    *,
    intensity: float = 1.0,
    device: str = "cpu",
) -> torch.Tensor:
    """Predict a bounded residual for one ECAPA vector.

    Returns a 256-D CPU tensor.  ``intensity <= 0`` is an exact zero tensor.
    """
    key = str(accent).strip().lower()
    if key not in ACCENT_TO_INDEX:
        raise ValueError(f"unknown accent {accent!r}")
    if float(intensity) <= 0.0:
        return torch.zeros(STYLE_DIM, dtype=torch.float32)
    feature = torch.as_tensor(ecapa, dtype=torch.float32).reshape(1, -1)
    if feature.shape[-1] != ECAPA_DIM:
        raise ValueError(f"ecapa must be {ECAPA_DIM}-D, got {feature.shape[-1]}")
    onehot = accent_one_hot(key).reshape(1, -1)
    strength = torch.tensor([[float(intensity)]], dtype=torch.float32)
    model = model.to(device).eval()
    with torch.no_grad():
        delta = model(
            feature.to(device), onehot.to(device), strength.to(device)
        )
    return delta.squeeze(0).detach().cpu()


def weighted_style_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    timbre_weight: float = 0.25,
) -> torch.Tensor:
    weights = torch.ones(STYLE_DIM, dtype=prediction.dtype, device=prediction.device)
    weights[:PROSODY_START] = float(timbre_weight)
    weights = weights / weights.mean()
    return (weights * (prediction - target) ** 2).mean()


def residual_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    timbre_weight: float = 0.25,
) -> Dict[str, float]:
    with torch.no_grad():
        cosine = F.cosine_similarity(prediction, target, dim=-1).mean().item()
        mse = weighted_style_mse(
            prediction, target, timbre_weight=timbre_weight
        ).item()
        return {
            "cosine_mean": float(cosine),
            "weighted_mse": float(mse),
            "prediction_norm_mean": float(prediction.norm(dim=-1).mean().item()),
            "target_norm_mean": float(target.norm(dim=-1).mean().item()),
            "n": int(prediction.shape[0]),
        }


def save_accent_adapter(
    path: Path,
    model: AccentResidualAdapter,
    *,
    accents: Sequence[str] = ACCENTS,
    metadata: Optional[Mapping[str, Any]] = None,
) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": ACCENT_SCHEMA,
        "accents": list(accents),
        "feature_dim": model.feature_dim,
        "style_dim": model.style_dim,
        "hidden_dim": model.hidden_dim,
        "max_norm": model.max_norm,
        "state_dict": model.state_dict(),
        "metadata": dict(metadata or {}),
    }
    torch.save(payload, path)
    return _sha256(path)


def load_accent_adapter(
    path: Path,
) -> Tuple[AccentResidualAdapter, Dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = AccentResidualAdapter(
        feature_dim=int(payload.get("feature_dim", ECAPA_DIM)),
        style_dim=int(payload.get("style_dim", STYLE_DIM)),
        hidden_dim=int(payload.get("hidden_dim", 256)),
        n_accents=len(payload.get("accents", ACCENTS)),
        max_norm=float(payload.get("max_norm", 0.5)),
    )
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()
