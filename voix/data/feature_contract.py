"""Canonical face-feature representation contract (FaceFeatureV2).

Every path that feeds the face-to-voice stack must produce the same 560-D
representation:

    [ArcFace 512 | craniofacial 32 | demographics 16]
        -> per-modality L2 normalization
        -> FaceFusionLayer (deterministic seed-42 projection)
        -> 560-D fused raw vector
        -> z-score with the speaker-disjoint FaceFeatureV2 scaler
        -> 560-D normalized vector consumed by CFVN

The V1 corpus was built without per-modality normalization, so geometry and
demographics dominated the projection and the fused vector was nearly constant
across speakers.  V2 normalizes each stream first; runtime probes show
face->face R@1 rising from ~9% to ~90% on the same clips.

The HDF5 corpus is built with the same deterministic ``FaceFusionLayer``
initialization (no trained fusion checkpoint exists). This module makes that
fact explicit, versioned, and shared by image inference, presets, training, and
evaluation instead of relying on path-specific heuristics.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import numpy as np
import torch
import torch.nn.functional as F

from voix.models.fusion import FaceFusionLayer

SCHEMA_VERSION = "FaceFeatureV2"
ID_DIM = 512
GEO_DIM = 32
DEMO_DIM = 16
FACE_DIM = ID_DIM + GEO_DIM + DEMO_DIM  # 560

# The HDF5 corpus was extracted with the deterministic seed-42 fusion layer.
# This is a fixed projection, not a learned fusion model. Documenting it here
# prevents a future caller from silently substituting a different initializer.
FUSION_INIT_SEED = 42


@dataclass(frozen=True)
class FaceFeature:
    """A single face representation in raw and normalized form."""

    raw: torch.Tensor                 # (B, 560), FaceFusionLayer output
    normalized: torch.Tensor          # (B, 560), z-scored with FaceFeatureV2
    schema_version: str = SCHEMA_VERSION


def validate_face_vector(vector: torch.Tensor, name: str = "face vector") -> None:
    """Validate shape, dtype, and finiteness of a 560-D face representation."""
    if vector.ndim != 2 or vector.shape[-1] != FACE_DIM:
        raise ValueError(f"{name} must have shape (B, {FACE_DIM}), got {tuple(vector.shape)}")
    if not torch.isfinite(vector).all():
        raise ValueError(f"{name} contains NaN or infinite values")


def canonicalize(
    raw: torch.Tensor,
    face_mean: torch.Tensor,
    face_std: torch.Tensor,
) -> torch.Tensor:
    """Apply the canonical FaceFeatureV2 z-score normalization.

    Args:
        raw: (B, 560) fused raw vector.
        face_mean: (560,) training-split mean.
        face_std: (560,) training-split standard deviation.

    Returns:
        (B, 560) normalized vector.
    """
    validate_face_vector(raw, "raw face feature")
    if face_mean.shape != (FACE_DIM,) or face_std.shape != (FACE_DIM,):
        raise ValueError(
            f"Scaler must provide ({FACE_DIM},) face_mean/face_std; got "
            f"{tuple(face_mean.shape)} / {tuple(face_std.shape)}"
        )
    std = torch.clamp(face_std.to(raw.device, raw.dtype), min=1e-6)
    return (raw - face_mean.to(raw.device, raw.dtype)) / std


def load_scaler_payload(
    path: str | Path,
    *,
    expected_dim: Optional[int] = None,
    face_key: Optional[str] = None,
) -> Dict[str, Any]:
    """Load a scaler payload and validate its dimension and representation key.

    ``load_face_scaler`` is the 560-D serving contract.  Trainers that support
    the raw-ArcFace diagnostic (512-D) need the same validation without the
    fixed 560-D assumption, so the generic loader is exposed separately.
    """
    scaler_path = Path(path)
    if not scaler_path.exists():
        raise FileNotFoundError(f"FaceFeatureV2 scaler not found: {scaler_path}")
    try:
        payload = torch.load(scaler_path, map_location="cpu", weights_only=True)
    except Exception:
        payload = torch.load(scaler_path, map_location="cpu", weights_only=False)
    if "face_mean" not in payload or "face_std" not in payload:
        raise ValueError(f"Scaler {scaler_path} is missing face_mean/face_std")
    payload["face_mean"] = payload["face_mean"].float()
    payload["face_std"] = payload["face_std"].float()
    if payload["face_mean"].shape != payload["face_std"].shape:
        raise ValueError(
            f"Scaler {scaler_path} has mismatched mean/std shapes: "
            f"{tuple(payload['face_mean'].shape)} / {tuple(payload['face_std'].shape)}"
        )
    if expected_dim is not None and payload["face_mean"].shape != (expected_dim,):
        raise ValueError(
            f"Scaler {scaler_path} expects {expected_dim}-D features, got "
            f"{tuple(payload['face_mean'].shape)}"
        )
    if face_key is not None and payload.get("face_key", face_key) != face_key:
        raise ValueError(
            f"Scaler {scaler_path} was built for face_key="
            f"{payload.get('face_key')!r}, not {face_key!r}"
        )
    if not torch.isfinite(payload["face_mean"]).all() or not torch.isfinite(payload["face_std"]).all():
        raise ValueError(f"Scaler {scaler_path} contains non-finite values")
    payload.setdefault("schema_version", SCHEMA_VERSION)
    return payload


def load_face_scaler(path: str | Path) -> Dict[str, Any]:
    """Load the canonical 560-D FaceFeatureV2 serving scaler."""
    return load_scaler_payload(
        path,
        expected_dim=FACE_DIM,
        face_key="face_features",
    )


class FaceFeatureBuilder:
    """Shared builder for all face representations in VOIX."""

    def __init__(self, device: str | torch.device = "cpu") -> None:
        self.device = torch.device(device)
        self.fusion = FaceFusionLayer().to(self.device)
        self.fusion.eval()
        self.fusion.freeze()

    @torch.no_grad()
    def fuse(
        self,
        e_id: np.ndarray | torch.Tensor,
        e_geo: np.ndarray | torch.Tensor,
        e_demo: np.ndarray | torch.Tensor,
    ) -> torch.Tensor:
        """Fuse the three raw modalities into a (B, 560) raw vector."""
        tensors = []
        for name, value, dim in (
            ("e_id", e_id, ID_DIM),
            ("e_geo", e_geo, GEO_DIM),
            ("e_demo", e_demo, DEMO_DIM),
        ):
            tensor = torch.as_tensor(value, dtype=torch.float32, device=self.device)
            if tensor.ndim == 1:
                tensor = tensor.unsqueeze(0)
            if tensor.ndim != 2 or tensor.shape[-1] != dim:
                raise ValueError(f"{name} must have shape (B, {dim}), got {tuple(tensor.shape)}")
            if not torch.isfinite(tensor).all():
                raise ValueError(f"{name} contains NaN or infinite values")
            tensors.append(tensor)
        fused = self.fusion(*tensors)
        validate_face_vector(fused, "fused face feature")
        return fused

    def build(
        self,
        e_id: np.ndarray | torch.Tensor,
        e_geo: np.ndarray | torch.Tensor,
        e_demo: np.ndarray | torch.Tensor,
        face_mean: torch.Tensor,
        face_std: torch.Tensor,
    ) -> FaceFeature:
        raw = self.fuse(e_id, e_geo, e_demo)
        return FaceFeature(
            raw=raw,
            normalized=canonicalize(raw, face_mean, face_std),
        )


def verify_path_parity(
    builder: FaceFeatureBuilder,
    e_id: np.ndarray,
    e_geo: np.ndarray,
    e_demo: np.ndarray,
    hdf5_raw: np.ndarray,
    atol: float = 1e-4,
) -> float:
    """Return max absolute deviation between a fresh fusion and HDF5 features.

    Used by contract tests to prove that image/preset preprocessing and the
    corpus representation are the same contract.
    """
    fresh = builder.fuse(e_id, e_geo, e_demo).cpu().numpy().squeeze(0)
    reference = np.asarray(hdf5_raw, dtype=np.float32).squeeze()
    if fresh.shape != reference.shape:
        raise ValueError(f"Shape mismatch: {fresh.shape} vs {reference.shape}")
    return float(np.max(np.abs(fresh - reference)))
