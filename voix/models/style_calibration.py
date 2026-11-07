"""Experimental style-manifold calibration for face-derived StyleTTS2 styles.

The canonical face-to-voice path (CFVN -> CVAE -> StyleAdapter) produces style
vectors that sit off the real-speech style manifold.  A gender-matched mean
translation

    s' = s + alpha * (mean_real_gender - mean_face_gender)

moves the face-style cloud onto the real-style mean.  Within a gender it
preserves every pairwise difference exactly, because the same constant is
added to every vector.  Cross-gender differences change by the difference of
the two gender deltas, so the transform is not a global rigid translation.

This module is deliberately separate from the serving pipeline.  It does not
load, modify, or replace any canonical checkpoint.  The demo generator applies
it explicitly and records the artifact hash in its metadata.

Schema: ``voix.style_calibration.v1``.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
import torch

CALIBRATION_SCHEMA = "voix.style_calibration.v1"
METHOD = "gender_matched_mean_translation"
STYLE_DIM = 256
GENDERS = ("female", "male")


def vector_sha256(vector: Any) -> str:
    """Deterministic hash for a float style/residual vector."""
    array = np.asarray(vector, dtype=np.float32).reshape(-1).round(8)
    return hashlib.sha256(array.tobytes()).hexdigest()


def _as_matrix(values: Any, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a 1-D or 2-D array, got {array.shape}")
    if array.shape[1] != STYLE_DIM:
        raise ValueError(
            f"{name} must have width {STYLE_DIM}, got {array.shape[1]}"
        )
    if array.shape[0] == 0:
        raise ValueError(f"{name} is empty")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return array


def _pairwise_stats(array: np.ndarray) -> Dict[str, Any]:
    norms = np.linalg.norm(array, axis=1)
    unit = array / np.maximum(norms[:, None], 1e-12)
    cosine = unit @ unit.T
    values = cosine[np.triu_indices_from(cosine, k=1)]
    return {
        "n": int(array.shape[0]),
        "norm_mean": float(norms.mean()),
        "norm_std": float(norms.std()),
        "pairwise_cosine_mean": float(values.mean()) if values.size else None,
        "pairwise_cosine_min": float(values.min()) if values.size else None,
        "pairwise_cosine_max": float(values.max()) if values.size else None,
        "centered_rms": float(
            np.sqrt(np.mean((array - array.mean(axis=0)) ** 2))
        ),
    }


def _mean_entry(face: np.ndarray, real: np.ndarray) -> Dict[str, Any]:
    face_mean = face.mean(axis=0)
    real_mean = real.mean(axis=0)
    delta = real_mean - face_mean
    denominator = max(
        float(np.linalg.norm(face_mean) * np.linalg.norm(real_mean)), 1e-12
    )
    return {
        "face": _pairwise_stats(face),
        "real": _pairwise_stats(real),
        "face_mean": face_mean.tolist(),
        "real_mean": real_mean.tolist(),
        "delta": delta.tolist(),
        "face_mean_norm": float(np.linalg.norm(face_mean)),
        "real_mean_norm": float(np.linalg.norm(real_mean)),
        "delta_norm": float(np.linalg.norm(delta)),
        "delta_timbre_norm": float(np.linalg.norm(delta[:128])),
        "delta_prosody_norm": float(np.linalg.norm(delta[128:])),
        "cos_face_mean_real_mean": float(
            np.dot(face_mean, real_mean) / denominator
        ),
        "face_mean_sha256": vector_sha256(face_mean),
        "real_mean_sha256": vector_sha256(real_mean),
        "delta_sha256": vector_sha256(delta),
    }


def fit_translation(
    face_styles: Mapping[str, Any],
    real_styles: Mapping[str, Any],
    *,
    alpha: float = 1.0,
    face_source: Optional[Mapping[str, Any]] = None,
    real_source: Optional[Mapping[str, Any]] = None,
    notes: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Fit a gender-matched mean translation.

    Args:
        face_styles: mapping gender -> array-like of face-derived styles.
        real_styles: mapping gender -> array-like of real-speech styles.
        alpha: default translation strength stored in the artifact.  The
            application function can override it.
        face_source / real_source: provenance mappings, stored verbatim.
        notes: optional extra notes.

    Returns:
        A JSON-serializable calibration artifact.
    """
    missing = [gender for gender in GENDERS if gender not in face_styles or gender not in real_styles]
    if missing:
        raise KeyError(f"missing styles for genders: {missing}")
    by_gender: Dict[str, Any] = {}
    for gender in GENDERS:
        face = _as_matrix(face_styles[gender], f"face_styles[{gender}]")
        real = _as_matrix(real_styles[gender], f"real_styles[{gender}]")
        by_gender[gender] = _mean_entry(face, real)
    return {
        "schema_version": CALIBRATION_SCHEMA,
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "level": "exploratory",
        "method": METHOD,
        "alpha": float(alpha),
        "style_dim": STYLE_DIM,
        "genders": list(GENDERS),
        "face_source": dict(face_source or {}),
        "real_source": dict(real_source or {}),
        "by_gender": by_gender,
        "notes": list(notes or []),
    }


def validate_calibration(calibration: Mapping[str, Any]) -> None:
    if calibration.get("schema_version") != CALIBRATION_SCHEMA:
        raise ValueError(
            f"unexpected calibration schema {calibration.get('schema_version')!r}"
        )
    by_gender = calibration.get("by_gender")
    if not isinstance(by_gender, Mapping):
        raise ValueError("calibration has no by_gender mapping")
    for gender in GENDERS:
        entry = by_gender.get(gender)
        if not isinstance(entry, Mapping):
            raise KeyError(f"calibration missing gender {gender!r}")
        delta = np.asarray(entry.get("delta"), dtype=np.float64).reshape(-1)
        if delta.shape != (STYLE_DIM,):
            raise ValueError(
                f"calibration delta for {gender!r} has shape {delta.shape}, "
                f"expected ({STYLE_DIM},)"
            )
        if not np.isfinite(delta).all():
            raise ValueError(f"calibration delta for {gender!r} is not finite")


def apply_translation(
    style: torch.Tensor,
    calibration: Mapping[str, Any],
    gender: str,
    *,
    alpha: Optional[float] = None,
) -> torch.Tensor:
    """Return ``style + alpha * delta_gender`` with shape/device/dtype preserved."""
    validate_calibration(calibration)
    if gender not in GENDERS:
        raise ValueError(f"gender must be one of {GENDERS}, got {gender!r}")
    if style.shape[-1] != STYLE_DIM:
        raise ValueError(
            f"style must have last dimension {STYLE_DIM}, got {tuple(style.shape)}"
        )
    entry = calibration["by_gender"][gender]
    delta = torch.as_tensor(
        entry["delta"], dtype=style.dtype, device=style.device
    ).reshape(-1)
    strength = float(calibration.get("alpha", 1.0) if alpha is None else alpha)
    return style + strength * delta


def save_calibration(path: Any, calibration: Mapping[str, Any]) -> Path:
    validate_calibration(calibration)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(calibration, indent=2) + "\n", encoding="utf-8"
    )
    return target


def load_calibration(path: Any) -> Dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_calibration(payload)
    return payload


def gender_for_code(gender_code: int) -> str:
    """Map the pipeline's 0=female / 1=male convention to a calibration key."""
    return "female" if int(gender_code) == 0 else "male"
