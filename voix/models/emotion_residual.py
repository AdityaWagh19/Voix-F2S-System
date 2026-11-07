"""Class-mean emotion residuals in the frozen StyleTTS2 style space.

The fixed emotion deltas in ``checkpoints/emotion_deltas`` are prosody-only
vectors estimated outside this experiment.  E1 asks a simpler question before
training anything: what if we estimate the residual directly from real paired
CREMA-D speech, average it per emotion over training speakers, and apply that?

This module contains only the deterministic, model-free pieces:

* deterministic clip selection when a speaker/sentence/emotion has several
  intensity variants;
* neutral/emotion pair construction;
* speaker-balanced class-mean estimation;
* neutral-zero and norm-bounded residual application;
* artifact save/load.

The script ``scripts/training/build_emotion_class_mean.py`` supplies the StyleTTS2 style
encoder and the real audio.  Nothing here reads the canonical face-to-voice
artifacts.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

STYLE_DIM = 256
PROSODY_START = 128
RESIDUAL_SCHEMA = "voix.emotion_class_mean.v1"
CREMA_EMOTIONS: Tuple[str, ...] = (
    "neutral",
    "happy",
    "sad",
    "angry",
    "fear",
    "disgust",
)
NON_NEUTRAL_EMOTIONS: Tuple[str, ...] = tuple(
    emotion for emotion in CREMA_EMOTIONS if emotion != "neutral"
)
INTENSITY_ORDER: Dict[str, int] = {
    "low": 0,
    "medium": 1,
    "high": 2,
    "unspecified": 3,
}


@dataclass(frozen=True)
class ResidualStats:
    """Per-emotion class-mean residual plus its provenance counts."""

    emotion: str
    delta: torch.Tensor
    delta_prosody_only: torch.Tensor
    n_pairs: int
    n_speakers: int
    speaker_norm_mean: float
    speaker_norm_ci_low: Optional[float]
    speaker_norm_ci_high: Optional[float]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "emotion": self.emotion,
            "n_pairs": self.n_pairs,
            "n_speakers": self.n_speakers,
            "norm": float(self.delta.norm().item()),
            "timbre_norm": float(self.delta[:PROSODY_START].norm().item()),
            "prosody_norm": float(self.delta[PROSODY_START:].norm().item()),
            "prosody_only_norm": float(self.delta_prosody_only.norm().item()),
            "speaker_norm_mean": self.speaker_norm_mean,
            "speaker_norm_ci_low": self.speaker_norm_ci_low,
            "speaker_norm_ci_high": self.speaker_norm_ci_high,
        }


def select_one_per_group(
    frame: Any,
    *,
    emotion_column: str = "intended_emotion",
    intensity_column: str = "intensity",
    clip_column: str = "clip_id",
    group_columns: Optional[Sequence[str]] = None,
    priority: Mapping[str, int] = INTENSITY_ORDER,
) -> Any:
    """Keep one deterministic clip per speaker/sentence/emotion group.

    CREMA-D has up to three intensity variants for the same sentence.  We prefer
    the unspecified (natural) recording, then high, medium, and low, with
    ``clip_id`` as the final deterministic tie-break.  This is the same ordering
    used by the paired evaluation grid.
    """
    import pandas as pd

    if group_columns is None:
        group_columns = ("actor_id", "sentence", emotion_column)
    missing = [column for column in group_columns if column not in frame.columns]
    if missing:
        raise ValueError(f"frame is missing columns: {missing}")
    if intensity_column not in frame.columns:
        raise ValueError(f"frame is missing column: {intensity_column}")
    if clip_column not in frame.columns:
        raise ValueError(f"frame is missing column: {clip_column}")

    ranked = frame.copy()
    ranked["_intensity_rank"] = (
        ranked[intensity_column].astype(str).str.lower().map(priority).fillna(-1)
    )
    ranked = ranked.sort_values(
        list(group_columns) + ["_intensity_rank", clip_column],
        ascending=[True] * len(group_columns) + [False, True],
    )
    selected = ranked.drop_duplicates(subset=list(group_columns), keep="first")
    return selected.drop(columns=["_intensity_rank"]).reset_index(drop=True)


def build_pairs(
    clips: Any,
    *,
    emotion_column: str = "intended_emotion",
    neutral_label: str = "neutral",
    emotions: Sequence[str] = NON_NEUTRAL_EMOTIONS,
) -> Any:
    """Join each non-neutral clip to the neutral clip of the same speaker/sentence."""
    import pandas as pd

    required = {
        "actor_id",
        "sentence",
        emotion_column,
        "clip_id",
        "path",
    }
    missing = sorted(required - set(clips.columns))
    if missing:
        raise ValueError(f"clips are missing columns: {missing}")

    neutral = clips[clips[emotion_column].astype(str) == neutral_label]
    neutral = neutral[["actor_id", "sentence", "clip_id", "path"]].rename(
        columns={"clip_id": "neutral_clip_id", "path": "neutral_path"}
    )
    if neutral.duplicated(subset=["actor_id", "sentence"]).any():
        raise ValueError("neutral clips are not unique per actor/sentence")

    emotion_rows = clips[clips[emotion_column].astype(str).isin(set(emotions))]
    emotion_rows = emotion_rows[
        ["actor_id", "sentence", emotion_column, "clip_id", "path"]
    ].rename(
        columns={
            emotion_column: "emotion",
            "clip_id": "emotion_clip_id",
            "path": "emotion_path",
        }
    )
    pairs = emotion_rows.merge(neutral, on=["actor_id", "sentence"], how="inner")
    if pairs.empty:
        raise ValueError("no neutral/emotion pairs could be built")
    pairs = pairs.sort_values(["actor_id", "sentence", "emotion"]).reset_index(drop=True)
    return pairs


def class_mean_residuals(
    records: Iterable[Mapping[str, Any]],
    *,
    prosody_only: bool = True,
    n_resamples: int = 2_000,
    confidence: float = 0.95,
    seed: int = 0,
) -> Dict[str, ResidualStats]:
    """Estimate a speaker-balanced mean residual per emotion.

    ``records`` contains one mapping per paired clip with keys ``emotion``,
    ``speaker_id``, and ``delta`` (a 256-D vector).  Each speaker contributes
    one mean vector, so a speaker with more usable sentences cannot dominate the
    class mean.  The prosody-only variant zeroes the first 128 dimensions, which
    keeps the comparison against the existing prosody-only fixed deltas fair.
    """
    from voix.evaluation.stats import bootstrap_mean_ci

    grouped: Dict[str, Dict[str, List[np.ndarray]]] = {}
    for record in records:
        emotion = str(record["emotion"])
        speaker = str(record["speaker_id"])
        delta = np.asarray(record["delta"], dtype=np.float64).reshape(-1)
        if delta.shape != (STYLE_DIM,):
            raise ValueError(
                f"delta for {emotion}/{speaker} has shape {delta.shape}, expected ({STYLE_DIM},)"
            )
        if not np.isfinite(delta).all():
            raise ValueError(f"delta for {emotion}/{speaker} contains non-finite values")
        grouped.setdefault(emotion, {}).setdefault(speaker, []).append(delta)

    results: Dict[str, ResidualStats] = {}
    for emotion, by_speaker in sorted(grouped.items()):
        speaker_means = [
            np.mean(np.stack(vectors, axis=0), axis=0)
            for vectors in by_speaker.values()
        ]
        mean = np.mean(np.stack(speaker_means, axis=0), axis=0)
        norms = [float(np.linalg.norm(vector)) for vector in speaker_means]
        if len(norms) >= 2:
            interval = bootstrap_mean_ci(
                norms,
                n_resamples=n_resamples,
                confidence=confidence,
                seed=seed,
            )
            ci_low: Optional[float] = interval.ci_low
            ci_high: Optional[float] = interval.ci_high
        else:
            ci_low = None
            ci_high = None
        full = torch.from_numpy(mean.astype(np.float32))
        prosody = full.clone()
        if prosody_only:
            prosody[:PROSODY_START] = 0.0
        results[emotion] = ResidualStats(
            emotion=emotion,
            delta=full,
            delta_prosody_only=prosody,
            n_pairs=sum(len(vectors) for vectors in by_speaker.values()),
            n_speakers=len(by_speaker),
            speaker_norm_mean=float(np.mean(norms)),
            speaker_norm_ci_low=ci_low,
            speaker_norm_ci_high=ci_high,
        )
    if not results:
        raise ValueError("no residual records were supplied")
    return results


def apply_style_residual(
    style: torch.Tensor,
    residual: torch.Tensor,
    *,
    intensity: float = 1.0,
    scale: float = 0.28,
    max_norm: Optional[float] = None,
) -> torch.Tensor:
    """Apply a residual with exact neutral-zero and optional norm bounding.

    ``intensity <= 0`` and a zero residual are exact no-ops.  The returned
    tensor has the same shape as the input style.
    """
    if residual.shape[-1] != style.shape[-1]:
        raise ValueError(
            f"residual width {residual.shape[-1]} does not match style width {style.shape[-1]}"
        )
    if intensity <= 0.0 or not torch.any(residual != 0):
        return style
    delta = residual.to(device=style.device, dtype=style.dtype) * float(scale) * float(intensity)
    if max_norm is not None:
        norm = float(delta.norm().item())
        if norm > float(max_norm) and norm > 0.0:
            delta = delta * (float(max_norm) / norm)
    return style + delta


def shuffled_residuals(
    records: Iterable[Mapping[str, Any]],
    *,
    seed: int = 0,
    prosody_only: bool = True,
) -> Dict[str, ResidualStats]:
    """Label-shuffled control: permute emotion labels across paired records.

    The permutation is deterministic for a seed and is applied to the residual
    vectors, not to the audio.  It answers whether a class-specific mean is
    needed at all: if the shuffled control performs as well as the real
    class-mean, the apparent effect is not class-specific.
    """
    records_list = list(records)
    if not records_list:
        raise ValueError("no residual records were supplied")
    rng = np.random.default_rng(seed)
    emotions = np.array([str(record["emotion"]) for record in records_list], dtype=object)
    shuffled = emotions[rng.permutation(len(emotions))]
    remapped = []
    for record, emotion in zip(records_list, shuffled):
        updated = dict(record)
        updated["emotion"] = str(emotion)
        remapped.append(updated)
    return class_mean_residuals(remapped, prosody_only=prosody_only, seed=seed)


def save_artifact(
    path: Any,
    results: Mapping[str, ResidualStats],
    *,
    metadata: Optional[Mapping[str, Any]] = None,
) -> None:
    """Write the class-mean artifact as a ``torch.save`` payload."""
    from pathlib import Path

    payload: Dict[str, Any] = {
        "schema_version": RESIDUAL_SCHEMA,
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "style_dim": STYLE_DIM,
        "prosody_start": PROSODY_START,
        "emotions": {
            emotion: {
                "delta": stats.delta,
                "delta_prosody_only": stats.delta_prosody_only,
                **stats.to_dict(),
            }
            for emotion, stats in sorted(results.items())
        },
        "metadata": dict(metadata or {}),
    }
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, target)


def load_artifact(path: Any) -> Dict[str, Any]:
    """Load and minimally validate a class-mean artifact."""
    from pathlib import Path

    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError("residual artifact must be a mapping")
    if payload.get("schema_version") != RESIDUAL_SCHEMA:
        raise ValueError(
            f"unexpected residual schema {payload.get('schema_version')!r}, "
            f"expected {RESIDUAL_SCHEMA!r}"
        )
    emotions = payload.get("emotions")
    if not isinstance(emotions, dict) or not emotions:
        raise ValueError("residual artifact contains no emotions")
    for emotion, entry in emotions.items():
        for key in ("delta", "delta_prosody_only"):
            value = entry.get(key)
            if not isinstance(value, torch.Tensor) or value.shape != (STYLE_DIM,):
                raise ValueError(f"invalid {key} for emotion {emotion!r}")
    return payload


def residual_from_artifact(
    payload: Mapping[str, Any],
    emotion: str,
    *,
    variant: str = "full",
) -> torch.Tensor:
    """Select a full or prosody-only residual from a loaded artifact."""
    entry = payload["emotions"].get(str(emotion))
    if entry is None:
        raise KeyError(f"emotion {emotion!r} is not present in the residual artifact")
    key = "delta_prosody_only" if variant == "prosody_only" else "delta"
    return entry[key].float()
