"""Learned emotion residuals on the frozen StyleTTS2 style space (E2).

The canonical face-to-voice path is not used here.  The identity base is the
frozen StyleTTS2 style/predictor vector of a real neutral clip; the adapter
predicts a bounded 256-D residual from that style, an emotion label, and an
intensity.  Neutral and zero intensity are exact no-ops.

The model is deliberately small (<1M parameters) and is trained on real paired
CREMA-D residuals.  It is a research artifact, not a serving checkpoint.
"""
from __future__ import annotations

import hashlib
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

STYLE_DIM = 256
PROSODY_START = 128
ADAPTER_SCHEMA = "voix.emotion_adapter.v1"
EMOTIONS: Tuple[str, ...] = (
    "neutral",
    "happy",
    "sad",
    "angry",
    "fear",
    "disgust",
)
NON_NEUTRAL_EMOTIONS: Tuple[str, ...] = tuple(
    emotion for emotion in EMOTIONS if emotion != "neutral"
)
EMOTION_TO_INDEX: Dict[str, int] = {
    emotion: index for index, emotion in enumerate(EMOTIONS)
}


class EmotionResidualAdapter(nn.Module):
    """Small MLP that predicts a bounded style residual.

    Args:
        style_dim: Width of the frozen style vector.
        hidden_dim: Hidden width.
        n_emotions: Number of emotion one-hot entries (including neutral).
        max_norm: Hard L2 bound on the predicted residual.
        dropout: Dropout in the hidden layers.
    """

    def __init__(
        self,
        style_dim: int = STYLE_DIM,
        hidden_dim: int = 256,
        n_emotions: int = len(EMOTIONS),
        max_norm: float = 0.5,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if style_dim <= 0 or hidden_dim <= 0:
            raise ValueError("style_dim and hidden_dim must be positive")
        if n_emotions < 2:
            raise ValueError("n_emotions must include neutral and at least one emotion")
        if max_norm <= 0.0:
            raise ValueError("max_norm must be positive")
        self.style_dim = int(style_dim)
        self.hidden_dim = int(hidden_dim)
        self.n_emotions = int(n_emotions)
        self.max_norm = float(max_norm)
        input_dim = self.style_dim + self.n_emotions + 1
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
        neutral_style: torch.Tensor,
        emotion_onehot: torch.Tensor,
        intensity: torch.Tensor,
    ) -> torch.Tensor:
        if neutral_style.dim() == 1:
            neutral_style = neutral_style.unsqueeze(0)
        if emotion_onehot.dim() == 1:
            emotion_onehot = emotion_onehot.unsqueeze(0)
        if intensity.dim() == 0:
            intensity = intensity.unsqueeze(0)
        if neutral_style.shape[-1] != self.style_dim:
            raise ValueError(
                f"neutral_style must have width {self.style_dim}, "
                f"got {neutral_style.shape[-1]}"
            )
        if emotion_onehot.shape[-1] != self.n_emotions:
            raise ValueError(
                f"emotion_onehot must have width {self.n_emotions}, "
                f"got {emotion_onehot.shape[-1]}"
            )
        if intensity.shape[-1] != 1:
            intensity = intensity.reshape(-1, 1)
        batch = neutral_style.shape[0]
        if emotion_onehot.shape[0] != batch or intensity.shape[0] != batch:
            raise ValueError("neutral_style, emotion_onehot, and intensity must agree")

        inputs = torch.cat(
            [
                neutral_style,
                emotion_onehot.to(neutral_style.dtype),
                intensity.to(neutral_style.dtype).reshape(batch, 1),
            ],
            dim=-1,
        )
        raw = self.net(inputs)
        delta = torch.tanh(raw)
        norm = delta.norm(dim=-1, keepdim=True)
        # Hard L2 bound while keeping the direction learned by the network.
        delta = delta * torch.clamp(self.max_norm / (norm + 1e-8), max=1.0)

        is_neutral = (
            (emotion_onehot[:, 0] > 0.5)
            | (emotion_onehot.sum(dim=-1) <= 0.0)
            | (intensity.reshape(-1) <= 0.0)
        )
        return delta * (~is_neutral).unsqueeze(-1).to(delta.dtype)

    @property
    def n_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


def emotion_one_hot(
    emotion: str,
    *,
    emotions: Sequence[str] = EMOTIONS,
) -> torch.Tensor:
    """Return a float one-hot vector, raising for unknown labels."""
    key = str(emotion).strip().lower()
    if key not in emotions:
        raise ValueError(f"unknown emotion {emotion!r}; expected one of {list(emotions)}")
    index = list(emotions).index(key)
    vector = torch.zeros(len(emotions), dtype=torch.float32)
    vector[index] = 1.0
    return vector


def build_examples(
    pairs: Any,
    style_by_path: Mapping[str, Any],
    *,
    intensity: float = 1.0,
) -> Dict[str, Any]:
    """Turn neutral/emotion pairs into adapter training examples.

    ``pairs`` is the output of ``voix.models.emotion_residual.build_pairs``.
    Every target is ``style(emotion) - style(neutral)`` for the same actor and
    sentence.  The main pilot treats the selected clip as full expression
    (intensity 1.0); explicit-intensity monotonicity is a separate E3 analysis.
    """
    required = {"actor_id", "sentence", "emotion", "neutral_path", "emotion_path"}
    missing = sorted(required - set(pairs.columns))
    if missing:
        raise ValueError(f"pairs are missing columns: {missing}")

    neutral_styles: List[np.ndarray] = []
    targets: List[np.ndarray] = []
    emotion_indices: List[int] = []
    intensities: List[float] = []
    speakers: List[str] = []
    pair_ids: List[str] = []
    emotions: List[str] = []

    for _, pair in pairs.iterrows():
        emotion = str(pair["emotion"]).strip().lower()
        if emotion not in NON_NEUTRAL_EMOTIONS:
            raise ValueError(f"pair emotion must be non-neutral, got {emotion!r}")
        neutral_key = str(pair["neutral_path"])
        emotion_key = str(pair["emotion_path"])
        if neutral_key not in style_by_path:
            raise KeyError(f"missing neutral style for {neutral_key}")
        if emotion_key not in style_by_path:
            raise KeyError(f"missing emotion style for {emotion_key}")
        neutral = np.asarray(style_by_path[neutral_key], dtype=np.float32).reshape(-1)
        target_style = np.asarray(style_by_path[emotion_key], dtype=np.float32).reshape(-1)
        if neutral.shape != (STYLE_DIM,) or target_style.shape != (STYLE_DIM,):
            raise ValueError("style vectors must be 256-D")
        neutral_styles.append(neutral)
        targets.append(target_style - neutral)
        emotion_indices.append(EMOTION_TO_INDEX[emotion])
        intensities.append(float(intensity))
        speakers.append(str(pair["actor_id"]))
        pair_ids.append(
            f"{pair['actor_id']}:{pair['sentence']}:{emotion}"
        )
        emotions.append(emotion)

    if not neutral_styles:
        raise ValueError("no examples were built")
    return {
        "neutral_styles": np.stack(neutral_styles).astype(np.float32),
        "targets": np.stack(targets).astype(np.float32),
        "emotion_indices": np.asarray(emotion_indices, dtype=np.int64),
        "intensities": np.asarray(intensities, dtype=np.float32),
        "speakers": np.asarray(speakers, dtype=object),
        "pair_ids": np.asarray(pair_ids, dtype=object),
        "emotions": np.asarray(emotions, dtype=object),
    }


def speaker_disjoint_split(
    speakers: Sequence[str],
    *,
    val_fraction: float = 0.15,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Deterministic speaker-disjoint fit/validation masks.

    The split is by speaker, never by clip, so an actor's style cannot leak
    across the early-stopping boundary.
    """
    if not 0.0 < float(val_fraction) < 1.0:
        raise ValueError("val_fraction must be in (0, 1)")
    unique = sorted({str(speaker) for speaker in speakers})
    if len(unique) < 2:
        raise ValueError("speaker-disjoint split needs at least two speakers")
    ranked = sorted(
        unique,
        key=lambda speaker: hashlib.sha256(
            f"{int(seed)}:{speaker}".encode("utf-8")
        ).hexdigest(),
    )
    n_val = max(1, int(round(len(unique) * float(val_fraction))))
    n_val = min(n_val, len(unique) - 1)
    validation = set(ranked[:n_val])
    fit_mask = np.asarray(
        [str(speaker) not in validation for speaker in speakers], dtype=bool
    )
    val_mask = ~fit_mask
    if not fit_mask.any() or not val_mask.any():
        raise ValueError("split produced an empty side")
    return fit_mask, val_mask, sorted(validation)


def _weighted_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    timbre_weight: float,
) -> torch.Tensor:
    weights = torch.ones(
        prediction.shape[-1], device=prediction.device, dtype=prediction.dtype
    )
    weights[:PROSODY_START] = float(timbre_weight)
    return (((prediction - target) ** 2) * weights).mean()


def residual_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    timbre_weight: float = 0.25,
    mse_weight: float = 1.0,
    cosine_weight: float = 0.5,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Weighted L2 plus cosine loss; the timbre half is down-weighted."""
    mse = _weighted_mse(prediction, target, timbre_weight=timbre_weight)
    cosine = 1.0 - F.cosine_similarity(prediction, target, dim=-1).mean()
    loss = float(mse_weight) * mse + float(cosine_weight) * cosine
    return loss, {
        "mse": float(mse.detach().cpu()),
        "cosine_loss": float(cosine.detach().cpu()),
    }


def residual_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    timbre_weight: float = 0.25,
    emotions: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Direction, norm, and per-emotion diagnostics for held-out residuals."""
    mse = _weighted_mse(prediction, target, timbre_weight=timbre_weight)
    cosine = F.cosine_similarity(prediction, target, dim=-1)
    result: Dict[str, Any] = {
        "weighted_mse": float(mse.detach().cpu()),
        "cosine_mean": float(cosine.mean().detach().cpu()),
        "prediction_norm_mean": float(
            prediction.norm(dim=-1).mean().detach().cpu()
        ),
        "target_norm_mean": float(target.norm(dim=-1).mean().detach().cpu()),
        "n": int(prediction.shape[0]),
    }
    if emotions is not None and len(emotions) == prediction.shape[0]:
        per_emotion: Dict[str, Any] = {}
        labels = np.asarray([str(item) for item in emotions], dtype=object)
        for emotion in sorted(set(labels.tolist())):
            mask = torch.from_numpy(labels == emotion).to(cosine.device)
            per_emotion[emotion] = {
                "n": int(mask.sum().item()),
                "cosine_mean": float(cosine[mask].mean().detach().cpu()),
                "weighted_mse": float(
                    _weighted_mse(
                        prediction[mask],
                        target[mask],
                        timbre_weight=timbre_weight,
                    ).detach().cpu()
                ),
            }
        result["per_emotion"] = per_emotion
    return result


def train_adapter(
    examples: Mapping[str, Any],
    *,
    hidden_dim: int = 256,
    max_norm: float = 0.5,
    dropout: float = 0.0,
    epochs: int = 200,
    batch_size: int = 128,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    timbre_weight: float = 0.25,
    mse_weight: float = 1.0,
    cosine_weight: float = 0.5,
    patience: int = 20,
    val_fraction: float = 0.15,
    seed: int = 0,
    device: str = "cpu",
) -> Dict[str, Any]:
    """Train the adapter with speaker-disjoint early stopping."""
    neutral_styles = np.asarray(examples["neutral_styles"], dtype=np.float32)
    targets = np.asarray(examples["targets"], dtype=np.float32)
    emotion_indices = np.asarray(examples["emotion_indices"], dtype=np.int64)
    intensities = np.asarray(examples["intensities"], dtype=np.float32)
    speakers = np.asarray(examples["speakers"], dtype=object)
    emotions = np.asarray(examples["emotions"], dtype=object)

    fit_mask, val_mask, val_speakers = speaker_disjoint_split(
        speakers.tolist(), val_fraction=val_fraction, seed=seed
    )
    style_mean = neutral_styles[fit_mask].mean(axis=0)
    style_std = neutral_styles[fit_mask].std(axis=0)
    style_std = np.maximum(style_std, 1e-6)

    def prepare(mask: np.ndarray) -> Dict[str, torch.Tensor]:
        styles = torch.from_numpy(
            (neutral_styles[mask] - style_mean) / style_std
        ).to(device)
        labels = emotion_indices[mask]
        onehot = F.one_hot(
            torch.from_numpy(labels), num_classes=len(EMOTIONS)
        ).float().to(device)
        return {
            "styles": styles,
            "onehot": onehot,
            "intensity": torch.from_numpy(intensities[mask]).to(device),
            "target": torch.from_numpy(targets[mask]).to(device),
            "emotions": emotions[mask],
        }

    fit = prepare(fit_mask)
    val = prepare(val_mask)
    n_fit = fit["styles"].shape[0]
    if n_fit < 2 or val["styles"].shape[0] < 1:
        raise ValueError("not enough examples after the speaker-disjoint split")

    torch.manual_seed(int(seed))
    model = EmotionResidualAdapter(
        style_dim=STYLE_DIM,
        hidden_dim=hidden_dim,
        n_emotions=len(EMOTIONS),
        max_norm=max_norm,
        dropout=dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(learning_rate),
        weight_decay=float(weight_decay),
    )

    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_val = math.inf
    best_epoch = 0
    stale = 0
    history: List[Dict[str, Any]] = []

    for epoch in range(1, int(epochs) + 1):
        model.train()
        permutation = torch.randperm(n_fit, device=device)
        total_loss = 0.0
        total_mse = 0.0
        total_cosine = 0.0
        batches = 0
        for start in range(0, n_fit, int(batch_size)):
            index = permutation[start : start + int(batch_size)]
            prediction = model(
                fit["styles"][index],
                fit["onehot"][index],
                fit["intensity"][index],
            )
            loss, parts = residual_loss(
                prediction,
                fit["target"][index],
                timbre_weight=timbre_weight,
                mse_weight=mse_weight,
                cosine_weight=cosine_weight,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu())
            total_mse += parts["mse"]
            total_cosine += parts["cosine_loss"]
            batches += 1

        model.eval()
        with torch.no_grad():
            val_prediction = model(val["styles"], val["onehot"], val["intensity"])
            val_loss, val_parts = residual_loss(
                val_prediction,
                val["target"],
                timbre_weight=timbre_weight,
                mse_weight=mse_weight,
                cosine_weight=cosine_weight,
            )
        history.append(
            {
                "epoch": epoch,
                "train_loss": total_loss / max(1, batches),
                "train_mse": total_mse / max(1, batches),
                "train_cosine_loss": total_cosine / max(1, batches),
                "val_loss": float(val_loss.detach().cpu()),
                "val_mse": val_parts["mse"],
                "val_cosine_loss": val_parts["cosine_loss"],
            }
        )
        if float(val_loss.detach().cpu()) < best_val - 1e-7:
            best_val = float(val_loss.detach().cpu())
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
            if stale >= int(patience):
                break

    if best_state is None:
        raise RuntimeError("training produced no finite validation checkpoint")
    model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        val_prediction = model(val["styles"], val["onehot"], val["intensity"])
        val_metrics = residual_metrics(
            val_prediction,
            val["target"],
            timbre_weight=timbre_weight,
            emotions=val["emotions"],
        )

    return {
        "model": model,
        "style_mean": style_mean.astype(np.float32),
        "style_std": style_std.astype(np.float32),
        "history": history,
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val),
        "val_metrics": val_metrics,
        "fit_speakers": sorted(
            set(speakers[fit_mask].astype(str).tolist())
        ),
        "val_speakers": val_speakers,
        "n_fit": int(fit_mask.sum()),
        "n_val": int(val_mask.sum()),
        "n_parameters": int(model.n_parameters),
        "config": {
            "style_dim": STYLE_DIM,
            "hidden_dim": int(hidden_dim),
            "n_emotions": len(EMOTIONS),
            "max_norm": float(max_norm),
            "dropout": float(dropout),
            "epochs": int(epochs),
            "batch_size": int(batch_size),
            "learning_rate": float(learning_rate),
            "weight_decay": float(weight_decay),
            "timbre_weight": float(timbre_weight),
            "mse_weight": float(mse_weight),
            "cosine_weight": float(cosine_weight),
            "patience": int(patience),
            "val_fraction": float(val_fraction),
            "seed": int(seed),
        },
    }


def save_adapter(
    path: Union[str, Path],
    model: EmotionResidualAdapter,
    *,
    style_mean: Any,
    style_std: Any,
    metadata: Optional[Mapping[str, Any]] = None,
) -> None:
    """Write a self-contained adapter artifact."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": ADAPTER_SCHEMA,
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "emotions": list(EMOTIONS),
        "emotion_to_index": dict(EMOTION_TO_INDEX),
        "style_dim": STYLE_DIM,
        "prosody_start": PROSODY_START,
        "config": {
            "style_dim": model.style_dim,
            "hidden_dim": model.hidden_dim,
            "n_emotions": model.n_emotions,
            "max_norm": model.max_norm,
        },
        "state_dict": {
            key: value.detach().cpu() for key, value in model.state_dict().items()
        },
        "style_mean": torch.as_tensor(style_mean, dtype=torch.float32).reshape(-1),
        "style_std": torch.as_tensor(style_std, dtype=torch.float32).reshape(-1),
        "metadata": dict(metadata or {}),
    }
    if payload["style_mean"].shape != (STYLE_DIM,):
        raise ValueError("style_mean must be 256-D")
    if payload["style_std"].shape != (STYLE_DIM,):
        raise ValueError("style_std must be 256-D")
    torch.save(payload, target)


def load_adapter(
    path: Union[str, Path],
) -> Tuple[EmotionResidualAdapter, Dict[str, Any]]:
    """Load and validate an adapter artifact."""
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError("adapter artifact must be a mapping")
    if payload.get("schema_version") != ADAPTER_SCHEMA:
        raise ValueError(
            f"unexpected adapter schema {payload.get('schema_version')!r}, "
            f"expected {ADAPTER_SCHEMA!r}"
        )
    config = payload.get("config") or {}
    model = EmotionResidualAdapter(
        style_dim=int(config.get("style_dim", STYLE_DIM)),
        hidden_dim=int(config.get("hidden_dim", 256)),
        n_emotions=int(config.get("n_emotions", len(EMOTIONS))),
        max_norm=float(config.get("max_norm", 0.5)),
        dropout=0.0,
    )
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    for key in ("style_mean", "style_std"):
        value = payload.get(key)
        if not isinstance(value, torch.Tensor) or value.shape != (STYLE_DIM,):
            raise ValueError(f"adapter artifact has an invalid {key}")
    if payload.get("emotions") != list(EMOTIONS):
        raise ValueError("adapter artifact emotion order does not match this module")
    return model, payload


def predict_residual(
    model: EmotionResidualAdapter,
    payload: Mapping[str, Any],
    neutral_style: Any,
    emotion: str,
    *,
    intensity: float = 1.0,
    device: str = "cpu",
) -> torch.Tensor:
    """Predict a bounded residual for one neutral style.

    Returns a 256-D CPU tensor.  Neutral or ``intensity <= 0`` returns an exact
    zero tensor, never a small learned vector.
    """
    key = str(emotion).strip().lower()
    if key not in EMOTION_TO_INDEX:
        raise ValueError(f"unknown emotion {emotion!r}")
    if key == "neutral" or float(intensity) <= 0.0:
        return torch.zeros(STYLE_DIM, dtype=torch.float32)

    style = torch.as_tensor(neutral_style, dtype=torch.float32).reshape(1, -1)
    if style.shape[-1] != STYLE_DIM:
        raise ValueError(f"neutral_style must be 256-D, got {style.shape[-1]}")
    mean = payload["style_mean"].to(
        device=style.device, dtype=torch.float32
    ).reshape(1, -1)
    std = payload["style_std"].to(
        device=style.device, dtype=torch.float32
    ).reshape(1, -1)
    normalized = (style - mean) / std.clamp_min(1e-6)
    onehot = emotion_one_hot(key).reshape(1, -1)
    strength = torch.tensor([[float(intensity)]], dtype=torch.float32)
    model = model.to(device).eval()
    with torch.no_grad():
        delta = model(
            normalized.to(device),
            onehot.to(device),
            strength.to(device),
        )
    return delta.squeeze(0).detach().cpu()


def apply_learned_residual(style: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
    """Add a learned residual without an extra scale factor.

    The adapter output is already bounded at ``max_norm``.  Neutral-zero is
    preserved by returning the input object unchanged when the delta is zero.
    """
    if delta.shape[-1] != style.shape[-1]:
        raise ValueError("style and residual widths differ")
    if not torch.any(delta != 0):
        return style
    return style + delta.to(device=style.device, dtype=style.dtype)


def shuffle_labels(
    examples: Mapping[str, Any],
    *,
    seed: int = 0,
) -> Dict[str, Any]:
    """Label-shuffled null: permute emotion labels, keep styles and targets.

    Each neutral style keeps its real target residual but receives a random
    emotion label.  The label therefore carries no information about the
    target, which is the null needed to test whether the direction gate is
    specific to class-conditioned learning.  ``emotions`` is permuted with
    ``emotion_indices`` so downstream per-emotion diagnostics use the same
    labels as training.
    """
    rng = np.random.default_rng(int(seed))
    n = len(examples["emotion_indices"])
    permutation = rng.permutation(n)
    shuffled = {key: value for key, value in examples.items()}
    shuffled["emotion_indices"] = np.asarray(
        examples["emotion_indices"], dtype=np.int64
    )[permutation]
    shuffled["emotions"] = np.asarray(
        examples["emotions"], dtype=object
    )[permutation]
    shuffled["label_shuffle_seed"] = int(seed)
    return shuffled


def shuffle_style_assignments(
    examples: Mapping[str, Any],
    *,
    seed: int = 0,
) -> Dict[str, Any]:
    """Style-shuffled diagnostic: reassign label/target pairs to neutral styles.

    This preserves the label-to-target association but breaks the association
    between the input neutral style and the target.  It tests whether
    personalization from the neutral style matters; it is **not** a valid
    label-shuffled null because each label still sees its real target
    distribution.
    """
    rng = np.random.default_rng(int(seed))
    n = len(examples["emotion_indices"])
    permutation = rng.permutation(n)
    shuffled = {key: value for key, value in examples.items()}
    for key in ("emotion_indices", "intensities", "targets"):
        shuffled[key] = np.asarray(examples[key])[permutation]
    for key in ("emotions", "pair_ids"):
        shuffled[key] = np.asarray(examples[key], dtype=object)[permutation]
    shuffled["style_shuffle_seed"] = int(seed)
    return shuffled


def shuffle_examples(
    examples: Mapping[str, Any],
    *,
    seed: int = 0,
) -> Dict[str, Any]:
    """Backward-compatible alias for the corrected label-shuffled null."""
    return shuffle_labels(examples, seed=seed)
