"""Frozen self-supervised accent features for A0.

The cheap ECAPA + prosody block in :mod:`voix.evaluation.accent_features` is
speaker-heavy and confuses US, Canadian, and Australian English.  This module
adds a frozen Wav2Vec2 representation: mean-pooled layer-8 hidden states from
the torchaudio ``WAV2VEC2_BASE`` bundle.  Nothing is fine-tuned.

The representation is deliberately fixed and documented so A0, A2, and any
later generated-audio evaluation all use the same features.  It is a research
feature extractor, not a serving component.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import numpy as np

SAMPLE_RATE = 16_000
MAX_SECONDS = 12.0
DEFAULT_BUNDLE = "WAV2VEC2_BASE"
DEFAULT_NUM_LAYERS = 8


@dataclass(frozen=True)
class SSLAccentConfig:
    """Configuration recorded next to every SSL feature cache."""

    bundle: str = DEFAULT_BUNDLE
    num_layers: int = DEFAULT_NUM_LAYERS
    sample_rate: int = SAMPLE_RATE
    max_seconds: float = MAX_SECONDS
    l2_normalize: bool = True

    def as_dict(self) -> Dict[str, object]:
        return {
            "bundle": self.bundle,
            "num_layers": int(self.num_layers),
            "sample_rate": int(self.sample_rate),
            "max_seconds": float(self.max_seconds),
            "l2_normalize": bool(self.l2_normalize),
            "pooling": "mean",
            "feature_dim": int(self.feature_dim),
        }

    @property
    def feature_dim(self) -> int:
        return 768


def ssl_feature_names(config: Optional[SSLAccentConfig] = None) -> List[str]:
    config = config or SSLAccentConfig()
    return [f"ssl_{index}" for index in range(config.feature_dim)]


class SSLAccentFeatureExtractor:
    """Extract frozen Wav2Vec2 mean-pooled features for a list of wav paths."""

    def __init__(
        self,
        config: Optional[SSLAccentConfig] = None,
        *,
        device: str = "cuda",
        batch_size: int = 8,
    ) -> None:
        self.config = config or SSLAccentConfig()
        self.device = str(device)
        self.batch_size = max(1, int(batch_size))
        self._model = None
        self._torch = None

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch
        import torchaudio.pipelines as pipelines

        bundle = getattr(pipelines, self.config.bundle, None)
        if bundle is None:
            available = [
                name
                for name in dir(pipelines)
                if name.startswith("WAV2VEC2")
            ]
            raise ValueError(
                f"unknown torchaudio bundle {self.config.bundle!r}; "
                f"available: {available}"
            )
        model = bundle.get_model()
        model.eval().to(self.device)
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        self._model = model
        self._torch = torch

    def _load_wave(self, path: Union[str, Path]) -> np.ndarray:
        import librosa

        wave, _ = librosa.load(
            str(path), sr=int(self.config.sample_rate), mono=True
        )
        limit = int(self.config.sample_rate * float(self.config.max_seconds))
        if limit > 0 and wave.size > limit:
            wave = wave[:limit]
        if wave.size == 0:
            raise ValueError(f"empty audio: {path}")
        return np.asarray(wave, dtype=np.float32)

    def extract_paths(self, paths: Sequence[Union[str, Path]]) -> np.ndarray:
        """Return an ``(N, 768)`` float32 matrix in input order."""
        self._load()
        torch = self._torch
        assert torch is not None and self._model is not None
        rows: List[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(paths), self.batch_size):
                batch_paths = paths[start : start + self.batch_size]
                waves = [self._load_wave(path) for path in batch_paths]
                max_length = max(int(wave.size) for wave in waves)
                padded = np.stack(
                    [
                        np.pad(wave, (0, max_length - wave.size))
                        for wave in waves
                    ]
                )
                # torchaudio's CUDA attention path requires the length tensor
                # on the same device as the waveforms; leaving it on CPU
                # raises a mixed-device RuntimeError for every batch.
                lengths = torch.tensor(
                    [int(wave.size) for wave in waves],
                    dtype=torch.long,
                    device=self.device,
                )
                waveforms = torch.from_numpy(padded).float().to(self.device)
                features, output_lengths = self._model.extract_features(
                    waveforms,
                    lengths=lengths,
                    num_layers=int(self.config.num_layers),
                )
                hidden = features[-1]
                if output_lengths is None:
                    output_lengths = torch.full(
                        (hidden.shape[0],), hidden.shape[1], dtype=torch.long
                    )
                mask = (
                    torch.arange(hidden.shape[1], device=hidden.device)
                    .unsqueeze(0)
                    < output_lengths.to(hidden.device).unsqueeze(1)
                )
                pooled = (hidden * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(
                    dim=1, keepdim=True
                ).clamp(min=1)
                pooled = pooled.detach().cpu().numpy().astype(np.float32)
                if self.config.l2_normalize:
                    norms = np.linalg.norm(pooled, axis=1, keepdims=True)
                    pooled = pooled / np.clip(norms, 1e-8, None)
                rows.append(pooled)
        if not rows:
            return np.zeros((0, self.config.feature_dim), dtype=np.float32)
        return np.concatenate(rows, axis=0)
