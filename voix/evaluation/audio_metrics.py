"""Objective audio metrics with explicit availability reporting.

This module adds the metrics the audit found missing from the evaluation path:

* real word error rate (Whisper transcription + Levenshtein alignment),
* a clearly-labelled Fréchet audio-distance *proxy* over mel-spectrogram
  statistics (not the VGGish FAD),
* ECAPA embedding diversity for candidate sets.

Nothing is fabricated.  When a dependency or reference corpus is unavailable the
result is ``None`` plus a machine-readable ``status`` string.
"""
from __future__ import annotations

import os
import re
import shutil
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Union

import numpy as np


def ensure_ffmpeg_on_path() -> Optional[str]:
    """Return an ``ffmpeg`` executable path, adding an alias to ``PATH``.

    Whisper shells out to a command literally named ``ffmpeg``.  Windows
    installations frequently provide it through ``imageio-ffmpeg`` under a
    versioned filename (``ffmpeg-win-x86_64-v7.1.exe``), which is not
    discoverable by name.  Create a stable alias in a temp directory and
    prepend it to ``PATH``.  Returns ``None`` when no executable is available.
    """
    existing = shutil.which("ffmpeg")
    if existing:
        return existing

    try:
        import imageio_ffmpeg

        source = Path(imageio_ffmpeg.get_ffmpeg_exe())
    except Exception:
        return None
    if not source.exists():
        return None

    shim_dir = Path(tempfile.gettempdir()) / "voix_ffmpeg"
    shim_dir.mkdir(parents=True, exist_ok=True)
    target = shim_dir / ("ffmpeg.exe" if os.name == "nt" else "ffmpeg")
    try:
        if not target.exists() or target.stat().st_size != source.stat().st_size:
            try:
                os.link(source, target)
            except OSError:
                shutil.copy2(source, target)
    except OSError:
        return None

    os.environ["PATH"] = str(shim_dir) + os.pathsep + os.environ.get("PATH", "")
    return str(target)


@lru_cache(maxsize=2)
@lru_cache(maxsize=2)
def _load_whisper_model(model_size: str, device: str):
    import whisper

    return whisper.load_model(model_size, device=device)


def normalize_text(text: str) -> str:
    """Lowercase and strip punctuation for WER comparison."""
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9'\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _edit_distance(reference: Sequence[str], hypothesis: Sequence[str]) -> int:
    """Levenshtein distance over word tokens."""
    if not reference:
        return len(hypothesis)
    if not hypothesis:
        return len(reference)
    previous = list(range(len(hypothesis) + 1))
    for i, ref_word in enumerate(reference, start=1):
        current = [i]
        for j, hyp_word in enumerate(hypothesis, start=1):
            insert = current[j - 1] + 1
            delete = previous[j] + 1
            substitute = previous[j - 1] + (ref_word != hyp_word)
            current.append(min(insert, delete, substitute))
        previous = current
    return previous[-1]


def word_error_rate(reference: str, hypothesis: str) -> float:
    """Word error rate in [0, 1+] using normalized text."""
    ref_tokens = normalize_text(reference).split()
    hyp_tokens = normalize_text(hypothesis).split()
    if not ref_tokens:
        raise ValueError("reference text is empty after normalization")
    return _edit_distance(ref_tokens, hyp_tokens) / len(ref_tokens)


def transcribe_wer(
    audio_paths: Iterable[Union[str, Path]],
    reference_text: Union[str, Sequence[str]],
    *,
    model_size: str = "base",
    device: str = "cpu",
) -> Dict[str, object]:
    """Transcribe generated audio with Whisper and compute WER.

    ``reference_text`` may be a single string (all clips share the prompt) or
    one reference per clip.  Per-clip references are required for the showcase,
    where cross-lingual samples have different prompts.

    Returns ``status="unavailable"`` when Whisper is not installed.  A missing
    transcription is never reported as a WER of 0.
    """
    paths = [str(path) for path in audio_paths]
    if not paths:
        return {"status": "unavailable", "reason": "no audio paths", "wer": None}
    if isinstance(reference_text, str):
        references = [reference_text] * len(paths)
    else:
        references = list(reference_text)
        if len(references) != len(paths):
            raise ValueError(
                f"expected {len(paths)} reference texts, got {len(references)}"
            )
    if any(not str(reference).strip() for reference in references):
        raise ValueError("reference_text contains an empty prompt")
    try:
        import whisper  # noqa: F401
    except ImportError:
        return {
            "status": "unavailable",
            "reason": "openai-whisper is not installed",
            "wer": None,
        }

    if ensure_ffmpeg_on_path() is None:
        return {
            "status": "unavailable",
            "reason": "ffmpeg executable not found (install ffmpeg or imageio-ffmpeg)",
            "wer": None,
        }

    model = _load_whisper_model(model_size, device)
    transcripts: List[str] = []
    scores: List[float] = []
    for path, reference in zip(paths, references):
        result = model.transcribe(path, fp16=device != "cpu")
        text = str(result.get("text", "")).strip()
        transcripts.append(text)
        scores.append(word_error_rate(reference, text))
    return {
        "status": "computed",
        "model": f"whisper-{model_size}",
        "n": len(scores),
        "wer": float(np.mean(scores)),
        "wer_std": float(np.std(scores)),
        "per_sample_wer": scores,
        "transcripts": transcripts,
    }


def _mel_statistics(
    path: Union[str, Path],
    *,
    sample_rate: int = 16000,
    n_mels: int = 80,
) -> Optional[tuple[np.ndarray, np.ndarray]]:
    try:
        import librosa

        wav, _ = librosa.load(str(path), sr=sample_rate, mono=True)
        if wav.size < sample_rate // 4:
            return None
        mel = librosa.feature.melspectrogram(
            y=wav, sr=sample_rate, n_mels=n_mels, n_fft=1024, hop_length=256
        )
        log_mel = librosa.power_to_db(mel, ref=np.max)
        # Treat time frames as samples of a mel-frequency distribution.
        vectors = log_mel.T
        mean = vectors.mean(axis=0)
        covariance = np.cov(vectors, rowvar=False)
        return mean, covariance
    except Exception:
        return None


def frechet_distance(
    mean_a: np.ndarray,
    covariance_a: np.ndarray,
    mean_b: np.ndarray,
    covariance_b: np.ndarray,
) -> float:
    """Fréchet distance between two Gaussians."""
    from scipy.linalg import sqrtm

    diff = mean_a - mean_b
    covmean = sqrtm(covariance_a @ covariance_b)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    value = diff @ diff + np.trace(covariance_a + covariance_b - 2.0 * covmean)
    return float(max(0.0, value))


def fad_proxy(
    reference_paths: Iterable[Union[str, Path]],
    generated_paths: Iterable[Union[str, Path]],
    *,
    sample_rate: int = 16000,
    n_mels: int = 80,
) -> Dict[str, object]:
    """Fréchet distance over mel-spectrogram statistics.

    This is a reproducible proxy for FAD, not the official VGGish-based metric.
    """
    reference_paths = [Path(path) for path in reference_paths]
    generated_paths = [Path(path) for path in generated_paths]
    if not reference_paths or not generated_paths:
        return {
            "status": "unavailable",
            "reason": "reference and generated audio are both required",
            "fad_proxy": None,
        }

    def pooled(paths: List[Path]) -> Optional[tuple[np.ndarray, np.ndarray]]:
        means, covariances, weights = [], [], []
        for path in paths:
            stats = _mel_statistics(
                path, sample_rate=sample_rate, n_mels=n_mels
            )
            if stats is None:
                continue
            mean, covariance = stats
            means.append(mean)
            covariances.append(covariance)
            weights.append(1.0)
        if not means:
            return None
        weights_arr = np.asarray(weights) / np.sum(weights)
        pooled_mean = np.sum(
            [weight * mean for weight, mean in zip(weights_arr, means)], axis=0
        )
        pooled_cov = np.sum(
            [
                weight * covariance
                for weight, covariance in zip(weights_arr, covariances)
            ],
            axis=0,
        )
        return pooled_mean, pooled_cov

    ref_stats = pooled(reference_paths)
    gen_stats = pooled(generated_paths)
    if ref_stats is None or gen_stats is None:
        return {
            "status": "unavailable",
            "reason": "could not read enough audio for mel statistics",
            "fad_proxy": None,
        }
    value = frechet_distance(*ref_stats, *gen_stats)
    return {
        "status": "computed",
        "metric": "mel-spectrogram frechet proxy (not VGGish FAD)",
        "n_reference": len(reference_paths),
        "n_generated": len(generated_paths),
        "fad_proxy": value,
    }


def embedding_diversity(
    audio_paths: Iterable[Union[str, Path]],
    *,
    device: str = "cpu",
) -> Dict[str, object]:
    """Mean pairwise cosine distance between ECAPA embeddings."""
    paths = [str(path) for path in audio_paths]
    if len(paths) < 2:
        return {
            "status": "unavailable",
            "reason": "at least two audio files are required",
            "pairwise_cosine": None,
        }
    try:
        import torch
        import torch.nn.functional as F

        from voix.evaluation.metrics import _get_ecapa, _load_wav

        extractor = _get_ecapa(device)
        embeddings = []
        for path in paths:
            wav, _ = _load_wav(path, 16000)
            embedding = extractor.extract(
                torch.from_numpy(np.ascontiguousarray(wav)).unsqueeze(0)
            ).squeeze(0)
            embeddings.append(embedding)
        normed = F.normalize(torch.stack(embeddings).float(), dim=-1)
        similarity = normed @ normed.T
        n = len(normed)
        off_diagonal = ~torch.eye(n, dtype=torch.bool)
        pairwise = float(similarity[off_diagonal].mean().item())
        return {
            "status": "computed",
            "n": n,
            "pairwise_cosine": pairwise,
            "pairwise_cosine_distance": 1.0 - pairwise,
        }
    except Exception as exc:
        return {
            "status": "unavailable",
            "reason": f"{type(exc).__name__}: {exc}",
            "pairwise_cosine": None,
        }
