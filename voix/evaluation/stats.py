"""Bootstrap and multiple-comparison utilities for research reports.

The functions here are deliberately model-free and deterministic given a seed.
They implement the analysis contract in ``docs/MASTER.md`` section 5.3:

* paired bootstrap by speaker for TTS metrics;
* speaker-cluster bootstrap for classifier metrics;
* percentile confidence intervals, not normal-approximation intervals;
* Holm-Bonferroni correction across pre-registered primary metrics.

No function here invents a metric.  Empty or non-finite inputs raise a
``ValueError`` rather than returning a placeholder score.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np


@dataclass(frozen=True)
class BootstrapResult:
    """A point estimate and percentile bootstrap interval."""

    estimate: float
    ci_low: float
    ci_high: float
    n: int
    n_resamples: int
    confidence: float
    seed: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "estimate": self.estimate,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "n": self.n,
            "n_resamples": self.n_resamples,
            "confidence": self.confidence,
            "seed": self.seed,
        }


def _finite_1d(values: Iterable[float], *, name: str) -> np.ndarray:
    array = np.asarray(list(values), dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if array.size == 0:
        raise ValueError(f"{name} must not be empty")
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        raise ValueError(f"{name} contains no finite values")
    return finite


def _percentile_interval(
    samples: np.ndarray,
    *,
    confidence: float,
) -> tuple[float, float]:
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    alpha = 1.0 - confidence
    low, high = np.percentile(samples, [100.0 * alpha / 2.0, 100.0 * (1.0 - alpha / 2.0)])
    return float(low), float(high)


def bootstrap_mean_ci(
    values: Iterable[float],
    *,
    n_resamples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 0,
) -> BootstrapResult:
    """Percentile bootstrap CI for the mean of independent observations."""
    array = _finite_1d(values, name="values")
    if n_resamples <= 0:
        raise ValueError("n_resamples must be positive")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, array.size, size=(n_resamples, array.size))
    means = array[indices].mean(axis=1)
    low, high = _percentile_interval(means, confidence=confidence)
    return BootstrapResult(
        estimate=float(array.mean()),
        ci_low=low,
        ci_high=high,
        n=int(array.size),
        n_resamples=int(n_resamples),
        confidence=float(confidence),
        seed=int(seed),
    )


def paired_bootstrap_difference(
    a: Iterable[float],
    b: Iterable[float],
    *,
    n_resamples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 0,
) -> BootstrapResult:
    """Percentile bootstrap CI for the mean paired difference ``a - b``.

    Pairs with a non-finite value on either side are dropped before resampling.
    The returned ``n`` is the number of complete pairs actually used.
    """
    left = np.asarray(list(a), dtype=np.float64)
    right = np.asarray(list(b), dtype=np.float64)
    if left.shape != right.shape:
        raise ValueError("paired inputs must have the same length")
    if left.ndim != 1:
        raise ValueError("paired inputs must be one-dimensional")
    valid = np.isfinite(left) & np.isfinite(right)
    if not valid.any():
        raise ValueError("no complete finite pairs")
    differences = left[valid] - right[valid]
    if n_resamples <= 0:
        raise ValueError("n_resamples must be positive")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, differences.size, size=(n_resamples, differences.size))
    means = differences[indices].mean(axis=1)
    low, high = _percentile_interval(means, confidence=confidence)
    return BootstrapResult(
        estimate=float(differences.mean()),
        ci_low=low,
        ci_high=high,
        n=int(differences.size),
        n_resamples=int(n_resamples),
        confidence=float(confidence),
        seed=int(seed),
    )


def paired_effect_size(a: Iterable[float], b: Iterable[float]) -> Dict[str, Any]:
    """Cohen's dz for paired observations (mean difference / SD of differences)."""
    left = np.asarray(list(a), dtype=np.float64)
    right = np.asarray(list(b), dtype=np.float64)
    if left.shape != right.shape or left.ndim != 1:
        raise ValueError("paired inputs must be one-dimensional and the same length")
    valid = np.isfinite(left) & np.isfinite(right)
    differences = left[valid] - right[valid]
    if differences.size < 2:
        raise ValueError("at least two complete pairs are required")
    sd = float(differences.std(ddof=1))
    mean = float(differences.mean())
    if sd == 0.0:
        return {
            "mean_difference": mean,
            "sd_difference": 0.0,
            "cohens_dz": None,
            "n": int(differences.size),
            "reason": "zero variance in paired differences",
        }
    return {
        "mean_difference": mean,
        "sd_difference": sd,
        "cohens_dz": mean / sd,
        "n": int(differences.size),
    }


def speaker_cluster_bootstrap(
    rows: Sequence[Mapping[str, Any]],
    metric_fn: Callable[[Sequence[Mapping[str, Any]]], float],
    *,
    speaker_key: str = "speaker_id",
    n_resamples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 0,
) -> BootstrapResult:
    """Cluster bootstrap that resamples speakers, not individual rows.

    ``metric_fn`` receives the pooled rows for the sampled speakers and must
    return a finite float.  Duplicate speakers are allowed, matching the
    standard cluster-bootstrap construction.
    """
    if not rows:
        raise ValueError("rows must not be empty")
    if n_resamples <= 0:
        raise ValueError("n_resamples must be positive")
    by_speaker: Dict[str, List[Mapping[str, Any]]] = {}
    for row in rows:
        if speaker_key not in row:
            raise ValueError(f"row is missing {speaker_key!r}")
        speaker = str(row[speaker_key])
        by_speaker.setdefault(speaker, []).append(row)
    speakers = sorted(by_speaker)
    if len(speakers) < 2:
        raise ValueError("at least two speakers are required for a cluster bootstrap")
    point = float(metric_fn(rows))
    if not np.isfinite(point):
        raise ValueError("metric_fn returned a non-finite point estimate")

    rng = np.random.default_rng(seed)
    samples = np.empty(n_resamples, dtype=np.float64)
    for index in range(n_resamples):
        drawn = rng.integers(0, len(speakers), size=len(speakers))
        pooled: List[Mapping[str, Any]] = []
        for speaker_index in drawn:
            pooled.extend(by_speaker[speakers[int(speaker_index)]])
        value = float(metric_fn(pooled))
        samples[index] = value if np.isfinite(value) else np.nan

    finite = samples[np.isfinite(samples)]
    if finite.size == 0:
        raise ValueError("metric_fn returned non-finite values for every resample")
    low, high = _percentile_interval(finite, confidence=confidence)
    return BootstrapResult(
        estimate=point,
        ci_low=low,
        ci_high=high,
        n=len(speakers),
        n_resamples=int(n_resamples),
        confidence=float(confidence),
        seed=int(seed),
    )


def holm_bonferroni(
    pvalues: Mapping[str, float],
    *,
    alpha: float = 0.05,
) -> Dict[str, Dict[str, Any]]:
    """Holm-Bonferroni step-down correction.

    Returns one entry per input metric with the raw p-value, adjusted p-value,
    rank, and rejection decision.  Ties are ordered by name for determinism.
    """
    if not pvalues:
        raise ValueError("pvalues must not be empty")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    cleaned: Dict[str, float] = {}
    for name, value in pvalues.items():
        number = float(value)
        if not np.isfinite(number) or not 0.0 <= number <= 1.0:
            raise ValueError(f"p-value for {name!r} must be finite and in [0, 1]")
        cleaned[str(name)] = number

    ordered = sorted(cleaned.items(), key=lambda item: (item[1], item[0]))
    count = len(ordered)
    adjusted: Dict[str, float] = {}
    running = 0.0
    for rank, (name, pvalue) in enumerate(ordered, start=1):
        candidate = min(1.0, (count - rank + 1) * pvalue)
        running = max(running, candidate)
        adjusted[name] = running

    return {
        name: {
            "p_value": cleaned[name],
            "adjusted_p_value": adjusted[name],
            "rank": next(
                rank for rank, (candidate, _) in enumerate(ordered, start=1) if candidate == name
            ),
            "rejected": adjusted[name] <= alpha,
            "alpha": alpha,
        }
        for name in cleaned
    }
