"""Objective naturalness proxies and paired prompt/delivery analysis.

These metrics are acoustic proxies.  They do not measure perceived naturalness,
and this module never fabricates MOS or CMOS.  Human listening remains the
confirmatory naturalness path.

The N0a audit holds the face, voice sample, diffusion settings, and every style
control fixed, varies only the text register (formal vs conversational), and
asks whether the resulting delivery shift is larger than the pipeline's own
seed-resampling noise.  Prompt pairs are the independent unit for inference.
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from voix.evaluation.stats import bootstrap_mean_ci

SAMPLE_RATE = 24000
F0_MIN_HZ = 50.0
F0_MAX_HZ = 600.0
PAUSE_MIN_SECONDS = 0.15
SILENCE_TOP_DB = 30.0
SPEECH_DYNAMIC_RANGE_DB = 30.0
SPECTRAL_TILT_MIN_KHZ = 0.1
SPECTRAL_TILT_MAX_KHZ = 8.0

PROMPT_REGISTERS: Tuple[str, str] = ("formal", "conversational")

CLIP_METRICS: Tuple[str, ...] = (
    "duration_s",
    "speech_duration_s",
    "word_count",
    "words_per_second",
    "f0_median_hz",
    "f0_std_st",
    "f0_range_st",
    "f0_voiced_ratio",
    "energy_std_db",
    "pause_count",
    "pause_ratio",
    "spectral_tilt_db_per_khz",
)

PROMPT_EFFECT_METRICS: Tuple[str, ...] = (
    "f0_std_st",
    "f0_range_st",
    "words_per_second",
    "energy_std_db",
    "pause_ratio",
    "spectral_tilt_db_per_khz",
)

_WORD_RE = re.compile(r"[A-Za-z]+(?:['\u2019][A-Za-z]+)?")


def word_count(text: Optional[str]) -> int:
    """Count whitespace-delimited English word tokens, keeping contractions."""
    if text is None:
        return 0
    return len(_WORD_RE.findall(str(text)))


def load_prompt_pairs(path: Union[str, Path]) -> List[Dict[str, str]]:
    """Load and validate the N0a formal/conversational prompt pairs."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != "voix.n0a_prompts.v1":
        raise ValueError(
            "prompt config schema_version must be 'voix.n0a_prompts.v1'"
        )
    pairs = payload.get("pairs")
    if not isinstance(pairs, list) or not pairs:
        raise ValueError("prompt config must contain a non-empty 'pairs' list")
    seen: set[str] = set()
    result: List[Dict[str, str]] = []
    for index, item in enumerate(pairs):
        if not isinstance(item, Mapping):
            raise ValueError(f"prompt pair {index} must be an object")
        pair_id = str(item.get("pair_id", "")).strip()
        formal = str(item.get("formal", "")).strip()
        conversational = str(item.get("conversational", "")).strip()
        if not pair_id:
            raise ValueError(f"prompt pair {index} is missing pair_id")
        if pair_id in seen:
            raise ValueError(f"duplicate prompt pair_id: {pair_id}")
        if not formal or not conversational:
            raise ValueError(
                f"prompt pair {pair_id} requires formal and conversational text"
            )
        seen.add(pair_id)
        result.append(
            {
                "pair_id": pair_id,
                "topic": str(item.get("topic", "")).strip(),
                "formal": formal,
                "conversational": conversational,
            }
        )
    return result


def _load_audio(
    audio: Union[str, Path, np.ndarray],
    sample_rate: int,
) -> np.ndarray:
    if isinstance(audio, (str, Path)):
        import librosa

        path = Path(audio)
        if not path.is_file():
            raise FileNotFoundError(f"audio not found: {path}")
        wav, _ = librosa.load(str(path), sr=sample_rate, mono=True)
    else:
        wav = np.asarray(audio, dtype=np.float64)
        if wav.ndim == 2 and 1 in wav.shape:
            wav = wav.reshape(-1)
        if wav.ndim != 1:
            raise ValueError("audio array must be one-dimensional")
    wav = np.asarray(wav, dtype=np.float64)
    if wav.size == 0:
        raise ValueError("empty audio")
    if not np.all(np.isfinite(wav)):
        wav = np.nan_to_num(wav, nan=0.0, posinf=0.0, neginf=0.0)
    return wav


def _f0_stats(wav: np.ndarray, sample_rate: int) -> Dict[str, Optional[float]]:
    import librosa

    f0, voiced_flag, _ = librosa.pyin(
        wav,
        fmin=F0_MIN_HZ,
        fmax=F0_MAX_HZ,
        sr=sample_rate,
        frame_length=2048,
        hop_length=512,
        fill_na=np.nan,
    )
    voiced = f0[
        np.asarray(voiced_flag, dtype=bool) & np.isfinite(f0) & (f0 > 0.0)
    ]
    voiced_ratio = float(voiced.size) / max(1, f0.size)
    if voiced.size < 3:
        return {
            "f0_median_hz": None,
            "f0_std_st": None,
            "f0_range_st": None,
            "f0_voiced_ratio": voiced_ratio,
        }
    median = float(np.median(voiced))
    semitones = 12.0 * np.log2(voiced / median)
    return {
        "f0_median_hz": median,
        "f0_std_st": float(np.std(semitones, ddof=0)),
        "f0_range_st": float(
            np.percentile(semitones, 95) - np.percentile(semitones, 5)
        ),
        "f0_voiced_ratio": voiced_ratio,
    }


def _pause_stats(wav: np.ndarray, sample_rate: int) -> Dict[str, float]:
    import librosa

    # ``librosa.effects.split`` returns one whole-clip interval for an
    # all-zero signal, which would otherwise invent a speech rate for silence.
    if float(np.max(np.abs(wav))) < 1e-6:
        return {
            "speech_duration_s": 0.0,
            "pause_count": 0.0,
            "pause_ratio": 0.0,
        }
    intervals = librosa.effects.split(wav, top_db=SILENCE_TOP_DB)
    if len(intervals) == 0:
        return {
            "speech_duration_s": 0.0,
            "pause_count": 0.0,
            "pause_ratio": 0.0,
        }
    speech_samples = sum(int(end - start) for start, end in intervals)
    span_samples = int(intervals[-1][1] - intervals[0][0])
    gaps = [
        (intervals[index + 1][0] - intervals[index][1]) / sample_rate
        for index in range(len(intervals) - 1)
    ]
    gaps = [float(gap) for gap in gaps if gap > 0.0]
    return {
        "speech_duration_s": float(speech_samples / sample_rate),
        "pause_count": float(
            sum(1 for gap in gaps if gap >= PAUSE_MIN_SECONDS)
        ),
        "pause_ratio": float(sum(gaps) / (span_samples / sample_rate))
        if span_samples > 0
        else 0.0,
    }


def _energy_and_tilt(
    wav: np.ndarray,
    sample_rate: int,
) -> Dict[str, Optional[float]]:
    import librosa

    n_fft = 2048
    hop_length = 512
    rms = librosa.feature.rms(
        y=wav, frame_length=n_fft, hop_length=hop_length
    )[0]
    if rms.size == 0:
        return {"energy_std_db": None, "spectral_tilt_db_per_khz": None}
    rms_db = 20.0 * np.log10(np.maximum(rms, 1e-10))
    threshold = float(
        np.percentile(rms_db, 95) - SPEECH_DYNAMIC_RANGE_DB
    )
    speech_mask = rms_db >= threshold
    if not speech_mask.any():
        speech_mask = np.ones_like(rms_db, dtype=bool)
    energy_std_db = float(np.std(rms_db[speech_mask], ddof=0))

    stft = np.abs(librosa.stft(wav, n_fft=n_fft, hop_length=hop_length)) ** 2
    frequencies = librosa.fft_frequencies(sr=sample_rate, n_fft=n_fft)
    band = (frequencies >= SPECTRAL_TILT_MIN_KHZ * 1000.0) & (
        frequencies <= SPECTRAL_TILT_MAX_KHZ * 1000.0
    )
    frames = min(stft.shape[1], speech_mask.size)
    tilt: Optional[float]
    if frames == 0 or not band.any():
        tilt = None
    else:
        power = stft[band][:, :frames][:, speech_mask[:frames]].mean(axis=1)
        power_db = 10.0 * np.log10(np.maximum(power, 1e-12))
        tilt = float(
            np.polyfit(frequencies[band] / 1000.0, power_db, 1)[0]
        )
    return {
        "energy_std_db": energy_std_db,
        "spectral_tilt_db_per_khz": tilt,
    }


def extract_naturalness_metrics(
    audio: Union[str, Path, np.ndarray],
    *,
    text: Optional[str] = None,
    sample_rate: int = SAMPLE_RATE,
) -> Dict[str, Optional[float]]:
    """Compute objective delivery proxies for one clip.

    ``words_per_second`` uses the speech span (pauses excluded) as the
    denominator, so a slower speaking rate is not confused with longer pauses.
    """
    wav = _load_audio(audio, sample_rate)
    metrics: Dict[str, Optional[float]] = {
        "duration_s": float(wav.size / sample_rate),
        "word_count": float(word_count(text)),
    }
    metrics.update(_pause_stats(wav, sample_rate))
    metrics.update(_f0_stats(wav, sample_rate))
    metrics.update(_energy_and_tilt(wav, sample_rate))
    speech_duration = metrics.get("speech_duration_s") or 0.0
    if text is not None and speech_duration > 0.0:
        metrics["words_per_second"] = float(
            metrics["word_count"] / speech_duration
        )
    else:
        metrics["words_per_second"] = None
    for name in CLIP_METRICS:
        metrics.setdefault(name, None)
    return {name: metrics[name] for name in CLIP_METRICS}


def _as_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _pooled_within_sd(
    cells: Mapping[Tuple[str, str], Mapping[int, Mapping[str, Optional[float]]]],
    metric: str,
) -> Optional[float]:
    """Pooled within-cell SD across seeds (cells are pair x register)."""
    numerator = 0.0
    denominator = 0
    for by_seed in cells.values():
        values = [
            entry.get(metric)
            for entry in by_seed.values()
            if entry.get(metric) is not None
        ]
        if len(values) < 2:
            continue
        numerator += float(np.var(values, ddof=1)) * (len(values) - 1)
        denominator += len(values) - 1
    if denominator == 0:
        return None
    return math.sqrt(numerator / denominator)


def paired_prompt_effects(
    rows: Sequence[Mapping[str, Any]],
    *,
    metrics: Sequence[str] = PROMPT_EFFECT_METRICS,
    n_resamples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 0,
) -> Dict[str, Any]:
    """Compare conversational-minus-formal delivery across prompt pairs.

    Each row is one generated clip and must contain ``prompt_pair_id``,
    ``register`` (formal/conversational), ``seed``, and the metric columns.
    A pair contributes one delta per metric, computed as the mean of the
    per-seed conversational-minus-formal differences.  The bootstrap resamples
    prompt pairs, not clips or seeds.

    ``seed_noise_sd`` is the pooled within-cell SD across seeds.  The
    ``seed_difference_noise_sd`` scales that to a difference between two
    independent cells.  ``effect_to_noise`` is descriptive: seed variation
    includes CVAE identity resampling, so it is a conservative noise floor for
    a prompt effect measured at a fixed seed.
    """
    if not rows:
        raise ValueError("rows must not be empty")
    if not metrics:
        raise ValueError("metrics must not be empty")

    cells: Dict[Tuple[str, str], Dict[int, Dict[str, Optional[float]]]] = {}
    pair_ids: set[str] = set()
    for row in rows:
        pair_id = str(row.get("prompt_pair_id") or "").strip()
        register = str(row.get("register") or "").strip().lower()
        if not pair_id:
            raise ValueError("every row requires prompt_pair_id")
        if register not in PROMPT_REGISTERS:
            raise ValueError(
                f"register must be one of {PROMPT_REGISTERS}, got {register!r}"
            )
        if row.get("seed") is None:
            raise ValueError(f"row {pair_id}/{register} requires a seed")
        seed_value = int(row["seed"])
        key = (pair_id, register)
        by_seed = cells.setdefault(key, {})
        if seed_value in by_seed:
            raise ValueError(
                f"duplicate row for pair={pair_id} register={register} "
                f"seed={seed_value}"
            )
        by_seed[seed_value] = {
            metric: _as_float(row.get(metric)) for metric in metrics
        }
        pair_ids.add(pair_id)

    per_metric_deltas: Dict[str, List[float]] = {metric: [] for metric in metrics}
    used_pairs: List[Dict[str, Any]] = []
    skipped_pairs: List[Dict[str, str]] = []
    for pair_id in sorted(pair_ids):
        formal = cells.get((pair_id, "formal"), {})
        conversational = cells.get((pair_id, "conversational"), {})
        common_seeds = sorted(set(formal) & set(conversational))
        if not common_seeds:
            skipped_pairs.append(
                {
                    "pair_id": pair_id,
                    "reason": "no common seeds across registers",
                }
            )
            continue
        pair_deltas: Dict[str, float] = {}
        for metric in metrics:
            differences: List[float] = []
            for seed_value in common_seeds:
                left = conversational[seed_value].get(metric)
                right = formal[seed_value].get(metric)
                if left is not None and right is not None:
                    differences.append(float(left) - float(right))
            if differences:
                delta = float(np.mean(differences))
                pair_deltas[metric] = delta
                per_metric_deltas[metric].append(delta)
        if pair_deltas:
            used_pairs.append(
                {
                    "pair_id": pair_id,
                    "n_seeds": len(common_seeds),
                    "deltas": pair_deltas,
                }
            )
        else:
            skipped_pairs.append(
                {
                    "pair_id": pair_id,
                    "reason": "no finite metric values for common seeds",
                }
            )

    effects: Dict[str, Dict[str, Any]] = {}
    for metric in metrics:
        values = per_metric_deltas[metric]
        if len(values) < 2:
            effects[metric] = {
                "status": "unavailable",
                "reason": "fewer than two complete prompt pairs",
                "n_pairs": len(values),
            }
            continue
        interval = bootstrap_mean_ci(
            values,
            n_resamples=n_resamples,
            confidence=confidence,
            seed=seed,
        )
        within_sd = _pooled_within_sd(cells, metric)
        difference_noise = (
            math.sqrt(2.0) * within_sd if within_sd is not None else None
        )
        mean_delta = float(np.mean(values))
        sd_delta = float(np.std(values, ddof=1))
        effect_to_noise = (
            abs(mean_delta) / difference_noise
            if difference_noise is not None and difference_noise > 1e-12
            else None
        )
        cohens_dz = (
            mean_delta / sd_delta if sd_delta > 1e-12 else None
        )
        positives = sum(1 for value in values if value > 0.0)
        negatives = sum(1 for value in values if value < 0.0)
        sign_consistency = max(positives, negatives) / len(values)
        detected = interval.ci_low > 0.0 or interval.ci_high < 0.0
        effects[metric] = {
            "status": "computed",
            "mean_delta": mean_delta,
            "ci_low": interval.ci_low,
            "ci_high": interval.ci_high,
            "n_pairs": interval.n,
            "n_resamples": interval.n_resamples,
            "confidence": interval.confidence,
            "bootstrap_seed": interval.seed,
            "sd_delta": sd_delta,
            "cohens_dz": cohens_dz,
            "seed_noise_sd": within_sd,
            "seed_difference_noise_sd": difference_noise,
            "effect_to_noise": effect_to_noise,
            "sign_consistency": sign_consistency,
            "detected": bool(detected),
            "exceeds_seed_noise": bool(
                effect_to_noise is not None and effect_to_noise >= 1.0
            ),
            "robustly_detected": bool(
                detected
                and effect_to_noise is not None
                and effect_to_noise >= 1.0
            ),
            "direction": (
                "conversational_higher"
                if mean_delta > 0.0
                else "conversational_lower"
                if mean_delta < 0.0
                else "none"
            ),
        }

    return {
        "metrics": effects,
        "pairs": used_pairs,
        "skipped_pairs": skipped_pairs,
        "n_pairs": len(used_pairs),
        "registers": list(PROMPT_REGISTERS),
        "seed_noise_definition": (
            "pooled within-cell SD across seeds; difference noise = "
            "sqrt(2) * within-cell SD"
        ),
        "n_resamples": int(n_resamples),
        "confidence": float(confidence),
        "seed": int(seed),
    }


def summarize_prompt_effects(effects: Mapping[str, Any]) -> Dict[str, Any]:
    """Count detected and robustly detected metrics for the N0a decision."""
    metrics = effects.get("metrics") or {}
    computed = {
        name: entry
        for name, entry in metrics.items()
        if isinstance(entry, Mapping) and entry.get("status") == "computed"
    }
    detected = sorted(
        name for name, entry in computed.items() if entry.get("detected")
    )
    robust = sorted(
        name
        for name, entry in computed.items()
        if entry.get("robustly_detected")
    )
    return {
        "metrics_computed": sorted(computed),
        "metrics_detected": detected,
        "metrics_robustly_detected": robust,
        "n_metrics_computed": len(computed),
        "n_metrics_detected": len(detected),
        "n_metrics_robustly_detected": len(robust),
    }
