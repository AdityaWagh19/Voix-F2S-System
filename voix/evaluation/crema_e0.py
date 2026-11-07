"""Metric aggregation for the CREMA-D E0 paired evaluation.

All functions are model-free.  They take measured per-clip rows and return
speaker-cluster bootstrap intervals, paired effects, and explicit
``unavailable`` entries when the sample is too small for an interval.

Nothing here fabricates a score.  A condition with no measured value is
reported as unavailable with a reason.
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from voix.evaluation.stats import (
    paired_effect_size,
    speaker_cluster_bootstrap,
)

EMOTION_ORDER: Tuple[str, ...] = (
    "neutral",
    "happy",
    "sad",
    "angry",
    "fear",
    "disgust",
)

# Only the classes with a current fixed-delta artifact have a pre-registered
# direction.  Fear and disgust are reported as unavailable for E0.
EXPECTED_DIRECTIONS: Dict[str, Dict[str, int]] = {
    "happy": {"f0_mean_hz": 1, "duration_seconds": -1},
    "sad": {"f0_mean_hz": -1, "duration_seconds": 1},
    "angry": {"f0_mean_hz": 1, "duration_seconds": -1},
}

METRIC_LABELS: Dict[str, str] = {
    "f0_mean_hz": "F0 mean (Hz)",
    "f0_std_hz": "F0 SD (Hz)",
    "duration_seconds": "Duration (s)",
    "energy_mean": "RMS energy",
    "energy_std": "RMS energy SD",
    "ecapa_generated_neutral": "ECAPA to generated neutral",
    "ecapa_identity_reference": "ECAPA to identity reference",
    "ecapa_real_neutral": "ECAPA to real neutral",
}


def _finite(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _metric_summary(
    values: Sequence[float],
    *,
    groups: Sequence[str],
    n_resamples: int,
    confidence: float,
    seed: int,
) -> Dict[str, Any]:
    finite = np.asarray([value for value in values if _finite(value) is not None], dtype=np.float64)
    if finite.size == 0:
        return {"status": "unavailable", "reason": "no finite values"}
    summary: Dict[str, Any] = {
        "status": "computed",
        "mean": float(finite.mean()),
        "std": float(finite.std(ddof=1)) if finite.size > 1 else None,
        "n": int(finite.size),
    }
    rows = [
        {"speaker_id": str(group), "value": float(value)}
        for value, group in zip(values, groups)
        if _finite(value) is not None
    ]
    speakers = {row["speaker_id"] for row in rows}
    summary["n_speakers"] = len(speakers)
    if len(speakers) < 2:
        summary.update(
            {
                "ci_low": None,
                "ci_high": None,
                "ci_status": "unavailable",
                "ci_reason": "fewer than two speakers",
            }
        )
        return summary
    result = speaker_cluster_bootstrap(
        rows,
        lambda selected: float(np.mean([row["value"] for row in selected])),
        speaker_key="speaker_id",
        n_resamples=n_resamples,
        confidence=confidence,
        seed=seed,
    )
    summary.update(
        {
            "ci_low": result.ci_low,
            "ci_high": result.ci_high,
            "ci_status": "computed",
            "ci_confidence": confidence,
            "ci_n_resamples": n_resamples,
            "ci_seed": seed,
        }
    )
    return summary


def aggregate_conditions(
    frame: pd.DataFrame,
    *,
    metrics: Sequence[str],
    n_resamples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 0,
) -> Dict[str, Any]:
    """Aggregate measured metrics by ``(condition, emotion)``."""
    required = {"condition", "emotion", "actor_id"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"frame is missing columns: {sorted(missing)}")
    conditions: Dict[str, Any] = {}
    for (condition, emotion), subset in frame.groupby(["condition", "emotion"], sort=False):
        entry: Dict[str, Any] = {
            "condition": str(condition),
            "emotion": str(emotion),
            "n": int(len(subset)),
            "actors": int(subset["actor_id"].nunique()),
            "metrics": {},
        }
        for metric in metrics:
            if metric not in subset.columns:
                entry["metrics"][metric] = {
                    "status": "unavailable",
                    "reason": f"column {metric!r} is missing",
                }
                continue
            entry["metrics"][metric] = _metric_summary(
                subset[metric].tolist(),
                groups=subset["actor_id"].astype(str).tolist(),
                n_resamples=n_resamples,
                confidence=confidence,
                seed=seed,
            )
        conditions[f"{condition}:{emotion}"] = entry
    return conditions


def feature_domain_shift(
    train_features: np.ndarray,
    real_features: np.ndarray,
    generated_features: np.ndarray,
    *,
    threshold: float = 3.0,
) -> Dict[str, Any]:
    """Measure how far generated features are from the real-speech distribution.

    The emotion classifier is trained on real CREMA-D features.  A generated
    clip can be classified with high confidence and still be meaningless if the
    whole generated domain lies outside that feature distribution.  We fit a
    scaler on the training features, compare the generated centroid to the real
    validation centroid, and normalize by the real validation spread.

    A ratio above ``threshold`` marks the classifier proxy invalid for that
    run.  It does not say the generated emotion is wrong; it says the classifier
    cannot measure it.
    """
    from sklearn.preprocessing import StandardScaler

    train = np.asarray(train_features, dtype=np.float64)
    real = np.asarray(real_features, dtype=np.float64)
    generated = np.asarray(generated_features, dtype=np.float64)
    if train.ndim != 2 or real.ndim != 2 or generated.ndim != 2:
        raise ValueError("feature matrices must be two-dimensional")
    if train.shape[1] != real.shape[1] or train.shape[1] != generated.shape[1]:
        raise ValueError("feature matrices must have the same width")
    if train.shape[0] < 2 or real.shape[0] < 1 or generated.shape[0] < 1:
        return {
            "status": "unavailable",
            "reason": "need at least two train rows, one real row, and one generated row",
        }
    if not (
        np.isfinite(train).all()
        and np.isfinite(real).all()
        and np.isfinite(generated).all()
    ):
        return {"status": "unavailable", "reason": "feature matrices contain non-finite values"}

    scaler = StandardScaler().fit(train)
    real_scaled = scaler.transform(real)
    generated_scaled = scaler.transform(generated)
    real_centroid = real_scaled.mean(axis=0)
    generated_centroid = generated_scaled.mean(axis=0)
    real_spread = float(
        np.mean(np.linalg.norm(real_scaled - real_centroid, axis=1))
    )
    centroid_distance = float(np.linalg.norm(generated_centroid - real_centroid))
    ratio = centroid_distance / max(real_spread, 1e-9)
    valid = bool(ratio <= float(threshold))
    return {
        "status": "computed",
        "standardized_centroid_distance": centroid_distance,
        "real_val_spread": real_spread,
        "ratio": ratio,
        "threshold": float(threshold),
        "valid": valid,
        "reason": (
            "generated features are within the real-speech spread"
            if valid
            else (
                "generated features are far outside the real-speech spread; "
                "classifier predictions are not a valid emotion measure"
            )
        ),
    }


def classification_metrics(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    groups: Sequence[str],
    *,
    n_resamples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 0,
) -> Dict[str, Any]:
    """Accuracy, macro-F1, per-class recall, confusion matrix, cluster CIs."""
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        f1_score,
        precision_recall_fscore_support,
    )

    if len(y_true) != len(y_pred) or len(y_true) != len(groups):
        raise ValueError("y_true, y_pred, and groups must have the same length")
    if not y_true:
        return {"status": "unavailable", "reason": "no rows"}
    labels = [label for label in EMOTION_ORDER if label in set(y_true) | set(y_pred)]
    labels += sorted((set(y_true) | set(y_pred)) - set(labels))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    per_class = {
        label: {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(support[index]),
        }
        for index, label in enumerate(labels)
    }
    rows = [
        {"speaker_id": str(group), "correct": float(str(true) == str(pred))}
        for true, pred, group in zip(y_true, y_pred, groups)
    ]
    speakers = {row["speaker_id"] for row in rows}
    accuracy = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0))
    result: Dict[str, Any] = {
        "status": "computed",
        "n": len(y_true),
        "n_speakers": len(speakers),
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "per_class": per_class,
        "confusion_matrix": {
            "labels": labels,
            "matrix": confusion_matrix(y_true, y_pred, labels=labels).astype(int).tolist(),
        },
        "speaker_cluster_ci": {
            "status": "unavailable",
            "reason": "fewer than two speakers",
        },
    }
    if len(speakers) >= 2:
        accuracy_result = speaker_cluster_bootstrap(
            rows,
            lambda selected: float(np.mean([row["correct"] for row in selected])),
            speaker_key="speaker_id",
            n_resamples=n_resamples,
            confidence=confidence,
            seed=seed,
        )
        result["speaker_cluster_ci"] = {
            "status": "computed",
            "accuracy_low": accuracy_result.ci_low,
            "accuracy_high": accuracy_result.ci_high,
            "confidence": confidence,
            "n_resamples": n_resamples,
            "seed": seed,
        }
    return result


def _paired_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Return one row per (actor, sentence, seed) with neutral and emotion values.

    The frame must contain one neutral row and one row for every non-neutral
    condition being compared.  The function pivots on ``condition`` so the
    fixed-delta and any future residual condition can be compared against the
    same generated neutral.
    """
    required = {"condition", "emotion", "actor_id", "sentence", "seed"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"frame is missing columns: {sorted(missing)}")
    neutral = frame[frame["condition"] == "neutral"].copy()
    if neutral.empty:
        return pd.DataFrame()
    neutral = neutral.rename(
        columns={
            column: f"neutral__{column}"
            for column in frame.columns
            if column not in {"condition", "emotion", "actor_id", "sentence", "seed"}
        }
    )
    key = ["actor_id", "sentence", "seed"]
    merged = frame[frame["condition"] != "neutral"].merge(
        neutral,
        on=key,
        how="inner",
        suffixes=("", "__neutral"),
    )
    return merged


def paired_delta_effects(
    frame: pd.DataFrame,
    *,
    metrics: Sequence[str],
    n_resamples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 0,
) -> Dict[str, Any]:
    """Paired emotion-minus-neutral effects with speaker-cluster intervals."""
    paired = _paired_frame(frame)
    if paired.empty:
        return {"status": "unavailable", "reason": "no paired neutral rows"}
    effects: Dict[str, Any] = {}
    for (condition, emotion), subset in paired.groupby(["condition", "emotion"], sort=False):
        condition_effects: Dict[str, Any] = {}
        for metric in metrics:
            neutral_column = f"neutral__{metric}"
            if metric not in subset.columns or neutral_column not in subset.columns:
                condition_effects[metric] = {
                    "status": "unavailable",
                    "reason": f"missing {metric!r} or its neutral counterpart",
                }
                continue
            left = pd.to_numeric(subset[metric], errors="coerce").to_numpy(dtype=np.float64)
            right = pd.to_numeric(subset[neutral_column], errors="coerce").to_numpy(dtype=np.float64)
            valid = np.isfinite(left) & np.isfinite(right)
            if not valid.any():
                condition_effects[metric] = {
                    "status": "unavailable",
                    "reason": "no complete pairs",
                }
                continue
            differences = left[valid] - right[valid]
            groups = subset.loc[valid, "actor_id"].astype(str).to_numpy()
            rows = [
                {"speaker_id": group, "delta": float(delta)}
                for group, delta in zip(groups, differences)
            ]
            speakers = {row["speaker_id"] for row in rows}
            effect: Dict[str, Any] = {
                "status": "computed",
                "n_pairs": int(differences.size),
                "n_speakers": len(speakers),
                "mean_delta": float(differences.mean()),
                "sd_delta": float(differences.std(ddof=1)) if differences.size > 1 else None,
                "cohens_dz": None,
            }
            if differences.size >= 2:
                effect["cohens_dz"] = paired_effect_size(left[valid], right[valid])["cohens_dz"]
            if len(speakers) >= 2:
                result = speaker_cluster_bootstrap(
                    rows,
                    lambda selected: float(np.mean([row["delta"] for row in selected])),
                    speaker_key="speaker_id",
                    n_resamples=n_resamples,
                    confidence=confidence,
                    seed=seed,
                )
                effect.update(
                    {
                        "ci_low": result.ci_low,
                        "ci_high": result.ci_high,
                        "ci_status": "computed",
                        "ci_confidence": confidence,
                        "ci_n_resamples": n_resamples,
                        "ci_seed": seed,
                    }
                )
            else:
                effect.update(
                    {
                        "ci_low": None,
                        "ci_high": None,
                        "ci_status": "unavailable",
                        "ci_reason": "fewer than two speakers",
                    }
                )
            expected = EXPECTED_DIRECTIONS.get(str(emotion), {}).get(metric)
            if expected is not None:
                effect["expected_direction"] = int(expected)
                effect["direction_match"] = bool(np.sign(effect["mean_delta"]) == expected)
                if effect.get("ci_status") == "computed":
                    if expected > 0:
                        effect["ci_excludes_zero_in_expected_direction"] = bool(
                            effect["ci_low"] > 0.0
                        )
                    else:
                        effect["ci_excludes_zero_in_expected_direction"] = bool(
                            effect["ci_high"] < 0.0
                        )
            condition_effects[metric] = effect
        effects[f"{condition}:{emotion}"] = condition_effects
    return {"status": "computed", "effects": effects}


def condition_gate_summary(
    paired_effects: Mapping[str, Any],
    *,
    conditions: Sequence[str] = ("happy", "sad", "angry"),
    condition_prefix: Optional[str] = None,
) -> Dict[str, Any]:
    """Check the pre-registered F0 and duration directions.

    ``condition_prefix`` selects which residual condition is gated.  When it is
    omitted, a single condition prefix in the effects table is inferred.  E0
    used ``fixed_delta``; E1 uses ``class_mean_prosody``/``class_mean_full``.
    Hard-coding one prefix would silently mark every other condition
    ``unavailable`` instead of evaluating it.
    """
    effects = paired_effects.get("effects", {}) if paired_effects.get("status") == "computed" else {}
    if condition_prefix is None:
        prefixes = sorted(
            {
                str(key).split(":", 1)[0]
                for key in effects
                if ":" in str(key)
            }
        )
        if len(prefixes) == 1:
            condition_prefix = prefixes[0]
        elif "fixed_delta" in prefixes:
            condition_prefix = "fixed_delta"
        else:
            condition_prefix = "fixed_delta"
    checks: Dict[str, Any] = {}
    failures: List[str] = []
    for emotion in conditions:
        key = f"{condition_prefix}:{emotion}"
        entry = effects.get(key, {})
        f0 = entry.get("f0_mean_hz", {})
        duration = entry.get("duration_seconds", {})
        emotion_checks = {}
        for metric, effect in (("f0_mean_hz", f0), ("duration_seconds", duration)):
            if effect.get("status") != "computed":
                emotion_checks[metric] = {
                    "status": "unavailable",
                    "reason": effect.get("reason", "not measured"),
                }
                failures.append(f"{emotion}: {metric} unavailable")
                continue
            direction_match = bool(effect.get("direction_match"))
            ci_confirms = bool(effect.get("ci_excludes_zero_in_expected_direction"))
            emotion_checks[metric] = {
                "status": "computed",
                "mean_delta": effect["mean_delta"],
                "ci_low": effect.get("ci_low"),
                "ci_high": effect.get("ci_high"),
                "n_pairs": effect.get("n_pairs"),
                "n_speakers": effect.get("n_speakers"),
                "expected_direction": effect["expected_direction"],
                "direction_match": direction_match,
                "ci_excludes_zero_in_expected_direction": ci_confirms,
                "passed": direction_match and ci_confirms,
            }
            if not direction_match:
                failures.append(
                    f"{emotion}: {metric} moved opposite to the pre-registered direction"
                )
            elif not ci_confirms:
                failures.append(
                    f"{emotion}: {metric} direction is correct but the CI includes zero"
                )
        checks[emotion] = emotion_checks
    return {
        "status": "computed",
        "checks": checks,
        "passed": not failures,
        "failures": failures,
    }


def identity_rows(
    frame: pd.DataFrame,
    *,
    neutral_column: str = "ecapa_generated_neutral",
    reference_column: str = "ecapa_identity_reference",
) -> List[Dict[str, Any]]:
    """Build guardrail rows from measured per-clip ECAPA similarities."""
    rows: List[Dict[str, Any]] = []
    for _, row in frame.iterrows():
        condition = str(row.get("condition", ""))
        emotion = str(row.get("emotion", ""))
        label = "neutral" if condition == "neutral" else emotion
        neutral_value = _finite(row.get(neutral_column))
        reference_value = _finite(row.get(reference_column))
        if neutral_value is None and reference_value is None:
            continue
        entry = {
            "condition": label,
            "speaker_id": str(row.get("actor_id", "")),
        }
        if neutral_value is not None:
            entry["ecapa_neutral"] = neutral_value
        if reference_value is not None:
            entry["ecapa_reference"] = reference_value
        rows.append(entry)
    return rows
