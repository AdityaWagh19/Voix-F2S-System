"""Evaluation metrics for accent and emotion expression controls.

This module is deliberately dependency-injected: prosody extraction, speaker
similarity, WER, and classifiers can all be replaced in tests or in an offline
environment. Missing models or references return ``status="unavailable"``
instead of a fabricated score.

Manifest schema (``voix.expression.v1``)::

    {
      "schema_version": "voix.expression.v1",
      "reference_text": "...",
      "samples": [
        {
          "audio": "demo/03_emotions/emotion_02_happy.wav",
          "control": "emotion",              // "emotion" or "accent"
          "label": "happy",
          "intensity": 0.9,
          "speaker_id": "subject_face",      // optional
          "neutral_reference": "demo/03_emotions/emotion_01_neutral.wav",
          "reference_text": "..."            // optional per-sample override
        }
      ]
    }
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from voix.evaluation.metrics import prosody_statistics, speaker_similarity

EXPRESSION_MANIFEST_SCHEMA = "voix.expression.v1"
SUPPORTED_CONTROLS = ("emotion", "accent")

# Proposed expected directions for intensity sweeps. These are hypotheses to be
# frozen before training; they are not measured results.
EMOTION_EXPECTED_DIRECTIONS: Dict[str, Dict[str, int]] = {
    "happy": {"f0_mean_hz": 1, "duration_seconds": -1},
    "sad": {"f0_mean_hz": -1, "duration_seconds": 1},
    "angry": {"f0_mean_hz": 1, "duration_seconds": -1},
    "whispered": {"energy_mean": -1, "duration_seconds": 1},
    "surprised": {"f0_mean_hz": 1, "duration_seconds": -1},
}


def _resolve(path: str, root: Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return (root / candidate).resolve()


def _label(value: Any) -> str:
    return str(value).strip().lower()


def validate_expression_manifest(
    payload: Dict[str, Any],
    *,
    require_schema: bool = False,
) -> List[str]:
    """Return a list of validation errors; an empty list means valid."""
    errors: List[str] = []
    if require_schema and payload.get("schema_version") != EXPRESSION_MANIFEST_SCHEMA:
        errors.append(
            f"schema_version must be {EXPRESSION_MANIFEST_SCHEMA!r}, "
            f"got {payload.get('schema_version')!r}"
        )
    samples = payload.get("samples")
    if not isinstance(samples, list) or not samples:
        errors.append("samples must be a non-empty list")
        return errors
    for index, sample in enumerate(samples):
        prefix = f"samples[{index}]"
        if not isinstance(sample, dict):
            errors.append(f"{prefix} must be an object")
            continue
        if not sample.get("audio"):
            errors.append(f"{prefix}.audio is required")
        control = _label(sample.get("control", ""))
        if control not in SUPPORTED_CONTROLS:
            errors.append(f"{prefix}.control must be one of {SUPPORTED_CONTROLS}")
        if not _label(sample.get("label", "")):
            errors.append(f"{prefix}.label is required")
        intensity = sample.get("intensity")
        if intensity is not None:
            try:
                value = float(intensity)
            except (TypeError, ValueError):
                errors.append(f"{prefix}.intensity must be numeric")
            else:
                if not 0.0 <= value <= 1.0:
                    errors.append(f"{prefix}.intensity must be in [0, 1]")
    return errors


def load_expression_manifest(
    path: Path,
    *,
    require_schema: bool = True,
) -> Dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    errors = validate_expression_manifest(payload, require_schema=require_schema)
    if errors:
        raise ValueError("invalid expression manifest: " + "; ".join(errors))
    return payload


def _safe_prosody(
    path: Path,
    prosody_fn: Callable[[Any], Dict[str, Any]],
) -> Dict[str, Any]:
    if not path.exists():
        return {"status": "unavailable", "reason": f"missing audio: {path}"}
    try:
        return {"status": "computed", **prosody_fn(str(path))}
    except Exception as exc:  # pragma: no cover - depends on audio backends
        return {"status": "unavailable", "reason": f"{type(exc).__name__}: {exc}"}


def aggregate_conditions(
    samples: Sequence[Dict[str, Any]],
    *,
    root: Path,
    prosody_fn: Callable[[Any], Dict[str, Any]] = prosody_statistics,
) -> Dict[str, Any]:
    """Compute per-sample prosody and per-condition aggregates."""
    per_sample: List[Dict[str, Any]] = []
    groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)

    for sample in samples:
        audio = _resolve(str(sample["audio"]), root)
        control = _label(sample["control"])
        label = _label(sample["label"])
        entry = {
            "audio": str(audio),
            "control": control,
            "label": label,
            "intensity": sample.get("intensity"),
            "speaker_id": sample.get("speaker_id"),
            "prosody": _safe_prosody(audio, prosody_fn),
        }
        per_sample.append(entry)
        if entry["prosody"]["status"] == "computed":
            groups[(control, label)].append(entry)

    aggregates: Dict[str, Any] = {}
    for (control, label), entries in sorted(groups.items()):
        def values(key: str) -> List[float]:
            return [
                float(entry["prosody"][key])
                for entry in entries
                if key in entry["prosody"]
                and isinstance(entry["prosody"][key], (int, float))
                and np.isfinite(entry["prosody"][key])
            ]

        duration = values("duration_seconds")
        f0 = values("f0_mean_hz")
        f0_std = values("f0_std_hz")
        energy = values("energy_mean")
        voiced = values("f0_voiced_ratio")
        aggregates[f"{control}:{label}"] = {
            "control": control,
            "label": label,
            "n": len(entries),
            "duration_seconds_mean": float(np.mean(duration)) if duration else None,
            "f0_mean_hz_mean": float(np.mean(f0)) if f0 else None,
            "f0_std_hz_mean": float(np.mean(f0_std)) if f0_std else None,
            "energy_mean_mean": float(np.mean(energy)) if energy else None,
            "voiced_ratio_mean": float(np.mean(voiced)) if voiced else None,
        }

    return {
        "status": "computed" if aggregates else "unavailable",
        "n_samples": len(per_sample),
        "per_sample": per_sample,
        "conditions": aggregates,
        "reason": None if aggregates else "no readable audio sample",
    }


def intensity_monotonicity(
    samples: Sequence[Dict[str, Any]],
    *,
    root: Path,
    prosody_fn: Callable[[Any], Dict[str, Any]] = prosody_statistics,
    expected_directions: Optional[Dict[str, Dict[str, int]]] = None,
) -> Dict[str, Any]:
    """Measure whether intensity moves the intended acoustic dimension.

    Spearman correlation is computed only when a condition has at least two
    distinct intensity values and at least three finite measurements.
    """
    expected_directions = expected_directions or EMOTION_EXPECTED_DIRECTIONS
    groups: Dict[Tuple[str, str], List[Tuple[float, Dict[str, Any]]]] = defaultdict(list)
    for sample in samples:
        intensity = sample.get("intensity")
        if intensity is None:
            continue
        audio = _resolve(str(sample["audio"]), root)
        prosody = _safe_prosody(audio, prosody_fn)
        if prosody["status"] != "computed":
            continue
        groups[(_label(sample["control"]), _label(sample["label"]))].append(
            (float(intensity), prosody)
        )

    try:
        from scipy.stats import spearmanr
    except Exception:  # pragma: no cover - scipy is a project dependency
        return {"status": "unavailable", "reason": "scipy is not installed"}

    results: Dict[str, Any] = {}
    for (control, label), entries in sorted(groups.items()):
        intensities = [entry[0] for entry in entries]
        if len(set(intensities)) < 2 or len(entries) < 3:
            results[f"{control}:{label}"] = {
                "status": "insufficient_data",
                "n": len(entries),
                "distinct_intensities": len(set(intensities)),
            }
            continue
        metrics = sorted(
            set().union(*(entry[1].keys() for entry in entries))
            & {"f0_mean_hz", "duration_seconds", "energy_mean", "f0_std_hz"}
        )
        condition_result: Dict[str, Any] = {
            "status": "computed",
            "n": len(entries),
            "metrics": {},
        }
        directions = expected_directions.get(label, {}) if control == "emotion" else {}
        for metric in metrics:
            pairs = [
                (intensity, float(prosody[metric]))
                for intensity, prosody in entries
                if metric in prosody
                and isinstance(prosody[metric], (int, float))
                and np.isfinite(prosody[metric])
            ]
            if len(pairs) < 3 or len({pair[0] for pair in pairs}) < 2:
                continue
            rho, p_value = spearmanr(
                [pair[0] for pair in pairs], [pair[1] for pair in pairs]
            )
            expected = directions.get(metric)
            condition_result["metrics"][metric] = {
                "rho": float(rho) if np.isfinite(rho) else None,
                "p_value": float(p_value) if np.isfinite(p_value) else None,
                "expected_direction": expected,
                "direction_matches": (
                    None if expected is None or not np.isfinite(rho)
                    else bool(np.sign(rho) == expected)
                ),
            }
        results[f"{control}:{label}"] = condition_result
    return {"status": "computed" if results else "unavailable", "conditions": results}


def speaker_preservation(
    samples: Sequence[Dict[str, Any]],
    *,
    root: Path,
    similarity_fn: Callable[[Any, Any], Optional[float]] = speaker_similarity,
    device: str = "cpu",
) -> Dict[str, Any]:
    """Compare each conditioned sample to its same-speaker neutral reference."""
    scores: List[float] = []
    per_sample: List[Dict[str, Any]] = []
    for sample in samples:
        reference = sample.get("neutral_reference")
        if not reference:
            continue
        audio = _resolve(str(sample["audio"]), root)
        neutral = _resolve(str(reference), root)
        if not audio.exists() or not neutral.exists():
            per_sample.append(
                {
                    "audio": str(audio),
                    "neutral_reference": str(neutral),
                    "status": "unavailable",
                    "reason": "missing audio or reference",
                }
            )
            continue
        try:
            similarity = similarity_fn(str(audio), str(neutral), device=device)
        except TypeError:
            # Allow simple two-argument test doubles.
            try:
                similarity = similarity_fn(str(audio), str(neutral))
            except Exception as exc:  # pragma: no cover - backend dependent
                similarity = None
                reason = f"{type(exc).__name__}: {exc}"
            else:
                reason = (
                    None if similarity is not None else "similarity unavailable"
                )
        except Exception as exc:  # pragma: no cover - backend dependent
            similarity = None
            reason = f"{type(exc).__name__}: {exc}"
        else:
            reason = None if similarity is not None else "similarity unavailable"
        entry = {
            "audio": str(audio),
            "neutral_reference": str(neutral),
            "control": _label(sample["control"]),
            "label": _label(sample["label"]),
            "speaker_id": sample.get("speaker_id"),
            "status": "computed" if similarity is not None else "unavailable",
            "similarity": None if similarity is None else float(similarity),
        }
        if reason:
            entry["reason"] = reason
        per_sample.append(entry)
        if similarity is not None:
            scores.append(float(similarity))

    return {
        "status": "computed" if scores else "unavailable",
        "n": len(scores),
        "mean": float(np.mean(scores)) if scores else None,
        "min": float(np.min(scores)) if scores else None,
        "per_sample": per_sample,
        "reason": None if scores else "no sample had a readable neutral reference",
    }


def classifier_accuracy(
    samples: Sequence[Dict[str, Any]],
    *,
    root: Path,
    classifier: Optional[Callable[[str], str]] = None,
    control: Optional[str] = None,
) -> Dict[str, Any]:
    """Evaluate a pluggable audio classifier against manifest labels."""
    if classifier is None:
        return {
            "status": "unavailable",
            "reason": "no classifier supplied",
            "accuracy": None,
        }

    selected = [
        sample
        for sample in samples
        if control is None or _label(sample["control"]) == _label(control)
    ]
    if not selected:
        return {
            "status": "unavailable",
            "reason": f"no samples for control={control!r}",
            "accuracy": None,
        }

    predictions: List[Dict[str, Any]] = []
    for sample in selected:
        audio = _resolve(str(sample["audio"]), root)
        expected = _label(sample["label"])
        if not audio.exists():
            predictions.append(
                {
                    "audio": str(audio),
                    "expected": expected,
                    "predicted": None,
                    "status": "unavailable",
                    "reason": "missing audio",
                }
            )
            continue
        try:
            predicted = _label(classifier(str(audio)))
        except Exception as exc:  # pragma: no cover - classifier dependent
            predictions.append(
                {
                    "audio": str(audio),
                    "expected": expected,
                    "predicted": None,
                    "status": "unavailable",
                    "reason": f"{type(exc).__name__}: {exc}",
                }
            )
            continue
        predictions.append(
            {
                "audio": str(audio),
                "expected": expected,
                "predicted": predicted,
                "status": "computed",
                "correct": predicted == expected,
            }
        )

    scored = [entry for entry in predictions if entry["status"] == "computed"]
    if not scored:
        return {
            "status": "unavailable",
            "reason": "classifier produced no predictions",
            "accuracy": None,
            "per_sample": predictions,
        }

    labels = sorted({entry["expected"] for entry in scored} | {entry["predicted"] for entry in scored})
    confusion = {label: {other: 0 for other in labels} for label in labels}
    for entry in scored:
        confusion[entry["expected"]][entry["predicted"]] += 1
    per_class = {}
    for label in labels:
        total = sum(confusion[label].values())
        correct = confusion[label].get(label, 0)
        per_class[label] = {
            "n": total,
            "correct": correct,
            "accuracy": float(correct / total) if total else None,
        }
    correct = sum(1 for entry in scored if entry["correct"])
    return {
        "status": "computed",
        "n": len(scored),
        "accuracy": float(correct / len(scored)),
        "per_class": per_class,
        "confusion": confusion,
        "per_sample": predictions,
    }


def evaluate_expression_manifest(
    manifest: Dict[str, Any],
    *,
    root: Path,
    device: str = "cpu",
    prosody_fn: Callable[[Any], Dict[str, Any]] = prosody_statistics,
    similarity_fn: Callable[..., Optional[float]] = speaker_similarity,
    emotion_classifier: Optional[Callable[[str], str]] = None,
    accent_classifier: Optional[Callable[[str], str]] = None,
) -> Dict[str, Any]:
    """Run the expression-control evaluation suite for one manifest."""
    errors = validate_expression_manifest(manifest)
    if errors:
        raise ValueError("invalid expression manifest: " + "; ".join(errors))
    samples = list(manifest["samples"])
    return {
        "schema_version": manifest.get("schema_version"),
        "n_samples": len(samples),
        "prosody": aggregate_conditions(samples, root=root, prosody_fn=prosody_fn),
        "intensity_monotonicity": intensity_monotonicity(
            samples, root=root, prosody_fn=prosody_fn
        ),
        "speaker_preservation": speaker_preservation(
            samples, root=root, similarity_fn=similarity_fn, device=device
        ),
        "emotion_classifier": classifier_accuracy(
            samples, root=root, classifier=emotion_classifier, control="emotion"
        ),
        "accent_classifier": classifier_accuracy(
            samples, root=root, classifier=accent_classifier, control="accent"
        ),
        "mos": {
            "status": "unavailable",
            "reason": "awaiting real human ratings",
        },
    }


def check_expression_gates(
    report: Dict[str, Any],
    *,
    wer: Optional[float] = None,
    max_wer: Optional[float] = None,
    min_speaker_similarity: float = 0.80,
    min_classifier_accuracy: float = 0.50,
    min_monotonicity_rho: float = 0.80,
) -> Dict[str, Any]:
    """Evaluate the pre-registered production gates that are measurable here.

    WER is computed by the synthesis evaluator and must be passed in explicitly.
    A gate whose metric is unavailable is reported as ``unavailable``; the
    overall status is ``unavailable`` when no gate failed but some gate could
    not be measured. Nothing unavailable is ever counted as passed.
    """
    gates: Dict[str, Any] = {}

    preservation = report.get("speaker_preservation", {})
    if preservation.get("status") == "computed":
        gates["speaker_preservation"] = {
            "value": preservation.get("mean"),
            "threshold": min_speaker_similarity,
            "passed": bool(preservation.get("mean", 0.0) >= min_speaker_similarity),
        }
    else:
        gates["speaker_preservation"] = {
            "status": "unavailable",
            "passed": None,
            "reason": preservation.get("reason"),
        }

    for key in ("emotion_classifier", "accent_classifier"):
        metric = report.get(key, {})
        if metric.get("status") == "computed":
            gates[key] = {
                "value": metric.get("accuracy"),
                "threshold": min_classifier_accuracy,
                "passed": bool(metric.get("accuracy", 0.0) >= min_classifier_accuracy),
            }
        else:
            gates[key] = {
                "status": "unavailable",
                "passed": None,
                "reason": metric.get("reason"),
            }

    monotonic = report.get("intensity_monotonicity", {})
    conditions = monotonic.get("conditions", {}) if monotonic.get("status") == "computed" else {}
    checks = []
    for condition in conditions.values():
        for metric_name, metric in condition.get("metrics", {}).items():
            expected = metric.get("expected_direction")
            rho = metric.get("rho")
            if expected is None or rho is None:
                continue
            checks.append(
                {
                    "metric": metric_name,
                    "rho": rho,
                    "expected_direction": expected,
                    "direction_matches": bool(np.sign(rho) == expected),
                    "magnitude_ok": bool(abs(rho) >= min_monotonicity_rho),
                }
            )
    if checks:
        gates["intensity_monotonicity"] = {
            # The weakest intended dimension must still clear the threshold.
            "value": min(abs(check["rho"]) for check in checks),
            "threshold": min_monotonicity_rho,
            "passed": all(
                check["direction_matches"] and check["magnitude_ok"]
                for check in checks
            ),
            "n_checks": len(checks),
            "checks": checks,
        }
    else:
        gates["intensity_monotonicity"] = {
            "status": "unavailable",
            "passed": None,
            "reason": "no intensity sweeps with expected directions",
        }

    if max_wer is not None:
        if wer is None:
            gates["wer"] = {
                "status": "unavailable",
                "threshold": max_wer,
                "passed": None,
                "reason": "no WER value supplied",
            }
        else:
            gates["wer"] = {
                "value": wer,
                "threshold": max_wer,
                "passed": bool(wer <= max_wer),
            }

    if not gates:
        status = "unavailable"
    elif any(gate.get("passed") is False for gate in gates.values()):
        status = "fail"
    elif all(gate.get("passed") is True for gate in gates.values()):
        status = "pass"
    else:
        status = "unavailable"

    return {"status": status, "gates": gates}
