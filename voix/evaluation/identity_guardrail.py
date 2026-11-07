"""Cross-track identity guardrail for TTS research experiments.

The canonical face-to-voice artifacts are frozen.  Every naturalness, emotion,
or accent experiment must show that it did not move the identity reference
outside a pre-registered budget.  This module is pure Python/NumPy and can be
unit-tested without a GPU or a checkpoint.

It does not load the CFVN, CVAE, adapter, or StyleTTS2.  Callers pass measured
retrieval metrics and per-condition ECAPA similarities.
"""
from __future__ import annotations

import json
import math
import statistics
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Union

import numpy as np

from voix.evaluation.research_report import sha256_file
from voix.evaluation.stats import speaker_cluster_bootstrap
from voix.utils.paths import project_path

CANONICAL_CHECKPOINTS: Dict[str, str] = {
    "cfvn": "results/checkpoints/cfvn_v2_best.pt",
    "cvae": "checkpoints/cvae_cfvn_face192_latent128_diversity.pt",
    "style_adapter": "checkpoints/style_adapter_cfvn_best.pt",
}
CANONICAL_HASH_SCHEMA = "voix.canonical_hashes.v1"
IDENTITY_GUARDRAIL_SCHEMA = "voix.identity_guardrail.v1"

# Measured reference values from the canonical stack.  These are baselines,
# not targets: the guardrail fails when a new experiment moves away from them.
CANONICAL_METRICS: Dict[str, float] = {
    "cfvn_recall_at_1": 0.0932,
    "cfvn_recall_at_10": 0.4068,
    "face_face_recall_at_1": 0.8475,
    "raw_face_face_recall_at_1": 0.9153,
}


@dataclass(frozen=True)
class GuardrailThresholds:
    """Pre-registered identity budget for a TTS experiment."""

    min_ecapa_neutral: float = 0.80
    max_reference_drop: float = 0.05
    max_canonical_drop: float = 0.05
    cfvn_tolerance: float = 0.005
    min_pairwise_diversity: Optional[float] = None
    min_neutral_reference: Optional[float] = 0.50

    def to_dict(self) -> Dict[str, Any]:
        return {
            "min_ecapa_neutral": self.min_ecapa_neutral,
            "max_reference_drop": self.max_reference_drop,
            "max_canonical_drop": self.max_canonical_drop,
            "cfvn_tolerance": self.cfvn_tolerance,
            "min_pairwise_diversity": self.min_pairwise_diversity,
            "min_neutral_reference": self.min_neutral_reference,
        }


def _resolve(path: Union[str, Path], root: Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else root / candidate


def write_canonical_hashes(
    path: Union[str, Path] = project_path("results", "canonical_hashes.json"),
    *,
    root: Union[str, Path] = project_path(),
) -> Dict[str, Any]:
    """Hash the three canonical checkpoints and write the reference file.

    All three files must exist; a missing canonical artifact is an error, not a
    value to paper over.
    """
    root_path = Path(root)
    files: Dict[str, Any] = {}
    missing: List[str] = []
    for name, relative in CANONICAL_CHECKPOINTS.items():
        resolved = _resolve(relative, root_path)
        if not resolved.exists():
            missing.append(str(resolved))
            continue
        files[name] = {
            "path": relative,
            "sha256": sha256_file(resolved),
            "size_bytes": resolved.stat().st_size,
        }
    if missing:
        raise FileNotFoundError("canonical checkpoint(s) missing: " + ", ".join(missing))

    payload = {
        "schema_version": CANONICAL_HASH_SCHEMA,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "files": files,
    }
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def verify_canonical_hashes(
    path: Union[str, Path] = project_path("results", "canonical_hashes.json"),
    *,
    root: Union[str, Path] = project_path(),
) -> Dict[str, Any]:
    """Compare the canonical checkpoints against the recorded hashes."""
    root_path = Path(root)
    reference_path = Path(path)
    if not reference_path.exists():
        return {
            "status": "unavailable",
            "passed": False,
            "reason": f"canonical hash reference missing: {reference_path}",
            "files": {},
        }
    payload = json.loads(reference_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != CANONICAL_HASH_SCHEMA:
        return {
            "status": "invalid",
            "passed": False,
            "reason": f"unexpected schema: {payload.get('schema_version')!r}",
            "files": {},
        }

    expected_files = payload.get("files") or {}
    files: Dict[str, Any] = {}
    mismatched: List[str] = []
    missing: List[str] = []
    for name, expected in expected_files.items():
        relative = expected.get("path")
        resolved = _resolve(relative, root_path) if relative else None
        actual = sha256_file(resolved) if resolved is not None and resolved.exists() else None
        matches = actual is not None and actual.lower() == str(expected.get("sha256", "")).lower()
        if actual is None:
            missing.append(name)
        elif not matches:
            mismatched.append(name)
        files[name] = {
            "path": relative,
            "expected_sha256": expected.get("sha256"),
            "actual_sha256": actual,
            "match": bool(matches),
        }

    passed = bool(files) and not mismatched and not missing
    return {
        "status": "computed",
        "passed": passed,
        "files": files,
        "missing": missing,
        "mismatched": mismatched,
    }


def neutral_zero_error(canonical_style: Any, residual: Any) -> float:
    """Maximum absolute difference between the canonical style and ``style + 0``."""
    canonical = np.asarray(canonical_style, dtype=np.float64)
    delta = np.asarray(residual, dtype=np.float64)
    if canonical.shape != delta.shape:
        raise ValueError(
            f"shape mismatch: canonical_style {canonical.shape} vs residual {delta.shape}"
        )
    if canonical.size == 0:
        raise ValueError("style tensors must not be empty")
    return float(np.max(np.abs(canonical - (canonical + delta))))


def neutral_zero_check(
    canonical_style: Any,
    residual: Any,
    *,
    atol: float = 1e-5,
) -> Dict[str, Any]:
    """Check that a zero-residual condition reproduces the canonical style."""
    error = neutral_zero_error(canonical_style, residual)
    return {
        "status": "computed",
        "max_abs_error": error,
        "atol": float(atol),
        "passed": bool(error <= atol),
    }


def _finite(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _mean(rows: Sequence[Mapping[str, Any]], key: str) -> float:
    values = [_finite(row.get(key)) for row in rows]
    finite = [value for value in values if value is not None]
    if not finite:
        raise ValueError(f"no finite {key!r} values")
    return float(statistics.fmean(finite))


def _cluster_summary(
    rows: Sequence[Mapping[str, Any]],
    key: str,
    *,
    n_resamples: int,
    confidence: float,
    seed: int,
) -> Dict[str, Any]:
    usable = [row for row in rows if _finite(row.get(key)) is not None]
    if not usable:
        return {"status": "unavailable", "reason": f"no finite {key!r} values"}
    speakers = {str(row.get("speaker_id")) for row in usable if row.get("speaker_id")}
    if len(speakers) < 2:
        values = [_finite(row.get(key)) for row in usable]
        finite = [value for value in values if value is not None]
        return {
            "status": "computed",
            "mean": float(statistics.fmean(finite)),
            "n": len(finite),
            "n_speakers": len(speakers),
            "ci_low": None,
            "ci_high": None,
            "ci_status": "unavailable",
            "reason": "fewer than two speakers; cluster CI unavailable",
        }
    result = speaker_cluster_bootstrap(
        usable,
        lambda selected: _mean(selected, key),
        speaker_key="speaker_id",
        n_resamples=n_resamples,
        confidence=confidence,
        seed=seed,
    )
    return {
        "status": "computed",
        "mean": result.estimate,
        "n": result.n,
        "n_speakers": len(speakers),
        "ci_low": result.ci_low,
        "ci_high": result.ci_high,
        "ci_status": "computed",
    }


def _paired_cluster_summary(
    rows: Sequence[Mapping[str, Any]],
    left_key: str,
    right_key: str,
    *,
    n_resamples: int,
    confidence: float,
    seed: int,
) -> Dict[str, Any]:
    paired = [
        row
        for row in rows
        if _finite(row.get(left_key)) is not None and _finite(row.get(right_key)) is not None
    ]
    if not paired:
        return {
            "status": "unavailable",
            "reason": f"no rows contain both {left_key!r} and {right_key!r}",
        }

    def difference(selected: Sequence[Mapping[str, Any]]) -> float:
        return float(
            statistics.fmean(
                float(row[left_key]) - float(row[right_key]) for row in selected
            )
        )

    speakers = {str(row.get("speaker_id")) for row in paired if row.get("speaker_id")}
    if len(speakers) < 2:
        return {
            "status": "computed",
            "mean": difference(paired),
            "n": len(paired),
            "n_speakers": len(speakers),
            "ci_low": None,
            "ci_high": None,
            "ci_status": "unavailable",
            "reason": "fewer than two speakers; cluster CI unavailable",
        }
    result = speaker_cluster_bootstrap(
        paired,
        difference,
        speaker_key="speaker_id",
        n_resamples=n_resamples,
        confidence=confidence,
        seed=seed,
    )
    return {
        "status": "computed",
        "mean": result.estimate,
        "n": result.n,
        "n_speakers": len(speakers),
        "ci_low": result.ci_low,
        "ci_high": result.ci_high,
        "ci_status": "computed",
    }


def evaluate_cfvn_drift(
    metrics: Mapping[str, Any],
    *,
    reference: Optional[Mapping[str, float]] = None,
    tolerance: float = 0.005,
) -> Dict[str, Any]:
    """Check CFVN retrieval against the recorded canonical baseline."""
    baseline = dict(CANONICAL_METRICS if reference is None else reference)
    source = metrics.get("cfvn", metrics) if isinstance(metrics, Mapping) else {}
    checks: Dict[str, Any] = {}
    failures: List[str] = []
    for key in ("recall_at_1", "recall_at_10"):
        expected_key = f"cfvn_{key}"
        expected = _finite(baseline.get(expected_key))
        actual = _finite(source.get(key))
        if expected is None:
            checks[key] = {
                "status": "unavailable",
                "reason": f"no canonical baseline for {expected_key}",
            }
            failures.append(f"{key}: missing canonical baseline")
            continue
        if actual is None:
            checks[key] = {
                "status": "unavailable",
                "reason": f"no measured {key}",
                "baseline": expected,
            }
            failures.append(f"{key}: missing measurement")
            continue
        delta = actual - expected
        passed = abs(delta) <= tolerance
        checks[key] = {
            "status": "computed",
            "value": actual,
            "baseline": expected,
            "delta": delta,
            "tolerance": tolerance,
            "passed": bool(passed),
        }
        if not passed:
            failures.append(f"{key}: delta {delta:+.4f} exceeds +/-{tolerance:.4f}")
    return {
        "status": "computed" if checks else "unavailable",
        "passed": not failures,
        "checks": checks,
        "failures": failures,
    }


def evaluate_identity_guardrail(
    rows: Sequence[Mapping[str, Any]],
    *,
    thresholds: Optional[GuardrailThresholds] = None,
    cfvn_metrics: Optional[Mapping[str, Any]] = None,
    canonical_hashes_path: Optional[Union[str, Path]] = None,
    require_cfvn: bool = True,
    root: Union[str, Path] = project_path(),
    n_resamples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 0,
) -> Dict[str, Any]:
    """Evaluate per-condition identity preservation and CFVN drift.

    Each row describes one generated sample and must include ``condition`` and
    ``speaker_id``.  Required measured fields are ``ecapa_neutral`` and
    ``ecapa_reference``; optional fields are ``canonical_ecapa`` and
    ``pairwise_diversity``.

    Set ``require_cfvn=False`` only for an experiment that is explicitly
    identity-conditioned and does not execute the face-to-voice CFVN.  The
    report then records CFVN as ``not_applicable`` instead of pretending it was
    measured or failing the run for a metric the experiment never used.
    """
    thresholds = thresholds or GuardrailThresholds()
    if not rows:
        raise ValueError("rows must not be empty")

    grouped: Dict[str, List[Mapping[str, Any]]] = {}
    for row in rows:
        if not row.get("condition"):
            raise ValueError("every row requires a condition")
        if not row.get("speaker_id"):
            raise ValueError("every row requires a speaker_id")
        grouped.setdefault(str(row["condition"]), []).append(row)

    # Per-speaker neutral reference similarity is the baseline for the
    # reference-drop check.  A condition that is further from the real
    # speaker than its own generated neutral is the drift this guardrail is
    # meant to catch.
    neutral_reference_values: Dict[str, List[float]] = {}
    for row in rows:
        if str(row.get("condition")) != "neutral":
            continue
        speaker = str(row.get("speaker_id"))
        value = _finite(row.get("ecapa_reference"))
        if value is not None:
            neutral_reference_values.setdefault(speaker, []).append(value)
    neutral_reference_by_speaker = {
        speaker: float(statistics.fmean(values))
        for speaker, values in neutral_reference_values.items()
    }

    conditions: Dict[str, Any] = {}
    failures: List[str] = []
    for condition, condition_rows in sorted(grouped.items()):
        neutral = _cluster_summary(
            condition_rows,
            "ecapa_neutral",
            n_resamples=n_resamples,
            confidence=confidence,
            seed=seed,
        )
        reference = _cluster_summary(
            condition_rows,
            "ecapa_reference",
            n_resamples=n_resamples,
            confidence=confidence,
            seed=seed,
        )
        if condition == "neutral":
            reference_drop = {
                "status": "not_applicable",
                "reason": "neutral is the reference baseline",
                "mean": 0.0,
                "n": len(condition_rows),
                "n_speakers": len(
                    {
                        str(row.get("speaker_id"))
                        for row in condition_rows
                        if row.get("speaker_id")
                    }
                ),
                "ci_low": None,
                "ci_high": None,
                "ci_status": "not_applicable",
            }
        else:
            baseline_rows: List[Mapping[str, Any]] = []
            for row in condition_rows:
                speaker = str(row.get("speaker_id"))
                baseline = neutral_reference_by_speaker.get(speaker)
                if baseline is None or _finite(row.get("ecapa_reference")) is None:
                    continue
                enriched = dict(row)
                enriched["neutral_reference_baseline"] = baseline
                baseline_rows.append(enriched)
            reference_drop = _paired_cluster_summary(
                baseline_rows,
                "neutral_reference_baseline",
                "ecapa_reference",
                n_resamples=n_resamples,
                confidence=confidence,
                seed=seed,
            )
        canonical_drop = _paired_cluster_summary(
            condition_rows,
            "ecapa_neutral",
            "canonical_ecapa",
            n_resamples=n_resamples,
            confidence=confidence,
            seed=seed,
        )
        diversity = _cluster_summary(
            condition_rows,
            "pairwise_diversity",
            n_resamples=n_resamples,
            confidence=confidence,
            seed=seed,
        )

        condition_failures: List[str] = []
        if neutral.get("status") != "computed":
            condition_failures.append("ECAPA similarity to generated neutral is unavailable")
        elif neutral["mean"] < thresholds.min_ecapa_neutral:
            condition_failures.append(
                f"ECAPA neutral {neutral['mean']:.4f} < {thresholds.min_ecapa_neutral:.4f}"
            )
        if reference_drop.get("status") == "computed":
            if reference_drop["mean"] > thresholds.max_reference_drop:
                condition_failures.append(
                    f"reference drop {reference_drop['mean']:+.4f} exceeds "
                    f"{thresholds.max_reference_drop:.4f}"
                )
        if canonical_drop.get("status") == "computed":
            if canonical_drop["mean"] < -thresholds.max_canonical_drop:
                condition_failures.append(
                    f"canonical drop {canonical_drop['mean']:+.4f} exceeds "
                    f"-{thresholds.max_canonical_drop:.4f}"
                )
        if (
            thresholds.min_pairwise_diversity is not None
            and diversity.get("status") == "computed"
            and diversity["mean"] < thresholds.min_pairwise_diversity
        ):
            condition_failures.append(
                f"pairwise diversity {diversity['mean']:.4f} < "
                f"{thresholds.min_pairwise_diversity:.4f}"
            )

        conditions[condition] = {
            "n": len(condition_rows),
            "ecapa_neutral": neutral,
            "ecapa_reference": reference,
            "reference_drop": reference_drop,
            "canonical_drop": canonical_drop,
            "pairwise_diversity": diversity,
            "passed": not condition_failures,
            "failures": condition_failures,
        }
        failures.extend(f"{condition}: {message}" for message in condition_failures)

    # Neutral sanity is a domain-shift diagnostic, not a reason to throw away a
    # relative comparison.  When the generated neutral is already far from the
    # real speaker, condition comparisons remain valid relative to that neutral
    # but absolute identity claims are not.
    neutral_condition = conditions.get("neutral")
    if neutral_condition is None:
        neutral_sanity = {
            "status": "unavailable",
            "reason": "no neutral condition rows were supplied",
            "passed": None,
        }
    elif thresholds.min_neutral_reference is None:
        neutral_sanity = {
            "status": "not_applicable",
            "reason": "no neutral reference threshold was configured",
            "passed": True,
        }
    else:
        neutral_reference = neutral_condition.get("ecapa_reference", {})
        if neutral_reference.get("status") != "computed":
            neutral_sanity = {
                "status": "unavailable",
                "reason": "neutral ECAPA reference similarity is unavailable",
                "passed": False,
            }
        else:
            neutral_sanity = {
                "status": "computed",
                "mean": neutral_reference["mean"],
                "threshold": thresholds.min_neutral_reference,
                "passed": bool(
                    neutral_reference["mean"] >= thresholds.min_neutral_reference
                ),
            }
    warnings: List[str] = []
    if neutral_sanity.get("passed") is False:
        warnings.append(
            "generated neutral ECAPA reference "
            f"{neutral_sanity.get('mean')} is below "
            f"{thresholds.min_neutral_reference}; condition comparisons are "
            "relative only (domain shift)"
        )

    if cfvn_metrics is None and not require_cfvn:
        cfvn = {
            "status": "not_applicable",
            "reason": (
                "this experiment is identity-conditioned and does not execute "
                "the face-to-voice CFVN"
            ),
            "passed": True,
        }
    elif cfvn_metrics is None:
        cfvn = {
            "status": "unavailable",
            "reason": "CFVN retrieval metrics were not supplied",
            "passed": False,
        }
        failures.append("CFVN retrieval metrics were not supplied")
    else:
        cfvn = evaluate_cfvn_drift(
            cfvn_metrics,
            tolerance=thresholds.cfvn_tolerance,
        )
        failures.extend(f"CFVN: {message}" for message in cfvn["failures"])

    if canonical_hashes_path is None:
        hashes = {
            "status": "unavailable",
            "reason": "canonical hash reference was not supplied",
            "passed": False,
        }
        failures.append("canonical checkpoint hashes were not verified")
    else:
        hashes = verify_canonical_hashes(canonical_hashes_path, root=root)
        if not hashes.get("passed"):
            failures.append("canonical checkpoint hashes do not match the recorded reference")

    return {
        "schema_version": IDENTITY_GUARDRAIL_SCHEMA,
        "thresholds": thresholds.to_dict(),
        "conditions": conditions,
        "neutral_sanity": neutral_sanity,
        "cfvn": cfvn,
        "canonical_hashes": hashes,
        "passed": not failures,
        "failures": failures,
        "warnings": warnings,
    }
