"""CI gate for the FaceFeatureV2 face-to-voice stack.

The gate is deliberately a pure function over a report dictionary so it can be
unit-tested without a GPU or a trained checkpoint.  ``scripts/evaluation/ci_gate_v2.py``
builds the report from real artifacts and exits non-zero when any check fails.

A missing metric is a failure, not a skip.  The audit found that missing
artifacts and collapsed embeddings could still produce a green test run; this
gate is the opposite contract.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class GateThresholds:
    """Starting thresholds; calibrate on a healthy held-out baseline."""

    min_recall_at_1: float = 0.05
    min_recall_chance_multiple: float = 10.0
    max_negative_control_recall_at_1: float = 0.05
    max_negative_control_chance_multiple: float = 3.0
    max_between_speaker_cosine: float = 0.80
    min_within_between_margin: float = 0.05
    min_collapse_ratio: float = 0.50
    max_decoder_pairwise_cosine: float = 0.50
    min_face_margin: float = 0.10
    min_active_units: int = 8
    max_prior_pairwise_cosine: float = 0.90
    max_adapter_pairwise_cosine: float = 0.90
    min_adapter_per_dim_std: float = 0.01
    min_adapter_target_cosine: float = 0.85


@dataclass
class Check:
    name: str
    value: Any
    threshold: Any
    passed: bool
    message: str = ""


@dataclass
class GateResult:
    passed: bool
    checks: List[Check] = field(default_factory=list)

    @property
    def failures(self) -> List[Check]:
        return [check for check in self.checks if not check.passed]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "checks": [
                {
                    "name": check.name,
                    "value": check.value,
                    "threshold": check.threshold,
                    "passed": check.passed,
                    "message": check.message,
                }
                for check in self.checks
            ],
            "failures": [check.name for check in self.failures],
        }


def _number(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _check_min(name: str, value: Any, threshold: float) -> Check:
    number = _number(value)
    if number is None:
        return Check(name, value, threshold, False, f"{name} is missing or non-finite")
    passed = number >= threshold
    relation = ">=" if passed else "<"
    return Check(
        name,
        number,
        threshold,
        passed,
        f"{name}={number:.4f} {relation} {threshold:.4f} (required >=)",
    )


def _check_max(name: str, value: Any, threshold: float) -> Check:
    number = _number(value)
    if number is None:
        return Check(name, value, threshold, False, f"{name} is missing or non-finite")
    passed = number <= threshold
    relation = "<=" if passed else ">"
    return Check(
        name,
        number,
        threshold,
        passed,
        f"{name}={number:.4f} {relation} {threshold:.4f} (required <=)",
    )


def evaluate_gate(
    report: Dict[str, Any],
    thresholds: Optional[GateThresholds] = None,
) -> GateResult:
    """Evaluate a CI report against the gate contract.

    Expected report keys (all required unless noted):
      ``retrieval.cfvn``: recall_at_1, chance_at_1, shuffled_recall_at_1,
          constant_recall_at_1
      ``health.cfvn``: within_speaker_cosine, between_speaker_cosine,
          collapse_ratio
      ``cvae``: decoder_pairwise_cosine, face_margin, active_units,
          prior_pairwise_cosine
      ``adapter``: output_pairwise_cosine, output_per_dim_std,
          output_target_cosine
      ``artifacts.missing``: list of missing artifact names (optional)
    """
    thresholds = thresholds or GateThresholds()
    checks: List[Check] = []

    artifacts = report.get("artifacts", {})
    missing = artifacts.get("missing") or []
    checks.append(
        Check(
            "artifacts_present",
            list(missing),
            [],
            not missing,
            f"missing artifacts: {missing}",
        )
    )

    retrieval = report.get("retrieval", {}).get("cfvn", {})
    recall = _number(retrieval.get("recall_at_1"))
    chance = _number(retrieval.get("chance_at_1"))
    if recall is None or chance is None:
        checks.append(
            Check(
                "cfvn_recall_at_1",
                recall,
                thresholds.min_recall_at_1,
                False,
                "CFVN retrieval metrics are missing",
            )
        )
        checks.append(
            Check(
                "cfvn_chance_multiple",
                None,
                thresholds.min_recall_chance_multiple,
                False,
                "CFVN chance baseline is missing",
            )
        )
    else:
        checks.append(
            _check_min("cfvn_recall_at_1", recall, thresholds.min_recall_at_1)
        )
        multiple = recall / max(chance, 1e-9)
        checks.append(
            _check_min(
                "cfvn_chance_multiple",
                multiple,
                thresholds.min_recall_chance_multiple,
            )
        )
        control_limit = max(
            thresholds.max_negative_control_recall_at_1,
            thresholds.max_negative_control_chance_multiple * chance,
        )
        checks.append(
            _check_max(
                "shuffled_control_recall_at_1",
                retrieval.get("shuffled_recall_at_1"),
                control_limit,
            )
        )
        checks.append(
            _check_max(
                "constant_control_recall_at_1",
                retrieval.get("constant_recall_at_1"),
                control_limit,
            )
        )

    health = report.get("health", {}).get("cfvn", {})
    checks.append(
        _check_max(
            "between_speaker_cosine",
            health.get("between_speaker_cosine"),
            thresholds.max_between_speaker_cosine,
        )
    )
    within = _number(health.get("within_speaker_cosine"))
    between = _number(health.get("between_speaker_cosine"))
    margin = (
        within - between if within is not None and between is not None else None
    )
    checks.append(
        _check_min(
            "within_between_margin",
            margin,
            thresholds.min_within_between_margin,
        )
    )
    checks.append(
        _check_min(
            "collapse_ratio",
            health.get("collapse_ratio"),
            thresholds.min_collapse_ratio,
        )
    )

    cvae = report.get("cvae", {})
    checks.append(
        _check_max(
            "cvae_decoder_pairwise_cosine",
            cvae.get("decoder_pairwise_cosine"),
            thresholds.max_decoder_pairwise_cosine,
        )
    )
    checks.append(
        _check_min(
            "cvae_face_margin",
            cvae.get("face_margin"),
            thresholds.min_face_margin,
        )
    )
    checks.append(
        _check_min(
            "cvae_active_units",
            cvae.get("active_units"),
            thresholds.min_active_units,
        )
    )
    checks.append(
        _check_max(
            "cvae_prior_pairwise_cosine",
            cvae.get("prior_pairwise_cosine"),
            thresholds.max_prior_pairwise_cosine,
        )
    )

    adapter = report.get("adapter", {})
    checks.append(
        _check_max(
            "adapter_output_pairwise_cosine",
            adapter.get("output_pairwise_cosine"),
            thresholds.max_adapter_pairwise_cosine,
        )
    )
    checks.append(
        _check_min(
            "adapter_output_per_dim_std",
            adapter.get("output_per_dim_std"),
            thresholds.min_adapter_per_dim_std,
        )
    )
    # Diversity alone is not enough: an adapter can make every speaker's
    # residual orthogonal while shrinking the common style mean toward zero.
    # Requiring target cosine catches that failure mode, which previously
    # passed the diversity/std checks with a target cosine near 0.48.
    checks.append(
        _check_min(
            "adapter_target_cosine",
            adapter.get("output_target_cosine"),
            thresholds.min_adapter_target_cosine,
        )
    )

    return GateResult(passed=all(check.passed for check in checks), checks=checks)
