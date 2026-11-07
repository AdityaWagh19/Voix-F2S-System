"""Post-hoc Holm sensitivity for the emotion direction checks.

The pre-registered emotion gate uses paired bootstrap CIs over validation
actors. This script is a robustness check, not a replacement for that gate: it
reconstructs a one-sample t test from the reported paired mean, SD, and n, then
applies Holm correction across the six direction checks (three emotions x F0
and duration). It does not use per-pair rows, so it is a summary-level
sensitivity analysis.

Usage:
    python scripts/evaluation/emotion_holm_sensitivity.py \
        --e1 results/emotion/e1c/report_prosody.json \
        --e2 results/emotion/e2c/report_main.json \
        --output results/emotion/holm_sensitivity.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict

from scipy.stats import t as student_t

CONDITIONS = ("happy", "sad", "angry")
METRICS = ("f0_mean_hz", "duration_seconds")
RUNS = {
    "e1_class_mean_prosody": "results/emotion/e1c/report_prosody.json",
    "e2_learned_residual": "results/emotion/e2c/report_main.json",
    "e2_corrected_label_null": "results/emotion/e2c/report_label_shuffled.json",
}


def holm_adjust(pvalues: Dict[str, float]) -> Dict[str, float]:
    """Return Holm step-down adjusted p-values."""
    ordered = sorted(pvalues.items(), key=lambda item: item[1])
    count = len(ordered)
    adjusted: Dict[str, float] = {}
    previous = 0.0
    for index, (name, pvalue) in enumerate(ordered):
        value = min(1.0, (count - index) * float(pvalue))
        value = max(value, previous)
        adjusted[name] = value
        previous = value
    return adjusted


def _condition_key(effects: Dict[str, Any], emotion: str) -> str:
    matches = [key for key in effects if key.endswith(f":{emotion}")]
    if len(matches) != 1:
        raise ValueError(f"expected one condition for {emotion}, found {matches}")
    return matches[0]


def analyse_run(report: Dict[str, Any]) -> Dict[str, Any]:
    effects = report["paired_effects"]["effects"]
    checks: Dict[str, Any] = {}
    pvalues: Dict[str, float] = {}
    for emotion in CONDITIONS:
        condition = _condition_key(effects, emotion)
        for metric in METRICS:
            effect = effects[condition][metric]
            n = int(effect["n_pairs"])
            mean = float(effect["mean_delta"])
            sd = float(effect["sd_delta"])
            if n < 2 or sd <= 0:
                raise ValueError(f"invalid paired summary for {condition}:{metric}")
            statistic = mean / (sd / math.sqrt(n))
            pvalue = float(2.0 * student_t.sf(abs(statistic), n - 1))
            name = f"{emotion}:{metric}"
            checks[name] = {
                "mean_delta": mean,
                "sd_delta": sd,
                "n_pairs": n,
                "t_statistic": statistic,
                "p_value": pvalue,
                "cohens_dz": mean / sd,
                "expected_direction": int(effect.get("expected_direction", 0)),
                "direction_match": bool(effect.get("direction_match", False)),
                "pre_registered_ci_excludes_zero": bool(
                    effect.get("ci_excludes_zero_in_expected_direction", False)
                ),
            }
            pvalues[name] = pvalue

    adjusted = holm_adjust(pvalues)
    for name, value in adjusted.items():
        checks[name]["holm_adjusted_p"] = value
        checks[name]["holm_pass_0_05"] = value < 0.05
        expected = int(checks[name]["expected_direction"])
        directional = bool(
            value < 0.05
            and checks[name]["direction_match"]
            and checks[name]["mean_delta"] * expected > 0
        )
        checks[name]["holm_pass_expected_direction"] = directional

    return {
        "n_checks": len(checks),
        "checks": checks,
        "holm_passed_two_sided": sorted(
            name for name, value in adjusted.items() if value < 0.05
        ),
        "holm_passed_expected_direction": sorted(
            name
            for name, row in checks.items()
            if row["holm_pass_expected_direction"]
        ),
        "all_holm_pass_two_sided": all(value < 0.05 for value in adjusted.values()),
        "all_holm_pass_expected_direction": all(
            row["holm_pass_expected_direction"] for row in checks.values()
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--e1", default=RUNS["e1_class_mean_prosody"])
    parser.add_argument("--e2", default=RUNS["e2_learned_residual"])
    parser.add_argument("--null", default=RUNS["e2_corrected_label_null"])
    parser.add_argument("--output", default="results/emotion/holm_sensitivity.json")
    args = parser.parse_args()

    paths = {
        "e1_class_mean_prosody": Path(args.e1),
        "e2_learned_residual": Path(args.e2),
        "e2_corrected_label_null": Path(args.null),
    }
    result = {
        "schema_version": "voix.emotion_holm_sensitivity.v1",
        "method": (
            "One-sample t test reconstructed from reported paired mean, SD, and "
            "n_pairs; Holm correction across six direction checks per run. "
            "Post-hoc sensitivity only; the pre-registered gate remains the "
            "paired bootstrap CI."
        ),
        "runs": {},
    }
    for name, path in paths.items():
        report = json.loads(path.read_text(encoding="utf-8"))
        result["runs"][name] = {
            "report": str(path),
            "status": report.get("status"),
            **analyse_run(report),
        }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    for name, run in result["runs"].items():
        print(f"== {name} (status={run['status']})")
        for check, row in run["checks"].items():
            print(
                f"  {check:25s} delta={row['mean_delta']:+.4f} "
                f"t={row['t_statistic']:+.2f} p={row['p_value']:.3g} "
                f"p_holm={row['holm_adjusted_p']:.3g} "
                f"{'PASS' if row['holm_pass_0_05'] else 'FAIL'}"
                f"{' (direction)' if row['holm_pass_expected_direction'] else ''}"
            )
        print(f"  all two-sided Holm pass: {run['all_holm_pass_two_sided']}")
        print(
            "  all expected-direction Holm pass: "
            f"{run['all_holm_pass_expected_direction']}"
        )
    print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
