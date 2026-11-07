"""Recompute the emotion direction gate from a saved research report.

The first E1 reports were produced before ``condition_gate_summary`` learned to
infer a non-``fixed_delta`` condition prefix.  Their per-clip measurements and
paired effects are valid, but the gate looked for ``fixed_delta:*`` keys and
marked every E1 check ``unavailable``.

This utility does not re-measure audio and does not add evidence.  It loads the
saved ``paired_effects`` block, recomputes only the direction gate with the
fixed code, records that the gate was recomputed, and writes a corrected report.
Use it only for reports whose paired effects were already computed.

Examples::

    python scripts/evaluation/recompute_emotion_direction_gate.py \
        --report results/emotion/e1/report_prosody.json \
        --output results/emotion/e1/report_prosody.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from voix.evaluation.crema_e0 import condition_gate_summary
from voix.evaluation.research_report import (
    RESEARCH_REPORT_SCHEMA,
    validate_report,
)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recompute the direction gate from saved paired effects"
    )
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--condition-prefix",
        default=None,
        help="Residual condition prefix; inferred when exactly one is present",
    )
    return parser.parse_args(argv)


def _prefixes(effects: Dict[str, Any]) -> list[str]:
    return sorted(
        {
            str(key).split(":", 1)[0]
            for key in effects
            if ":" in str(key)
        }
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if not args.report.is_file():
        raise FileNotFoundError(f"report not found: {args.report}")
    report = json.loads(args.report.read_text(encoding="utf-8"))
    if report.get("schema_version") != RESEARCH_REPORT_SCHEMA:
        raise ValueError(
            f"unexpected schema {report.get('schema_version')!r}; "
            f"expected {RESEARCH_REPORT_SCHEMA!r}"
        )
    paired_effects = report.get("paired_effects")
    if not isinstance(paired_effects, dict):
        raise ValueError("report has no paired_effects block")
    if paired_effects.get("status") != "computed":
        raise ValueError("paired_effects status is not 'computed'")

    effects = paired_effects.get("effects") or {}
    prefixes = _prefixes(effects)
    prefix = args.condition_prefix
    if prefix is None:
        if len(prefixes) != 1:
            raise ValueError(
                "cannot infer a single condition prefix; "
                f"found {prefixes}; pass --condition-prefix"
            )
        prefix = prefixes[0]

    gate = condition_gate_summary(paired_effects, condition_prefix=prefix)
    gate["condition_prefix"] = prefix
    gate["recomputed_from_saved_paired_effects"] = True
    gate["recompute_reason"] = (
        "The original report was generated before the direction gate supported "
        "non-fixed condition prefixes. No measurement changed."
    )

    gates = report.setdefault("gates", {})
    previous = gates.get("e0_directions")
    if isinstance(previous, dict):
        gate["previous_gate"] = previous
    gates["e0_directions"] = gate

    limitations = report.setdefault("limitations", [])
    note = (
        "The direction gate was recomputed from the saved paired effects after "
        "fixing condition-prefix inference; audio was not re-measured."
    )
    if note not in limitations:
        limitations.append(note)

    errors = validate_report(report)
    if errors:
        raise ValueError(f"recomputed report is invalid: {errors}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(
        f"Wrote {args.output}: prefix={prefix} passed={gate['passed']} "
        f"failures={gate['failures']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
