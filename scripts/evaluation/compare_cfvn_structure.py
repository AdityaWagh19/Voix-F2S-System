"""Compare CFVN structure-preservation runs on validation, then report test.

The experiment asks whether a raw-face structure-distillation term can stop the
CFVN from losing face-face identity while it learns the voice mapping.  The
selection rule is fixed before looking at test:

1. A run is eligible when its selected-epoch validation ``cfvn_face_face_r1``
   is at least the raw-face baseline (no identity reduction).
2. Among eligible runs, prefer the highest validation cross-modal score
   ``val_r1 + 0.25 * val_r10``.
3. If no run is eligible, report the best face-face run and mark the identity
   gate false.

Test metrics are copied for every run but are not used to choose the winner.
The decision also reports ``test_identity_gate_winner`` and the number of runs
that pass the same face-face comparison on test, so a validation-only result
cannot be mistaken for a stable identity improvement.

Usage:
    python scripts/evaluation/compare_cfvn_structure.py
    python scripts/evaluation/compare_cfvn_structure.py --output results/cfvn_structure_comparison.json
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_OUTPUT = ROOT / "results" / "cfvn_structure_comparison.json"
DEFAULT_RUNS = [
    ROOT / "results" / "checkpoints" / f"cfvn_struct_s{index}.eval.json"
    for index in range(4)
] + [
    ROOT / "results" / "checkpoints" / f"cfvn_struct_{name}.eval.json"
    for name in ("s3_seed43", "s3_seed44", "f1", "f2")
]


def load_run(path: Path) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    test = payload.get("test", {})
    history = payload.get("history", [])
    best_epoch = int(test.get("best_epoch", 0))
    val = None
    for row in history:
        if int(row.get("epoch", -1)) == best_epoch:
            val = row
            break
    if val is None:
        return None
    return {
        "checkpoint": str(test.get("checkpoint", path.with_suffix(".pt"))),
        "eval": str(path),
        "best_epoch": best_epoch,
        "validation": {
            "val_r1": float(val["val_r1"]),
            "val_r10": float(val["val_r10"]),
            "val_chance_r1": float(val["val_chance_r1"]),
            "val_gender_controlled_r1": float(val["val_gender_controlled_r1"]),
            "cfvn_face_face_r1": float(val["cfvn_face_face_r1"]),
            "raw_face_face_r1": float(val["raw_face_face_r1"]),
            "selection_score": float(val["selection_score"]),
        },
        "test": {
            "val_r1": float(test["val_r1"]),
            "val_r5": float(test["val_r5"]),
            "val_r10": float(test["val_r10"]),
            "val_chance_r1": float(test["val_chance_r1"]),
            "val_gender_controlled_r1": float(test["val_gender_controlled_r1"]),
            "cfvn_face_face_r1": float(test["cfvn_face_face_r1"]),
            "raw_face_face_r1": float(test["raw_face_face_r1"]),
        },
    }


def choose(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Apply the fixed validation-only selection rule.

    Test metrics are attached to each run but are never used to pick the
    winner.  The returned ``test_identity_gate`` is a descriptive check on the
    already-chosen winner: validation eligibility does not guarantee that the
    same run preserves face identity on test.
    """
    eligible = [
        run
        for run in runs
        if run["validation"]["cfvn_face_face_r1"]
        >= run["validation"]["raw_face_face_r1"] - 1e-9
    ]
    pool = eligible if eligible else runs
    winner = max(
        pool,
        key=lambda run: (
            run["validation"]["val_r1"]
            + 0.25 * run["validation"]["val_r10"]
        ),
    )
    test_pass = [
        run
        for run in runs
        if run["test"]["cfvn_face_face_r1"]
        >= run["test"]["raw_face_face_r1"] - 1e-9
    ]
    return {
        "identity_gate": bool(eligible),
        "selection_split": "validation",
        "selection_rule": (
            "require cfvn_face_face_r1 >= raw_face_face_r1 on validation; "
            "then maximize val_r1 + 0.25 * val_r10"
        ),
        "winner": winner["checkpoint"],
        "winner_best_epoch": winner["best_epoch"],
        "eligible": [run["checkpoint"] for run in eligible],
        "test_identity_gate_winner": bool(
            winner["test"]["cfvn_face_face_r1"]
            >= winner["test"]["raw_face_face_r1"] - 1e-9
        ),
        "validation_eligible_runs": len(eligible),
        "test_identity_pass_runs": len(test_pass),
        "test_identity_pass": [run["checkpoint"] for run in test_pass],
    }


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("runs", nargs="*", type=Path, default=None)
    args = parser.parse_args(argv)
    paths = args.runs or DEFAULT_RUNS
    runs = [run for path in paths if (run := load_run(Path(path))) is not None]
    if not runs:
        raise SystemExit("no CFVN structure eval files found")
    payload = {
        "schema_version": "voix.cfvn_structure_comparison.v1",
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "runs": runs,
        "decision": choose(runs),
        "notes": [
            "The winner is chosen on validation; test is reported for every run but never used for selection.",
            "The original cfvn_v2_best.pt is not replaced by this experiment.",
            "A single seed is not evidence of a general improvement; check validation_eligible_runs vs test_identity_pass_runs.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["decision"], indent=2), flush=True)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
