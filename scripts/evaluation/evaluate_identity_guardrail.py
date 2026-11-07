"""Evaluate the cross-track identity guardrail or manage canonical hashes.

Examples::

    # Freeze the canonical checkpoint hashes once.
    python scripts/evaluation/evaluate_identity_guardrail.py --write-hashes

    # Verify the canonical artifacts have not changed.
    python scripts/evaluation/evaluate_identity_guardrail.py --verify-hashes --require-pass

    # Evaluate per-condition ECAPA rows and CFVN retrieval metrics.
    python scripts/evaluation/evaluate_identity_guardrail.py \
        --rows results/emotion/e0/identity_rows.csv \
        --cfvn-metrics results/emotion/e0/cfvn.json \
        --output results/emotion/e0/identity_guardrail.json --require-pass
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from voix.evaluation.identity_guardrail import (  # noqa: E402
    evaluate_identity_guardrail,
    verify_canonical_hashes,
    write_canonical_hashes,
)


def _load_rows(path: Path) -> List[Dict[str, Any]]:
    if path.suffix.lower() == ".csv":
        import pandas as pd

        return pd.read_csv(path).to_dict(orient="records")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and isinstance(payload.get("rows"), list):
        payload = payload["rows"]
    if not isinstance(payload, list):
        raise ValueError("rows JSON must be a list or contain a 'rows' list")
    return [dict(row) for row in payload]


def _load_json(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("expected a JSON object")
    return payload


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Identity guardrail and canonical hash utility")
    parser.add_argument("--write-hashes", action="store_true")
    parser.add_argument("--verify-hashes", action="store_true")
    parser.add_argument("--rows", type=Path, default=None)
    parser.add_argument("--cfvn-metrics", type=Path, default=None)
    parser.add_argument("--canonical-hashes", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--n-resamples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--require-pass", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.write_hashes:
        payload = write_canonical_hashes()
        print(json.dumps(payload, indent=2))
        return 0

    if args.verify_hashes:
        result = verify_canonical_hashes()
        print(json.dumps(result, indent=2))
        return 0 if result.get("passed") or not args.require_pass else 1

    if args.rows is None:
        raise SystemExit("--rows is required unless --write-hashes or --verify-hashes is used")

    rows = _load_rows(args.rows)
    cfvn_metrics = _load_json(args.cfvn_metrics) if args.cfvn_metrics else None
    hashes_path = args.canonical_hashes
    if hashes_path is None:
        default_path = ROOT / "results" / "canonical_hashes.json"
        hashes_path = default_path if default_path.exists() else None

    result = evaluate_identity_guardrail(
        rows,
        cfvn_metrics=cfvn_metrics,
        canonical_hashes_path=hashes_path,
        n_resamples=args.n_resamples,
        seed=args.seed,
    )
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"Wrote {args.output}")
    print(f"Passed: {result['passed']}")
    for failure in result["failures"]:
        print(f"ERROR: {failure}")
    return 0 if result["passed"] or not args.require_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
