"""Check that core face-to-voice training code never references AMI, CREMA-D,
GLOBE, or MSP-Podcast.

Usage::

    python scripts/audit/check_data_firewall.py
    python scripts/audit/check_data_firewall.py --output results/data_firewall.json --require-pass
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from voix.evaluation.data_firewall import scan_core_training  # noqa: E402


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scan core training code for data leakage")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--require-pass", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    report = scan_core_training(args.root)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Scanned {len(report['scanned'])} core training file(s)")
    print(f"Passed: {report['passed']}")
    for violation in report["violations"]:
        print(
            f"ERROR: {violation['path']}:{violation['line']} "
            f"references {violation['dataset']}: {violation['text']}"
        )
    if args.output is not None:
        print(f"Wrote {args.output}")
    return 0 if report["passed"] or not args.require_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
