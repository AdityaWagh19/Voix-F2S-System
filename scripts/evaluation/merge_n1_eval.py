"""Merge N1 metric parts into one evaluation payload for the gate script."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge N1 metric parts")
    parser.add_argument("--parts", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    rows: List[Dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for path in args.parts:
        payload = json.loads(path.read_text(encoding="utf-8"))
        for row in payload.get("rows") or []:
            key = (str(row["clip_id"]), str(row["condition"]))
            if key in seen:
                raise ValueError(f"duplicate row in parts: {key}")
            seen.add(key)
            rows.append(dict(row))
    if not rows:
        raise ValueError("no rows were merged")
    output = {
        "status": "computed",
        "summary": {
            "status": "computed",
            "n": len(rows),
            "note": "Merged from bounded-memory metric parts.",
        },
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.output} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
