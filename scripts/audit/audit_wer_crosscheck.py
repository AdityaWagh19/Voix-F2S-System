"""Cross-check the research demo WER with a second Whisper model.

The demo generator reports WER from whisper-base.  This script re-transcribes
the same 21 WAV files with a larger model and records per-clip results, the
report hash, and the audio hashes so the check is pinned to the exact demo.

Usage:
    python scripts/audit/audit_wer_crosscheck.py --model small
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from voix.evaluation.audio_metrics import transcribe_wer

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "results" / "demo" / "research_v2" / "report.json"
DEFAULT_OUTPUT = ROOT / "results" / "audit" / "wer_crosscheck" / "wer_small.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="small")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    report_bytes = REPORT.read_bytes()
    report = json.loads(report_bytes.decode("utf-8"))
    rows = report["conditions"]
    paths = [Path(row["wav_path"]) for row in rows]
    references = [row["text"] for row in rows]

    result = transcribe_wer(
        paths, references, model_size=args.model, device="cpu"
    )
    if result.get("status") != "computed":
        raise SystemExit(f"WER unavailable: {result.get('reason')}")

    payload = {
        "schema_version": "voix.wer_crosscheck.v1",
        "level": "exploratory",
        "source_report": str(REPORT),
        "source_report_sha256": hashlib.sha256(report_bytes).hexdigest(),
        "audio": [
            {
                "name": path.stem,
                "path": str(path),
                "sha256": _sha256(path),
                "reference": reference,
            }
            for path, reference in zip(paths, references)
        ],
        "model": result["model"],
        "n": result["n"],
        "mean_wer": result["wer"],
        "std_wer": result["wer_std"],
        "max_wer": max(result["per_sample_wer"]),
        "per_sample_wer": result["per_sample_wer"],
        "transcripts": result["transcripts"],
        "notes": [
            "This is an ASR cross-check, not a human intelligibility or naturalness rating.",
            "The reference text is identical for all 21 clips.",
            "A zero WER does not establish naturalness, emotion recognition, or accent recognition.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote {args.output}")
    print(
        f"{result['model']}: n={result['n']} mean WER={result['wer']:.4f} "
        f"max={max(result['per_sample_wer']):.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
