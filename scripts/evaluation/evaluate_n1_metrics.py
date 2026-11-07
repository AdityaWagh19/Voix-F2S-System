"""Compute N1 generation metrics in a bounded-memory slice.

The full 104-clip evaluation holds StyleTTS2, ECAPA, librosa, and Whisper at
once, which exceeds this machine's RAM.  Run generation with
``train_styletts2_lora.py --eval-generate-only``, then run this script on
small slices in separate processes.  Merge the parts with
``scripts/evaluation/merge_n1_eval.py`` and score them with ``evaluate_n1_gate.py``.
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_EVAL = ROOT / "results" / "naturalness" / "n1" / "eval_pilot.json"
DEFAULT_OUTPUT = ROOT / "results" / "naturalness" / "n1" / "parts" / "part_000.json"


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score an N1 evaluation slice")
    parser.add_argument("--eval", type=Path, default=DEFAULT_EVAL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--wer-model", default="base")
    parser.add_argument("--skip-wer", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    payload = json.loads(args.eval.read_text(encoding="utf-8"))
    all_rows = list(payload.get("rows") or [])
    if not all_rows:
        raise ValueError(f"{args.eval} has no rows")
    start = max(0, int(args.offset))
    stop = start + max(1, int(args.limit))
    rows = [dict(row) for row in all_rows[start:stop]]
    if not rows:
        raise ValueError(f"slice [{start}:{stop}] is empty")

    from voix.evaluation.audio_metrics import transcribe_wer
    from voix.evaluation.metrics import prosody_statistics, speaker_similarity

    for row in rows:
        path = str(row["wav_path"])
        if not Path(path).is_file():
            raise FileNotFoundError(path)
        stats = prosody_statistics(path, sample_rate=24000)
        row.update(stats)
        row["speaker_similarity_target"] = speaker_similarity(
            path, str(row["target_wav_path"]), device=args.device
        )
        row["speaker_similarity_ref"] = speaker_similarity(
            path, str(row["ref_wav_path"]), device=args.device
        )
        gc.collect()

    if args.skip_wer:
        for row in rows:
            row["wer"] = None
    else:
        result = transcribe_wer(
            [str(row["wav_path"]) for row in rows],
            [str(row["text"]) for row in rows],
            model_size=args.wer_model,
            device=args.device,
        )
        if result.get("status") == "computed":
            for row, score in zip(rows, result["per_sample_wer"]):
                row["wer"] = float(score)
        else:
            for row in rows:
                row["wer"] = None

    output = {
        "status": "computed",
        "part": {"offset": start, "limit": int(args.limit), "n": len(rows)},
        "wer_status": {
            "status": "skipped" if args.skip_wer else "computed",
            "model": f"whisper-{args.wer_model}",
        },
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.output} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
