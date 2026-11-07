"""Audit the downloaded GLOBE accent WAVs against the selection manifest.

This is the post-download health gate.  It checks that every manifest clip has
a readable 24 kHz mono WAV, that its duration matches the Common Voice
duration, and that the samples are finite and not clipped.  It does not
measure MOS or accent quality.

Usage:
    python scripts/audit/audit_globe_audio.py
    python scripts/audit/audit_globe_audio.py --workers 8
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import soundfile as sf

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_MANIFEST = ROOT / "data" / "processed" / "globe_accent_manifest.csv"
DEFAULT_AUDIO_DIR = ROOT / "data" / "processed" / "globe_audio"
DEFAULT_OUTPUT = ROOT / "data" / "processed" / "globe_accent_audio_audit.json"


def audio_path(audio_dir: Path, row: pd.Series) -> Path:
    return (
        audio_dir
        / str(row["speaker_split"])
        / str(row["accent_class"])
        / f"{row['clip_id']}.wav"
    )


def audit_one(
    row: pd.Series,
    *,
    audio_dir: Path,
    max_trim_sec: float,
    max_trim_fraction: float,
    max_overrun_sec: float,
    min_actual_sec: float,
) -> Dict[str, object]:
    path = audio_path(audio_dir, row)
    result: Dict[str, object] = {
        "clip_id": row["clip_id"],
        "accent_class": row["accent_class"],
        "speaker_split": row["speaker_split"],
        "client_id": row["client_id"],
        "path": str(path),
        "expected_duration_sec": float(row["duration_sec"]),
    }
    if not path.is_file():
        result.update(status="missing", error="file not found")
        return result
    try:
        info = sf.info(path)
        audio, sample_rate = sf.read(path, dtype="float32", always_2d=False)
    except Exception as exc:
        result.update(status="error", error=f"{type(exc).__name__}: {exc}")
        return result

    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    duration = float(len(audio)) / sample_rate if sample_rate else 0.0
    finite = bool(np.isfinite(audio).all())
    peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
    clipping = float(np.mean(np.abs(audio) >= 0.999)) if len(audio) else 0.0
    rms = float(np.sqrt(np.mean(np.square(audio)))) if len(audio) else 0.0
    errors: List[str] = []
    if sample_rate != 24000:
        errors.append(f"sample_rate={sample_rate}")
    if info.channels != 1:
        errors.append(f"channels={info.channels}")
    if not finite:
        errors.append("non-finite samples")
    if peak > 1.0:
        errors.append(f"peak={peak:.4f}")
    if clipping > 0.0:
        errors.append(f"clipping_fraction={clipping:.6f}")
    if rms < 1e-5:
        errors.append(f"near-silent rms={rms:.2e}")
    expected = float(row["duration_sec"])
    trim = expected - duration
    trim_fraction = trim / expected if expected > 0 else 0.0
    if duration < min_actual_sec:
        errors.append(f"actual duration {duration:.3f}s < {min_actual_sec:.3f}s")
    if trim > max_trim_sec:
        errors.append(f"trimmed {trim:.3f}s > {max_trim_sec:.3f}s")
    if trim_fraction > max_trim_fraction:
        errors.append(
            f"trimmed {trim_fraction:.3f} of expected > {max_trim_fraction:.3f}"
        )
    if -trim > max_overrun_sec:
        errors.append(f"longer than expected by {-trim:.3f}s")
    result.update(
        status="ok" if not errors else "error",
        error="; ".join(errors) if errors else None,
        sample_rate=int(sample_rate),
        channels=int(info.channels),
        duration_sec=round(duration, 4),
        trim_sec=round(trim, 4),
        trim_fraction=round(trim_fraction, 4),
        peak=round(peak, 6),
        rms=round(rms, 6),
        clipping_fraction=round(clipping, 8),
        finite=finite,
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--audio-dir", type=Path, default=DEFAULT_AUDIO_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--max-trim-sec",
        type=float,
        default=7.0,
        help=(
            "Maximum absolute VAD trim versus the Common Voice duration. "
            "GLOBE can remove long leading/trailing silence; 6.33 s was the "
            "largest observed value."
        ),
    )
    parser.add_argument(
        "--max-trim-fraction",
        type=float,
        default=0.85,
        help="Maximum VAD trim as a fraction of the Common Voice duration.",
    )
    parser.add_argument(
        "--max-overrun-sec",
        type=float,
        default=0.1,
        help="Maximum allowed excess over the Common Voice duration.",
    )
    parser.add_argument(
        "--min-actual-sec",
        type=float,
        default=1.0,
        help="Minimum actual speech duration after VAD trimming.",
    )
    parser.add_argument(
        "--min-hours",
        type=float,
        default=2.0,
        help="Authoritative post-trim hours required per trainable class.",
    )
    parser.add_argument(
        "--min-speakers",
        type=int,
        default=30,
        help="Authoritative post-trim speaker count required per trainable class.",
    )
    parser.add_argument(
        "--evaluation-only",
        nargs="*",
        default=["irish"],
        choices=[
            "us",
            "england",
            "australian",
            "canadian",
            "india_south_asia",
            "irish",
        ],
        help="Classes kept for evaluation but excluded from the training gate.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.manifest.is_file():
        raise SystemExit(f"missing manifest: {args.manifest}")
    manifest = pd.read_csv(args.manifest)
    rows = [row for _, row in manifest.iterrows()]
    results: List[Dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {
            pool.submit(
                audit_one,
                row,
                audio_dir=args.audio_dir,
                max_trim_sec=args.max_trim_sec,
                max_trim_fraction=args.max_trim_fraction,
                max_overrun_sec=args.max_overrun_sec,
                min_actual_sec=args.min_actual_sec,
            ): row
            for row in rows
        }
        for future in as_completed(futures):
            results.append(future.result())

    failures = [item for item in results if item["status"] != "ok"]
    ok = [item for item in results if item["status"] == "ok"]
    by_class: Dict[str, Dict[str, object]] = {}
    for accent in sorted(manifest["accent_class"].unique()):
        subset = [item for item in results if item["accent_class"] == accent]
        good = [item for item in subset if item["status"] == "ok"]
        by_class[accent] = {
            "clips": len(subset),
            "ok": len(good),
            "failed": len(subset) - len(good),
            "hours": round(sum(float(item.get("duration_sec", 0) or 0) for item in good) / 3600.0, 3),
            "speakers": len({item["client_id"] for item in good}),
        }
    trims = [float(item["trim_sec"]) for item in ok if "trim_sec" in item]
    evaluation_only = set(args.evaluation_only)
    class_gate_failures: List[str] = []
    for accent, stats in by_class.items():
        if accent in evaluation_only:
            continue
        if float(stats["hours"]) < args.min_hours:
            class_gate_failures.append(
                f"{accent}: {stats['hours']:.3f}h < {args.min_hours:.3f}h"
            )
        if int(stats["speakers"]) < args.min_speakers:
            class_gate_failures.append(
                f"{accent}: {stats['speakers']} speakers < {args.min_speakers}"
            )
    summary = {
        "manifest": str(args.manifest),
        "audio_dir": str(args.audio_dir),
        "clips": len(manifest),
        "ok": len(ok),
        "failed": len(failures),
        "hours": round(sum(float(item.get("duration_sec", 0) or 0) for item in ok) / 3600.0, 3),
        "max_trim_sec": args.max_trim_sec,
        "max_trim_fraction": args.max_trim_fraction,
        "max_overrun_sec": args.max_overrun_sec,
        "min_actual_sec": args.min_actual_sec,
        "mean_trim_sec": round(sum(trims) / len(trims), 4) if trims else None,
        "max_observed_trim_sec": round(max(trims), 4) if trims else None,
        "min_hours": args.min_hours,
        "min_speakers": args.min_speakers,
        "evaluation_only_classes": sorted(evaluation_only),
        "gate_pass": len(failures) == 0 and not class_gate_failures,
        "class_gate_failures": class_gate_failures,
        "by_class": by_class,
        "failure_count": len(failures),
        "failures": failures[:1000],
        "notes": [
            "Health gate only; this is not an accent-quality or MOS measurement.",
            "GLOBE audio is VAD-trimmed, so actual durations are shorter than Common Voice clip durations.",
            "The actual hours here are authoritative for the 2-hour class gate; the selection summary uses pre-trim durations.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Wrote {args.output}")
    if failures or class_gate_failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
