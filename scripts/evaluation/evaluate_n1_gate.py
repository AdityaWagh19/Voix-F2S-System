"""Evaluate the N1 AMI LoRA pilot against the pre-registered gate.

The trainer writes paired base/LoRA rows with WER, ECAPA similarity, and basic
prosody.  This script adds the naturalness metrics from
``voix.evaluation.naturalness``, compares each generated clip with its real AMI
target, and applies the N1 gate:

* WER is within ``base + 0.05``;
* target and reference ECAPA similarity are within ``base - 0.05``;
* at least one pre-registered prosody metric is closer to the real AMI target
  with a speaker-cluster bootstrap CI whose upper bound is below zero.

The gate is deliberately conservative.  It is a proxy result, not MOS.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from voix.evaluation.naturalness import extract_naturalness_metrics  # noqa: E402
from voix.evaluation.stats import speaker_cluster_bootstrap  # noqa: E402

DEFAULT_EVAL = ROOT / "results" / "naturalness" / "n1" / "eval_pilot.json"
DEFAULT_MANIFEST = ROOT / "data" / "processed" / "ami_lora" / "manifest.csv"
DEFAULT_REPORT = ROOT / "results" / "naturalness" / "n1" / "gate_report.json"

PROSODY_METRICS = (
    "f0_median_hz",
    "f0_range_st",
    "words_per_second",
    "energy_std_db",
    "pause_ratio",
    "spectral_tilt_db_per_khz",
)
DIRECT_METRICS = (
    "wer",
    "speaker_similarity_target",
    "speaker_similarity_ref",
)


def _finite(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _cluster_interval(
    rows: Sequence[Mapping[str, Any]],
    *,
    metric_key: str = "delta",
    n_resamples: int = 10_000,
    seed: int = 0,
) -> Dict[str, Any]:
    usable = [row for row in rows if _finite(row.get(metric_key)) is not None]
    if len(usable) < 2:
        return {
            "status": "unavailable",
            "reason": "fewer than two complete pairs",
            "n_pairs": len(usable),
        }
    speakers = {str(row["speaker_id"]) for row in usable}
    if len(speakers) < 2:
        return {
            "status": "unavailable",
            "reason": "fewer than two speakers",
            "n_pairs": len(usable),
        }

    def mean_delta(subset: Sequence[Mapping[str, Any]]) -> float:
        values = [_finite(row.get(metric_key)) for row in subset]
        clean = [value for value in values if value is not None]
        return float(np.mean(clean)) if clean else float("nan")

    interval = speaker_cluster_bootstrap(
        usable,
        mean_delta,
        n_resamples=n_resamples,
        seed=seed,
    )
    return {
        "status": "computed",
        "n_pairs": len(usable),
        "n_speakers": len(speakers),
        "n_clusters": int(interval.n),
        "mean_delta": float(interval.estimate),
        "ci_low": float(interval.ci_low),
        "ci_high": float(interval.ci_high),
        "ci_confidence": 0.95,
        "ci_n_resamples": int(interval.n_resamples),
        "ci_seed": int(interval.seed),
    }


def _by_speaker(
    rows: Sequence[Mapping[str, Any]],
    *,
    metric_key: str = "delta",
) -> Dict[str, Dict[str, Any]]:
    """Descriptive per-speaker deltas; not used by the gate decision."""
    grouped: Dict[str, List[float]] = {}
    for row in rows:
        value = _finite(row.get(metric_key))
        if value is None:
            continue
        grouped.setdefault(str(row["speaker_id"]), []).append(value)
    return {
        speaker: {
            "n_pairs": len(values),
            "mean_delta": float(np.mean(values)),
        }
        for speaker, values in sorted(grouped.items())
    }


def _target_metrics(
    target_paths: Mapping[str, str],
    target_texts: Mapping[str, str],
    *,
    sample_rate: int = 24000,
) -> Dict[str, Dict[str, Optional[float]]]:
    cache: Dict[str, Dict[str, Optional[float]]] = {}
    for clip_id, path in target_paths.items():
        if not Path(path).is_file():
            continue
        cache[str(clip_id)] = extract_naturalness_metrics(
            path, text=target_texts.get(str(clip_id)), sample_rate=sample_rate
        )
    return cache


def evaluate(
    eval_payload: Mapping[str, Any],
    manifest: pd.DataFrame,
    *,
    n_resamples: int = 10_000,
    seed: int = 0,
    min_paired_clips: int = 104,
) -> Dict[str, Any]:
    rows = list(eval_payload.get("rows") or [])
    if not rows:
        raise ValueError("evaluation payload has no rows")
    by_clip: Dict[str, Dict[str, Mapping[str, Any]]] = {}
    for row in rows:
        by_clip.setdefault(str(row["clip_id"]), {})[str(row["condition"])] = row

    manifest_by_clip = {
        str(row["clip_id"]): row for row in manifest.to_dict("records")
    }
    target_paths = {
        clip_id: str(manifest_by_clip[clip_id]["wav_path"])
        for clip_id in by_clip
        if clip_id in manifest_by_clip
    }
    target_texts = {
        clip_id: str(manifest_by_clip[clip_id].get("text", ""))
        for clip_id in by_clip
        if clip_id in manifest_by_clip
    }
    targets = _target_metrics(target_paths, target_texts)

    direct_rows: List[Dict[str, Any]] = []
    naturalness_rows: Dict[str, List[Dict[str, Any]]] = {
        metric: [] for metric in PROSODY_METRICS
    }
    generated_naturalness: Dict[str, Dict[str, Optional[float]]] = {}
    for clip_id, pair in sorted(by_clip.items()):
        if "base" not in pair or "lora" not in pair:
            continue
        base = pair["base"]
        lora = pair["lora"]
        speaker = str(base["speaker_id"])
        for metric in DIRECT_METRICS:
            left = _finite(lora.get(metric))
            right = _finite(base.get(metric))
            if left is None or right is None:
                continue
            direct_rows.append(
                {
                    "clip_id": clip_id,
                    "speaker_id": speaker,
                    "metric": metric,
                    "delta": left - right,
                }
            )

        for condition, row in (("base", base), ("lora", lora)):
            key = f"{clip_id}:{condition}"
            if key not in generated_naturalness:
                generated_naturalness[key] = extract_naturalness_metrics(
                    str(row["wav_path"]), text=str(row.get("text", "")), sample_rate=24000
                )
        target = targets.get(clip_id)
        if target is None:
            continue
        for metric in PROSODY_METRICS:
            base_value = generated_naturalness[f"{clip_id}:base"].get(metric)
            lora_value = generated_naturalness[f"{clip_id}:lora"].get(metric)
            target_value = target.get(metric)
            if base_value is None or lora_value is None or target_value is None:
                continue
            base_error = abs(float(base_value) - float(target_value))
            lora_error = abs(float(lora_value) - float(target_value))
            naturalness_rows[metric].append(
                {
                    "clip_id": clip_id,
                    "speaker_id": speaker,
                    "base_value": float(base_value),
                    "lora_value": float(lora_value),
                    "target_value": float(target_value),
                    "base_abs_error": base_error,
                    "lora_abs_error": lora_error,
                    "delta": lora_error - base_error,
                    "signed_delta": float(lora_value) - float(base_value),
                }
            )

    direct = {}
    for metric in DIRECT_METRICS:
        subset = [row for row in direct_rows if row["metric"] == metric]
        direct[metric] = _cluster_interval(
            subset, n_resamples=n_resamples, seed=seed
        )
        direct[metric]["by_speaker"] = _by_speaker(subset)

    naturalness = {}
    for metric in PROSODY_METRICS:
        subset = naturalness_rows[metric]
        interval = _cluster_interval(
            subset, n_resamples=n_resamples, seed=seed
        )
        if interval.get("status") == "computed":
            signed = [row["signed_delta"] for row in subset]
            interval["mean_signed_delta"] = float(np.mean(signed))
            interval["closer_to_ami"] = bool(
                interval["mean_delta"] < 0.0 and interval["ci_high"] < 0.0
            )
        else:
            interval["closer_to_ami"] = False
        interval["by_speaker"] = _by_speaker(subset)
        naturalness[metric] = interval

    wer = direct["wer"]
    target_sim = direct["speaker_similarity_target"]
    ref_sim = direct["speaker_similarity_ref"]
    wer_ok = (
        wer.get("status") == "computed"
        and float(wer["mean_delta"]) <= 0.05
    )
    target_ok = (
        target_sim.get("status") == "computed"
        and float(target_sim["mean_delta"]) >= -0.05
    )
    ref_ok = (
        ref_sim.get("status") == "computed"
        and float(ref_sim["mean_delta"]) >= -0.05
    )
    improved = [
        metric for metric, row in naturalness.items() if row.get("closer_to_ami")
    ]
    n_paired = sum(
        1 for pair in by_clip.values() if "base" in pair and "lora" in pair
    )
    coverage_ok = n_paired >= int(min_paired_clips)
    gate_passed = bool(
        wer_ok and target_ok and ref_ok and improved and coverage_ok
    )

    return {
        "schema_version": "voix.n1_gate.v1",
        "track": "naturalness",
        "level": "exploratory",
        "source_eval": str(DEFAULT_EVAL),
        "n_clips": len(by_clip),
        "n_paired_clips": n_paired,
        "direct_metrics": direct,
        "naturalness_metrics": naturalness,
        "gate": {
            "wer_within_base_plus_0.05": wer_ok,
            "target_ecapa_within_base_minus_0.05": target_ok,
            "reference_ecapa_within_base_minus_0.05": ref_ok,
            "improved_prosody_metrics": improved,
            "at_least_one_prosody_ci_below_zero": bool(improved),
            "minimum_paired_clips": int(min_paired_clips),
            "paired_clip_coverage_sufficient": coverage_ok,
            "passed": gate_passed,
        },
        "notes": [
            "Naturalness metrics compare generated audio with the real AMI target clip.",
            "Absolute error is |generated - target|; a negative delta means LoRA is closer to AMI.",
            "The gate requires the full paired-clip coverage, not just a subset with a favorable CI.",
            "The gate is a proxy result, not MOS/CMOS.",
            "The trainer adapts model.diffusion.unet only, not model.predictor.",
        ],
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval", type=Path, default=DEFAULT_EVAL)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--n-resamples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--min-paired-clips",
        type=int,
        default=104,
        help="Minimum base/LoRA pairs required before the gate can pass.",
    )
    args = parser.parse_args(argv)

    payload = json.loads(args.eval.read_text(encoding="utf-8"))
    manifest = pd.read_csv(args.manifest)
    result = evaluate(
        payload,
        manifest,
        n_resamples=args.n_resamples,
        seed=args.seed,
        min_paired_clips=args.min_paired_clips,
    )
    result["source_eval"] = str(args.eval)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )

    print(f"Wrote {args.output}")
    print(json.dumps(result["gate"], indent=2))
    for name, row in result["direct_metrics"].items():
        print(f"{name}: {row}")
    for name, row in result["naturalness_metrics"].items():
        print(f"{name}: {row}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
