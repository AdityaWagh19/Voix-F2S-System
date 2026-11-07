"""Run a multi-seed, multi-prompt diffusion-step ablation.

The single-seed diagnostic in ``diagnose_audio_quality.py`` showed that more
steps do not monotonically improve the noise proxy.  This driver repeats that
diagnostic across seeds and prompts so the tier-1 default is not chosen from
one sample.  It does not produce MOS or a perceptual claim.

Example:
    python scripts/evaluation/run_step_ablation.py --device cuda
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

ROOT = Path(__file__).resolve().parents[2]
DIAGNOSTIC = ROOT / "scripts" / "audit" / "diagnose_audio_quality.py"
DEFAULT_OUTPUT = ROOT / "results" / "step_ablation.json"
DEFAULT_WORK_DIR = ROOT / "results" / "step_ablation"

PROMPTS = {
    "formal": (
        "The human voice reflects both physical anatomy and lived culture, "
        "carrying the unique cadence of personal identity."
    ),
    "conversational": (
        "Hey, so I tried that new place down the street, and honestly? "
        "The coffee was better than I expected."
    ),
}


def _run_one(
    *,
    seed: int,
    prompt_id: str,
    prompt: str,
    steps: Iterable[int],
    device: str,
    emotion: str,
    emotion_intensity: float,
    work_dir: Path,
) -> Dict[str, Any]:
    output = work_dir / f"seed{seed}_{prompt_id}.json"
    if output.is_file():
        print(f"[skip] {output} exists")
        return json.loads(output.read_text(encoding="utf-8"))

    command = [
        sys.executable,
        str(DIAGNOSTIC),
        "--device",
        device,
        "--seed",
        str(seed),
        "--text",
        prompt,
        "--steps",
        *[str(step) for step in steps],
        "--emotion",
        emotion,
        "--emotion-intensity",
        str(emotion_intensity),
        "--output",
        str(output),
    ]
    print(f"[run] seed={seed} prompt={prompt_id} steps={list(steps)}")
    subprocess.run(command, cwd=str(ROOT), check=True)
    return json.loads(output.read_text(encoding="utf-8"))


def _stats(values: Iterable[float]) -> Dict[str, Any]:
    values = [float(value) for value in values]
    if not values:
        return {"n": 0, "mean": None, "std": None, "min": None, "max": None}
    return {
        "n": len(values),
        "mean": round(statistics.mean(values), 4),
        "std": round(statistics.pstdev(values), 4),
        "min": round(min(values), 4),
        "max": round(max(values), 4),
    }


def _condition_map(report: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {item["label"]: item for item in report.get("conditions", [])}


def aggregate(
    reports: List[Dict[str, Any]],
    *,
    seeds: Iterable[int],
    prompts: Dict[str, str],
    steps: Iterable[int],
    emotion: str,
) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    labels = set()
    for report in reports:
        prompt_id = next(
            (key for key, value in prompts.items() if value == report["text"]),
            "unknown",
        )
        for label, item in _condition_map(report).items():
            labels.add(label)
            raw = item["measurement"]["raw"]
            saved = item["measurement"]["save_current"]
            rows.append(
                {
                    "seed": report["seed"],
                    "prompt": prompt_id,
                    "condition": label,
                    "diffusion_steps": item["diffusion_steps"],
                    "emotion": item["emotion"],
                    "dsp": item["dsp"],
                    "raw_snr_proxy_db": raw["snr_proxy_db"],
                    "raw_noise_floor_dbfs": raw["noise_floor_dbfs"],
                    "raw_peak": raw["peak"],
                    "raw_clipping_fraction": raw["clipping_fraction"],
                    "saved_lufs": saved["lufs"],
                    "saved_noise_floor_dbfs": saved["noise_floor_dbfs"],
                }
            )

    by_label: Dict[str, Dict[str, Any]] = {}
    for label in sorted(labels):
        subset = [row for row in rows if row["condition"] == label]
        by_label[label] = {
            "n": len(subset),
            "diffusion_steps": subset[0]["diffusion_steps"] if subset else None,
            "emotion": subset[0]["emotion"] if subset else None,
            "dsp": subset[0]["dsp"] if subset else None,
            "raw_snr_proxy_db": _stats(row["raw_snr_proxy_db"] for row in subset),
            "raw_noise_floor_dbfs": _stats(
                row["raw_noise_floor_dbfs"] for row in subset
            ),
            "raw_peak": _stats(row["raw_peak"] for row in subset),
            "saved_lufs": _stats(row["saved_lufs"] for row in subset),
            "saved_noise_floor_dbfs": _stats(
                row["saved_noise_floor_dbfs"] for row in subset
            ),
        }

    step_summary: Dict[str, Any] = {}
    for step in steps:
        neutral = [
            row
            for row in rows
            if row["condition"] == f"neutral_steps{step}"
        ]
        whispered = [
            row
            for row in rows
            if row["condition"] == f"{emotion}_steps{step}"
        ]
        step_summary[str(step)] = {
            "n": len(neutral),
            "neutral_snr_proxy_db": _stats(
                row["raw_snr_proxy_db"] for row in neutral
            ),
            "neutral_noise_floor_dbfs": _stats(
                row["raw_noise_floor_dbfs"] for row in neutral
            ),
            "whispered_snr_proxy_db": _stats(
                row["raw_snr_proxy_db"] for row in whispered
            ),
            "whispered_noise_floor_dbfs": _stats(
                row["raw_noise_floor_dbfs"] for row in whispered
            ),
        }

    return {
        "schema_version": "voix.step_ablation.v1",
        "notes": [
            "This is a noise/loudness proxy ablation, not MOS or a perceptual naturalness claim.",
            "Each row holds the face, speaker embedding, text, and seed fixed within that run.",
            "The saved metrics use the current writer: no peak normalization, only -1 dBTP true-peak safety.",
            "Do not choose the tier-1 default from one seed or one prompt.",
        ],
        "seeds": list(seeds),
        "prompts": {key: value for key, value in prompts.items()},
        "steps": list(steps),
        "emotion": emotion,
        "by_condition": by_label,
        "by_step": step_summary,
        "rows": rows,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 2026])
    parser.add_argument("--steps", type=int, nargs="+", default=[5, 10, 20])
    parser.add_argument("--emotion", default="whispered")
    parser.add_argument("--emotion-intensity", type=float, default=0.9)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    reports = [
        _run_one(
            seed=seed,
            prompt_id=prompt_id,
            prompt=prompt,
            steps=args.steps,
            device=args.device,
            emotion=args.emotion,
            emotion_intensity=args.emotion_intensity,
            work_dir=args.work_dir,
        )
        for seed in args.seeds
        for prompt_id, prompt in PROMPTS.items()
    ]
    report = aggregate(
        reports,
        seeds=args.seeds,
        prompts=PROMPTS,
        steps=args.steps,
        emotion=args.emotion,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report["by_step"], indent=2, ensure_ascii=False))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
