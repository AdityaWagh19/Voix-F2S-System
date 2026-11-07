"""Evaluate the A3 accent style-residual sweep and select one scale.

The selection rule is declared before looking at the result:

1. The A0 classifier proxy must be valid on the generated audio.
2. WER must stay within ``baseline + 0.05``.
3. ECAPA similarity to the real style donor must stay within
   ``baseline - 0.05``.
4. Among eligible scales, pick the highest trainable-classifier accuracy.
   Ties go to the smaller scale.

If no scale beats the baseline accuracy while satisfying the WER and identity
budgets, the result is a null: no scale is selected and the demo must not show
an accent condition.  No human accent-identification study is claimed.
"""
from __future__ import annotations

import argparse
import importlib.util
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

DEFAULT_INDEX = ROOT / "results" / "accent" / "a3" / "generation_index.csv"
DEFAULT_REPORT = ROOT / "results" / "accent" / "a3" / "scale_selection.json"
DEFAULT_FEATURES = ROOT / "data" / "processed" / "globe_accent_features.npz"
DEFAULT_MODEL = ROOT / "checkpoints" / "accent_classifier_globe.joblib"
WER_BUDGET = 0.05
IDENTITY_BUDGET = -0.05


def _load_a2_module():
    path = ROOT / "scripts" / "evaluation" / "evaluate_accent_a2.py"
    spec = importlib.util.spec_from_file_location("evaluate_accent_a2", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _finite(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate A3 accent residuals")
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--generation-report", type=Path, default=None)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--baseline-condition", default="regex")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--compute-wer", action="store_true")
    parser.add_argument("--wer-model", default="base")
    parser.add_argument("--domain-shift-threshold", type=float, default=3.0)
    parser.add_argument("--n-resamples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args(argv)


def _condition_wer(
    frame: pd.DataFrame,
    *,
    model_size: str,
    device: str,
) -> Dict[str, Any]:
    from voix.evaluation.audio_metrics import transcribe_wer

    by_accent: Dict[str, Any] = {}
    for accent, subset in frame.groupby("accent_class", sort=True):
        entry: Dict[str, Any] = {}
        for condition, group in subset.groupby("condition", sort=True):
            group = group.sort_values(["prompt_id", "seed"])
            outcome = transcribe_wer(
                group["wav_path"].astype(str).tolist(),
                group["text"].astype(str).tolist(),
                model_size=model_size,
                device=device,
            )
            entry[str(condition)] = outcome
        by_accent[str(accent)] = entry
    return {"status": "computed" if by_accent else "unavailable", "by_accent": by_accent}


def _identity_similarity(
    frame: pd.DataFrame,
    features: np.ndarray,
    *,
    donor_path: str,
    device: str,
) -> Dict[str, Any]:
    a2 = _load_a2_module()
    donor_features = a2._extract_features([donor_path], device=device)[0]
    donor = donor_features[:192]
    ecapa = features[:, :192]
    similarities = ecapa @ donor
    by_accent: Dict[str, Any] = {}
    for accent, subset in frame.groupby("accent_class", sort=True):
        entry: Dict[str, Any] = {}
        for condition, group in subset.groupby("condition", sort=True):
            indices = group.index.to_numpy()
            entry[str(condition)] = {
                "n": int(len(indices)),
                "mean_ecapa_to_donor": float(np.mean(similarities[indices])),
            }
        by_accent[str(accent)] = entry
    values = {
        (
            str(row.accent_class),
            str(row.condition),
            int(row.prompt_id),
            int(row.seed),
        ): float(similarities[index])
        for index, row in enumerate(frame.itertuples(index=False))
    }
    return {
        "status": "computed",
        "donor": donor_path,
        "by_accent": by_accent,
        "values": values,
    }


def _paired_delta(
    values: Mapping[tuple[str, str, int, int], float],
    *,
    baseline: str,
    treatment: str,
    n_resamples: int,
    seed: int,
) -> Dict[str, Any]:
    from voix.evaluation.stats import paired_bootstrap_difference

    left: List[float] = []
    right: List[float] = []
    for (accent, condition, prompt, run_seed), value in values.items():
        if condition != treatment:
            continue
        other = values.get((accent, baseline, prompt, run_seed))
        if other is None:
            continue
        left.append(float(value))
        right.append(float(other))
    if len(left) < 2:
        return {"status": "unavailable", "reason": "fewer than two paired values"}
    interval = paired_bootstrap_difference(
        left, right, n_resamples=n_resamples, seed=seed
    )
    return {
        "status": "computed",
        "mean_delta": float(interval.estimate),
        "ci_low": float(interval.ci_low),
        "ci_high": float(interval.ci_high),
        "n_pairs": int(interval.n),
    }


def select_scale(
    candidates: Sequence[Mapping[str, Any]],
    *,
    proxy_valid: bool,
    baseline_accuracy: Optional[float],
    wer_budget: float = WER_BUDGET,
    identity_budget: float = IDENTITY_BUDGET,
) -> Dict[str, Any]:
    """Apply the pre-declared A3 selection rule to per-scale candidates."""
    if not proxy_valid or baseline_accuracy is None:
        return {
            "status": "unavailable",
            "reason": "A0 classifier proxy is invalid on generated audio",
            "selected_scale": None,
            "eligible": [],
        }
    eligible: List[Dict[str, Any]] = []
    for candidate in candidates:
        accuracy = _finite(candidate.get("trainable_accuracy"))
        wer_delta = _finite(candidate.get("wer_delta"))
        identity_delta = _finite(candidate.get("identity_delta"))
        if accuracy is None or wer_delta is None or identity_delta is None:
            continue
        gain = accuracy - float(baseline_accuracy)
        ok = (
            wer_delta <= float(wer_budget)
            and identity_delta >= float(identity_budget)
            and gain > 0.0
        )
        if ok:
            eligible.append(
                {
                    "condition": str(candidate["condition"]),
                    "scale": float(candidate["scale"]),
                    "trainable_accuracy": float(accuracy),
                    "accuracy_gain": float(gain),
                    "wer_delta": float(wer_delta),
                    "identity_delta": float(identity_delta),
                }
            )
    if not eligible:
        return {
            "status": "null",
            "reason": (
                "no scale improved classifier accuracy while staying inside "
                "the WER and identity budgets"
            ),
            "selected_scale": None,
            "eligible": [],
        }
    eligible.sort(key=lambda row: (-row["trainable_accuracy"], row["scale"]))
    best = eligible[0]
    return {
        "status": "selected",
        "selected_scale": float(best["scale"]),
        "selected_condition": best["condition"],
        "eligible": eligible,
        "rule": (
            "highest trainable-classifier accuracy among scales with "
            "WER delta <= +0.05 and ECAPA donor delta >= -0.05; ties go to "
            "the smaller scale"
        ),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    frame = pd.read_csv(args.index)
    required = {
        "clip_id",
        "accent_class",
        "condition",
        "prompt_id",
        "text",
        "wav_path",
        "seed",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"generation index is missing columns: {missing}")
    missing_audio = [
        str(path) for path in frame["wav_path"].astype(str) if not Path(path).is_file()
    ]
    if missing_audio:
        raise FileNotFoundError(f"missing generated audio: {missing_audio[:5]}")

    a2 = _load_a2_module()
    import joblib

    model_payload = joblib.load(args.model)
    feature_key = str(model_payload.get("feature_key", "base"))
    ssl_config = model_payload.get("ssl_config")
    print(f"Measuring {len(frame)} clip(s) ...", flush=True)
    features = a2._extract_features(
        frame["wav_path"].astype(str).tolist(),
        device=args.device,
        ssl_config=ssl_config,
    )
    feature_frame = a2._load_feature_rows(args.features, feature_key=feature_key)
    classifier = a2._classifier_block(
        frame,
        features,
        model_path=args.model,
        feature_frame=feature_frame,
        domain_shift_threshold=args.domain_shift_threshold,
    )
    proxy_valid = bool(classifier.get("proxy_valid"))
    baseline_accuracy = classifier["by_condition"].get(
        args.baseline_condition, {}
    ).get("trainable_accuracy")

    identity: Dict[str, Any] = {
        "status": "unavailable",
        "reason": "no generation report with an identity donor",
    }
    donor_path = None
    if args.generation_report and args.generation_report.is_file():
        generation = json.loads(args.generation_report.read_text(encoding="utf-8"))
        donor_path = (generation.get("identity") or {}).get("path")
    if donor_path and Path(donor_path).is_file():
        identity = _identity_similarity(
            frame, features, donor_path=str(donor_path), device=args.device
        )

    if args.compute_wer:
        print("Transcribing with Whisper ...", flush=True)
        wer = _condition_wer(
            frame, model_size=args.wer_model, device=args.device
        )
    else:
        wer = {
            "status": "unavailable",
            "reason": "WER was not requested; pass --compute-wer",
            "by_accent": {},
        }

    # Pair per-sample values by (accent, condition, prompt, seed).
    wer_values: Dict[tuple[str, str, int, int], float] = {}
    if wer.get("status") == "computed":
        for accent, conditions in wer["by_accent"].items():
            for condition, outcome in conditions.items():
                if outcome.get("status") != "computed":
                    continue
                subset = frame[
                    (frame["accent_class"].astype(str) == accent)
                    & (frame["condition"].astype(str) == condition)
                ].sort_values(["prompt_id", "seed"])
                for row, score in zip(
                    subset.itertuples(index=False), outcome["per_sample_wer"]
                ):
                    wer_values[(accent, condition, int(row.prompt_id), int(row.seed))] = float(score)

    identity_values: Dict[tuple[str, str, int, int], float] = {}
    if identity.get("status") == "computed":
        identity_values = identity.pop("values")

    treatments = [
        condition
        for condition in sorted(frame["condition"].unique())
        if condition != args.baseline_condition
    ]
    candidates: List[Dict[str, Any]] = []
    for condition in treatments:
        scale = None
        if condition.startswith("style_s"):
            try:
                scale = float(condition.split("style_s", 1)[1])
            except ValueError:
                scale = None
        if scale is None:
            continue
        wer_delta = (
            _paired_delta(
                wer_values,
                baseline=args.baseline_condition,
                treatment=condition,
                n_resamples=args.n_resamples,
                seed=args.seed,
            )
            if wer_values
            else {"status": "unavailable"}
        )
        identity_delta = (
            _paired_delta(
                identity_values,
                baseline=args.baseline_condition,
                treatment=condition,
                n_resamples=args.n_resamples,
                seed=args.seed,
            )
            if identity_values
            else {"status": "unavailable"}
        )
        candidates.append(
            {
                "condition": condition,
                "scale": scale,
                "trainable_accuracy": classifier["by_condition"]
                .get(condition, {})
                .get("trainable_accuracy"),
                "wer_delta": wer_delta.get("mean_delta"),
                "wer_ci_low": wer_delta.get("ci_low"),
                "wer_ci_high": wer_delta.get("ci_high"),
                "identity_delta": identity_delta.get("mean_delta"),
                "identity_ci_low": identity_delta.get("ci_low"),
                "identity_ci_high": identity_delta.get("ci_high"),
            }
        )

    selection = select_scale(
        candidates,
        proxy_valid=proxy_valid,
        baseline_accuracy=baseline_accuracy,
    )
    report = {
        "schema_version": "voix.accent_a3_selection.v1",
        "track": "accent",
        "experiment": "A3",
        "level": "exploratory",
        "index": str(args.index),
        "index_sha256": a2._sha256(args.index),
        "model": str(args.model),
        "model_sha256": a2._sha256(args.model),
        "baseline_condition": args.baseline_condition,
        "proxy_valid": proxy_valid,
        "domain_shift": classifier.get("domain_shift"),
        "baseline_accuracy": baseline_accuracy,
        "classifier_by_condition": classifier.get("by_condition"),
        "candidates": candidates,
        "selection": selection,
        "wer": wer,
        "identity": identity,
        "limitations": [
            "The A0 classifier is a proxy trained on real GLOBE speech.",
            "A3 uses a single validation US donor style; speaker generalization is not measured.",
            "No human accent-identification or MOS study was run.",
            "A3 has no Irish residual because Irish speakers are eval-only.",
            "The residual is a class contrast; accent and speaker identity are confounded because GLOBE has no within-speaker cross-accent pairs.",
        ],
        "mos": {"status": "unavailable", "reason": "no human ratings collected"},
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.report}")
    print(json.dumps(selection, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
