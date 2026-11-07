"""Evaluate the A2 accent G2P comparison.

Measures, for every generated accent clip:

* WER against the known prompt with Whisper;
* A0 accent-classifier predictions, with a generated-vs-real domain-shift
  validity check;
* ECAPA identity retention against the regex baseline and the real style donor;
* duration and F0.

The classifier is a proxy trained on real GLOBE speech.  If the generated
features are far outside the real-speech spread, classifier accuracy is
reported as unavailable rather than as a real accent result.

No human MOS or accent-identification study is claimed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_INDEX = ROOT / "results" / "accent" / "a2" / "generation_index.csv"
DEFAULT_REPORT = ROOT / "results" / "accent" / "a2" / "report.json"
DEFAULT_FEATURES = ROOT / "data" / "processed" / "globe_accent_features.npz"
DEFAULT_MODEL = ROOT / "checkpoints" / "accent_classifier_globe.joblib"
TRAINABLE = ("us", "england", "australian", "canadian", "india_south_asia")
# The frozen GLOBE classifier predates the four-accent rename.  Map the
# current serving label to the classifier's training label only for scoring;
# generated rows keep the canonical `indian_english` name.
EVAL_ACCENT_ALIASES = {"indian_english": "india_south_asia"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate A2 accent G2P clips")
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--generation-report", type=Path, default=None)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--compute-wer", action="store_true")
    parser.add_argument("--wer-model", default="base")
    parser.add_argument("--domain-shift-threshold", type=float, default=3.0)
    parser.add_argument("--n-resamples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args(argv)


def _load_feature_rows(path: Path, *, feature_key: str = "combined") -> pd.DataFrame:
    cache = np.load(path, allow_pickle=True)
    key = {
        "combined": "features",
        "ssl": "ssl_features",
        "base": "base_features",
    }.get(str(feature_key), "features")
    if key not in cache.files:
        if key == "base_features" and "features" in cache.files:
            key = "features"
        else:
            raise ValueError(f"feature cache has no {key!r} block")
    return pd.DataFrame(
        {
            "clip_id": cache["clip_id"].astype(str),
            "label": cache["label"].astype(str),
            "speaker_id": cache["speaker_id"].astype(str),
            "split": cache["split"].astype(str),
            "features": [np.asarray(row, dtype=np.float64) for row in cache[key]],
        }
    )


def _extract_features(
    paths: Sequence[str],
    *,
    device: str,
    ssl_config: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    from voix.data.speaker_extractor import ECAPAExtractor
    from voix.evaluation.accent_features import (
        combine_features,
        extract_prosody_spectral_features,
    )

    ecapa = ECAPAExtractor(device=device)
    rows: List[np.ndarray] = []
    for path in paths:
        embedding = ecapa.extract_from_path(str(path)).detach().cpu().numpy().reshape(-1)
        norm = float(np.linalg.norm(embedding))
        if norm > 0:
            embedding = embedding / norm
        prosody = extract_prosody_spectral_features(str(path))
        rows.append(combine_features(embedding, prosody))
    base = np.stack(rows)
    if not ssl_config:
        return base
    from voix.evaluation.accent_ssl import (
        SSLAccentConfig,
        SSLAccentFeatureExtractor,
    )

    allowed = {
        key: ssl_config[key]
        for key in (
            "bundle",
            "num_layers",
            "sample_rate",
            "max_seconds",
            "l2_normalize",
        )
        if key in ssl_config
    }
    extractor = SSLAccentFeatureExtractor(
        SSLAccentConfig(**allowed), device=device, batch_size=8
    )
    ssl = extractor.extract_paths(paths)
    return np.concatenate([base, ssl], axis=1)


def _domain_shift(
    train: np.ndarray,
    real_val: np.ndarray,
    generated: np.ndarray,
    *,
    threshold: float,
) -> Dict[str, Any]:
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(train)
    real_scaled = scaler.transform(real_val)
    generated_scaled = scaler.transform(generated)
    real_centroid = real_scaled.mean(axis=0)
    generated_centroid = generated_scaled.mean(axis=0)
    real_spread = float(np.mean(np.linalg.norm(real_scaled - real_centroid, axis=1)))
    distance = float(np.linalg.norm(generated_centroid - real_centroid))
    ratio = distance / max(real_spread, 1e-9)
    valid = bool(ratio <= float(threshold))
    return {
        "status": "computed",
        "standardized_centroid_distance": distance,
        "real_val_spread": real_spread,
        "ratio": ratio,
        "threshold": float(threshold),
        "valid": valid,
        "reason": (
            "generated features are within the real-speech spread"
            if valid
            else (
                "generated features are far outside the real-speech spread; "
                "classifier predictions are not a valid accent measure"
            )
        ),
    }


def _paired_wer(
    frame: pd.DataFrame,
    *,
    model_size: str,
    device: str,
    n_resamples: int,
    seed: int,
) -> Dict[str, Any]:
    from voix.evaluation.audio_metrics import transcribe_wer
    from voix.evaluation.stats import paired_bootstrap_difference

    results: Dict[str, Any] = {}
    for accent, subset in frame.groupby("accent_class", sort=True):
        entry: Dict[str, Any] = {}
        per_prompt: Dict[str, Dict[int, float]] = {}
        for condition, group in subset.groupby("condition", sort=True):
            group = group.sort_values(["prompt_id", "seed"])
            missing = [
                str(path)
                for path in group["wav_path"].astype(str)
                if not Path(path).is_file()
            ]
            if missing:
                entry[condition] = {
                    "status": "unavailable",
                    "reason": f"missing audio: {missing[:3]}",
                    "wer": None,
                }
                continue
            outcome = transcribe_wer(
                group["wav_path"].astype(str).tolist(),
                group["text"].astype(str).tolist(),
                model_size=model_size,
                device=device,
            )
            entry[condition] = outcome
            if outcome.get("status") == "computed":
                per_prompt[condition] = {
                    int(prompt): float(value)
                    for prompt, value in zip(
                        group["prompt_id"].astype(int), outcome["per_sample_wer"]
                    )
                }
        if "regex" in per_prompt and "g2p" in per_prompt:
            shared = sorted(set(per_prompt["regex"]) & set(per_prompt["g2p"]))
            if len(shared) >= 2:
                regex_values = [per_prompt["regex"][prompt] for prompt in shared]
                g2p_values = [per_prompt["g2p"][prompt] for prompt in shared]
                interval = paired_bootstrap_difference(
                    g2p_values,
                    regex_values,
                    n_resamples=n_resamples,
                    seed=seed,
                )
                entry["g2p_minus_regex"] = {
                    "mean": interval.estimate,
                    "ci_low": interval.ci_low,
                    "ci_high": interval.ci_high,
                    "n_pairs": interval.n,
                    "unit": "prompt",
                }
        results[str(accent)] = entry
    return {
        "status": "computed" if results else "unavailable",
        "by_accent": results,
        "reason": None if results else "no accent group had paired WER",
    }


def _classifier_block(
    frame: pd.DataFrame,
    features: np.ndarray,
    *,
    model_path: Path,
    feature_frame: pd.DataFrame,
    domain_shift_threshold: float,
) -> Dict[str, Any]:
    import joblib

    payload = joblib.load(model_path)
    model = payload["model"]
    predictions = model.predict(features).astype(str)
    probabilities = model.predict_proba(features)
    classes = [str(item) for item in payload.get("classes", model.classes_)]
    frame = frame.copy()
    frame["predicted_accent"] = predictions
    frame["max_probability"] = probabilities.max(axis=1)

    train_mask = feature_frame["split"].astype(str) == "train"
    val_mask = feature_frame["split"].astype(str) == "val"
    train_features = np.stack(feature_frame.loc[train_mask, "features"].to_list())
    val_features = np.stack(feature_frame.loc[val_mask, "features"].to_list())
    shift = _domain_shift(
        train_features,
        val_features,
        features,
        threshold=domain_shift_threshold,
    )
    valid = bool(shift.get("valid", False))

    by_condition: Dict[str, Any] = {}
    for condition, group in frame.groupby("condition", sort=True):
        group = group.copy()
        group["eval_accent_class"] = group["accent_class"].map(
            lambda value: EVAL_ACCENT_ALIASES.get(str(value), str(value))
        )
        trainable = group[group["eval_accent_class"].isin(TRAINABLE)].copy()
        entry: Dict[str, Any] = {
            "n": int(len(group)),
            "prediction_counts": {
                str(label): int((group["predicted_accent"] == label).sum())
                for label in classes
            },
            "mean_max_probability": float(group["max_probability"].mean()),
        }
        if valid and len(trainable):
            entry["trainable_accuracy"] = float(
                (trainable["predicted_accent"] == trainable["eval_accent_class"]).mean()
            )
            entry["per_class_recall"] = {
                accent: float(
                    (
                        trainable.loc[
                            trainable["eval_accent_class"] == accent,
                            "predicted_accent",
                        ]
                        == accent
                    ).mean()
                )
                for accent in TRAINABLE
                if (trainable["eval_accent_class"] == accent).any()
            }
        else:
            entry["trainable_accuracy"] = None
        by_condition[str(condition)] = entry

    irish = frame[frame["accent_class"] == "irish"]
    irish_block: Dict[str, Any] = {"status": "not_evaluated"}
    if len(irish):
        irish_block = {
            "status": "evaluated",
            "n": int(len(irish)),
            "prediction_counts": {
                str(label): int((irish["predicted_accent"] == label).sum())
                for label in classes
            },
            "note": "Irish is evaluation-only; the model has no Irish class.",
        }

    return {
        "model": str(model_path),
        "model_sha256": _sha256(model_path),
        "classes": classes,
        "proxy_valid": valid,
        "domain_shift": shift,
        "by_condition": by_condition,
        "irish": irish_block,
        "predictions": frame[
            ["clip_id", "accent_class", "condition", "predicted_accent", "max_probability"]
        ].to_dict(orient="records"),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    from voix.evaluation.research_report import (
        build_report,
        computed_metric,
        unavailable_metric,
        write_report,
    )

    args = parse_args(argv)
    frame = pd.read_csv(args.index)
    required = {"clip_id", "accent_class", "condition", "prompt_id", "text", "wav_path"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"generation index is missing columns: {missing}")
    missing_audio = [
        str(path) for path in frame["wav_path"].astype(str) if not Path(path).is_file()
    ]
    if missing_audio:
        raise FileNotFoundError(f"missing generated audio: {missing_audio[:5]}")

    print(f"Measuring {len(frame)} clip(s) ...", flush=True)
    print("Extracting ECAPA + prosody features ...", flush=True)
    import joblib

    model_payload = joblib.load(args.model)
    feature_key = str(model_payload.get("feature_key", "base"))
    ssl_config = model_payload.get("ssl_config")
    if ssl_config:
        print("Extracting frozen Wav2Vec2 features ...", flush=True)
    features = _extract_features(
        frame["wav_path"].astype(str).tolist(),
        device=args.device,
        ssl_config=ssl_config,
    )

    feature_frame = _load_feature_rows(args.features, feature_key=feature_key)
    classifier = _classifier_block(
        frame,
        features,
        model_path=args.model,
        feature_frame=feature_frame,
        domain_shift_threshold=args.domain_shift_threshold,
    )

    # Identity: the first 192 feature dimensions are L2-normalized ECAPA.
    ecapa = features[:, :192]
    identity_frame = frame.copy()
    identity_frame["ecapa_index"] = np.arange(len(identity_frame))
    paired_rows: List[Dict[str, Any]] = []
    for (accent, prompt, seed), group in identity_frame.groupby(
        ["accent_class", "prompt_id", "seed"], sort=True
    ):
        if set(group["condition"]) != {"regex", "g2p"}:
            continue
        regex_index = int(group.loc[group["condition"] == "regex", "ecapa_index"].iloc[0])
        g2p_index = int(group.loc[group["condition"] == "g2p", "ecapa_index"].iloc[0])
        paired_rows.append(
            {
                "accent_class": str(accent),
                "prompt_id": int(prompt),
                "seed": int(seed),
                "ecapa_regex_g2p": float(np.dot(ecapa[regex_index], ecapa[g2p_index])),
            }
        )
    paired = pd.DataFrame(paired_rows)
    identity_by_accent: Dict[str, Any] = {}
    if len(paired):
        for accent, group in paired.groupby("accent_class", sort=True):
            values = group["ecapa_regex_g2p"].astype(float).tolist()
            identity_by_accent[str(accent)] = {
                "n_pairs": len(values),
                "mean_ecapa_regex_g2p": float(np.mean(values)),
                "min_ecapa_regex_g2p": float(np.min(values)),
                "max_ecapa_regex_g2p": float(np.max(values)),
            }

    # Real donor identity: use the donor clip from the generation report.
    donor_block: Dict[str, Any] = {"status": "unavailable", "reason": "no generation report"}
    if args.generation_report and args.generation_report.is_file():
        generation = json.loads(args.generation_report.read_text(encoding="utf-8"))
        identity = generation.get("identity", {})
        donor_path = identity.get("path")
        if donor_path and Path(donor_path).is_file():
            donor_features = _extract_features([str(donor_path)], device=args.device)[0]
            donor_ecapa = donor_features[:192]
            donor_similarity = ecapa @ donor_ecapa
            donor_block = {
                "status": "computed",
                "clip_id": identity.get("clip_id"),
                "mean_ecapa_to_donor": float(np.mean(donor_similarity)),
                "by_condition": {
                    str(condition): float(np.mean(donor_similarity[frame["condition"].to_numpy() == condition]))
                    for condition in sorted(frame["condition"].unique())
                },
                "note": "Absolute similarity is low because generated and real speech are different domains.",
            }
    else:
        generation = {}

    # Prosody: duration and F0 are already in the feature cache.
    from voix.evaluation.accent_features import PROSODY_FEATURE_NAMES

    names = list(PROSODY_FEATURE_NAMES)
    duration_index = names.index("duration_seconds")
    f0_mean_index = names.index("f0_mean_hz")
    f0_std_index = names.index("f0_std_hz")
    prosody = features[:, 192:]
    prosody_block: Dict[str, Any] = {}
    for (accent, condition), group in frame.groupby(["accent_class", "condition"], sort=True):
        indices = group.index.to_numpy()
        prosody_block[f"{accent}:{condition}"] = {
            "duration_mean": float(prosody[indices, duration_index].mean()),
            "f0_mean_hz": float(prosody[indices, f0_mean_index].mean()),
            "f0_std_hz": float(prosody[indices, f0_std_index].mean()),
        }

    # WER is opt-in because Whisper is expensive.
    if args.compute_wer:
        print("Transcribing with Whisper ...", flush=True)
        wer = _paired_wer(
            frame,
            model_size=args.wer_model,
            device=args.device,
            n_resamples=args.n_resamples,
            seed=args.seed,
        )
    else:
        wer = {
            "status": "unavailable",
            "reason": "WER was not requested; pass --compute-wer",
            "by_accent": {},
        }

    # A2 gate: G2P must not cost more than +0.10 WER versus regex, and must
    # improve classifier accuracy when the classifier proxy is valid.
    gate_checks: Dict[str, Any] = {}
    if isinstance(wer, dict) and wer.get("status") != "unavailable":
        for accent, entry in wer.get("by_accent", {}).items():
            delta = entry.get("g2p_minus_regex")
            if not delta:
                gate_checks[str(accent)] = {
                    "status": "unavailable",
                    "reason": "paired WER was not computed",
                }
                continue
            gate_checks[str(accent)] = {
                "status": "computed",
                "wer_delta": delta["mean"],
                "wer_ci_low": delta["ci_low"],
                "wer_ci_high": delta["ci_high"],
                "wer_within_budget": bool(delta["mean"] <= 0.10),
            }
    classifier_valid = bool(classifier.get("proxy_valid"))
    accuracy_gain: Dict[str, Any] = {}
    if classifier_valid:
        regex_acc = classifier["by_condition"].get("regex", {}).get("trainable_accuracy")
        g2p_acc = classifier["by_condition"].get("g2p", {}).get("trainable_accuracy")
        if regex_acc is not None and g2p_acc is not None:
            accuracy_gain = {
                "status": "computed",
                "regex_accuracy": float(regex_acc),
                "g2p_accuracy": float(g2p_acc),
                "gain": float(g2p_acc - regex_acc),
                "passes_5_points": bool((g2p_acc - regex_acc) >= 0.05),
            }
    else:
        accuracy_gain = {
            "status": "invalid",
            "reason": "classifier proxy invalid because of generated-domain shift",
        }
    wer_ok = all(
        entry.get("wer_within_budget", False)
        for entry in gate_checks.values()
        if entry.get("status") == "computed"
    ) if gate_checks else False
    gate = {
        "status": "computed" if gate_checks else "unavailable",
        "wer_within_budget": bool(wer_ok),
        "classifier_valid": classifier_valid,
        "accuracy_gain": accuracy_gain,
        "passed": bool(
            wer_ok
            and classifier_valid
            and accuracy_gain.get("passes_5_points", False)
        ),
        "checks": gate_checks,
        "reason": (
            None
            if wer_ok and classifier_valid and accuracy_gain.get("passes_5_points")
            else (
                "accent improvement is not demonstrated: "
                + "; ".join(
                    part
                    for part in (
                        None if wer_ok else "WER budget failed",
                        (
                            "classifier proxy invalid"
                            if not classifier_valid
                            else (
                                None
                                if accuracy_gain.get("passes_5_points")
                                else "classifier accuracy gain below 5 points"
                            )
                        ),
                    )
                    if part
                )
            )
        ),
    }

    metrics: Dict[str, Any] = {
        "wer": (
            computed_metric(
                {
                    accent: {
                        condition: entry.get("wer")
                        for condition, entry in conditions.items()
                        if isinstance(entry, dict) and entry.get("status") == "computed"
                    }
                    for accent, conditions in wer.get("by_accent", {}).items()
                }
            )
            if wer.get("status") != "unavailable"
            else unavailable_metric(wer.get("reason", "WER unavailable"))
        ),
        "accent_classifier_accuracy": (
            computed_metric(
                {
                    condition: entry.get("trainable_accuracy")
                    for condition, entry in classifier["by_condition"].items()
                }
            )
            if classifier_valid
            else unavailable_metric(
                "classifier proxy invalid: generated features are far outside the real-speech spread"
            )
        ),
        "accent_classifier_domain_shift_ratio": computed_metric(
            classifier["domain_shift"].get("ratio"),
            threshold=args.domain_shift_threshold,
            valid=classifier_valid,
            standardized_centroid_distance=classifier["domain_shift"].get(
                "standardized_centroid_distance"
            ),
            real_val_spread=classifier["domain_shift"].get("real_val_spread"),
        ),
        "identity_regex_vs_g2p": computed_metric(identity_by_accent),
        "identity_to_real_donor": (
            computed_metric(donor_block)
            if donor_block.get("status") == "computed"
            else unavailable_metric(donor_block.get("reason", "donor unavailable"))
        ),
        "prosody": computed_metric(prosody_block),
        "phoneme_changes": computed_metric(
            generation.get("changed_prompts_by_accent", {}),
            total_prompts=generation.get("prompt_count"),
        ),
    }

    report = build_report(
        track="accent",
        level="exploratory",
        run_id="a2_g2p_comparison",
        status="completed" if not missing_audio else "partial",
        metrics=metrics,
        gates={"a2_g2p": gate},
        limitations=[
            "Six prompts are a diagnostic, not a confirmatory accent study.",
            "The accent classifier is trained on real read speech and is only a proxy for generated audio.",
            "Absolute ECAPA similarity to the real donor is low because generated and real audio are different domains.",
            "Irish is evaluation-only and has no classifier class.",
            "No human accent-identification or MOS study was run.",
        ],
        mos={
            "status": "unavailable",
            "reason": "no human ratings were collected for this run",
        },
        seeds=[int(seed) for seed in sorted(frame["seed"].unique())],
        manifest_path=args.index,
        per_class={
            "classifier": classifier["by_condition"],
            "irish_classifier": classifier["irish"],
        },
        artifacts={
            "generation_report": str(args.generation_report) if args.generation_report else None,
            "index_sha256": _sha256(args.index),
        },
        extra={
            "experiment": "A2",
            "generation": {
                "identity": generation.get("identity"),
                "conditions": generation.get("conditions"),
                "prompts": generation.get("prompts"),
                "rows": generation.get("rows"),
            },
            "classifier_predictions": classifier["predictions"],
        },
    )
    write_report(args.report, report)
    print(
        f"A2 gate passed={gate['passed']} "
        f"(WER budget={wer_ok}, classifier valid={classifier_valid})"
    )
    print(f"Wrote {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
