"""Evaluate the E0 fixed-delta CREMA-D paired generation.

The evaluator is deliberately independent of the face path.  It measures:

* emotion-classifier accuracy / macro-F1 / per-class recall, with the real
  validation clips as a speaker-disjoint calibration ceiling;
* F0, duration, and energy, paired against the generated neutral;
* ECAPA similarity to the generated neutral, the actor's identity reference,
  and the real neutral clip;
* the E0 direction gate (happy/angry raise F0 and shorten duration, sad lowers
  F0 and lengthens duration);
* the shared identity guardrail (ECAPA neutral retention, canonical hashes,
  neutral-zero).

WER is opt-in because Whisper and ffmpeg are expensive and may be absent.
No MOS is claimed.  Every unmeasured metric is written as ``unavailable`` with
a reason.

Examples::

    python scripts/evaluation/evaluate_crema_e0.py \
        --index results/emotion/e0_smoke/generation_index.csv \
        --output results/emotion/e0_smoke/report.json --device cuda

    python scripts/evaluation/evaluate_crema_e0.py \
        --index results/emotion/e0/generation_index.csv \
        --output results/emotion/e0/report.json --device cuda --compute-wer
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from voix.evaluation.crema_e0 import (  # noqa: E402
    EMOTION_ORDER,
    aggregate_conditions,
    classification_metrics,
    condition_gate_summary,
    feature_domain_shift,
    identity_rows,
    paired_delta_effects,
)
from voix.evaluation.crema_paired import validate_paired_grid  # noqa: E402
from voix.evaluation.identity_guardrail import (  # noqa: E402
    GuardrailThresholds,
    evaluate_identity_guardrail,
)
from voix.evaluation.research_report import (  # noqa: E402
    build_report,
    computed_metric,
    unavailable_metric,
    write_report,
)

DEFAULT_INDEX = ROOT / "results" / "emotion" / "e0" / "generation_index.csv"
DEFAULT_MANIFEST = ROOT / "data" / "processed" / "crema_d_paired_eval_val.csv"
DEFAULT_OUTPUT = ROOT / "results" / "emotion" / "e0" / "report.json"
DEFAULT_CLASSIFIER = ROOT / "checkpoints" / "emotion_classifier_crema_d.joblib"
DEFAULT_CLASSIFIER_REPORT = ROOT / "results" / "emotion_classifier_crema_d.json"
DEFAULT_FEATURES = ROOT / "data" / "processed" / "crema_d_features.npz"
DEFAULT_GENERATION_REPORT = ROOT / "results" / "emotion" / "e0" / "generation_report.json"
DEFAULT_CANONICAL_HASHES = ROOT / "results" / "canonical_hashes.json"

PROSODY_METRICS = ("f0_mean_hz", "f0_std_hz", "duration_seconds", "energy_mean", "energy_std")
ECAPA_METRICS = (
    "ecapa_generated_neutral",
    "ecapa_identity_reference",
    "ecapa_real_neutral",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _config_sha256(payload: Dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


def _resolve(path: str) -> Path:
    candidate = Path(str(path))
    return candidate if candidate.is_absolute() else ROOT / candidate


def _load_training_module():
    path = ROOT / "scripts" / "training" / "train_emotion_classifier.py"
    spec = importlib.util.spec_from_file_location("train_emotion_classifier", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fit_calibration_model(features: np.ndarray, labels: np.ndarray, params: Dict[str, Any]):
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                SVC(
                    kernel="rbf",
                    C=float(params.get("C", 3.0)),
                    gamma=params.get("gamma", "scale"),
                    class_weight="balanced",
                ),
            ),
        ]
    ).fit(features, labels)


def _measure_one(row: pd.Series, extract_features) -> Dict[str, Any]:
    from voix.evaluation.metrics import prosody_statistics

    path = _resolve(str(row["audio_path"]))
    result: Dict[str, Any] = {
        "audio_path": str(row["audio_path"]),
        "audio_exists": path.exists(),
    }
    if not path.exists():
        result["prosody"] = {"status": "unavailable", "reason": f"missing audio: {path}"}
        result["features"] = {"status": "unavailable", "reason": f"missing audio: {path}"}
        return result
    try:
        result["prosody"] = {"status": "computed", **prosody_statistics(str(path))}
    except Exception as exc:  # pragma: no cover - audio backend guard
        result["prosody"] = {
            "status": "unavailable",
            "reason": f"{type(exc).__name__}: {exc}",
        }
    try:
        result["features"] = {
            "status": "computed",
            "value": extract_features(str(path)),
        }
    except Exception as exc:  # pragma: no cover - audio backend guard
        result["features"] = {
            "status": "unavailable",
            "reason": f"{type(exc).__name__}: {exc}",
        }
    return result


def _cosine(left, right) -> Optional[float]:
    import torch.nn.functional as F

    if left is None or right is None:
        return None
    value = float(F.cosine_similarity(left, right).item())
    return value if math.isfinite(value) else None


def _wer_by_condition(
    frame: pd.DataFrame,
    *,
    model_size: str,
    device: str,
) -> Dict[str, Any]:
    from voix.evaluation.audio_metrics import transcribe_wer

    conditions: Dict[str, Any] = {}
    for (condition, emotion), subset in frame.groupby(["condition", "emotion"], sort=False):
        paths = [_resolve(path) for path in subset["audio_path"].astype(str)]
        references = subset["text"].astype(str).tolist()
        missing = [str(path) for path in paths if not path.exists()]
        if missing:
            conditions[f"{condition}:{emotion}"] = {
                "status": "unavailable",
                "reason": f"missing audio: {missing[:5]}",
                "wer": None,
            }
            continue
        conditions[f"{condition}:{emotion}"] = transcribe_wer(
            paths,
            references,
            model_size=model_size,
            device=device,
        )
    computed = {
        key: float(value["wer"])
        for key, value in conditions.items()
        if value.get("status") == "computed" and value.get("wer") is not None
    }
    if not computed:
        status = "unavailable"
    elif len(computed) == len(conditions):
        status = "computed"
    else:
        status = "partial"
    return {
        "status": status,
        "conditions": conditions,
        "mean": float(np.mean(list(computed.values()))) if computed else None,
        "worst": max(computed.values()) if computed else None,
        "reason": None if computed else "Whisper, ffmpeg, or audio unavailable",
    }


def _condition_recall(frame: pd.DataFrame, prediction_column: str) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for (condition, emotion), subset in frame.groupby(["condition", "emotion"], sort=False):
        prediction = subset[prediction_column].astype(str)
        target = str(emotion)
        result[f"{condition}:{emotion}"] = {
            "status": "computed",
            "n": int(len(subset)),
            "target_recall": float((prediction == target).mean()),
            "predicted_counts": prediction.value_counts().to_dict(),
        }
    return result


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate E0 fixed-delta CREMA-D audio")
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--classifier", type=Path, default=DEFAULT_CLASSIFIER)
    parser.add_argument("--classifier-report", type=Path, default=DEFAULT_CLASSIFIER_REPORT)
    parser.add_argument("--features-cache", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--generation-report", type=Path, default=DEFAULT_GENERATION_REPORT)
    parser.add_argument("--canonical-hashes", type=Path, default=DEFAULT_CANONICAL_HASHES)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--n-resamples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--max-clips", type=int, default=None)
    parser.add_argument("--compute-wer", action="store_true")
    parser.add_argument("--wer-model", default="base")
    parser.add_argument(
        "--min-ecapa-neutral",
        type=float,
        default=0.80,
        help="Pre-registered identity budget for generated-neutral retention",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    import joblib
    import torch
    import torch.nn.functional as F

    from voix.data.speaker_extractor import ECAPAExtractor

    if not args.index.is_file():
        raise FileNotFoundError(f"generation index not found: {args.index}")
    if not args.manifest.is_file():
        raise FileNotFoundError(f"paired evaluation manifest not found: {args.manifest}")
    if not args.classifier.is_file():
        raise FileNotFoundError(f"emotion classifier not found: {args.classifier}")
    if not args.features_cache.is_file():
        raise FileNotFoundError(f"CREMA-D feature cache not found: {args.features_cache}")

    index = pd.read_csv(args.index)
    manifest = pd.read_csv(args.manifest)
    grid_errors = validate_paired_grid(manifest, check_audio=True)
    if grid_errors:
        raise ValueError("invalid paired evaluation manifest: " + "; ".join(grid_errors))

    required_index = {
        "audio_id",
        "condition",
        "emotion",
        "actor_id",
        "speaker_id",
        "sentence",
        "text",
        "seed",
        "audio_path",
        "neutral_audio_path",
        "identity_path",
        "neutral_path",
        "emotion_path",
    }
    missing = required_index - set(index.columns)
    if missing:
        raise ValueError(f"generation index is missing columns: {sorted(missing)}")
    if args.max_clips is not None:
        index = index.head(int(args.max_clips)).copy()
    if index.empty:
        raise ValueError("generation index is empty")

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("--device cuda was requested but CUDA is unavailable")

    # ---- Real-speech calibration -----------------------------------------
    training_module = _load_training_module()
    extract_features = training_module.extract_features
    cache = np.load(args.features_cache, allow_pickle=True)
    cache_features = cache["features"]
    cache_split = cache["split"].astype(str)
    cache_actor = cache["actor_id"].astype(str)
    cache_label = cache["intended_emotion"].astype(str)
    cache_path = cache["path"].astype(str)

    actors = set(index["actor_id"].astype(str))
    train_mask = cache_split == "train"
    val_mask = (cache_split == "val") & np.isin(cache_actor, sorted(actors))
    if not train_mask.any() or not val_mask.any():
        raise ValueError("feature cache must contain train rows and validation rows for the evaluated actors")

    classifier_report = (
        json.loads(args.classifier_report.read_text(encoding="utf-8"))
        if args.classifier_report.is_file()
        else {}
    )
    best_params = classifier_report.get("best_params", {"C": 3.0, "gamma": "scale"})
    calibration_model = _fit_calibration_model(
        cache_features[train_mask],
        cache_label[train_mask],
        best_params,
    )
    saved_model = joblib.load(args.classifier)

    y_val = cache_label[val_mask]
    groups_val = cache_actor[val_mask]
    real_val_calibration = classification_metrics(
        y_val.tolist(),
        calibration_model.predict(cache_features[val_mask]).tolist(),
        groups_val.tolist(),
        n_resamples=args.n_resamples,
        seed=args.seed,
    )
    real_val_saved = classification_metrics(
        y_val.tolist(),
        saved_model.predict(cache_features[val_mask]).tolist(),
        groups_val.tolist(),
        n_resamples=args.n_resamples,
        seed=args.seed,
    )

    # ---- Per-clip measurements -------------------------------------------
    print(f"Measuring {len(index)} generated clip(s) with {args.workers} worker(s) ...", flush=True)
    rows = [row for _, row in index.iterrows()]
    with ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as pool:
        measured = list(pool.map(lambda row: _measure_one(row, extract_features), rows))

    print("Extracting ECAPA embeddings ...", flush=True)
    ecapa = ECAPAExtractor(device=device)
    embedding_cache: Dict[str, Any] = {}

    def embedding_for(path: str):
        key = str(path)
        if key in embedding_cache:
            return embedding_cache[key]
        resolved = _resolve(key)
        if not resolved.exists():
            embedding_cache[key] = None
            return None
        value = ecapa.extract_from_path(str(resolved))
        value = F.normalize(value, dim=-1)
        embedding_cache[key] = value
        return value

    records: List[Dict[str, Any]] = []
    generated_feature_vectors: List[np.ndarray] = []
    generated_feature_conditions: List[str] = []
    generated_feature_emotions: List[str] = []
    for source, result in zip(rows, measured):
        record = source.to_dict()
        record["audio_exists"] = bool(result["audio_exists"])
        prosody = result["prosody"]
        record["prosody_status"] = prosody.get("status")
        if prosody.get("status") == "computed":
            for metric in PROSODY_METRICS:
                record[metric] = prosody.get(metric)
        features = result["features"]
        record["feature_status"] = features.get("status")
        if features.get("status") == "computed":
            vector = np.asarray(features["value"], dtype=np.float32).reshape(1, -1)
            record["prediction_calibration"] = str(calibration_model.predict(vector)[0])
            record["prediction_saved"] = str(saved_model.predict(vector)[0])
            generated_feature_vectors.append(vector.reshape(-1))
            generated_feature_conditions.append(str(source["condition"]))
            generated_feature_emotions.append(str(source["emotion"]))
        else:
            record["prediction_calibration"] = None
            record["prediction_saved"] = None

        own_embedding = embedding_for(str(source["audio_path"])) if result["audio_exists"] else None
        neutral_embedding = embedding_for(str(source["neutral_audio_path"]))
        identity_embedding = embedding_for(str(source["identity_path"]))
        real_neutral_embedding = embedding_for(str(source["neutral_path"]))
        real_emotion_path = str(source.get("emotion_path", "") or "")
        real_emotion_embedding = (
            embedding_for(real_emotion_path)
            if real_emotion_path and bool(source.get("has_real_emotion_clip", False))
            else None
        )

        if str(source["condition"]) == "neutral":
            record["ecapa_generated_neutral"] = 1.0 if own_embedding is not None else None
        else:
            record["ecapa_generated_neutral"] = _cosine(own_embedding, neutral_embedding)
        record["ecapa_identity_reference"] = _cosine(own_embedding, identity_embedding)
        record["ecapa_real_neutral"] = _cosine(own_embedding, real_neutral_embedding)
        record["ecapa_real_emotion"] = _cosine(own_embedding, real_emotion_embedding)
        records.append(record)

    measured_frame = pd.DataFrame(records)

    # ---- Metrics ----------------------------------------------------------
    condition_metrics = aggregate_conditions(
        measured_frame,
        metrics=PROSODY_METRICS + ECAPA_METRICS,
        n_resamples=args.n_resamples,
        seed=args.seed,
    )
    paired_effects = paired_delta_effects(
        measured_frame,
        metrics=PROSODY_METRICS + ECAPA_METRICS,
        n_resamples=args.n_resamples,
        seed=args.seed,
    )
    direction_gate = condition_gate_summary(paired_effects)

    generated_rows = measured_frame[measured_frame["prediction_calibration"].notna()].copy()

    # ---- Classifier domain-shift diagnostic ------------------------------
    generated_matrix = (
        np.stack(generated_feature_vectors).astype(np.float64)
        if generated_feature_vectors
        else np.zeros((0, cache_features.shape[1]), dtype=np.float64)
    )
    generated_condition_array = np.asarray(generated_feature_conditions, dtype=str)
    real_val_features = cache_features[val_mask].astype(np.float64)
    real_val_labels = cache_label[val_mask].astype(str)
    real_neutral_features = real_val_features[real_val_labels == "neutral"]
    if real_neutral_features.size == 0:
        real_neutral_features = real_val_features
    generated_neutral_features = (
        generated_matrix[generated_condition_array == "neutral"]
        if generated_matrix.size
        else generated_matrix
    )
    domain_shift = feature_domain_shift(
        cache_features[train_mask].astype(np.float64),
        real_neutral_features,
        generated_neutral_features,
    )
    domain_shift_all = feature_domain_shift(
        cache_features[train_mask].astype(np.float64),
        real_val_features,
        generated_matrix,
    )
    classifier_proxy_valid = bool(domain_shift.get("valid", False))

    overall_generated = classification_metrics(
        generated_rows["emotion"].astype(str).tolist(),
        generated_rows["prediction_calibration"].astype(str).tolist(),
        generated_rows["actor_id"].astype(str).tolist(),
        n_resamples=args.n_resamples,
        seed=args.seed,
    )
    overall_saved = classification_metrics(
        generated_rows["emotion"].astype(str).tolist(),
        generated_rows["prediction_saved"].astype(str).tolist(),
        generated_rows["actor_id"].astype(str).tolist(),
        n_resamples=args.n_resamples,
        seed=args.seed,
    )
    target_recall = _condition_recall(generated_rows, "prediction_calibration")

    # ---- Identity guardrail ----------------------------------------------
    guardrail_rows = identity_rows(measured_frame)
    identity = evaluate_identity_guardrail(
        guardrail_rows,
        thresholds=GuardrailThresholds(min_ecapa_neutral=args.min_ecapa_neutral),
        require_cfvn=False,
        canonical_hashes_path=args.canonical_hashes,
        n_resamples=args.n_resamples,
        seed=args.seed,
    )

    # ---- WER --------------------------------------------------------------
    if args.compute_wer:
        # Free ECAPA before loading Whisper; the 4 GB card cannot reliably hold
        # both models, and the embedding loop has already finished.
        try:
            import torch as _torch

            del ecapa
            if _torch.cuda.is_available():
                _torch.cuda.empty_cache()
        except Exception:
            pass
        wer = _wer_by_condition(measured_frame, model_size=args.wer_model, device=device)
    else:
        wer = {
            "status": "unavailable",
            "reason": "WER was not requested; pass --compute-wer",
            "conditions": {},
            "mean": None,
            "worst": None,
        }

    # ---- Generation report ------------------------------------------------
    generation = {}
    if args.generation_report.is_file():
        generation = json.loads(args.generation_report.read_text(encoding="utf-8"))

    # ---- Report -----------------------------------------------------------
    def metric_values(metric: str) -> Dict[str, Any]:
        values = {}
        for key, entry in condition_metrics.items():
            value = entry["metrics"].get(metric, {})
            values[key] = value.get("mean") if value.get("status") == "computed" else None
        return values

    neutral_zero = generation.get("neutral_zero", {
        "status": "unavailable",
        "reason": "generation report not supplied",
    })
    e0_direction_pass = bool(direction_gate.get("passed"))
    identity_pass = bool(identity.get("passed"))
    neutral_zero_pass = bool(neutral_zero.get("passed", False))
    accuracy_above_chance = bool(
        classifier_proxy_valid
        and overall_generated.get("status") == "computed"
        and overall_generated["accuracy"] > 1.0 / len(EMOTION_ORDER)
    )
    classifier_proxy_reason = (
        "generated features are within the real-speech distribution"
        if classifier_proxy_valid
        else domain_shift.get(
            "reason",
            "generated features are outside the real-speech distribution",
        )
    )

    metrics: Dict[str, Any] = {
        "classifier_accuracy": computed_metric(
            overall_generated.get("accuracy"),
            n=overall_generated.get("n"),
            n_speakers=overall_generated.get("n_speakers"),
            ci=overall_generated.get("speaker_cluster_ci"),
        )
        if overall_generated.get("status") == "computed" and classifier_proxy_valid
        else unavailable_metric(
            f"classifier proxy invalid: {classifier_proxy_reason}"
        ),
        "classifier_macro_f1": computed_metric(
            overall_generated.get("macro_f1"),
            n=overall_generated.get("n"),
        )
        if overall_generated.get("status") == "computed" and classifier_proxy_valid
        else unavailable_metric(
            f"classifier proxy invalid: {classifier_proxy_reason}"
        ),
        "classifier_target_recall": computed_metric(target_recall)
        if classifier_proxy_valid
        else unavailable_metric(
            f"classifier proxy invalid: {classifier_proxy_reason}"
        ),
        "classifier_domain_shift_ratio": computed_metric(
            domain_shift.get("ratio"),
            threshold=domain_shift.get("threshold"),
            standardized_centroid_distance=domain_shift.get(
                "standardized_centroid_distance"
            ),
            real_val_spread=domain_shift.get("real_val_spread"),
        )
        if domain_shift.get("status") == "computed"
        else unavailable_metric(
            domain_shift.get("reason", "domain-shift diagnostic unavailable")
        ),
        "f0_mean_hz": computed_metric(metric_values("f0_mean_hz")),
        "f0_std_hz": computed_metric(metric_values("f0_std_hz")),
        "duration_seconds": computed_metric(metric_values("duration_seconds")),
        "energy_mean": computed_metric(metric_values("energy_mean")),
        "ecapa_generated_neutral": computed_metric(
            metric_values("ecapa_generated_neutral")
        ),
        "ecapa_identity_reference": computed_metric(
            metric_values("ecapa_identity_reference")
        ),
        "neutral_zero_max_abs_error": computed_metric(
            neutral_zero.get("max_abs_error"),
            tolerance=neutral_zero.get("tolerance"),
        )
        if neutral_zero.get("max_abs_error") is not None
        else unavailable_metric("generation report has no neutral-zero measurement"),
        "wer": computed_metric(
            {"mean": wer.get("mean"), "worst": wer.get("worst"), "conditions": wer.get("conditions")}
        )
        if wer.get("status") in ("computed", "partial")
        else unavailable_metric(wer.get("reason", "WER unavailable")),
    }

    gates = {
        "e0_directions": direction_gate,
        "identity": {
            "passed": identity_pass,
            "failures": identity.get("failures", []),
            "warnings": identity.get("warnings", []),
            "neutral_sanity": identity.get("neutral_sanity", {}),
            "thresholds": identity.get("thresholds", {}),
        },
        "neutral_zero": {
            "passed": neutral_zero_pass,
            "max_abs_error": neutral_zero.get("max_abs_error"),
            "tolerance": neutral_zero.get("tolerance"),
        },
        "classifier_above_chance": {
            "passed": accuracy_above_chance,
            "status": "computed" if classifier_proxy_valid else "invalid",
            "proxy_valid": classifier_proxy_valid,
            "reason": classifier_proxy_reason,
            "accuracy": overall_generated.get("accuracy"),
            "macro_f1": overall_generated.get("macro_f1"),
            "chance": 1.0 / len(EMOTION_ORDER),
            "domain_shift": domain_shift,
        },
    }

    limitations = [
        "Exploratory: CREMA-D is acted speech and every emotion class is below the 2 h confirmatory gate.",
        f"Only {measured_frame['actor_id'].nunique()} validation actor(s); speaker-cluster intervals are provisional.",
        "No human MOS or emotion-identification study was run; perceptual quality is unmeasured.",
        "Fear and disgust have no fixed-delta artifact in this checkout and are reported as unavailable for E0.",
        "The emotion classifier is an evaluator, not a learned control; real-speech calibration is train-only.",
        "The canonical face-to-voice artifacts are frozen and were not used to generate these clips.",
        "The identity base is StyleTTS2's frozen style/predictor encoder, not the face-to-voice adapter.",
    ]
    neutral_sanity = identity.get("neutral_sanity", {})
    if neutral_sanity.get("passed") is False:
        mean = neutral_sanity.get("mean")
        threshold = neutral_sanity.get("threshold")
        mean_text = f"{float(mean):.3f}" if mean is not None else "unavailable"
        threshold_text = f"{float(threshold):.3f}" if threshold is not None else "unavailable"
        limitations.append(
            "Generated neutral ECAPA similarity to the real speaker is "
            f"{mean_text} < {threshold_text}; condition comparisons are relative "
            "only (domain shift)."
        )
    if domain_shift.get("status") == "computed" and not classifier_proxy_valid:
        limitations.append(
            "Generated speech is outside the real-speech feature distribution "
            f"(standardized centroid distance {domain_shift['standardized_centroid_distance']:.1f}, "
            f"{domain_shift['ratio']:.1f}x the real validation spread); the emotion "
            "classifier proxy is invalid for this run and its accuracy is reported "
            "as unavailable for decision-making."
        )
    if len(generation.get("seeds", [])) <= 1:
        limitations.append("One generation seed; multi-seed stability is not established.")

    config_payload = {
        "index": str(args.index),
        "manifest": str(args.manifest),
        "classifier": str(args.classifier),
        "features_cache": str(args.features_cache),
        "device": device,
        "workers": int(args.workers),
        "n_resamples": int(args.n_resamples),
        "seed": int(args.seed),
        "min_ecapa_neutral": float(args.min_ecapa_neutral),
        "compute_wer": bool(args.compute_wer),
        "wer_model": args.wer_model,
    }
    run_id = args.run_id or f"crema_e0_{len(generation.get('seeds', [])) or 1}seed"
    report = build_report(
        track="emotion",
        level="exploratory",
        run_id=run_id,
        metrics=metrics,
        gates=gates,
        status=(
            "completed"
            if direction_gate.get("status") == "computed"
            and direction_gate.get("passed")
            and identity_pass
            and neutral_zero_pass
            else "partial"
        ),
        limitations=limitations,
        mos={
            "status": "unavailable",
            "reason": "no human ratings were collected for this run",
        },
        seeds=[int(seed) for seed in sorted(set(index["seed"].astype(int)))],
        config_sha256=_config_sha256(config_payload),
        manifest_path=args.manifest,
        per_class={
            "generated_confusion_matrix": overall_generated.get("confusion_matrix"),
            "generated_per_class": overall_generated.get("per_class"),
            "real_val_calibration": real_val_calibration,
            "real_val_saved_model_in_sample": real_val_saved,
            "generated_saved_model": overall_saved,
        },
        artifacts={
            "generation_index": str(args.index),
            "generation_report": str(args.generation_report),
            "audio_dir": str(Path(index.iloc[0]["audio_path"]).parent.parent),
            "classifier": str(args.classifier),
            "features_cache": str(args.features_cache),
            "canonical_hashes": str(args.canonical_hashes),
        },
        extra={
            "config": config_payload,
            "generation": {
                "seeds": generation.get("seeds"),
                "diffusion_steps": generation.get("diffusion_steps"),
                "scalar_modifiers": generation.get("scalar_modifiers"),
                "emotions_generated": generation.get("emotions_generated"),
                "emotions_unavailable": generation.get("emotions_unavailable"),
                "neutral_zero": generation.get("neutral_zero"),
                "counts": generation.get("counts"),
            },
            "conditions": condition_metrics,
            "paired_effects": paired_effects,
            "classification": {
                "generated_calibration_model": overall_generated,
                "generated_saved_model": overall_saved,
                "target_recall": target_recall,
                "real_val_calibration": real_val_calibration,
                "real_val_saved_model_in_sample": real_val_saved,
                "domain_shift_neutral": domain_shift,
                "domain_shift_all": domain_shift_all,
                "proxy_valid": classifier_proxy_valid,
            },
            "identity": identity,
            "wer": wer,
            "unavailable": [
                "human MOS",
                "human emotion-identification",
                "fear fixed delta",
                "disgust fixed delta",
            ],
        },
    )
    write_report(args.output, report)
    print(f"Run: {run_id}")
    print(f"Generated accuracy: {overall_generated.get('accuracy')}  macro-F1: {overall_generated.get('macro_f1')}")
    print(f"Classifier proxy valid: {classifier_proxy_valid}  domain-shift ratio: {domain_shift.get('ratio')}")
    print(f"Real val calibration accuracy: {real_val_calibration.get('accuracy')}")
    print(f"E0 direction gate: {direction_gate.get('passed')}")
    print(f"Identity guardrail: {identity_pass}")
    print(f"Neutral-zero: {neutral_zero_pass}")
    print(f"Wrote {args.output}")
    return 0 if report["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
