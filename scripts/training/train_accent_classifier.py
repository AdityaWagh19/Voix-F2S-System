"""Train and evaluate the A0 GLOBE accent classifier.

Model selection uses the speaker-disjoint validation split only.  The test
split is scored once, and only when ``--evaluate-test`` is passed, so the final
report is not the product of repeated test peeking.

Examples::

    # Development report on validation speakers
    python scripts/training/train_accent_classifier.py

    # Final A0 report; touch test once
    python scripts/training/train_accent_classifier.py --evaluate-test

    # Include the Irish evaluation-only class as an out-of-distribution check
    python scripts/training/train_accent_classifier.py --evaluate-test --evaluate-irish
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_FEATURES = ROOT / "data" / "processed" / "globe_accent_features.npz"
DEFAULT_MODEL = ROOT / "checkpoints" / "accent_classifier_globe.joblib"
DEFAULT_REPORT = ROOT / "results" / "accent" / "a0" / "report.json"
TRAINABLE_CLASSES = (
    "us",
    "england",
    "australian",
    "canadian",
    "india_south_asia",
)
EVAL_ONLY_CLASSES = ("irish",)
REPORT_SCHEMA = "voix.accent_classifier_report.v1"
C_GRID = (0.01, 0.1, 1.0, 10.0)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the A0 accent classifier")
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--n-resamples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--feature-key",
        choices=("combined", "ssl", "base"),
        default="combined",
        help=(
            "Which block of the accent cache to train on: combined "
            "(ECAPA+prosody+SSL), ssl, or base (ECAPA+prosody)"
        ),
    )
    parser.add_argument("--evaluate-test", action="store_true")
    parser.add_argument("--evaluate-irish", action="store_true")
    parser.add_argument(
        "--no-speaker-centered",
        action="store_true",
        help="Skip the transductive speaker-centered diagnostic",
    )
    return parser.parse_args(argv)


def _fit_model(features: np.ndarray, labels: np.ndarray, *, c_value: float):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    C=float(c_value),
                    max_iter=3000,
                    class_weight="balanced",
                    multi_class="multinomial",
                    n_jobs=-1,
                ),
            ),
        ]
    ).fit(features, labels)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    import joblib

    from voix.evaluation.accent import (
        classifier_metrics,
        majority_baseline,
        nearest_centroid_speaker_probe,
        speaker_centered_diagnostic,
    )

    if not args.features.is_file():
        raise FileNotFoundError(f"accent feature cache not found: {args.features}")
    cache = np.load(args.features, allow_pickle=True)
    feature_key = str(args.feature_key)
    key_map = {
        "combined": "features",
        "ssl": "ssl_features",
        "base": "base_features",
    }
    cache_key = key_map[feature_key]
    if cache_key not in cache.files:
        if feature_key == "base" and "features" in cache.files:
            cache_key = "features"
        else:
            raise ValueError(
                f"feature cache has no {cache_key!r} block; "
                "run scripts/data/extract_accent_ssl_features.py first"
            )
    features = cache[cache_key].astype(np.float64)
    name_key = {
        "features": "feature_names",
        "ssl_features": "ssl_feature_names",
        "base_features": "base_feature_names",
    }[cache_key]
    if name_key in cache.files:
        feature_names = [str(item) for item in cache[name_key]]
    else:
        feature_names = [f"feature_{index}" for index in range(features.shape[1])]
    ssl_config: Optional[Dict[str, Any]] = None
    if "ssl_config" in cache.files:
        try:
            ssl_config = json.loads(str(cache["ssl_config"].item()))
        except (ValueError, TypeError):
            ssl_config = None
    labels = cache["label"].astype(str)
    speakers = cache["speaker_id"].astype(str)
    splits = cache["split"].astype(str)
    clip_ids = cache["clip_id"].astype(str)
    if features.ndim != 2 or features.shape[0] != labels.size:
        raise ValueError("feature cache is malformed")
    if not np.isfinite(features).all():
        raise ValueError("feature cache contains non-finite values")

    # A failed SSL extraction leaves an all-zero row.  Those rows are not real
    # features and must never be trained on.  When an SSL block exists, apply
    # the same exclusion to every feature key so base/SSL/combined ablations use
    # an identical clip set.
    excluded_zero_ssl = 0
    if "ssl_features" in cache.files:
        ssl_block = cache["ssl_features"]
        if ssl_block.ndim != 2 or ssl_block.shape[0] != labels.size:
            raise ValueError("SSL feature block is malformed")
        zero_rows = ~np.any(ssl_block, axis=1)
        excluded_zero_ssl = int(zero_rows.sum())
        if excluded_zero_ssl:
            keep = ~zero_rows
            features = features[keep]
            labels = labels[keep]
            speakers = speakers[keep]
            splits = splits[keep]
            clip_ids = clip_ids[keep]
            if not np.isfinite(features).all():
                raise ValueError("feature cache contains non-finite values")

    trainable_mask = np.isin(labels, TRAINABLE_CLASSES)
    train_mask = trainable_mask & (splits == "train")
    val_mask = trainable_mask & (splits == "val")
    test_mask = trainable_mask & (splits == "test")
    if not train_mask.any() or not val_mask.any():
        raise ValueError("feature cache must contain train and val rows for the five trainable classes")

    train_labels = labels[train_mask]
    val_labels = labels[val_mask]
    train_speakers = speakers[train_mask]
    val_speakers = speakers[val_mask]
    overlap = set(train_speakers) & set(val_speakers)
    if overlap:
        raise ValueError(f"train/val speaker overlap: {sorted(overlap)[:10]}")

    selection: List[Dict[str, Any]] = []
    best: Optional[Dict[str, Any]] = None
    for c_value in C_GRID:
        model = _fit_model(features[train_mask], train_labels, c_value=c_value)
        predictions = model.predict(features[val_mask]).tolist()
        metrics = classifier_metrics(
            val_labels.tolist(),
            predictions,
            val_speakers.tolist(),
            labels=TRAINABLE_CLASSES,
            n_resamples=args.n_resamples,
            seed=args.seed,
        )
        entry = {
            "C": c_value,
            "val_accuracy": metrics["accuracy"],
            "val_macro_f1": metrics["macro_f1"],
        }
        selection.append(entry)
        key = (metrics["macro_f1"], metrics["accuracy"], -c_value)
        if best is None or key > best["key"]:
            best = {
                "key": key,
                "C": c_value,
                "model": model,
                "predictions": predictions,
                "metrics": metrics,
            }
    assert best is not None

    baseline = majority_baseline(train_labels.tolist(), val_labels.tolist())
    baseline_metrics = classifier_metrics(
        val_labels.tolist(),
        baseline["predictions"],
        val_speakers.tolist(),
        labels=TRAINABLE_CLASSES,
        n_resamples=args.n_resamples,
        seed=args.seed,
    )

    speaker_probe = nearest_centroid_speaker_probe(
        features[train_mask],
        train_speakers.tolist(),
        seed=args.seed,
    )

    centered = None
    if not args.no_speaker_centered:
        centered = speaker_centered_diagnostic(
            features,
            speakers,
            labels,
            train_mask,
            val_mask,
            classes=TRAINABLE_CLASSES,
            n_resamples=args.n_resamples,
            seed=args.seed,
        )

    args.model.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": best["model"],
            "classes": list(TRAINABLE_CLASSES),
            "C": best["C"],
            "feature_dim": int(features.shape[1]),
            "feature_key": feature_key,
            "feature_names": feature_names,
            "ssl_config": ssl_config,
            "schema_version": "voix.accent_classifier_model.v1",
            "train_speakers": int(len(set(train_speakers))),
        },
        args.model,
    )

    report: Dict[str, Any] = {
        "schema_version": REPORT_SCHEMA,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "level": "exploratory",
        "track": "accent",
        "experiment": "A0",
        "features": str(args.features),
        "features_sha256": _sha256(args.features),
        "model": str(args.model),
        "model_sha256": _sha256(args.model),
        "classes": list(TRAINABLE_CLASSES),
        "feature_dim": int(features.shape[1]),
        "feature_key": feature_key,
        "excluded_zero_ssl_rows": int(excluded_zero_ssl),
        "ssl_config": ssl_config,
        "train": {
            "clips": int(train_mask.sum()),
            "speakers": int(len(set(train_speakers))),
            "per_class": {
                label: int((train_labels == label).sum()) for label in TRAINABLE_CLASSES
            },
        },
        "validation": {
            "clips": int(val_mask.sum()),
            "speakers": int(len(set(val_speakers))),
            "metrics": best["metrics"],
            "majority_baseline": {
                "label": baseline["label"],
                "metrics": baseline_metrics,
            },
        },
        "model_selection": {
            "grid": list(C_GRID),
            "criterion": "validation macro-F1, then accuracy, then smaller C",
            "selected_C": best["C"],
            "results": selection,
        },
        "speaker_probe": speaker_probe,
        "speaker_centered_diagnostic": centered,
        "gates": {
            "beats_majority": {
                "accuracy": float(best["metrics"]["accuracy"]),
                "majority_accuracy": float(baseline_metrics["accuracy"]),
                "passed": bool(
                    best["metrics"]["accuracy"] > baseline_metrics["accuracy"]
                ),
            },
            "beats_2x_chance": {
                "accuracy": float(best["metrics"]["accuracy"]),
                "chance": float(1.0 / len(TRAINABLE_CLASSES)),
                "threshold": float(2.0 / len(TRAINABLE_CLASSES)),
                "passed": bool(
                    best["metrics"]["accuracy"] > 2.0 / len(TRAINABLE_CLASSES)
                ),
            },
            "ssl_fallback": {
                "validation_macro_f1": float(best["metrics"]["macro_f1"]),
                "threshold": 0.55,
                "triggered": bool(best["metrics"]["macro_f1"] < 0.55),
                "note": (
                    "A macro-F1 below 0.55 triggers the frozen SSL feature "
                    "fallback before any generated-audio accent evaluation."
                ),
            },
        },
        "test": {
            "status": "not_evaluated",
            "reason": "pass --evaluate-test for the one-time final evaluation",
        },
        "irish": {
            "status": "not_evaluated",
            "reason": "pass --evaluate-irish for the evaluation-only check",
        },
        "notes": [
            "Model selection used validation speakers only.",
            "Labels are self-reported GLOBE accent classes; predicted_accent was not used.",
            "The speaker probe measures feature speaker-discriminability, not accent leakage directly.",
            "The speaker-centered diagnostic is transductive and not deployable.",
            "No human accent-identification study was run.",
        ],
        "limitations": [
            "GLOBE is read speech from Common Voice, not conversational speech.",
            "Per-class test speakers are few; speaker-cluster intervals are wide.",
            "Generated-audio accent accuracy is a proxy, not proof of real-world accent quality.",
        ],
    }

    if args.evaluate_test:
        if not test_mask.any():
            raise ValueError("--evaluate-test was requested but no test rows exist")
        test_speakers = speakers[test_mask]
        if set(test_speakers) & (set(train_speakers) | set(val_speakers)):
            raise ValueError("test speakers overlap train or validation speakers")
        test_predictions = best["model"].predict(features[test_mask]).tolist()
        test_metrics = classifier_metrics(
            labels[test_mask].tolist(),
            test_predictions,
            test_speakers.tolist(),
            labels=TRAINABLE_CLASSES,
            n_resamples=args.n_resamples,
            seed=args.seed,
        )
        test_baseline = majority_baseline(train_labels.tolist(), labels[test_mask].tolist())
        report["test"] = {
            "status": "evaluated",
            "clips": int(test_mask.sum()),
            "speakers": int(len(set(test_speakers))),
            "metrics": test_metrics,
            "majority_baseline": {
                "label": test_baseline["label"],
                "accuracy": test_baseline["accuracy"],
                "predictions": test_baseline["predictions"],
            },
        }
        report["level"] = "confirmatory"

    if args.evaluate_irish:
        irish_mask = (labels == "irish") & (splits == "test")
        if not irish_mask.any():
            raise ValueError("--evaluate-irish was requested but no Irish test rows exist")
        irish_predictions = best["model"].predict(features[irish_mask]).tolist()
        report["irish"] = {
            "status": "evaluated",
            "clips": int(irish_mask.sum()),
            "speakers": int(len(set(speakers[irish_mask]))),
            "metrics": classifier_metrics(
                labels[irish_mask].tolist(),
                irish_predictions,
                speakers[irish_mask].tolist(),
                labels=TRAINABLE_CLASSES + EVAL_ONLY_CLASSES,
                n_resamples=args.n_resamples,
                seed=args.seed,
            ),
            "note": (
                "Irish is evaluation-only and is not part of the five-class "
                "training set or macro average."
            ),
        }

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Selected C={best['C']}", flush=True)
    print(
        f"Val accuracy={best['metrics']['accuracy']:.4f} "
        f"macro-F1={best['metrics']['macro_f1']:.4f} "
        f"(majority acc={baseline_metrics['accuracy']:.4f})",
        flush=True,
    )
    print(
        f"Speaker probe top-1={speaker_probe.get('accuracy')} "
        f"chance={speaker_probe.get('chance')}",
        flush=True,
    )
    if args.evaluate_test:
        print(
            f"Test accuracy={report['test']['metrics']['accuracy']:.4f} "
            f"macro-F1={report['test']['metrics']['macro_f1']:.4f}",
            flush=True,
        )
    print(f"Wrote {args.model}", flush=True)
    print(f"Wrote {args.report}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
