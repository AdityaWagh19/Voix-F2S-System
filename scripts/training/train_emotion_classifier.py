"""Train a pilot CREMA-D emotion classifier with speaker-disjoint evaluation.

This is an **evaluation classifier**, not a learned expression control.  CREMA-D
has 91 actors but only about 0.88 h per emotion, below the 2 h/class control
gate.  The report therefore keeps ``control_training_eligible: false`` and
reports real held-out-speaker accuracy instead of implying production readiness.

Features are hand-crafted acoustic descriptors (MFCC/delta, chroma, spectral
contrast, ZCR, RMS, and F0).  Hyper-parameters are selected on the validation
split; the test split is touched once for the final metrics.

Usage:
    python scripts/training/train_emotion_classifier.py
    python scripts/training/train_emotion_classifier.py --max-per-class 40  # smoke
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_MANIFEST = ROOT / "data" / "processed" / "crema_d_manifest.csv"
DEFAULT_CACHE = ROOT / "data" / "processed" / "crema_d_features.npz"
DEFAULT_MODEL = ROOT / "checkpoints" / "emotion_classifier_crema_d.joblib"
DEFAULT_OUTPUT = ROOT / "results" / "emotion_classifier_crema_d.json"
DEFAULT_CURATION = ROOT / "data" / "processed" / "crema_d_curation_report.json"

SCHEMA = "voix.emotion_classifier.v1"
EMOTIONS = ["neutral", "happy", "sad", "angry", "fear", "disgust"]


def extract_features(path: str, sample_rate: int = 16000, n_mfcc: int = 20) -> np.ndarray:
    """Return a fixed-length acoustic feature vector for one clip."""
    import librosa

    y, _ = librosa.load(path, sr=sample_rate, mono=True)
    y = np.nan_to_num(y.astype(np.float32))
    if y.size == 0:
        raise ValueError(f"empty audio: {path}")
    y, _ = librosa.effects.trim(y, top_db=30.0)
    if y.size < sample_rate // 10:
        raise ValueError(f"audio too short after trim: {path}")

    mfcc = librosa.feature.mfcc(y=y, sr=sample_rate, n_mfcc=n_mfcc)
    delta = librosa.feature.delta(mfcc)
    chroma = librosa.feature.chroma_stft(y=y, sr=sample_rate)
    contrast = librosa.feature.spectral_contrast(y=y, sr=sample_rate)
    zcr = librosa.feature.zero_crossing_rate(y)
    rms = librosa.feature.rms(y=y)
    f0 = librosa.yin(
        y,
        fmin=50.0,
        fmax=600.0,
        sr=sample_rate,
        frame_length=2048,
    )
    voiced = f0[(f0 >= 50.0) & (f0 <= 600.0)]
    if voiced.size == 0:
        f0_mean = f0_std = 0.0
    else:
        f0_mean = float(np.mean(voiced))
        f0_std = float(np.std(voiced))
    parts = [
        mfcc.mean(axis=1),
        mfcc.std(axis=1),
        delta.mean(axis=1),
        delta.std(axis=1),
        chroma.mean(axis=1),
        chroma.std(axis=1),
        contrast.mean(axis=1),
        contrast.std(axis=1),
        np.array(
            [
                float(zcr.mean()),
                float(zcr.std()),
                float(rms.mean()),
                float(rms.std()),
                f0_mean,
                f0_std,
                float(voiced.size) / max(1, f0.size),
            ]
        ),
    ]
    vector = np.concatenate(parts).astype(np.float32)
    return np.nan_to_num(vector)


def feature_names(n_mfcc: int = 20) -> List[str]:
    names: List[str] = []
    names += [f"mfcc_{i}_mean" for i in range(n_mfcc)]
    names += [f"mfcc_{i}_std" for i in range(n_mfcc)]
    names += [f"delta_mfcc_{i}_mean" for i in range(n_mfcc)]
    names += [f"delta_mfcc_{i}_std" for i in range(n_mfcc)]
    names += [f"chroma_{i}_mean" for i in range(12)]
    names += [f"chroma_{i}_std" for i in range(12)]
    names += [f"contrast_{i}_mean" for i in range(7)]
    names += [f"contrast_{i}_std" for i in range(7)]
    names += [
        "zcr_mean",
        "zcr_std",
        "rms_mean",
        "rms_std",
        "f0_mean",
        "f0_std",
        "f0_voiced_ratio",
    ]
    return names


def _resolve_path(path: str) -> str:
    """Resolve a manifest path against the repo root, not the process CWD."""
    candidate = Path(path)
    if candidate.is_absolute():
        return str(candidate)
    return str((ROOT / candidate).resolve())


def build_feature_matrix(
    manifest: pd.DataFrame,
    *,
    workers: int = 8,
    max_per_class: Optional[int] = None,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """Extract features for every row, optionally capping clips per emotion.

    The cap is applied per ``(split, emotion)`` rather than per emotion.  A
    per-emotion cap would take the first rows of a split-sorted manifest and
    could leave validation or test empty in a smoke run.
    """
    frame = manifest.copy()
    if max_per_class is not None:
        frame = (
            frame.groupby(["split", "intended_emotion"], group_keys=False)
            .head(max_per_class)
            .reset_index(drop=True)
        )
    paths = [_resolve_path(path) for path in frame["path"].astype(str)]

    def _one(path: str) -> Optional[np.ndarray]:
        try:
            return extract_features(path)
        except Exception as exc:  # pragma: no cover - per-file guard
            print(f"[features] skip {path}: {exc}", flush=True)
            return None

    started = time.time()
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        features = list(pool.map(_one, paths))
    keep = [index for index, value in enumerate(features) if value is not None]
    if not keep:
        raise RuntimeError("no features could be extracted")
    matrix = np.stack([features[index] for index in keep])
    frame = frame.iloc[keep].reset_index(drop=True)
    print(
        f"[features] {len(frame)}/{len(paths)} clips, dim={matrix.shape[1]}, "
        f"{time.time() - started:.1f}s",
        flush=True,
    )
    return matrix, frame


def _fit_model(X: np.ndarray, y: np.ndarray, C: float, gamma: str | float):
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", C=C, gamma=gamma, class_weight="balanced")),
        ]
    ).fit(X, y)


def _score(model, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        f1_score,
        precision_recall_fscore_support,
    )

    prediction = model.predict(X)
    labels = sorted(set(y.tolist()))
    precision, recall, f1, support = precision_recall_fscore_support(
        y, prediction, labels=labels, zero_division=0
    )
    per_class = {
        label: {
            "precision": round(float(precision[index]), 4),
            "recall": round(float(recall[index]), 4),
            "f1": round(float(f1[index]), 4),
            "support": int(support[index]),
        }
        for index, label in enumerate(labels)
    }
    return {
        "accuracy": round(float(accuracy_score(y, prediction)), 4),
        "macro_f1": round(float(f1_score(y, prediction, average="macro")), 4),
        "weighted_f1": round(float(f1_score(y, prediction, average="weighted")), 4),
        "per_class": per_class,
        "confusion_matrix": {
            "labels": labels,
            "matrix": confusion_matrix(y, prediction, labels=labels).astype(int).tolist(),
        },
        "n": int(len(y)),
        "prediction": prediction,
    }


def _baseline_accuracy(train_y: np.ndarray, test_y: np.ndarray) -> float:
    from sklearn.dummy import DummyClassifier

    model = DummyClassifier(strategy="most_frequent").fit(
        train_y.reshape(-1, 1), train_y
    )
    return round(float((model.predict(test_y.reshape(-1, 1)) == test_y).mean()), 4)


def _bootstrap_speaker_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: np.ndarray,
    *,
    iterations: int = 1000,
    seed: int = 0,
) -> Dict[str, Any]:
    """Cluster bootstrap over held-out speakers.

    The test split has only a handful of speakers, so a clip-level interval
    would be too narrow.  Resampling speakers with replacement keeps each
    speaker's clips together and shows how much the result depends on which
    speakers were held out.
    """
    from sklearn.metrics import accuracy_score, f1_score

    rng = np.random.default_rng(seed)
    speakers = np.unique(groups)
    if len(speakers) < 2:
        return {"status": "unavailable", "reason": "fewer than two test speakers"}
    indices = {speaker: np.flatnonzero(groups == speaker) for speaker in speakers}
    accuracy: List[float] = []
    macro_f1: List[float] = []
    for _ in range(iterations):
        sample = rng.choice(speakers, size=len(speakers), replace=True)
        rows = np.concatenate([indices[speaker] for speaker in sample])
        accuracy.append(float(accuracy_score(y_true[rows], y_pred[rows])))
        macro_f1.append(float(f1_score(y_true[rows], y_pred[rows], average="macro")))
    return {
        "status": "computed",
        "method": "speaker cluster bootstrap, 95% percentile",
        "iterations": iterations,
        "speakers": int(len(speakers)),
        "accuracy_95ci": [
            round(float(np.percentile(accuracy, 2.5)), 4),
            round(float(np.percentile(accuracy, 97.5)), 4),
        ],
        "macro_f1_95ci": [
            round(float(np.percentile(macro_f1, 2.5)), 4),
            round(float(np.percentile(macro_f1, 97.5)), 4),
        ],
    }


def train_classifier(
    manifest: pd.DataFrame,
    features: np.ndarray,
    *,
    grid: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[Any, Dict[str, Any], Dict[str, Any]]:
    """Select on validation, refit on train+val, evaluate once on test."""
    frame = manifest.reset_index(drop=True)
    train_mask = (frame["split"] == "train").to_numpy()
    val_mask = (frame["split"] == "val").to_numpy()
    test_mask = (frame["split"] == "test").to_numpy()
    if not train_mask.any() or not val_mask.any() or not test_mask.any():
        raise ValueError("manifest must contain train, val, and test rows")

    X_train, y_train = features[train_mask], frame.loc[train_mask, "intended_emotion"].to_numpy()
    X_val, y_val = features[val_mask], frame.loc[val_mask, "intended_emotion"].to_numpy()
    X_test, y_test = features[test_mask], frame.loc[test_mask, "intended_emotion"].to_numpy()

    if grid is None:
        grid = [
            {"C": 1.0, "gamma": "scale"},
            {"C": 3.0, "gamma": "scale"},
            {"C": 10.0, "gamma": "scale"},
            {"C": 3.0, "gamma": 0.01},
            {"C": 10.0, "gamma": 0.001},
        ]

    selection: List[Dict[str, Any]] = []
    best = None
    for params in grid:
        model = _fit_model(X_train, y_train, **params)
        score = _score(model, X_val, y_val)
        row = {
            "params": params,
            "val_accuracy": score["accuracy"],
            "val_macro_f1": score["macro_f1"],
        }
        selection.append(row)
        if best is None or score["macro_f1"] > best[0]:
            best = (score["macro_f1"], params)
    if best is None:
        raise RuntimeError("no model was fitted")
    best_params = best[1]

    X_fit = np.concatenate([X_train, X_val])
    y_fit = np.concatenate([y_train, y_val])
    final = _fit_model(X_fit, y_fit, **best_params)
    test = _score(final, X_test, y_test)

    test_prediction = test.pop("prediction")
    test_groups = frame.loc[test_mask, "actor_id"].astype(str).to_numpy()
    bootstrap = _bootstrap_speaker_ci(y_test, test_prediction, test_groups)
    report = {
        "best_params": best_params,
        "selection": selection,
        "selection_metric": "val_macro_f1",
        # The final model is refit on train+val, so its score on val is not a
        # held-out metric.  Only the test split is reported as evaluation.
        "test": test,
        "test_speaker_bootstrap": bootstrap,
        "majority_baseline_test_accuracy": _baseline_accuracy(y_fit, y_test),
        "test_predictions": test_prediction.tolist(),
        "split_counts": {
            "train": int(train_mask.sum()),
            "val": int(val_mask.sum()),
            "test": int(test_mask.sum()),
        },
        "split_speakers": {
            "train": int(frame.loc[train_mask, "actor_id"].nunique()),
            "val": int(frame.loc[val_mask, "actor_id"].nunique()),
            "test": int(frame.loc[test_mask, "actor_id"].nunique()),
        },
    }
    return final, report, {"y_test": y_test, "frame": frame[test_mask]}


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--curation", type=Path, default=DEFAULT_CURATION)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-per-class", type=int, default=None)
    parser.add_argument("--reuse-features", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    manifest = pd.read_csv(args.manifest)
    required = {"path", "intended_emotion", "split", "actor_id"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"manifest missing columns: {sorted(missing)}")

    if args.reuse_features and args.cache.is_file():
        cached = np.load(args.cache, allow_pickle=True)
        features = cached["features"]
        frame = pd.DataFrame(
            {
                "path": cached["path"],
                "intended_emotion": cached["intended_emotion"],
                "split": cached["split"],
                "actor_id": cached["actor_id"],
            }
        )
        if "duration_sec" in cached and len(cached["duration_sec"]) == len(frame):
            frame["duration_sec"] = cached["duration_sec"]
        if len(frame) != len(manifest):
            raise ValueError("feature cache does not match the manifest row count")
        cached_max = int(cached["max_per_class"]) if "max_per_class" in cached else -1
        requested_max = -1 if args.max_per_class is None else args.max_per_class
        if cached_max != requested_max:
            raise ValueError(
                "feature cache was built with a different --max-per-class"
            )
        print(f"[features] loaded {len(frame)} rows from {args.cache}", flush=True)
    else:
        features, frame = build_feature_matrix(
            manifest, workers=args.workers, max_per_class=args.max_per_class
        )
        args.cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.cache,
            features=features,
            path=frame["path"].astype(str).to_numpy(),
            intended_emotion=frame["intended_emotion"].astype(str).to_numpy(),
            split=frame["split"].astype(str).to_numpy(),
            actor_id=frame["actor_id"].astype(str).to_numpy(),
            max_per_class=np.array(-1 if args.max_per_class is None else args.max_per_class),
            duration_sec=frame["duration_sec"].to_numpy(dtype=np.float64)
            if "duration_sec" in frame.columns
            else np.array([], dtype=np.float64),
        )
        print(f"[features] wrote {args.cache}", flush=True)

    model, report, _ = train_classifier(frame, features)
    args.model.parent.mkdir(parents=True, exist_ok=True)
    import joblib

    joblib.dump(model, args.model)

    curation = None
    if args.curation.is_file():
        curation = json.loads(args.curation.read_text(encoding="utf-8"))
    control_eligible = bool(
        curation and curation.get("gate", {}).get("training_eligible", False)
    )
    payload = {
        "schema_version": SCHEMA,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "manifest": str(args.manifest),
        "model": str(args.model),
        "features_cache": str(args.cache),
        "feature_dim": int(features.shape[1]),
        "feature_names": feature_names(),
        "scope": (
            "Pilot evaluation classifier for acted CREMA-D emotion. "
            "It is not a learned expression control."
        ),
        "control_training_eligible": control_eligible,
        "curation_gate": curation.get("gate") if curation else None,
        "dataset": {
            "clips": int(len(frame)),
            "speakers": int(frame["actor_id"].nunique()),
            "hours": round(
                float(frame["duration_sec"].sum()) / 3600.0, 3
            )
            if "duration_sec" in frame.columns
            else None,
        },
        **report,
        "notes": [
            "Speaker-disjoint train/validation/test splits; test is evaluated once.",
            "CREMA-D is acted speech and 0.633-0.963 h/emotion after excluding source-clipped clips, so this is a pilot classifier, not a production control.",
            "Metrics are measured on held-out speakers; the 9-speaker test split is reported with a speaker-cluster bootstrap CI.",
            "No human MOS is claimed.",
            "Cross-corpus generalization is untested and reported as unavailable.",
        ],
        "unavailable": ["human MOS", "cross-corpus generalization"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({k: payload[k] for k in ("dataset", "best_params", "test", "majority_baseline_test_accuracy")}, indent=2), flush=True)
    print(f"Wrote {args.model}")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
