"""Accent-classifier metrics and diagnostics.

Kept separate from the emotion evaluator so the accent report uses a fixed
five-class label order and does not inherit emotion-specific label ordering.
"""
from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence

import numpy as np

from voix.evaluation.stats import speaker_cluster_bootstrap


def classifier_metrics(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    groups: Sequence[str],
    *,
    labels: Sequence[str],
    n_resamples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 0,
) -> Dict[str, Any]:
    """Accuracy, macro-F1, per-class recall, confusion matrix, cluster CI."""
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        f1_score,
        precision_recall_fscore_support,
    )

    if len(y_true) != len(y_pred) or len(y_true) != len(groups):
        raise ValueError("y_true, y_pred, and groups must have the same length")
    if not y_true:
        return {"status": "unavailable", "reason": "no rows"}
    label_list = [str(label) for label in labels]
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=label_list, zero_division=0
    )
    per_class = {
        label: {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(support[index]),
        }
        for index, label in enumerate(label_list)
    }
    rows = [
        {"speaker_id": str(group), "correct": float(str(true) == str(pred))}
        for true, pred, group in zip(y_true, y_pred, groups)
    ]
    speakers = {row["speaker_id"] for row in rows}
    result: Dict[str, Any] = {
        "status": "computed",
        "n": len(y_true),
        "n_speakers": len(speakers),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(
            f1_score(y_true, y_pred, average="macro", labels=label_list, zero_division=0)
        ),
        "per_class": per_class,
        "confusion_matrix": {
            "labels": label_list,
            "matrix": confusion_matrix(y_true, y_pred, labels=label_list).astype(int).tolist(),
        },
        "speaker_cluster_ci": {
            "status": "unavailable",
            "reason": "fewer than two speakers",
        },
    }
    if len(speakers) >= 2:
        interval = speaker_cluster_bootstrap(
            rows,
            lambda selected: float(np.mean([row["correct"] for row in selected])),
            speaker_key="speaker_id",
            n_resamples=n_resamples,
            confidence=confidence,
            seed=seed,
        )
        result["speaker_cluster_ci"] = {
            "status": "computed",
            "accuracy_low": interval.ci_low,
            "accuracy_high": interval.ci_high,
            "confidence": confidence,
            "n_resamples": n_resamples,
            "seed": seed,
        }
    return result


def majority_baseline(
    train_labels: Sequence[str],
    test_labels: Sequence[str],
) -> Dict[str, Any]:
    """Predict the most frequent training label for every test clip."""
    from collections import Counter

    if not train_labels or not test_labels:
        raise ValueError("train and test labels must be non-empty")
    counts = Counter(str(label) for label in train_labels)
    majority = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]
    predictions = [majority] * len(test_labels)
    return {
        "label": majority,
        "train_support": int(counts[majority]),
        "predictions": predictions,
        "accuracy": float(np.mean([str(y) == majority for y in test_labels])),
    }


def nearest_centroid_speaker_probe(
    features: np.ndarray,
    speakers: Sequence[str],
    *,
    train_fraction: float = 0.8,
    seed: int = 0,
) -> Dict[str, Any]:
    """Closed-set speaker probe on held-out clips of the training speakers.

    Each speaker's clips are split deterministically into probe-train and
    probe-validation.  A cosine nearest-centroid classifier then predicts the
    held-out clips.  High accuracy means the feature block is strongly
    speaker-discriminative; it does not by itself invalidate the accent
    classifier, but it is a reported limitation.
    """
    matrix = np.asarray(features, dtype=np.float64)
    speaker_array = np.asarray([str(speaker) for speaker in speakers], dtype=object)
    if matrix.ndim != 2 or matrix.shape[0] != speaker_array.size:
        raise ValueError("features and speakers must describe the same rows")
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be in (0, 1)")

    rng = np.random.default_rng(seed)
    train_indices: List[int] = []
    val_indices: List[int] = []
    for speaker in sorted(set(speaker_array)):
        indices = np.flatnonzero(speaker_array == speaker)
        if indices.size < 2:
            continue
        order = rng.permutation(indices)
        cut = max(1, min(indices.size - 1, int(round(indices.size * train_fraction))))
        train_indices.extend(order[:cut].tolist())
        val_indices.extend(order[cut:].tolist())
    if not train_indices or not val_indices:
        return {
            "status": "unavailable",
            "reason": "no speaker has at least two clips",
        }

    train_matrix = matrix[train_indices]
    train_speakers = speaker_array[train_indices]
    centroids: Dict[str, np.ndarray] = {}
    for speaker in sorted(set(train_speakers)):
        centroid = train_matrix[train_speakers == speaker].mean(axis=0)
        norm = float(np.linalg.norm(centroid))
        if norm > 0:
            centroid = centroid / norm
        centroids[speaker] = centroid
    centroid_labels = list(centroids)
    centroid_matrix = np.stack([centroids[label] for label in centroid_labels], axis=0)

    val_matrix = matrix[val_indices]
    val_norms = np.linalg.norm(val_matrix, axis=1, keepdims=True)
    val_norms[val_norms == 0] = 1.0
    similarities = (val_matrix / val_norms) @ centroid_matrix.T
    predictions = [centroid_labels[index] for index in np.argmax(similarities, axis=1)]
    truth = speaker_array[val_indices].tolist()
    correct = [float(str(pred) == str(true)) for pred, true in zip(predictions, truth)]
    return {
        "status": "computed",
        "n_train_clips": len(train_indices),
        "n_val_clips": len(val_indices),
        "n_speakers": len(centroid_labels),
        "accuracy": float(np.mean(correct)),
        "chance": float(1.0 / len(centroid_labels)),
        "train_fraction": train_fraction,
        "seed": seed,
    }


def speaker_centered_diagnostic(
    features: np.ndarray,
    speakers: Sequence[str],
    labels: Sequence[str],
    train_mask: np.ndarray,
    val_mask: np.ndarray,
    *,
    classes: Sequence[str],
    n_resamples: int = 2_000,
    seed: int = 0,
) -> Dict[str, Any]:
    """Transductive diagnostic: subtract each speaker's mean feature.

    This uses each validation speaker's own clips to estimate their mean, so it
    is not a deployable model.  It asks whether accent remains after removing
    the speaker-level offset.  Report it as a diagnostic only.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    matrix = np.asarray(features, dtype=np.float64)
    speaker_array = np.asarray([str(speaker) for speaker in speakers], dtype=object)
    label_array = np.asarray([str(label) for label in labels], dtype=object)
    if matrix.shape[0] != speaker_array.size or matrix.shape[0] != label_array.size:
        raise ValueError("features, speakers, and labels must describe the same rows")

    centered = matrix.copy()
    for speaker in sorted(set(speaker_array)):
        mask = speaker_array == speaker
        centered[mask] = matrix[mask] - matrix[mask].mean(axis=0, keepdims=True)

    train_mask = np.asarray(train_mask, dtype=bool)
    val_mask = np.asarray(val_mask, dtype=bool)
    if not train_mask.any() or not val_mask.any():
        return {"status": "unavailable", "reason": "empty train or validation split"}
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    multi_class="multinomial",
                ),
            ),
        ]
    ).fit(centered[train_mask], label_array[train_mask])
    predictions = model.predict(centered[val_mask]).tolist()
    metrics = classifier_metrics(
        label_array[val_mask].tolist(),
        predictions,
        speaker_array[val_mask].tolist(),
        labels=classes,
        n_resamples=n_resamples,
        seed=seed,
    )
    metrics["note"] = (
        "transductive diagnostic: speaker means are estimated from each "
        "speaker's own clips, including validation clips; not deployable"
    )
    return metrics
