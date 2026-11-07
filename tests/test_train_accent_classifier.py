"""Integration test for the A0 accent training CLI on synthetic features."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "training" / "train_accent_classifier.py"
CLASSES = ("us", "england", "australian", "canadian", "india_south_asia")


def _load_module():
    spec = importlib.util.spec_from_file_location("train_accent_classifier", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _synthetic_cache(path: Path, *, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    rows = []
    labels = []
    speakers = []
    splits = []
    clip_ids = []
    for split in ("train", "val", "test"):
        for class_index, label in enumerate(CLASSES):
            for speaker_index in range(3):
                speaker = f"{split}_{label}_{speaker_index}"
                center = np.zeros(10)
                center[class_index] = 2.0
                center[5 + class_index % 5] = 1.0
                for clip_index in range(4):
                    rows.append(center + rng.normal(scale=0.15, size=10))
                    labels.append(label)
                    speakers.append(speaker)
                    splits.append(split)
                    clip_ids.append(f"{speaker}_{clip_index}")
    np.savez_compressed(
        path,
        features=np.asarray(rows, dtype=np.float32),
        label=np.asarray(labels, dtype=str),
        speaker_id=np.asarray(speakers, dtype=str),
        split=np.asarray(splits, dtype=str),
        clip_id=np.asarray(clip_ids, dtype=str),
        feature_names=np.asarray([f"f{index}" for index in range(10)], dtype=str),
    )


def test_training_cli_writes_validation_and_test_reports(tmp_path):
    module = _load_module()
    cache = tmp_path / "features.npz"
    _synthetic_cache(cache)
    model = tmp_path / "model.joblib"
    report_path = tmp_path / "report.json"
    exit_code = module.main(
        [
            "--features",
            str(cache),
            "--model",
            str(model),
            "--report",
            str(report_path),
            "--n-resamples",
            "50",
            "--evaluate-test",
            "--no-speaker-centered",
        ]
    )
    assert exit_code == 0
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["level"] == "confirmatory"
    assert report["validation"]["metrics"]["status"] == "computed"
    assert report["test"]["status"] == "evaluated"
    assert report["test"]["metrics"]["status"] == "computed"
    assert report["speaker_probe"]["status"] == "computed"
    assert report["speaker_centered_diagnostic"] is None
    assert report["feature_key"] == "combined"
    assert report["gates"]["beats_majority"]["passed"] is True
    assert report["gates"]["beats_2x_chance"]["passed"] is True
    assert "triggered" in report["gates"]["ssl_fallback"]
    assert model.is_file()


def _synthetic_ssl_cache(path: Path, *, seed: int = 0) -> None:
    """Cache with base, SSL, and combined blocks plus one failed SSL row."""
    rng = np.random.default_rng(seed)
    rows = []
    labels = []
    speakers = []
    splits = []
    clip_ids = []
    for split in ("train", "val", "test"):
        for class_index, label in enumerate(CLASSES):
            for speaker_index in range(3):
                speaker = f"{split}_{label}_{speaker_index}"
                center = np.zeros(10)
                center[class_index] = 2.0
                center[5 + class_index % 5] = 1.0
                for clip_index in range(4):
                    rows.append(center + rng.normal(scale=0.15, size=10))
                    labels.append(label)
                    speakers.append(speaker)
                    splits.append(split)
                    clip_ids.append(f"{speaker}_{clip_index}")
    base = np.asarray(rows, dtype=np.float32)
    ssl = rng.normal(scale=0.2, size=(len(rows), 10)).astype(np.float32)
    # Simulate an extraction failure in the first training row.
    ssl[0] = 0.0
    combined = np.concatenate([base, ssl], axis=1)
    np.savez_compressed(
        path,
        base_features=base,
        ssl_features=ssl,
        features=combined,
        label=np.asarray(labels, dtype=str),
        speaker_id=np.asarray(speakers, dtype=str),
        split=np.asarray(splits, dtype=str),
        clip_id=np.asarray(clip_ids, dtype=str),
        base_feature_names=np.asarray([f"b{index}" for index in range(10)], dtype=str),
        ssl_feature_names=np.asarray([f"s{index}" for index in range(10)], dtype=str),
        feature_names=np.asarray(
            [f"b{index}" for index in range(10)]
            + [f"s{index}" for index in range(10)],
            dtype=str,
        ),
        ssl_config=np.asarray(json.dumps({"bundle": "synthetic"})),
    )


def test_training_excludes_zero_ssl_rows(tmp_path):
    module = _load_module()
    cache = tmp_path / "features_ssl.npz"
    _synthetic_ssl_cache(cache)
    report_path = tmp_path / "report.json"
    exit_code = module.main(
        [
            "--features",
            str(cache),
            "--model",
            str(tmp_path / "model.joblib"),
            "--report",
            str(report_path),
            "--n-resamples",
            "50",
            "--no-speaker-centered",
        ]
    )
    assert exit_code == 0
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["excluded_zero_ssl_rows"] == 1
    assert report["train"]["clips"] == 59
    assert report["feature_dim"] == 20
    assert report["ssl_config"] == {"bundle": "synthetic"}


def test_training_cli_defaults_to_validation_only(tmp_path):
    module = _load_module()
    cache = tmp_path / "features.npz"
    _synthetic_cache(cache)
    report_path = tmp_path / "report.json"
    exit_code = module.main(
        [
            "--features",
            str(cache),
            "--model",
            str(tmp_path / "model.joblib"),
            "--report",
            str(report_path),
            "--n-resamples",
            "50",
        ]
    )
    assert exit_code == 0
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["level"] == "exploratory"
    assert report["test"]["status"] == "not_evaluated"
    assert report["speaker_centered_diagnostic"]["status"] == "computed"
