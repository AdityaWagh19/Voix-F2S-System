"""Tests for the frozen SSL accent features and A0 feature-key ablation."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np

from voix.evaluation.accent_ssl import SSLAccentConfig, ssl_feature_names

ROOT = Path(__file__).resolve().parent.parent
TRAINER = ROOT / "scripts" / "training" / "train_accent_classifier.py"
CLASSES = ("us", "england", "australian", "canadian", "india_south_asia")


def _load_trainer():
    spec = importlib.util.spec_from_file_location(
        "train_accent_classifier_ssl_test", TRAINER
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _synthetic_cache(path: Path, *, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    base_rows = []
    ssl_rows = []
    labels = []
    speakers = []
    splits = []
    clip_ids = []
    for split in ("train", "val", "test"):
        for class_index, label in enumerate(CLASSES):
            for speaker_index in range(3):
                speaker = f"{split}_{label}_{speaker_index}"
                base_center = np.zeros(12)
                base_center[class_index] = 2.0
                ssl_center = np.zeros(16)
                ssl_center[class_index] = 1.5
                for clip_index in range(3):
                    base_rows.append(
                        base_center + rng.normal(scale=0.15, size=12)
                    )
                    ssl_rows.append(
                        ssl_center + rng.normal(scale=0.15, size=16)
                    )
                    labels.append(label)
                    speakers.append(speaker)
                    splits.append(split)
                    clip_ids.append(f"{speaker}_{clip_index}")
    base = np.asarray(base_rows, dtype=np.float32)
    ssl = np.asarray(ssl_rows, dtype=np.float32)
    np.savez_compressed(
        path,
        features=np.concatenate([base, ssl], axis=1),
        base_features=base,
        ssl_features=ssl,
        label=np.asarray(labels, dtype=str),
        speaker_id=np.asarray(speakers, dtype=str),
        split=np.asarray(splits, dtype=str),
        clip_id=np.asarray(clip_ids, dtype=str),
        feature_names=np.asarray(
            [f"base_{index}" for index in range(12)]
            + [f"ssl_{index}" for index in range(16)],
            dtype=str,
        ),
        base_feature_names=np.asarray(
            [f"base_{index}" for index in range(12)], dtype=str
        ),
        ssl_feature_names=np.asarray(
            [f"ssl_{index}" for index in range(16)], dtype=str
        ),
        ssl_config=np.asarray(
            json.dumps(
                {
                    "bundle": "WAV2VEC2_BASE",
                    "num_layers": 8,
                    "sample_rate": 16000,
                    "max_seconds": 12.0,
                    "l2_normalize": True,
                },
                sort_keys=True,
            )
        ),
    )


def test_ssl_config_is_documented_and_stable():
    config = SSLAccentConfig()
    payload = config.as_dict()
    assert payload["feature_dim"] == 768
    assert payload["pooling"] == "mean"
    assert len(ssl_feature_names(config)) == 768


def test_extractor_keeps_lengths_on_the_waveform_device(monkeypatch):
    import torch

    from voix.evaluation.accent_ssl import SSLAccentFeatureExtractor

    extractor = SSLAccentFeatureExtractor(device="cpu", batch_size=2)
    observed = {}

    class StubModel:
        def extract_features(self, waveforms, lengths=None, num_layers=8):
            observed["same_device"] = lengths.device == waveforms.device
            observed["lengths"] = lengths.tolist()
            hidden = torch.ones(
                (waveforms.shape[0], 4, 8), dtype=torch.float32
            )
            output_lengths = torch.full(
                (waveforms.shape[0],), 4, dtype=torch.long
            )
            return [hidden], output_lengths

    monkeypatch.setattr(extractor, "_load", lambda: None)
    extractor._model = StubModel()
    extractor._torch = torch
    monkeypatch.setattr(
        extractor,
        "_load_wave",
        lambda path: np.ones(1600, dtype=np.float32),
    )
    features = extractor.extract_paths(["a.wav", "b.wav"])
    assert features.shape == (2, 8)
    assert observed["same_device"] is True
    assert observed["lengths"] == [1600, 1600]


def test_trainer_selects_each_feature_block(tmp_path):
    module = _load_trainer()
    cache = tmp_path / "features.npz"
    _synthetic_cache(cache)
    expected_dims = {"combined": 28, "base": 12, "ssl": 16}
    for feature_key, expected_dim in expected_dims.items():
        model = tmp_path / f"model_{feature_key}.joblib"
        report_path = tmp_path / f"report_{feature_key}.json"
        exit_code = module.main(
            [
                "--features",
                str(cache),
                "--model",
                str(model),
                "--report",
                str(report_path),
                "--feature-key",
                feature_key,
                "--n-resamples",
                "20",
                "--no-speaker-centered",
            ]
        )
        assert exit_code == 0
        report = json.loads(report_path.read_text(encoding="utf-8"))
        assert report["feature_key"] == feature_key
        assert report["feature_dim"] == expected_dim
        assert report["ssl_config"]["num_layers"] == 8
        payload = __import__("joblib").load(model)
        assert payload["feature_key"] == feature_key
        assert payload["feature_dim"] == expected_dim
