"""Tests for the E0 evaluator's pure helpers and CLI defaults."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "evaluation" / "evaluate_crema_e0.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("evaluate_crema_e0", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


evaluate = _load_module()


def test_parse_args_defaults():
    args = evaluate.parse_args([])
    assert args.index.name == "generation_index.csv"
    assert args.manifest.name == "crema_d_paired_eval_val.csv"
    assert args.device == "cuda"
    assert args.compute_wer is False
    assert args.min_ecapa_neutral == pytest.approx(0.80)
    assert args.n_resamples == 10_000


def test_parse_args_flags():
    args = evaluate.parse_args(
        ["--device", "cpu", "--compute-wer", "--n-resamples", "100", "--seed", "7"]
    )
    assert args.device == "cpu"
    assert args.compute_wer is True
    assert args.n_resamples == 100
    assert args.seed == 7


def test_condition_recall_and_predicted_counts():
    frame = pd.DataFrame(
        {
            "condition": ["fixed_delta", "fixed_delta", "neutral"],
            "emotion": ["happy", "happy", "neutral"],
            "prediction_calibration": ["happy", "sad", "neutral"],
        }
    )
    result = evaluate._condition_recall(frame, "prediction_calibration")
    assert result["fixed_delta:happy"]["target_recall"] == pytest.approx(0.5)
    assert result["fixed_delta:happy"]["predicted_counts"] == {"happy": 1, "sad": 1}
    assert result["neutral:neutral"]["target_recall"] == pytest.approx(1.0)


def test_wer_by_condition_uses_injected_transcriber(tmp_path: Path, monkeypatch):
    paths = []
    for name in ("a.wav", "b.wav"):
        path = tmp_path / name
        path.write_bytes(b"not real audio")
        paths.append(path)
    frame = pd.DataFrame(
        {
            "condition": ["fixed_delta", "neutral"],
            "emotion": ["happy", "neutral"],
            "audio_path": [str(paths[0]), str(paths[1])],
            "text": ["hello", "hello"],
        }
    )

    import voix.evaluation.audio_metrics as audio_metrics

    def fake_transcribe(audio_paths, references, *, model_size, device):
        assert len(list(audio_paths)) == 1
        assert references == ["hello"]
        return {"status": "computed", "wer": 0.2, "model": model_size}

    monkeypatch.setattr(audio_metrics, "transcribe_wer", fake_transcribe)
    result = evaluate._wer_by_condition(frame, model_size="tiny", device="cpu")
    assert result["status"] == "computed"
    assert result["mean"] == pytest.approx(0.2)
    assert result["worst"] == pytest.approx(0.2)


def test_wer_by_condition_reports_missing_audio(tmp_path: Path):
    frame = pd.DataFrame(
        {
            "condition": ["fixed_delta"],
            "emotion": ["happy"],
            "audio_path": [str(tmp_path / "missing.wav")],
            "text": ["hello"],
        }
    )
    result = evaluate._wer_by_condition(frame, model_size="tiny", device="cpu")
    assert result["status"] == "unavailable"
    assert "missing audio" in result["conditions"]["fixed_delta:happy"]["reason"]


def test_measure_one_handles_missing_audio(tmp_path: Path):
    row = pd.Series({"audio_path": str(tmp_path / "missing.wav")})
    result = evaluate._measure_one(row, lambda path: [0.0])
    assert result["audio_exists"] is False
    assert result["prosody"]["status"] == "unavailable"
    assert result["features"]["status"] == "unavailable"


def test_cosine_handles_none_and_orthogonal_vectors():
    assert evaluate._cosine(None, torch.ones(1, 4)) is None
    assert evaluate._cosine(torch.ones(1, 4), None) is None
    assert evaluate._cosine(torch.tensor([[1.0, 0.0]]), torch.tensor([[1.0, 0.0]])) == pytest.approx(1.0)
    assert evaluate._cosine(torch.tensor([[1.0, 0.0]]), torch.tensor([[0.0, 1.0]])) == pytest.approx(0.0)


def test_config_sha256_is_deterministic():
    first = evaluate._config_sha256({"b": 2, "a": 1})
    second = evaluate._config_sha256({"a": 1, "b": 2})
    assert first == second
    assert len(first) == 64
    assert first != evaluate._config_sha256({"a": 1, "b": 3})
