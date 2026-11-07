"""Tests for the StyleTTS2 LoRA trainer's data and reporting helpers."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "training" / "train_styletts2_lora.py"

spec = importlib.util.spec_from_file_location("train_styletts2_lora", SCRIPT)
assert spec and spec.loader
trainer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trainer)


def _manifest(tmp_path: Path, overlap: bool = False) -> Path:
    wav = tmp_path / "clip.wav"
    sf.write(str(wav), np.zeros(2400, dtype=np.float32), 24000)
    rows = []
    for index in range(4):
        split = "train" if index < 2 else "val"
        speaker = "id00000" if (overlap and index == 2) else f"id{index:05d}"
        rows.append(
            {
                "clip_id": f"{split}/{speaker}/{index}",
                "speaker_id": speaker,
                "split": split,
                "wav_path": str(wav),
                "ref_clip_id": f"{split}/{speaker}/ref",
                "ref_wav_path": str(wav),
                "duration_sec": 0.1,
                "text": "this is a test sentence",
            }
        )
    path = tmp_path / "manifest.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_load_manifest_validates_speaker_disjointness(tmp_path):
    frame = trainer.load_manifest(_manifest(tmp_path))
    assert set(frame["split"]) == {"train", "val"}
    assert len(frame) == 4

    with pytest.raises(ValueError, match="not speaker-disjoint"):
        trainer.load_manifest(_manifest(tmp_path, overlap=True))


def test_load_manifest_rejects_missing_audio(tmp_path):
    path = _manifest(tmp_path)
    frame = pd.read_csv(path)
    frame.loc[0, "wav_path"] = str(tmp_path / "missing.wav")
    frame.to_csv(path, index=False)
    with pytest.raises(FileNotFoundError, match="missing wav"):
        trainer.load_manifest(path)


def test_warmup_cosine_has_warmup_peak_and_decay():
    assert trainer.warmup_cosine(0, 100, 10, 0.05) == pytest.approx(0.1)
    assert trainer.warmup_cosine(9, 100, 10, 0.05) == pytest.approx(1.0)
    assert trainer.warmup_cosine(10, 100, 10, 0.05) == pytest.approx(1.0)
    values = [trainer.warmup_cosine(step, 100, 10, 0.05) for step in range(10, 100)]
    assert all(0.05 <= value <= 1.0 for value in values)
    assert all(a >= b - 1e-12 for a, b in zip(values, values[1:]))
    assert values[-1] < 0.06


def test_select_eval_records_is_deterministic_and_held_out():
    frame = pd.DataFrame(
        [
            {
                "clip_id": f"val/{index}",
                "speaker_id": f"id{index}",
                "split": "val",
            }
            for index in range(6)
        ]
    )
    first = trainer.select_eval_records(frame, 3, seed=7)
    second = trainer.select_eval_records(frame, 3, seed=7)
    assert [row["clip_id"] for row in first] == [row["clip_id"] for row in second]
    assert len(first) == 3
    assert all(str(row["clip_id"]).startswith("val/") for row in first)
    assert trainer.select_eval_records(frame, 0, seed=7) == []


def test_summarize_generation_reports_means_and_paired_delta():
    rows = [
        {
            "clip_id": "a",
            "condition": "base",
            "wer": 0.4,
            "speaker_similarity_target": 0.5,
            "f0_mean_hz": 100.0,
            "duration_seconds": 2.0,
        },
        {
            "clip_id": "b",
            "condition": "base",
            "wer": 0.2,
            "speaker_similarity_target": 0.7,
            "f0_mean_hz": 120.0,
            "duration_seconds": 2.2,
        },
        {
            "clip_id": "a",
            "condition": "lora",
            "wer": 0.1,
            "speaker_similarity_target": 0.8,
            "f0_mean_hz": 130.0,
            "duration_seconds": 2.1,
        },
        {
            "clip_id": "b",
            "condition": "lora",
            "wer": None,
            "speaker_similarity_target": 0.6,
            "f0_mean_hz": None,
            "duration_seconds": 2.3,
        },
    ]
    summary = trainer.summarize_generation(rows)
    assert summary["status"] == "computed"
    assert summary["conditions"]["base"]["metrics"]["wer"]["mean"] == pytest.approx(0.3)
    assert summary["conditions"]["lora"]["metrics"]["wer"]["mean"] == pytest.approx(0.1)
    assert summary["conditions"]["lora"]["metrics"]["wer"]["n"] == 1
    assert summary["delta_lora_minus_base"]["wer"] == pytest.approx(-0.2)
    assert summary["delta_lora_minus_base"]["f0_mean_hz"] == pytest.approx(20.0)
    assert summary["paired_delta_lora_minus_base"]["wer"]["mean"] == pytest.approx(-0.3)
    assert summary["paired_delta_lora_minus_base"]["f0_mean_hz"]["mean"] == pytest.approx(30.0)
    assert trainer.summarize_generation([])["status"] == "unavailable"


def test_evaluate_generation_runs_paired_base_and_lora(tmp_path, monkeypatch):
    import argparse

    import torch

    from voix.training.lora import inject_lora

    class FakeSynthesizer:
        def inference(self, text, ref_s, alpha, beta, diffusion_steps):
            return np.sin(np.linspace(0.0, 20.0, 2400)).astype(np.float32)

    model = torch.nn.Module()
    model.diffusion = torch.nn.Module()
    model.diffusion.unet = torch.nn.Sequential(torch.nn.Linear(2, 2))
    inject_lora(model.diffusion.unet, rank=1, alpha=1.0)
    monkeypatch.setattr(
        trainer,
        "extract_style",
        lambda _model, _path, _device: torch.zeros(1, 256),
    )

    import voix.evaluation.audio_metrics as audio_metrics
    import voix.evaluation.metrics as metrics
    import voix.models.tts_backend as backend

    monkeypatch.setattr(
        metrics,
        "prosody_statistics",
        lambda _path, sample_rate=24000: {
            "duration_seconds": 0.1,
            "f0_mean_hz": 100.0,
            "f0_std_hz": 5.0,
            "f0_range_hz": 20.0,
            "f0_voiced_ratio": 0.8,
            "energy_mean": 0.01,
            "energy_std": 0.002,
        },
    )
    monkeypatch.setattr(
        metrics,
        "speaker_similarity",
        lambda _a, _b, device="cpu": 0.5,
    )
    monkeypatch.setattr(
        audio_metrics,
        "transcribe_wer",
        lambda paths, references, model_size="base", device="cpu": {
            "status": "computed",
            "per_sample_wer": [0.1] * len(paths),
        },
    )
    monkeypatch.setattr(
        backend,
        "_lufs_normalize",
        lambda wav, sample_rate, target_lufs=-23.0: wav.unsqueeze(0)
        if wav.dim() == 1
        else wav,
    )

    args = argparse.Namespace(
        seed=1,
        device="cpu",
        eval_device="cpu",
        alpha=0.18,
        beta=0.22,
        eval_diffusion_steps=2,
        whisper_model="base",
    )
    records = [
        {
            "clip_id": "val/id00000/0",
            "speaker_id": "id00000",
            "text": "this is a test",
            "wav_path": str(tmp_path / "target.wav"),
            "ref_clip_id": "val/id00000/ref",
            "ref_wav_path": str(tmp_path / "ref.wav"),
        }
    ]
    result = trainer.evaluate_generation(
        FakeSynthesizer(), model, records, tmp_path / "out", args
    )
    assert result["status"] == "computed"
    assert len(result["rows"]) == 2
    assert {row["condition"] for row in result["rows"]} == {"base", "lora"}
    assert result["summary"]["conditions"]["lora"]["metrics"]["wer"]["mean"] == pytest.approx(0.1)
    assert result["summary"]["wer_status"]["base"]["status"] == "computed"
    assert (tmp_path / "out").is_dir()


def test_build_tokens_prepends_bos_and_rejects_long_sequences():
    class Converter:
        def phonemize(self, text):
            return "h ə l oʊ"

    class Cleaner:
        def __call__(self, text):
            return [5, 6, 7]

    tokens = trainer.build_tokens("hello", Converter(), Cleaner())
    assert tokens.tolist() == [0, 5, 6, 7]
    with pytest.raises(ValueError, match="too long"):
        trainer.build_tokens("hello", Converter(), Cleaner(), max_tokens=3)
