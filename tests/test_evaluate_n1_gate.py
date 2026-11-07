"""Tests for the N1 gate evaluator."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "evaluation" / "evaluate_n1_gate.py"
PROSODY = (
    "f0_median_hz",
    "f0_range_st",
    "words_per_second",
    "energy_std_db",
    "pause_ratio",
    "spectral_tilt_db_per_khz",
)


def _load_module():
    spec = importlib.util.spec_from_file_location("evaluate_n1_gate", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _payload(tmp_path: Path, *, lora_metric: float, lora_wer: float, lora_sim: float):
    rows = []
    for index, speaker in enumerate(("S1", "S2", "S3", "S4")):
        clip = f"clip{index}"
        for condition in ("base", "lora"):
            wav = tmp_path / f"{clip}_{condition}.wav"
            wav.write_bytes(b"")
            rows.append(
                {
                    "clip_id": clip,
                    "condition": condition,
                    "speaker_id": speaker,
                    "wav_path": str(wav),
                    "text": "hello there",
                    "wer": 0.20 if condition == "base" else lora_wer,
                    "speaker_similarity_target": (
                        0.60 if condition == "base" else lora_sim
                    ),
                    "speaker_similarity_ref": (
                        0.60 if condition == "base" else lora_sim
                    ),
                }
            )
    manifest_rows = []
    for index, speaker in enumerate(("S1", "S2", "S3", "S4")):
        clip = f"clip{index}"
        target = tmp_path / f"{clip}_target.wav"
        target.write_bytes(b"")
        manifest_rows.append(
            {
                "clip_id": clip,
                "speaker_id": speaker,
                "wav_path": str(target),
                "text": "hello there",
            }
        )
    return {"rows": rows}, pd.DataFrame(manifest_rows), lora_metric


def _install_metric(module, monkeypatch, lora_metric: float):
    def fake_metrics(audio, *, text=None, sample_rate=24000):
        name = Path(audio).name
        if "target" in name:
            value = 5.0
        elif "lora" in name:
            value = lora_metric
        else:
            value = 20.0
        return {metric: value for metric in PROSODY}

    monkeypatch.setattr(module, "extract_naturalness_metrics", fake_metrics)


def test_gate_passes_when_lora_moves_toward_ami_without_identity_loss(
    tmp_path, monkeypatch
):
    module = _load_module()
    payload, manifest, lora_metric = _payload(
        tmp_path, lora_metric=10.0, lora_wer=0.22, lora_sim=0.58
    )
    _install_metric(module, monkeypatch, lora_metric)
    result = module.evaluate(
        payload, manifest, n_resamples=200, seed=0, min_paired_clips=4
    )
    gate = result["gate"]
    assert gate["wer_within_base_plus_0.05"] is True
    assert gate["target_ecapa_within_base_minus_0.05"] is True
    assert gate["reference_ecapa_within_base_minus_0.05"] is True
    assert gate["at_least_one_prosody_ci_below_zero"] is True
    assert gate["passed"] is True
    for metric in PROSODY:
        row = result["naturalness_metrics"][metric]
        assert row["mean_delta"] == -10.0
        assert row["ci_high"] < 0.0


def test_gate_fails_on_identity_loss(tmp_path, monkeypatch):
    module = _load_module()
    payload, manifest, lora_metric = _payload(
        tmp_path, lora_metric=10.0, lora_wer=0.22, lora_sim=0.40
    )
    _install_metric(module, monkeypatch, lora_metric)
    result = module.evaluate(
        payload, manifest, n_resamples=200, seed=0, min_paired_clips=4
    )
    assert result["gate"]["target_ecapa_within_base_minus_0.05"] is False
    assert result["gate"]["passed"] is False


def test_gate_fails_when_prosody_moves_away_from_ami(tmp_path, monkeypatch):
    module = _load_module()
    payload, manifest, lora_metric = _payload(
        tmp_path, lora_metric=30.0, lora_wer=0.22, lora_sim=0.58
    )
    _install_metric(module, monkeypatch, lora_metric)
    result = module.evaluate(
        payload, manifest, n_resamples=200, seed=0, min_paired_clips=4
    )
    assert result["gate"]["at_least_one_prosody_ci_below_zero"] is False
    assert result["gate"]["passed"] is False


def test_cluster_interval_reports_pairs_and_clusters_separately():
    module = _load_module()
    rows = [
        {"speaker_id": f"S{index % 4}", "delta": float(index)}
        for index in range(8)
    ]
    result = module._cluster_interval(rows, n_resamples=100, seed=0)
    assert result["n_pairs"] == 8
    assert result["n_clusters"] == 4
    assert result["n_speakers"] == 4


def test_gate_fails_on_wer_budget(tmp_path, monkeypatch):
    module = _load_module()
    payload, manifest, lora_metric = _payload(
        tmp_path, lora_metric=10.0, lora_wer=0.40, lora_sim=0.58
    )
    _install_metric(module, monkeypatch, lora_metric)
    result = module.evaluate(
        payload, manifest, n_resamples=200, seed=0, min_paired_clips=4
    )
    assert result["gate"]["wer_within_base_plus_0.05"] is False
    assert result["gate"]["passed"] is False


def test_gate_fails_when_paired_coverage_is_incomplete(tmp_path, monkeypatch):
    module = _load_module()
    payload, manifest, lora_metric = _payload(
        tmp_path, lora_metric=10.0, lora_wer=0.22, lora_sim=0.58
    )
    _install_metric(module, monkeypatch, lora_metric)
    result = module.evaluate(
        payload, manifest, n_resamples=200, seed=0, min_paired_clips=5
    )
    assert result["gate"]["paired_clip_coverage_sufficient"] is False
    assert result["gate"]["passed"] is False
