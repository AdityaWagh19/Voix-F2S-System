"""Tests for objective naturalness proxies and the paired prompt audit."""
from __future__ import annotations

import json
import math

import numpy as np
import pytest

from voix.evaluation.naturalness import (
    CLIP_METRICS,
    PROMPT_EFFECT_METRICS,
    extract_naturalness_metrics,
    load_prompt_pairs,
    paired_prompt_effects,
    summarize_prompt_effects,
    word_count,
)

SAMPLE_RATE = 24000


def _synthetic_speech(
    *,
    duration_s: float = 2.0,
    f0_hz: float = 150.0,
    gap: tuple[float, float] | None = (0.9, 1.3),
    sample_rate: int = SAMPLE_RATE,
) -> np.ndarray:
    """Harmonic source with vibrato, an amplitude envelope, and an optional gap."""
    time = np.arange(int(duration_s * sample_rate)) / sample_rate
    vibrato = 1.0 + 0.02 * np.sin(2.0 * np.pi * 5.0 * time)
    phase = 2.0 * np.pi * np.cumsum(f0_hz * vibrato) / sample_rate
    signal = np.zeros_like(time)
    for harmonic in range(1, 9):
        signal += (1.0 / harmonic) * np.sin(harmonic * phase)
    signal /= np.max(np.abs(signal)) + 1e-9
    envelope = np.ones_like(time)
    fade = int(0.01 * sample_rate)
    envelope[:fade] = np.linspace(0.0, 1.0, fade)
    envelope[-fade:] = np.linspace(1.0, 0.0, fade)
    signal *= envelope
    if gap is not None:
        start = int(gap[0] * sample_rate)
        end = int(gap[1] * sample_rate)
        signal[start:end] = 0.0
    rng = np.random.default_rng(0)
    signal += 1e-4 * rng.standard_normal(signal.size)
    return signal.astype(np.float32)


def test_word_count_keeps_contractions():
    assert word_count("I don't think it's ready.") == 5
    assert word_count("") == 0
    assert word_count(None) == 0


def test_real_prompt_config_has_ten_unique_pairs():
    pairs = load_prompt_pairs("configs/research/n0a_prompt_pairs.json")
    assert len(pairs) == 10
    assert len({pair["pair_id"] for pair in pairs}) == 10
    for pair in pairs:
        assert pair["formal"] != pair["conversational"]
        assert pair["formal"] and pair["conversational"]


def test_prompt_config_validation(tmp_path):
    valid = {
        "schema_version": "voix.n0a_prompts.v1",
        "pairs": [
            {
                "pair_id": "a",
                "formal": "Formal text.",
                "conversational": "Casual text.",
            }
        ],
    }
    path = tmp_path / "prompts.json"
    path.write_text(json.dumps(valid), encoding="utf-8")
    assert load_prompt_pairs(path)[0]["pair_id"] == "a"

    duplicate = dict(valid)
    duplicate["pairs"] = valid["pairs"] * 2
    path.write_text(json.dumps(duplicate), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        load_prompt_pairs(path)

    missing = dict(valid)
    missing["pairs"] = [{"pair_id": "a", "formal": "only formal"}]
    path.write_text(json.dumps(missing), encoding="utf-8")
    with pytest.raises(ValueError, match="formal and conversational"):
        load_prompt_pairs(path)


def test_metrics_capture_duration_pauses_f0_and_rate():
    signal = _synthetic_speech()
    metrics = extract_naturalness_metrics(
        signal,
        text="one two three four five six",
        sample_rate=SAMPLE_RATE,
    )
    assert set(metrics) == set(CLIP_METRICS)
    assert metrics["duration_s"] == pytest.approx(2.0, abs=0.01)
    assert metrics["speech_duration_s"] < metrics["duration_s"]
    assert metrics["pause_count"] >= 1.0
    assert metrics["pause_ratio"] > 0.0
    assert 120.0 < metrics["f0_median_hz"] < 180.0
    assert metrics["f0_std_st"] > 0.0
    assert metrics["f0_range_st"] >= 0.0
    assert metrics["energy_std_db"] > 0.0
    assert math.isfinite(metrics["spectral_tilt_db_per_khz"])
    assert metrics["spectral_tilt_db_per_khz"] < 0.0
    assert metrics["word_count"] == 6.0
    assert metrics["words_per_second"] > 0.0


def test_silent_clip_does_not_invent_f0():
    metrics = extract_naturalness_metrics(
        np.zeros(SAMPLE_RATE // 4, dtype=np.float32),
        text="silence",
        sample_rate=SAMPLE_RATE,
    )
    assert metrics["f0_median_hz"] is None
    assert metrics["f0_std_st"] is None
    assert metrics["words_per_second"] is None


def _audit_rows(deltas, *, n_seeds=3, seed_offsets=(-0.01, 0.0, 0.01)):
    rows = []
    for pair_index, delta in enumerate(deltas):
        baseline = 1.0 + 0.1 * pair_index
        for register in ("formal", "conversational"):
            shift = 0.0 if register == "formal" else delta
            for seed_index, offset in enumerate(seed_offsets[:n_seeds]):
                rows.append(
                    {
                        "prompt_pair_id": f"pair_{pair_index:02d}",
                        "register": register,
                        "seed": seed_index,
                        "f0_std_st": baseline + shift + offset,
                        "f0_range_st": 2.0 * baseline + shift + offset,
                        "words_per_second": 4.0 + shift + offset,
                        "energy_std_db": 6.0 + shift + offset,
                        "pause_ratio": 0.2 + shift + offset,
                        "spectral_tilt_db_per_khz": -8.0 + shift + offset,
                    }
                )
    return rows


def test_paired_effect_detects_consistent_shift_above_seed_noise():
    effects = paired_prompt_effects(
        _audit_rows([0.5] * 10),
        n_resamples=500,
        seed=0,
    )
    assert effects["n_pairs"] == 10
    entry = effects["metrics"]["f0_std_st"]
    assert entry["status"] == "computed"
    assert entry["mean_delta"] == pytest.approx(0.5)
    assert entry["ci_low"] == pytest.approx(0.5)
    assert entry["ci_high"] == pytest.approx(0.5)
    assert entry["detected"] is True
    assert entry["seed_noise_sd"] == pytest.approx(0.01)
    assert entry["effect_to_noise"] > 1.0
    assert entry["robustly_detected"] is True
    assert entry["direction"] == "conversational_higher"


def test_paired_effect_does_not_call_balanced_noise_a_shift():
    deltas = [0.2 if index % 2 == 0 else -0.2 for index in range(10)]
    effects = paired_prompt_effects(
        _audit_rows(deltas),
        n_resamples=500,
        seed=0,
    )
    entry = effects["metrics"]["f0_std_st"]
    assert entry["mean_delta"] == pytest.approx(0.0)
    assert entry["ci_low"] < 0.0 < entry["ci_high"]
    assert entry["detected"] is False
    assert entry["sign_consistency"] == pytest.approx(0.5)


def test_paired_effect_reports_unavailable_without_both_registers():
    rows = [
        {
            "prompt_pair_id": "pair_00",
            "register": "formal",
            "seed": 0,
            "f0_std_st": 1.0,
        }
    ]
    effects = paired_prompt_effects(rows, n_resamples=100)
    assert effects["n_pairs"] == 0
    assert effects["metrics"]["f0_std_st"]["status"] == "unavailable"
    assert effects["skipped_pairs"][0]["pair_id"] == "pair_00"


def test_duplicate_clip_rows_raise():
    rows = _audit_rows([0.1])[:2]
    rows.append(dict(rows[0]))
    with pytest.raises(ValueError, match="duplicate row"):
        paired_prompt_effects(rows, n_resamples=100)


def test_summary_counts_detected_and_robust_metrics():
    effects = paired_prompt_effects(_audit_rows([0.5] * 10), n_resamples=200)
    summary = summarize_prompt_effects(effects)
    assert summary["n_metrics_computed"] == len(PROMPT_EFFECT_METRICS)
    assert "f0_std_st" in summary["metrics_detected"]
    assert "f0_std_st" in summary["metrics_robustly_detected"]
