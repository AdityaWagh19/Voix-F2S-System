"""Tests for the multi-seed step-ablation aggregation.

No model is loaded here.  The synthetic reports exercise the pure aggregation
path so a future change cannot silently drop a condition or report a single
run as if it were a multi-seed mean.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "evaluation" / "run_step_ablation.py"

spec = importlib.util.spec_from_file_location("run_step_ablation", SCRIPT)
assert spec and spec.loader
script = importlib.util.module_from_spec(spec)
spec.loader.exec_module(script)


def _measurement(snr: float, floor: float, lufs: float) -> dict:
    raw = {
        "snr_proxy_db": snr,
        "noise_floor_dbfs": floor,
        "peak": 0.5,
        "clipping_fraction": 0.0,
    }
    saved = {
        "lufs": lufs,
        "noise_floor_dbfs": floor,
    }
    return {"raw": raw, "save_current": saved}


def _report(seed: int, text: str, snr: float, floor: float, lufs: float) -> dict:
    return {
        "seed": seed,
        "text": text,
        "conditions": [
            {
                "label": "neutral_steps5",
                "diffusion_steps": 5,
                "emotion": None,
                "dsp": "none",
                "measurement": _measurement(snr, floor, lufs),
            },
            {
                "label": "whispered_steps5",
                "diffusion_steps": 5,
                "emotion": "whispered",
                "dsp": "pitch+time_stretch+energy",
                "measurement": _measurement(snr - 5.0, floor - 1.0, lufs - 2.0),
            },
        ],
    }


def test_aggregate_averages_across_seeds_and_prompts():
    reports = [
        _report(42, script.PROMPTS["formal"], 27.0, -45.0, -23.0),
        _report(123, script.PROMPTS["conversational"], 29.0, -47.0, -23.0),
    ]

    result = script.aggregate(
        reports,
        seeds=[42, 123],
        prompts=script.PROMPTS,
        steps=[5],
        emotion="whispered",
    )

    neutral = result["by_condition"]["neutral_steps5"]
    assert neutral["n"] == 2
    assert neutral["raw_snr_proxy_db"] == {
        "n": 2,
        "mean": 28.0,
        "std": 1.0,
        "min": 27.0,
        "max": 29.0,
    }
    assert neutral["raw_noise_floor_dbfs"]["mean"] == -46.0
    assert neutral["saved_lufs"]["mean"] == -23.0

    whispered = result["by_condition"]["whispered_steps5"]
    assert whispered["raw_snr_proxy_db"]["mean"] == 23.0
    assert result["by_step"]["5"]["neutral_snr_proxy_db"]["mean"] == 28.0
    assert result["by_step"]["5"]["whispered_snr_proxy_db"]["mean"] == 23.0
    assert len(result["rows"]) == 4
    assert {row["prompt"] for row in result["rows"]} == {
        "formal",
        "conversational",
    }


def test_aggregate_keeps_unavailable_stats_explicit():
    result = script.aggregate(
        [],
        seeds=[42],
        prompts=script.PROMPTS,
        steps=[5],
        emotion="whispered",
    )

    assert result["by_step"]["5"]["neutral_snr_proxy_db"] == {
        "n": 0,
        "mean": None,
        "std": None,
        "min": None,
        "max": None,
    }
    assert result["rows"] == []
