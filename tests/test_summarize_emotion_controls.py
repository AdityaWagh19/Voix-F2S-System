"""Tests for the emotion controlled-comparison summary."""
from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "evaluation" / "summarize_emotion_controls.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("summarize_emotion_controls", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_summary_carries_wer_into_condition_and_emotion_rows():
    module = _load_module()
    report = {
        "run_id": "demo",
        "generation": {"scalar_modifiers": "neutral", "emotions_generated": ["happy"]},
        "gates": {
            "e0_directions": {
                "checks": {
                    "happy": {
                        "f0_mean_hz": {
                            "mean_delta": 12.0,
                            "ci_low": 4.0,
                            "ci_high": 20.0,
                            "passed": True,
                        }
                    }
                }
            },
            "neutral_zero": {"passed": True},
            "identity": {"passed": True},
        },
        "metrics": {
            "ecapa_generated_neutral": {
                "value": {"neutral:neutral": 1.0, "class_mean:happy": 0.9}
            },
            "classifier_accuracy": {"status": "unavailable"},
        },
        "wer": {
            "status": "computed",
            "mean": 0.1234,
            "worst": 0.2,
            "conditions": {
                "neutral:neutral": {"status": "computed", "wer": 0.1},
                "class_mean:happy": {"status": "computed", "wer": 0.1234},
            },
        },
    }
    summary = module.summarize(
        [{"label": "demo", "path": "demo.json", "report": report}]
    )
    condition = summary["conditions"]["demo"]
    assert condition["wer_mean"] == 0.1234
    assert condition["rows"]["happy"]["wer"]["wer"] == 0.1234
    text = module.markdown(summary)
    assert "| WER mean |" in text
    assert "| demo | neutral | 1/1 | True | True | 0.1234 |" in text
    assert "| demo | happy | +12.00 [+4.00, +20.00]" in text
    assert "| 0.1234 |" in text
