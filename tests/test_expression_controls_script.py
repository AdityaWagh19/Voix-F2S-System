"""Tests for the expression-control evaluation CLI.

The script is loaded by path so it can be tested without installing it as a
module. Whisper and the model-backed evaluators are replaced with deterministic
test doubles; no audio is transcribed and no checkpoint is loaded.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "evaluation" / "evaluate_expression_controls.py"

spec = importlib.util.spec_from_file_location(
    "evaluate_expression_controls", SCRIPT
)
assert spec and spec.loader
script = importlib.util.module_from_spec(spec)
spec.loader.exec_module(script)


def _touch(tmp_path: Path, name: str) -> Path:
    path = tmp_path / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"RIFF")
    return path


def _manifest(tmp_path: Path, *, reference: str = "The quick brown fox.") -> dict:
    _touch(tmp_path, "happy.wav")
    _touch(tmp_path, "irish.wav")
    return {
        "schema_version": "voix.expression.v1",
        "reference_text": reference,
        "samples": [
            {
                "audio": "happy.wav",
                "control": "emotion",
                "label": "happy",
                "intensity": 0.8,
            },
            {
                "audio": "irish.wav",
                "control": "accent",
                "label": "irish",
                "intensity": 0.8,
            },
        ],
    }


def _passing_report() -> dict:
    return {
        "prosody": {"status": "computed"},
        "speaker_preservation": {"status": "computed", "mean": 0.9},
        "emotion_classifier": {"status": "computed", "accuracy": 0.6},
        "accent_classifier": {"status": "computed", "accuracy": 0.6},
        "intensity_monotonicity": {
            "status": "computed",
            "conditions": {
                "emotion:happy": {
                    "metrics": {
                        "f0_mean_hz": {"rho": 0.9, "expected_direction": 1}
                    }
                }
            },
        },
        "mos": {"status": "unavailable", "reason": "awaiting real human ratings"},
    }


class TestComputeWerByCondition:
    def test_groups_by_condition_and_reports_mean_and_worst(
        self, tmp_path, monkeypatch
    ):
        _touch(tmp_path, "happy_a.wav")
        _touch(tmp_path, "happy_b.wav")
        _touch(tmp_path, "irish.wav")
        manifest = {
            "reference_text": "shared prompt",
            "samples": [
                {"audio": "happy_a.wav", "control": "emotion", "label": "happy"},
                {"audio": "happy_b.wav", "control": "emotion", "label": "happy"},
                {"audio": "irish.wav", "control": "accent", "label": "irish"},
            ],
        }
        calls = []

        def fake_transcribe(paths, references, *, model_size="base", device="cpu"):
            calls.append((list(paths), list(references), model_size, device))
            wer = 0.1 if len(paths) == 2 else 0.3
            return {"status": "computed", "wer": wer, "n": len(paths)}

        from voix.evaluation import audio_metrics

        monkeypatch.setattr(audio_metrics, "transcribe_wer", fake_transcribe)

        result = script.compute_wer_by_condition(
            manifest, root=tmp_path, device="cpu", model_size="tiny"
        )

        assert result["status"] == "computed"
        assert result["worst"] == 0.3
        assert result["mean"] == 0.2
        assert result["conditions"]["emotion:happy"]["wer"] == 0.1
        assert result["conditions"]["accent:irish"]["wer"] == 0.3
        assert len(calls) == 2
        assert all(call[1] == ["shared prompt", "shared prompt"] for call in calls if len(call[1]) == 2)
        assert all(call[2] == "tiny" for call in calls)

    def test_missing_audio_is_unavailable_and_other_groups_are_partial(
        self, tmp_path, monkeypatch
    ):
        _touch(tmp_path, "happy.wav")
        manifest = {
            "reference_text": "prompt",
            "samples": [
                {"audio": "happy.wav", "control": "emotion", "label": "happy"},
                {"audio": "missing.wav", "control": "accent", "label": "irish"},
            ],
        }

        def fake_transcribe(paths, references, *, model_size="base", device="cpu"):
            return {"status": "computed", "wer": 0.1, "n": len(paths)}

        from voix.evaluation import audio_metrics

        monkeypatch.setattr(audio_metrics, "transcribe_wer", fake_transcribe)

        result = script.compute_wer_by_condition(manifest, root=tmp_path)

        assert result["status"] == "partial"
        assert result["worst"] == 0.1
        assert result["conditions"]["accent:irish"]["status"] == "unavailable"
        assert "missing audio" in result["conditions"]["accent:irish"]["reason"]

    def test_missing_reference_text_is_unavailable(self, tmp_path, monkeypatch):
        _touch(tmp_path, "happy.wav")
        manifest = {
            "samples": [
                {"audio": "happy.wav", "control": "emotion", "label": "happy"}
            ]
        }
        from voix.evaluation import audio_metrics

        monkeypatch.setattr(
            audio_metrics,
            "transcribe_wer",
            lambda *args, **kwargs: {"status": "computed", "wer": 0.0},
        )

        result = script.compute_wer_by_condition(manifest, root=tmp_path)

        assert result["status"] == "unavailable"
        assert result["worst"] is None
        assert result["conditions"]["emotion:happy"]["reason"] == "missing reference_text"


class TestMain:
    def test_writes_report_and_requires_pass(self, tmp_path, monkeypatch):
        manifest_path = tmp_path / "manifest.json"
        manifest_path.write_text(
            json.dumps(_manifest(tmp_path)), encoding="utf-8"
        )
        output = tmp_path / "out" / "expression.json"
        monkeypatch.setattr(
            script, "evaluate_expression_manifest", lambda *a, **k: _passing_report()
        )

        code = script.main(
            [
                "--manifest",
                str(manifest_path),
                "--output",
                str(output),
                "--root",
                str(tmp_path),
                "--require-pass",
            ]
        )

        assert code == 0
        payload = json.loads(output.read_text(encoding="utf-8"))
        assert payload["manifest"] == str(manifest_path)
        assert payload["gates"]["status"] == "pass"
        assert payload["report"]["mos"]["status"] == "unavailable"

    def test_unavailable_gates_exit_nonzero_when_required(
        self, tmp_path, monkeypatch
    ):
        manifest_path = tmp_path / "manifest.json"
        manifest_path.write_text(
            json.dumps(_manifest(tmp_path)), encoding="utf-8"
        )
        report = _passing_report()
        report["accent_classifier"] = {
            "status": "unavailable",
            "reason": "no classifier supplied",
        }
        monkeypatch.setattr(
            script, "evaluate_expression_manifest", lambda *a, **k: report
        )

        code = script.main(
            [
                "--manifest",
                str(manifest_path),
                "--output",
                str(tmp_path / "expression.json"),
                "--root",
                str(tmp_path),
                "--require-pass",
            ]
        )

        assert code == 1

    def test_invalid_manifest_is_rejected(self, tmp_path):
        manifest_path = tmp_path / "manifest.json"
        manifest_path.write_text(json.dumps({"samples": []}), encoding="utf-8")
        with pytest.raises(ValueError, match="invalid expression manifest"):
            script.main(
                [
                    "--manifest",
                    str(manifest_path),
                    "--output",
                    str(tmp_path / "expression.json"),
                ]
            )
