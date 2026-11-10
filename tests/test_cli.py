"""CLI contract tests: parser bounds, preflight gating, and opt-in watermarking.

These tests deliberately use a fake service so they exercise argument handling
and routing without loading any neural checkpoint or touching the GPU.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from voix.cli import (
    _watermark_requested,
    build_parser,
    handle_capabilities,
    handle_preflight,
    handle_synth,
)
from voix.service import (
    SynthesisResponse,
    VoiceCandidate,
    validate_output_destination,
)


class FakeService:
    """Minimal stand-in for VoixService used by CLI routing tests."""

    def __init__(self, report=None, response=None):
        self.report = report or {
            "ok": True,
            "errors": [],
            "warnings": [],
            "checks": [],
            "device": "cpu",
        }
        self.response = response
        self.preflight_calls = []
        self.synth_calls = []

    def preflight(self, request=None):
        self.preflight_calls.append(request)
        return dict(self.report)

    def synthesize(self, request, progress_callback=None):
        self.synth_calls.append(request)
        return self.response

    def get_capabilities(self):
        return {
            "accents": ["us", "england", "indian_english", "australian"],
            "emotions": ["neutral", "happy", "sad", "angry"],
        }


def _success_response(device: str = "cpu") -> SynthesisResponse:
    return SynthesisResponse(
        request_id="test",
        face_image="face.jpg",
        text="Hello",
        candidates=[
            VoiceCandidate(
                index=1,
                waveform=np.zeros(2400, dtype=np.float32),
                duration_seconds=0.1,
                backend="test",
            )
        ],
        latency_seconds=0.01,
        execution_device=device,
    )


def test_help_lists_operational_commands(capsys):
    with pytest.raises(SystemExit) as exc:
        build_parser().parse_args(["--help"])
    assert exc.value.code == 0
    output = capsys.readouterr().out
    for command in (
        "synth",
        "batch",
        "interactive",
        "info",
        "capabilities",
        "preflight",
    ):
        assert command in output


def test_capabilities_json_reports_four_accents_and_emotions(capsys):
    code = handle_capabilities(FakeService(), as_json=True)
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["accents"] == ["us", "england", "indian_english", "australian"]
    assert payload["emotions"] == ["neutral", "happy", "sad", "angry"]


def test_preflight_json_is_machine_readable(capsys):
    service = FakeService()
    args = build_parser().parse_args(
        [
            "preflight",
            "-f",
            "face.jpg",
            "-o",
            "out.wav",
            "--accent",
            "england",
            "--emotion",
            "happy",
            "--json",
        ]
    )
    code = handle_preflight(args, service)
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["errors"] == []
    request = service.preflight_calls[0]
    assert request.accent == "england"
    assert request.emotion == "happy"
    assert request.watermark is False


def test_preflight_failure_returns_nonzero(capsys):
    service = FakeService(
        report={
            "ok": False,
            "errors": ["missing checkpoint"],
            "warnings": [],
            "checks": [],
            "device": "cpu",
        }
    )
    args = build_parser().parse_args(["preflight", "--json"])
    code = handle_preflight(args, service)
    assert code == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["errors"] == ["missing checkpoint"]


def test_multi_voice_output_must_be_a_directory(tmp_path):
    with pytest.raises(ValueError, match="Pass a directory"):
        validate_output_destination(tmp_path / "many.wav", num_voices=2)

    destination = tmp_path / "many"
    validate_output_destination(destination, num_voices=2)
    assert destination.is_dir()


def test_single_voice_output_creates_parent_directory(tmp_path):
    destination = tmp_path / "nested" / "one.wav"
    validate_output_destination(destination, num_voices=1)
    assert destination.parent.is_dir()


def test_output_rejects_existing_file_where_directory_expected(tmp_path):
    destination = tmp_path / "not_a_directory"
    destination.write_text("x", encoding="utf-8")
    with pytest.raises(NotADirectoryError):
        validate_output_destination(destination, num_voices=2)


@pytest.mark.parametrize(
    "argv",
    [
        ["synth", "-f", "face.jpg", "-v", "0"],
        ["synth", "-f", "face.jpg", "-v", "11"],
        ["synth", "-f", "face.jpg", "--temp", "0"],
        ["synth", "-f", "face.jpg", "--temp", "3.5"],
        ["synth", "-f", "face.jpg", "--accent", "klingon"],
        ["synth", "-f", "face.jpg", "--emotion", "elated"],
        ["synth", "-f", "face.jpg", "--emotion-intensity", "1.5"],
    ],
)
def test_out_of_range_arguments_are_rejected(argv):
    with pytest.raises(SystemExit) as exc:
        build_parser().parse_args(argv)
    assert exc.value.code == 2


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        (["synth", "-f", "face.jpg"], False),
        (["synth", "-f", "face.jpg", "--watermark"], True),
        (["synth", "-f", "face.jpg", "--no-watermark"], False),
    ],
)
def test_watermark_is_opt_in(argv, expected):
    args = build_parser().parse_args(argv)
    assert _watermark_requested(args) is expected


def test_device_flag_works_before_or_after_subcommand():
    parser = build_parser()
    assert parser.parse_args(["--device", "cpu", "synth", "-f", "face.jpg"]).device == "cpu"
    assert parser.parse_args(["synth", "-f", "face.jpg", "--device", "cuda"]).device == "cuda"
    assert parser.parse_args(["synth", "-f", "face.jpg"]).device is None


def test_synth_forwards_watermark_and_device(capsys):
    args = build_parser().parse_args(
        ["synth", "-f", "face.jpg", "--watermark", "--device", "cpu", "--json"]
    )
    service = FakeService(response=_success_response())
    code = handle_synth(args, service)
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["execution_device"] == "cpu"
    assert len(service.synth_calls) == 1
    assert service.synth_calls[0].watermark is True


def test_synth_defaults_to_no_watermark(capsys):
    args = build_parser().parse_args(["synth", "-f", "face.jpg", "--json"])
    service = FakeService(response=_success_response())
    code = handle_synth(args, service)
    assert code == 0
    json.loads(capsys.readouterr().out)
    assert service.synth_calls[0].watermark is False


def test_preflight_failure_prevents_synthesis(capsys):
    service = FakeService(
        report={
            "ok": False,
            "errors": ["bad output destination"],
            "warnings": [],
            "checks": [],
            "device": "cpu",
        }
    )
    args = build_parser().parse_args(["synth", "-f", "face.jpg", "-q"])
    code = handle_synth(args, service)
    assert code == 1
    assert service.synth_calls == []
    assert "bad output destination" in capsys.readouterr().err


def test_missing_text_file_fails_before_preflight(tmp_path, capsys):
    missing = tmp_path / "missing.txt"
    args = build_parser().parse_args(
        ["synth", "-f", "face.jpg", "--text-file", str(missing)]
    )
    service = FakeService()
    code = handle_synth(args, service)
    assert code == 1
    assert service.preflight_calls == []
    assert "Text file not found" in capsys.readouterr().err
