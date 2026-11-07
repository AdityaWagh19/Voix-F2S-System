"""Tests for the high-level VoixService DTO contract (no model loading)."""

from pathlib import Path

import numpy as np
import pytest

from voix.service import SynthesisRequest, SynthesisResponse, VoiceCandidate


def test_preset_id_is_valid_face_input():
    req = SynthesisRequest(face_image="young_female_resonant", text="Hello")
    req.validate()  # must not raise FileNotFoundError


def test_unknown_face_input_is_rejected():
    req = SynthesisRequest(face_image="not_a_file_or_preset", text="Hello")
    with pytest.raises(FileNotFoundError, match="not a known demographic preset"):
        req.validate()


def test_save_all_single_wav(tmp_path):
    response = SynthesisResponse(
        request_id="x",
        face_image="preset",
        text="Hello",
        candidates=[VoiceCandidate(index=1, waveform=np.zeros(2400, dtype=np.float32))],
    )
    paths = response.save_all(tmp_path / "one.wav")
    assert len(paths) == 1
    assert paths[0].name == "one.wav"
    assert paths[0].exists()


def test_save_all_multiple_requires_directory(tmp_path):
    response = SynthesisResponse(
        request_id="x",
        face_image="preset",
        text="Hello",
        candidates=[
            VoiceCandidate(index=1, waveform=np.zeros(2400, dtype=np.float32)),
            VoiceCandidate(index=2, waveform=np.zeros(2400, dtype=np.float32)),
        ],
    )
    with pytest.raises(ValueError, match="Pass a directory"):
        response.save_all(tmp_path / "many.wav")

    paths = response.save_all(tmp_path / "many")
    assert [p.name for p in paths] == ["voice_candidate_1.wav", "voice_candidate_2.wav"]
    assert all(p.exists() for p in paths)


def test_save_all_rejects_file_as_directory(tmp_path):
    target = tmp_path / "already_a_file"
    target.write_text("x", encoding="utf-8")
    response = SynthesisResponse(
        request_id="x",
        face_image="preset",
        text="Hello",
        candidates=[
            VoiceCandidate(index=1, waveform=np.zeros(2400, dtype=np.float32)),
            VoiceCandidate(index=2, waveform=np.zeros(2400, dtype=np.float32)),
        ],
    )
    with pytest.raises(NotADirectoryError):
        response.save_all(target)
