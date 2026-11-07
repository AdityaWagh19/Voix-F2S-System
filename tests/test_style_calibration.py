"""Tests for the experimental style-manifold calibration."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from voix.models.style_calibration import (
    CALIBRATION_SCHEMA,
    apply_translation,
    fit_translation,
    gender_for_code,
    load_calibration,
    save_calibration,
    validate_calibration,
    vector_sha256,
)


def _styles(mean: float, offsets, gender_bias: float = 0.0):
    base = np.full(256, mean, dtype=np.float64)
    rows = []
    for offset in offsets:
        row = base.copy()
        row[0] += offset
        row[1] += gender_bias
        rows.append(row)
    return np.asarray(rows)


def _calibration():
    face = {
        "female": _styles(1.0, [0.0, 0.5, -0.5]),
        "male": _styles(2.0, [0.0, 0.25, -0.25]),
    }
    real = {
        "female": _styles(3.0, [0.0, 0.5, -0.5]),
        "male": _styles(4.0, [0.0, 0.25, -0.25]),
    }
    return fit_translation(face, real, alpha=1.0)


def test_fit_recovers_gender_matched_means_and_deltas():
    calibration = _calibration()
    assert calibration["schema_version"] == CALIBRATION_SCHEMA
    female = calibration["by_gender"]["female"]
    male = calibration["by_gender"]["male"]
    assert female["delta_norm"] == pytest.approx(2.0 * np.sqrt(256), rel=1e-6)
    assert male["delta_norm"] == pytest.approx(2.0 * np.sqrt(256), rel=1e-6)
    assert female["cos_face_mean_real_mean"] == pytest.approx(1.0, rel=1e-6)
    assert female["face_mean_sha256"] == vector_sha256(
        np.asarray(female["face_mean"])
    )
    assert female["delta_sha256"] == vector_sha256(np.asarray(female["delta"]))
    validate_calibration(calibration)


def test_apply_translation_moves_mean_and_preserves_within_gender_differences():
    calibration = _calibration()
    face = torch.from_numpy(_styles(1.0, [0.0, 0.5, -0.5])).float()
    calibrated = apply_translation(face, calibration, "female", alpha=1.0)
    before = face - face.mean(dim=0, keepdim=True)
    after = calibrated - calibrated.mean(dim=0, keepdim=True)
    assert torch.allclose(before, after, atol=1e-6)
    shifted_mean = calibrated.mean(dim=0) - face.mean(dim=0)
    delta = torch.as_tensor(calibration["by_gender"]["female"]["delta"]).float()
    assert torch.allclose(shifted_mean, delta, atol=1e-6)
    # The calibrated mean is the real mean by construction.
    real_mean = torch.as_tensor(calibration["by_gender"]["female"]["real_mean"]).float()
    assert torch.allclose(calibrated.mean(dim=0), real_mean, atol=1e-6)


def test_alpha_override_and_zero_is_identity():
    calibration = _calibration()
    style = torch.zeros(1, 256)
    assert torch.equal(apply_translation(style, calibration, "male", alpha=0.0), style)
    shifted = apply_translation(style, calibration, "male", alpha=0.5)
    delta = torch.as_tensor(calibration["by_gender"]["male"]["delta"]).float()
    assert torch.allclose(shifted, 0.5 * delta, atol=1e-6)


def test_save_load_roundtrip(tmp_path: Path):
    calibration = _calibration()
    path = save_calibration(tmp_path / "calibration.json", calibration)
    loaded = load_calibration(path)
    assert loaded["schema_version"] == CALIBRATION_SCHEMA
    assert loaded["by_gender"]["female"]["delta"] == calibration["by_gender"]["female"]["delta"]
    assert json.loads(path.read_text(encoding="utf-8"))["method"] == calibration["method"]


def test_validation_rejects_wrong_width():
    calibration = _calibration()
    calibration["by_gender"]["female"]["delta"] = [0.0] * 128
    with pytest.raises(ValueError, match="expected \\(256,\\)"):
        validate_calibration(calibration)


def test_apply_rejects_wrong_style_width():
    calibration = _calibration()
    with pytest.raises(ValueError, match="last dimension 256"):
        apply_translation(torch.zeros(1, 128), calibration, "female")


def test_gender_code_mapping():
    assert gender_for_code(0) == "female"
    assert gender_for_code(1) == "male"
