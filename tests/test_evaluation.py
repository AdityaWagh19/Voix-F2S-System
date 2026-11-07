"""Unit tests for VOIX Evaluation Suite and Style-Space Modulation.

Tests:
  - Evaluation metrics: prosody_statistics, speaker_similarity, face_voice_consistency, evaluate_synthesis
  - Accent style deltas: get_accent_style_delta, apply_accent_style_delta
  - Emotion style deltas: get_emotion_style_delta, apply_emotion_style_delta
"""

from __future__ import annotations

import os
from pathlib import Path
import numpy as np
import pytest
import torch

from voix.evaluation.metrics import (
    prosody_statistics,
    speaker_similarity,
    face_voice_consistency,
    evaluate_synthesis,
)
from voix.models.accent_control import (
    AccentType,
    get_accent_style_delta,
    apply_accent_style_delta,
)
from voix.models.emotion_control import (
    EmotionType,
    get_emotion_style_delta,
    apply_emotion_style_delta,
)


class TestEvaluationMetrics:
    """Test suite for voix/evaluation/metrics.py."""

    def test_prosody_statistics_shape_and_keys(self):
        sr = 24000
        t = np.linspace(0, 1.0, sr, endpoint=False)
        wav = 0.5 * np.sin(2 * np.pi * 200 * t) + 0.2 * np.sin(2 * np.pi * 400 * t)

        stats = prosody_statistics(wav, sample_rate=sr)
        assert isinstance(stats, dict)
        assert "f0_mean_hz" in stats
        assert "f0_std_hz" in stats
        assert "f0_range_hz" in stats
        assert "f0_voiced_ratio" in stats
        assert "f0_smoothness" in stats
        assert "energy_mean" in stats
        assert "energy_std" in stats
        assert "duration_seconds" in stats
        assert stats["f0_voiced_ratio"] > 0.5

    def test_speaker_similarity_identical_wavs(self, tmp_path):
        import soundfile as sf
        sr = 24000
        t = np.linspace(0, 0.5, int(sr * 0.5), endpoint=False)
        wav = 0.3 * np.sin(2 * np.pi * 300 * t)
        p1 = tmp_path / "tone1.wav"
        p2 = tmp_path / "tone2.wav"
        sf.write(str(p1), wav, sr)
        sf.write(str(p2), wav, sr)

        sim = speaker_similarity(p1, p2)
        assert sim >= 0.85

    def test_evaluate_synthesis_report(self, tmp_path):
        import soundfile as sf
        sr = 24000
        t = np.linspace(0, 0.5, int(sr * 0.5), endpoint=False)
        wav = 0.3 * np.sin(2 * np.pi * 220 * t)
        p = tmp_path / "test_synth.wav"
        sf.write(str(p), wav, sr)

        report = evaluate_synthesis(p, reference_wav=p)
        assert "wav_path" in report
        assert "prosody" in report
        assert "speaker_similarity" in report
        assert report["speaker_similarity"] >= 0.85


class TestStyleSpaceModulation:
    """Test suite for style-space accent and emotion delta transformations."""

    def test_accent_style_delta_application(self):
        ref_s = torch.ones(1, 256)
        delta = get_accent_style_delta("british_rp")
        if delta is not None:
            assert delta.shape == (256,)
            s_mod = apply_accent_style_delta(ref_s, "british_rp", strength=0.5)
            assert s_mod.shape == (1, 256)
            assert torch.allclose(s_mod[:, :128], ref_s[:, :128])
            assert not torch.allclose(s_mod[:, 128:], ref_s[:, 128:])

    def test_accent_zero_strength_identity(self):
        ref_s = torch.randn(1, 256)
        s_mod = apply_accent_style_delta(ref_s, "british_rp", strength=0.0)
        assert torch.allclose(s_mod, ref_s)

    def test_accent_general_american_identity(self):
        ref_s = torch.randn(1, 256)
        s_mod = apply_accent_style_delta(ref_s, "general_american", strength=0.8)
        assert torch.allclose(s_mod, ref_s)

    def test_emotion_style_delta_application(self):
        ref_s = torch.ones(1, 256)
        delta = get_emotion_style_delta("happy")
        assert delta is not None
        assert delta.shape == (256,)

        s_mod = apply_emotion_style_delta(ref_s, "happy", intensity=0.6)
        assert s_mod.shape == (1, 256)
        assert torch.allclose(s_mod[:, :128], ref_s[:, :128])
        assert not torch.allclose(s_mod[:, 128:], ref_s[:, 128:])

    def test_emotion_neutral_identity(self):
        ref_s = torch.randn(1, 256)
        s_mod = apply_emotion_style_delta(ref_s, "neutral", intensity=0.7)
        assert torch.allclose(s_mod, ref_s)


class TestCfvnFaceScalerSelection:
    """The 512-D diagnostic CFVN must use the ArcFace scaler, not the 560-D one."""

    @staticmethod
    def _write_scaler(path: Path, dim: int, face_key: str) -> None:
        torch.save(
            {
                "schema_version": "FaceFeatureV2",
                "face_key": face_key,
                "feature_dim": dim,
                "face_mean": torch.zeros(dim),
                "face_std": torch.ones(dim),
                "n_train": 10,
                "n_train_speakers": 2,
            },
            path,
        )

    @staticmethod
    def _install_fake_extractor(monkeypatch, value: np.ndarray) -> None:
        import cv2

        from voix.data import face_extractor

        class _FakeArcFace:
            def __init__(self, *args, **kwargs):
                pass

            def extract_full_with_bbox(self, image):
                return value, 0, 30.0, None

        monkeypatch.setattr(cv2, "imread", lambda _path: np.zeros((8, 8, 3), dtype=np.uint8))
        monkeypatch.setattr(face_extractor, "ArcFaceExtractor", _FakeArcFace)

    def test_arcface_cfvn_uses_explicit_512d_scaler(self, tmp_path, monkeypatch):
        from types import SimpleNamespace

        from voix.evaluation.metrics import _face_embedding_for_cfvn

        scaler_path = tmp_path / "face_scaler_arcface_v2.pt"
        self._write_scaler(scaler_path, 512, "arcface")
        value = np.linspace(-1.0, 1.0, 512, dtype=np.float32)
        self._install_fake_extractor(monkeypatch, value)

        net = SimpleNamespace(face_dim=512)
        face = _face_embedding_for_cfvn(
            tmp_path / "face.png",
            net,
            scaler_path,
            "cpu",
        )

        assert face is not None
        assert tuple(face.shape) == (1, 512)
        assert torch.allclose(face.squeeze(0), torch.from_numpy(value))

    def test_arcface_cfvn_switches_from_canonical_560d_scaler(
        self, tmp_path, monkeypatch
    ):
        from types import SimpleNamespace

        from voix.evaluation import metrics as metrics_module

        canonical = tmp_path / "face_scaler_v2.pt"
        arcface = tmp_path / "face_scaler_arcface_v2.pt"
        self._write_scaler(canonical, 560, "face_features")
        self._write_scaler(arcface, 512, "arcface")
        monkeypatch.setattr(metrics_module, "DEFAULT_SCALER", canonical)
        monkeypatch.setattr(metrics_module, "DEFAULT_SCALER_ARCFACE", arcface)

        value = np.full(512, 0.5, dtype=np.float32)
        self._install_fake_extractor(monkeypatch, value)

        # Caller passes the default 560-D path; the function must redirect the
        # 512-D model to the ArcFace scaler instead of failing.
        face = metrics_module._face_embedding_for_cfvn(
            tmp_path / "face.png",
            SimpleNamespace(face_dim=512),
            canonical,
            "cpu",
        )

        assert face is not None
        assert tuple(face.shape) == (1, 512)

    def test_missing_arcface_scaler_returns_none(self, tmp_path, monkeypatch):
        from types import SimpleNamespace

        from voix.evaluation import metrics as metrics_module

        canonical = tmp_path / "face_scaler_v2.pt"
        self._write_scaler(canonical, 560, "face_features")
        monkeypatch.setattr(metrics_module, "DEFAULT_SCALER", canonical)
        monkeypatch.setattr(
            metrics_module, "DEFAULT_SCALER_ARCFACE", tmp_path / "missing.pt"
        )
        self._install_fake_extractor(
            monkeypatch, np.zeros(512, dtype=np.float32)
        )

        face = metrics_module._face_embedding_for_cfvn(
            tmp_path / "face.png",
            SimpleNamespace(face_dim=512),
            canonical,
            "cpu",
        )
        assert face is None


class TestCfvnCache:
    def test_cfvn_checkpoint_is_loaded_once_per_path_and_device(
        self, tmp_path, monkeypatch
    ):
        from voix.evaluation import metrics as metrics_module

        calls = []

        class FakeCfvn:
            @staticmethod
            def from_checkpoint(path, device="cpu"):
                calls.append((path, device))
                return object()

        monkeypatch.setattr(
            "voix.models.contrastive_net.ContrastiveFaceVoiceNet", FakeCfvn
        )
        metrics_module._CFVN_CACHE.clear()
        checkpoint = tmp_path / "cfvn.pt"
        checkpoint.touch()

        first = metrics_module._get_cfvn(checkpoint, "cpu")
        second = metrics_module._get_cfvn(checkpoint, "cpu")
        metrics_module._get_cfvn(checkpoint, "cuda")

        assert first is second
        assert len(calls) == 2
        assert calls[0][0] == str(checkpoint.resolve())
        assert calls[1][1] == "cuda"
