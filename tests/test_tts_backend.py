"""Unit test suite for StyleTTS 2 Acoustic Synthesis Backend (Phase 4).

These tests exercise the explicitly-requested offline fallback. Neural tests are
marked ``neural`` and only run when ``VOIX_RUN_NEURAL_TESTS=1`` is set, so a
missing StyleTTS2 install can never masquerade as a passing neural test.
"""

import os
import sys

import pytest
import torch

from voix.models.tts_backend import StyleTTS2Backend


def test_einops_torch_backend_is_primed_before_tensorflow():
    """The broken TF install must not capture einops' backend selection."""
    from einops import _backends

    assert torch.Tensor in _backends._type2backend


def _fallback_backend() -> StyleTTS2Backend:
    return StyleTTS2Backend(device="cpu", use_neural=False)


class TestStyleTTS2Backend:
    """Verify StyleTTS 2 interface, constants, and explicit fallback behavior."""

    def test_import(self):
        from voix.models import StyleTTS2Backend
        assert StyleTTS2Backend is not None

    def test_constants(self):
        assert StyleTTS2Backend.STYLE_DIM == 128
        assert StyleTTS2Backend.SAMPLE_RATE == 24000

    def test_instantiation(self):
        backend = _fallback_backend()
        assert backend.sample_rate == 24000
        assert backend.style_dim in (128, 256)
        assert not backend._loaded
        assert backend.backend_name == "unloaded"

    def test_compute_style_requires_neural_backend(self):
        backend = _fallback_backend()
        with pytest.raises(RuntimeError, match="neural backend"):
            backend.compute_style("unused.wav")

    def test_synthesize_shape_and_dtype(self):
        backend = _fallback_backend()
        style = torch.randn(1, 128)
        wav = backend.synthesize("Hello world, this is VOIX.", style=style)

        assert isinstance(wav, torch.Tensor)
        assert wav.dim() == 2
        assert wav.shape[0] == 1
        assert wav.shape[1] > 10_000  # More than 0.4s at 24 kHz
        assert wav.dtype == torch.float32
        assert backend.backend_name == "fallback"

    def test_synthesize_1d_vector(self):
        backend = _fallback_backend()
        style = torch.randn(128)
        wav = backend.synthesize("Testing one-dimensional style vector input.", style=style)

        assert wav.shape[0] == 1
        assert wav.shape[1] > 0

    def test_amplitude_clamped(self):
        backend = _fallback_backend()
        style = torch.randn(1, 128) * 10.0
        wav = backend.synthesize("Checking amplitude bounding.", style=style)

        assert wav.max().item() <= 1.0
        assert wav.min().item() >= -1.0
        assert torch.isfinite(wav).all()

    def test_different_styles_different_audio(self):
        backend = _fallback_backend()
        s1 = torch.ones(1, 128) * 2.0
        s2 = -torch.ones(1, 128) * 2.0

        wav1 = backend.synthesize("Consistent text prompt.", style=s1)
        wav2 = backend.synthesize("Consistent text prompt.", style=s2)

        min_len = min(wav1.shape[1], wav2.shape[1])
        assert not torch.allclose(wav1[:, :min_len], wav2[:, :min_len], atol=1e-3)

    def test_missing_neural_model_raises_by_default(self, monkeypatch):
        """A broken neural install must not silently become the fallback."""
        monkeypatch.setitem(sys.modules, "styletts2", None)
        backend = StyleTTS2Backend(device="cpu", use_neural=True)
        with pytest.raises(RuntimeError, match="Failed to load the neural StyleTTS2"):
            backend.load()
        assert backend.backend_name == "unloaded"

    def test_explicit_fallback_opt_in(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "styletts2", None)
        backend = StyleTTS2Backend(device="cpu", use_neural=True, allow_fallback=True)
        backend.load()
        assert backend.backend_name == "fallback"
        assert backend.last_error is not None

    def test_duration_scale_changes_length(self):
        backend = _fallback_backend()
        style = torch.zeros(1, 128)
        short = backend.synthesize("A fixed sentence for duration.", style=style, duration_scale=0.5)
        long = backend.synthesize("A fixed sentence for duration.", style=style, duration_scale=2.0)
        assert long.shape[1] > short.shape[1] * 2

    def _measure_f0(self, wav: torch.Tensor, sample_rate: int = 24000) -> float:
        import librosa
        import numpy as np

        mono = wav.squeeze(0).detach().cpu().numpy().astype("float32")
        f0, voiced, _ = librosa.pyin(
            mono, fmin=50.0, fmax=600.0, sr=sample_rate, frame_length=2048
        )
        values = f0[voiced]
        assert values.size > 0
        return float(np.nanmedian(values))

    def test_f0_scale_changes_pitch_measurably(self):
        backend = _fallback_backend()
        style = torch.zeros(1, 128)
        base = backend.synthesize(
            "Measuring pitch control.", style=style, f0_scale=1.0
        )
        higher = backend.synthesize(
            "Measuring pitch control.", style=style, f0_scale=1.5
        )
        assert self._measure_f0(higher) > self._measure_f0(base) * 1.25

    def test_target_f0_sets_pitch_measurably(self):
        backend = _fallback_backend()
        style = torch.zeros(1, 128)
        wav = backend.synthesize(
            "Measuring absolute pitch.", style=style, target_f0=220.0
        )
        assert abs(self._measure_f0(wav) - 220.0) < 15.0

    def test_lufs_target_is_measurable(self):
        import pyloudnorm as pyln

        backend = _fallback_backend()
        style = torch.zeros(1, 128)
        target = -20.0
        wav = backend.synthesize(
            "Measuring loudness normalization.",
            style=style,
            loudness_lufs=target,
        )
        meter = pyln.Meter(backend.sample_rate)
        loudness = meter.integrated_loudness(
            wav.squeeze(0).detach().cpu().numpy().astype("float64")
        )
        assert abs(loudness - target) <= 1.5

    def test_lufs_normalization_uses_true_peak_limiter(self):
        import numpy as np

        from voix.models.tts_backend import _lufs_normalize
        from voix.utils.audio import true_peak_dbfs

        sample_rate = 24000
        t = np.arange(sample_rate, dtype=np.float32) / sample_rate
        hot = torch.from_numpy(
            (3.0 * np.sin(2.0 * np.pi * 220.0 * t)).astype(np.float32)
        ).unsqueeze(0)

        wav = _lufs_normalize(hot, sample_rate, target_lufs=-10.0)

        assert true_peak_dbfs(wav.numpy()) <= -0.99
        assert float(wav.abs().max()) < 1.0

    def test_lufs_normalization_does_not_boost_near_silence(self):
        from voix.models.tts_backend import _lufs_normalize

        silence = torch.zeros(1, 24000)
        wav = _lufs_normalize(silence, 24000, target_lufs=-23.0)

        assert torch.equal(wav, silence)


@pytest.mark.neural
@pytest.mark.skipif(
    os.environ.get("VOIX_RUN_NEURAL_TESTS") != "1",
    reason="Set VOIX_RUN_NEURAL_TESTS=1 to run real StyleTTS2 tests",
)
def test_neural_synthesis_runs():
    backend = StyleTTS2Backend(device="cpu", use_neural=True)
    backend.load()
    assert backend.backend_name == "neural"
    wav = backend.synthesize("Neural backend integration test.", style=torch.zeros(1, 256))
    assert wav.shape[0] == 1 and wav.shape[1] > 0
