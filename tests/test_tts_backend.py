"""Unit test suite for StyleTTS 2 Acoustic Synthesis Backend (Phase 4)."""

import pytest
import torch

from voix.models.tts_backend import StyleTTS2Backend


class TestStyleTTS2Backend:
    """Verify StyleTTS 2 interface, constants, and synthesis execution."""

    def test_import(self):
        from voix.models import StyleTTS2Backend
        assert StyleTTS2Backend is not None

    def test_constants(self):
        assert StyleTTS2Backend.STYLE_DIM == 128
        assert StyleTTS2Backend.SAMPLE_RATE == 24000

    def test_instantiation(self):
        backend = StyleTTS2Backend(device="cpu")
        assert backend.sample_rate == 24000
        assert backend.style_dim == 128
        assert not backend._loaded

    def test_synthesize_shape_and_dtype(self):
        backend = StyleTTS2Backend(device="cpu")
        style = torch.randn(1, 128)
        wav = backend.synthesize("Hello world, this is VOIX.", style=style)

        assert isinstance(wav, torch.Tensor)
        assert wav.dim() == 2
        assert wav.shape[0] == 1
        assert wav.shape[1] > 10_000  # More than 0.4s at 24 kHz
        assert wav.dtype == torch.float32

    def test_synthesize_1d_vector(self):
        backend = StyleTTS2Backend(device="cpu")
        style = torch.randn(128)
        wav = backend.synthesize("Testing one-dimensional style vector input.", style=style)

        assert wav.shape[0] == 1
        assert wav.shape[1] > 0

    def test_amplitude_clamped(self):
        backend = StyleTTS2Backend(device="cpu")
        style = torch.randn(1, 128) * 10.0
        wav = backend.synthesize("Checking amplitude bounding.", style=style)

        assert wav.max().item() <= 1.0
        assert wav.min().item() >= -1.0
        assert torch.isfinite(wav).all()

    def test_different_styles_different_audio(self):
        backend = StyleTTS2Backend(device="cpu")
        s1 = torch.ones(1, 128) * 2.0
        s2 = -torch.ones(1, 128) * 2.0

        wav1 = backend.synthesize("Consistent text prompt.", style=s1)
        wav2 = backend.synthesize("Consistent text prompt.", style=s2)

        # Min length matching
        min_len = min(wav1.shape[1], wav2.shape[1])
        assert not torch.allclose(wav1[:, :min_len], wav2[:, :min_len], atol=1e-3)
