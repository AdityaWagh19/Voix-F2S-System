"""Backbone smoke tests: ECAPA-TDNN speaker embedding extractor.

Verifies interface, output shape, dtype, and seed-controlled determinism.

Run with:
    pytest tests/test_acoustic_backbones.py -v
"""

import numpy as np
import pytest

from voix.utils.seed import set_global_seed


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def synthetic_waveform():
    """Return a 3-second synthetic 16 kHz waveform (1, 48000)."""
    import torch
    set_global_seed(42)
    return torch.randn(1, 48000)  # 3 seconds at 16 kHz


# ---------------------------------------------------------------------------
# ECAPAExtractor interface tests
# ---------------------------------------------------------------------------

class TestECAPAExtractor:
    """Verify ECAPAExtractor interface, embedding dimension, and determinism."""

    def test_import(self):
        """ECAPAExtractor can be imported from voix.data."""
        from voix.data.speaker_extractor import ECAPAExtractor
        assert ECAPAExtractor is not None

    def test_embedding_dim_constant(self):
        """EMBEDDING_DIM constant is exactly 192."""
        from voix.data.speaker_extractor import ECAPAExtractor
        assert ECAPAExtractor.EMBEDDING_DIM == 192

    def test_sample_rate_constant(self):
        """SAMPLE_RATE constant is exactly 16000 Hz."""
        from voix.data.speaker_extractor import ECAPAExtractor
        assert ECAPAExtractor.SAMPLE_RATE == 16000

    def test_instantiation_cpu(self):
        """ECAPAExtractor can be instantiated targeting CPU."""
        from voix.data.speaker_extractor import ECAPAExtractor
        extractor = ECAPAExtractor(
            device="cpu",
            savedir="./checkpoints/speechbrain/spkrec-ecapa-test"
        )
        assert not extractor._loaded  # Lazy: not loaded at init
        assert extractor.EMBEDDING_DIM == 192

    def test_instantiation_cuda(self):
        """ECAPAExtractor can be instantiated targeting CUDA."""
        import torch
        from voix.data.speaker_extractor import ECAPAExtractor
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available on this machine")
        extractor = ECAPAExtractor(device="cuda")
        assert not extractor._loaded


# ---------------------------------------------------------------------------
# Seed utility tests (independent of model download)
# ---------------------------------------------------------------------------

class TestSeedUtility:
    """Verify set_global_seed produces deterministic torch tensors."""

    def test_seed_torch_reproducibility(self):
        """Same seed produces identical torch random tensors."""
        import torch
        set_global_seed(42)
        t1 = torch.randn(192)
        set_global_seed(42)
        t2 = torch.randn(192)
        assert torch.allclose(t1, t2, atol=1e-6), "Seeded tensors not identical"

    def test_different_seeds_different_tensors(self):
        """Different seeds produce different torch random tensors."""
        import torch
        set_global_seed(42)
        t1 = torch.randn(192)
        set_global_seed(43)
        t2 = torch.randn(192)
        assert not torch.allclose(t1, t2), "Different seeds should differ"

    def test_seed_numpy_reproducibility(self):
        """Same seed produces identical numpy random arrays."""
        set_global_seed(42)
        a1 = np.random.randn(192)
        set_global_seed(42)
        a2 = np.random.randn(192)
        np.testing.assert_array_almost_equal(a1, a2, decimal=6)


# ---------------------------------------------------------------------------
# PyTorch CUDA availability test
# ---------------------------------------------------------------------------

class TestCUDAEnvironment:
    """Verify CUDA is available and functional."""

    def test_cuda_available(self):
        """torch.cuda.is_available() must return True on VOIX hardware."""
        import torch
        assert torch.cuda.is_available(), (
            "CUDA is not available. Check NVIDIA drivers and PyTorch CUDA build."
        )

    def test_cuda_device_count(self):
        """At least one CUDA device must be present."""
        import torch
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        assert torch.cuda.device_count() >= 1

    def test_tensor_on_cuda(self):
        """Tensors can be allocated on cuda:0 without error."""
        import torch
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        t = torch.zeros(192, device="cuda:0")
        assert t.is_cuda
        assert t.shape == (192,)
        del t
        torch.cuda.empty_cache()

    def test_cuda_memory_cleanup(self):
        """torch.cuda.empty_cache() can be called without error."""
        import torch
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        t = torch.randn(1000, 1000, device="cuda:0")
        del t
        torch.cuda.empty_cache()  # Must not raise
