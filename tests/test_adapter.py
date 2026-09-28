"""Unit test suite for VOIX Style Projection Adapter (Phase 4)."""

import pytest
import torch
import torch.nn as nn

from voix.models.adapter import StyleAdapter


class TestStyleAdapter:
    """Test StyleAdapter shapes, gradient flow, and normalization."""

    def test_parameter_count_and_shapes(self):
        adapter = StyleAdapter(in_dim=192, hidden_dim=256, out_dim=128)
        total_params = sum(p.numel() for p in adapter.parameters())
        trainable = sum(p.numel() for p in adapter.parameters() if p.requires_grad)

        assert total_params == trainable
        assert 100_000 < total_params < 200_000, f"Expected ~148k parameters, got {total_params}"

    def test_forward_batch_2d(self):
        adapter = StyleAdapter(in_dim=192, hidden_dim=256, out_dim=128)
        B = 16
        e_s = torch.randn(B, 192)
        s_hat = adapter(e_s)

        assert s_hat.shape == (B, 128)
        assert torch.isfinite(s_hat).all()

    def test_forward_single_vector_1d(self):
        adapter = StyleAdapter(in_dim=192, hidden_dim=256, out_dim=128)
        e_s = torch.randn(192)
        s_hat = adapter(e_s)

        assert s_hat.shape == (128,)
        assert torch.isfinite(s_hat).all()

    def test_gradient_flow(self):
        adapter = StyleAdapter(in_dim=192, hidden_dim=256, out_dim=128)
        B = 8
        e_s = torch.randn(B, 192, requires_grad=True)
        target = torch.randn(B, 128)

        s_hat = adapter(e_s)
        loss = nn.functional.mse_loss(s_hat, target)
        loss.backward()

        for name, p in adapter.named_parameters():
            assert p.grad is not None, f"No gradient for {name}"
            assert torch.isfinite(p.grad).all(), f"Non-finite gradient in {name}"

        assert e_s.grad is not None

    def test_output_normalization(self):
        adapter = StyleAdapter(in_dim=192, hidden_dim=256, out_dim=128, normalize_output=True)
        B = 4
        e_s = torch.randn(B, 192)
        s_hat = adapter(e_s)

        norms = torch.norm(s_hat, p=2, dim=-1)
        assert torch.allclose(norms, torch.ones(B), atol=1e-5)

    def test_numerical_stability(self):
        adapter = StyleAdapter(in_dim=192, hidden_dim=256, out_dim=128)
        extreme = torch.cat([
            torch.zeros(4, 192),
            torch.ones(4, 192) * 50.0,
            torch.randn(4, 192) * 10.0,
        ], dim=0)

        out = adapter(extreme)
        assert torch.isfinite(out).all()
