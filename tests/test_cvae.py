"""Unit test suite for VOIX CVAE Architecture (Phase 3)."""

import pytest
import torch
import torch.nn as nn

from voix.models.cvae import CVAE, CVAEEncoder, CVAEDecoder, CVAEOutput
from voix.training.losses import CVAELoss, CosineReconstructionLoss, FreeBitsKLLoss


class TestCVAEArchitecture:
    """Test structural definitions, parameter shapes, and forward/backward passes."""

    def test_parameter_count_and_shapes(self):
        model = CVAE(face_dim=560, speaker_dim=192, latent_dim=192)
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        assert total_params == trainable_params
        assert total_params > 500_000, f"Expected >500k parameters, got {total_params}"

        # Verify encoder heads
        assert model.encoder.fc_mu.out_features == 192
        assert model.encoder.fc_logvar.out_features == 192

    def test_forward_pass_training_mode(self):
        model = CVAE(face_dim=560, speaker_dim=192, latent_dim=192)
        model.train()

        B = 16
        e_f = torch.randn(B, 560)
        out = model(e_f)

        assert isinstance(out, CVAEOutput)
        assert out.e_s_hat.shape == (B, 192)
        assert out.mu.shape == (B, 192)
        assert out.logvar.shape == (B, 192)
        assert out.z.shape == (B, 192)

        # Reparameterization noise should be present during training
        assert not torch.allclose(out.z, out.mu, atol=1e-5)

    def test_forward_pass_deterministic_mode(self):
        model = CVAE(face_dim=560, speaker_dim=192, latent_dim=192)
        model.eval()

        B = 8
        e_f = torch.randn(B, 560)

        out1 = model(e_f, deterministic=True)
        out2 = model(e_f, deterministic=True)

        assert torch.allclose(out1.z, out1.mu)
        assert torch.allclose(out1.e_s_hat, out2.e_s_hat)

    def test_numerical_stability(self):
        model = CVAE(face_dim=560, speaker_dim=192, latent_dim=192)
        model.train()

        # Test extreme large inputs and zeros
        e_f_extreme = torch.cat([
            torch.zeros(4, 560),
            torch.ones(4, 560) * 100.0,
            torch.randn(4, 560) * 10.0,
        ], dim=0)

        out = model(e_f_extreme)
        assert torch.isfinite(out.e_s_hat).all()
        assert torch.isfinite(out.mu).all()
        assert torch.isfinite(out.logvar).all()
        assert (out.logvar >= -10.0).all() and (out.logvar <= 10.0).all()

    def test_gradient_flow(self):
        model = CVAE(face_dim=560, speaker_dim=192, latent_dim=192)
        model.train()

        B = 8
        e_f = torch.randn(B, 560, requires_grad=True)
        e_s_gt = torch.randn(B, 192)

        out = model(e_f)
        recon_loss = 1.0 - torch.nn.functional.cosine_similarity(out.e_s_hat, e_s_gt).mean()
        kl_loss = -0.5 * torch.mean(1 + out.logvar - out.mu.pow(2) - out.logvar.exp())
        loss = recon_loss + kl_loss

        loss.backward()

        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert torch.isfinite(param.grad).all(), f"Non-finite gradient in {name}"

        assert e_f.grad is not None

    def test_sample_voices_diversity(self):
        model = CVAE(face_dim=560, speaker_dim=192, latent_dim=192)
        model.eval()

        e_f = torch.randn(560)
        K = 7
        candidates = model.sample_voices(e_f, num_samples=K, temperature=1.0)

        assert candidates.shape == (K, 192)
        assert torch.isfinite(candidates).all()

        # Candidates must be distinct
        pairwise_diffs = torch.cdist(candidates, candidates)
        # Upper triangle without diagonal
        mask = torch.triu(torch.ones(K, K, dtype=torch.bool), diagonal=1)
        min_dist = pairwise_diffs[mask].min().item()
        assert min_dist > 0.05, f"Candidate voices collapsed! Min dist: {min_dist}"

    def test_temperature_scaling(self):
        model = CVAE(face_dim=560, speaker_dim=192, latent_dim=192)
        model.eval()

        e_f = torch.randn(560)
        low_t = model.sample_voices(e_f, num_samples=100, temperature=0.1)
        high_t = model.sample_voices(e_f, num_samples=100, temperature=2.0)

        var_low = torch.var(low_t, dim=0).mean().item()
        var_high = torch.var(high_t, dim=0).mean().item()

        assert var_high > var_low * 10, f"Expected var_high >> var_low, got {var_high} vs {var_low}"

    def test_cvae_loss_integration(self):
        model = CVAE(face_dim=560, speaker_dim=192, latent_dim=192)
        cvae_loss_fn = CVAELoss(lambda_fb=0.5, beta_max=1.0, warmup_steps=1000)

        B = 16
        e_f = torch.randn(B, 560)
        e_s = torch.randn(B, 192)

        out = model(e_f)
        total_loss, metrics = cvae_loss_fn(
            e_s_hat=out.e_s_hat,
            e_s=e_s,
            mu=out.mu,
            logvar=out.logvar,
            step=500,
        )

        assert total_loss.ndim == 0
        assert total_loss.item() > 0
        assert "loss_recon" in metrics
        assert "loss_kl" in metrics
        assert "active_units" in metrics
        assert metrics["beta"] == 0.5
