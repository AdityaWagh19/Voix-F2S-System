"""CVAE loss functions for VOIX Phase 3.

Implements:
  - CosineReconstructionLoss  : 1 - cosine_similarity(e_s_hat, e_s)
  - FreeBitsKLLoss            : sum_d max(lambda_fb, KL_d)
  - CVAELoss                  : combined loss with beta annealing
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Individual loss components
# ---------------------------------------------------------------------------

class CosineReconstructionLoss(nn.Module):
    """Directional reconstruction loss in ECAPA-TDNN embedding space.

    ECAPA embeddings live on a hypersphere where identity is purely angular.
    MSE penalises magnitude deviations that carry no acoustic meaning.
    Cosine distance directly optimises the direction of the predicted vector.

    Loss = mean(1 - cosine_similarity(e_s_hat, e_s))

    Range: [0, 2] where 0 = perfect alignment, 2 = anti-parallel.
    """

    def __init__(self, eps: float = 1e-8) -> None:
        super().__init__()
        self.eps = eps

    def forward(
        self,
        e_s_hat: torch.Tensor,   # (B, 192) predicted speaker embedding
        e_s:     torch.Tensor,   # (B, 192) ground-truth speaker embedding
    ) -> torch.Tensor:
        cos_sim = F.cosine_similarity(e_s_hat, e_s, dim=1, eps=self.eps)  # (B,)
        return (1.0 - cos_sim).mean()


class FreeBitsKLLoss(nn.Module):
    """Dimension-wise free-bits KL divergence.

    Prevents posterior collapse by guaranteeing each latent dimension
    incurs zero KL penalty until it encodes at least lambda_fb nats of
    information.

    KL_d = -0.5 * (1 + logvar_d - mu_d^2 - exp(logvar_d))

    Loss = sum_d max(lambda_fb, KL_d)     (per sample, then mean over batch)

    Args:
        lambda_fb: Free-bits threshold in nats per dimension. Default: 0.5.
    """

    def __init__(self, lambda_fb: float = 0.5, reduction: str = "mean") -> None:
        super().__init__()
        self.lambda_fb = lambda_fb
        self.reduction = reduction

    def forward(
        self,
        mu:     torch.Tensor,   # (B, latent_dim) posterior mean
        logvar: torch.Tensor,   # (B, latent_dim) posterior log-variance
        prior_mu:     torch.Tensor,  # (B, latent_dim) prior mean
        prior_logvar: torch.Tensor,  # (B, latent_dim) prior log-variance
    ) -> torch.Tensor:
        # KL between N(mu, sigma^2) and N(prior_mu, prior_sigma^2)
        # KL_d = 0.5 * [prior_logvar - logvar - 1
        #                + exp(logvar) / exp(prior_logvar)
        #                + (mu - prior_mu)^2 / exp(prior_logvar)]
        kl_per_dim = 0.5 * (
            prior_logvar - logvar - 1.0
            + torch.exp(logvar - prior_logvar)
            + (mu - prior_mu).pow(2) * torch.exp(-prior_logvar)
        )  # (B, latent_dim)

        # Free-bits clamp: zero gradient below threshold
        kl_clamped = torch.clamp(kl_per_dim, min=self.lambda_fb)  # (B, latent_dim)

        # Average or sum across latent dimensions
        if self.reduction == "mean":
            return kl_clamped.mean()
        return kl_clamped.sum(dim=1).mean()

    def active_units(
        self,
        mu:     torch.Tensor,
        logvar: torch.Tensor,
        prior_mu:     torch.Tensor,
        prior_logvar: torch.Tensor,
        threshold: float = 0.1,
    ) -> int:
        """Count latent dimensions with KL > threshold (active units).

        A healthy CVAE should have >= 32 active units out of 192 after training.
        Posterior collapse shows as near-zero active units.
        """
        with torch.no_grad():
            kl_per_dim = 0.5 * (
                prior_logvar - logvar - 1.0
                + torch.exp(logvar - prior_logvar)
                + (mu - prior_mu).pow(2) * torch.exp(-prior_logvar)
            )
            mean_kl = kl_per_dim.mean(dim=0)   # (latent_dim,)
            return int((mean_kl > threshold).sum().item())


# ---------------------------------------------------------------------------
# Combined CVAE loss with linear beta annealing
# ---------------------------------------------------------------------------

class CVAELoss(nn.Module):
    """Combined CVAE objective with cosine reconstruction and free-bits KL.

    Total loss = L_recon + beta(t) * L_KL

    where:
      L_recon = CosineReconstructionLoss(e_s_hat, e_s)
      L_KL    = FreeBitsKLLoss(mu, logvar, prior_mu, prior_logvar)
      beta(t) = beta_max * min(1, t / warmup_steps)

    Args:
        lambda_fb:     Free-bits threshold per latent dim. Default: 0.5.
        beta_max:      Maximum KL weight after warmup. Default: 1.0.
        warmup_steps:  Steps over which beta linearly ramps 0 -> beta_max.
                       Typically 30% of total training steps.
    """

    def __init__(
        self,
        lambda_fb:    float = 0.5,
        beta_max:     float = 1.0,
        warmup_steps: int   = 5000,
        reduction:    str   = "mean",
    ) -> None:
        super().__init__()
        self.recon_loss = CosineReconstructionLoss()
        self.kl_loss    = FreeBitsKLLoss(lambda_fb=lambda_fb, reduction=reduction)
        self.beta_max     = beta_max
        self.warmup_steps = warmup_steps
        self._step        = 0   # internal step counter (call .step() each train step)

    def beta(self, step: int | None = None) -> float:
        """Compute current beta value (linear warmup from 0 to beta_max)."""
        t = step if step is not None else self._step
        return self.beta_max * min(1.0, t / max(1, self.warmup_steps))

    def step(self) -> None:
        """Advance the internal step counter by 1."""
        self._step += 1

    def forward(
        self,
        e_s_hat:      torch.Tensor,                  # (B, 192) decoder output
        e_s:          torch.Tensor,                  # (B, 192) ground-truth embedding
        mu:           torch.Tensor,                  # (B, latent_dim) recognition posterior mu
        logvar:       torch.Tensor,                  # (B, latent_dim) recognition posterior logvar
        prior_mu:     torch.Tensor | None = None,    # (B, latent_dim) prior mu (default: zeros)
        prior_logvar: torch.Tensor | None = None,    # (B, latent_dim) prior logvar (default: zeros)
        step:         int | None = None,
    ) -> tuple[torch.Tensor, dict]:
        """Compute total loss and return per-component breakdown.

        Returns:
            (total_loss, metrics_dict)
            where metrics_dict contains:
              - loss_total  : scalar total loss
              - loss_recon  : cosine reconstruction loss
              - loss_kl     : free-bits KL loss
              - beta        : current annealing weight
              - active_units: number of active latent dims (KL > 0.1 nats)
        """
        if prior_mu is None:
            prior_mu = torch.zeros_like(mu)
        if prior_logvar is None:
            prior_logvar = torch.zeros_like(logvar)

        beta_val = self.beta(step)

        l_recon = self.recon_loss(e_s_hat, e_s)
        l_kl    = self.kl_loss(mu, logvar, prior_mu, prior_logvar)
        total   = l_recon + beta_val * l_kl

        au = self.kl_loss.active_units(mu, logvar, prior_mu, prior_logvar)

        metrics = {
            "loss_total":   total.item(),
            "loss_recon":   l_recon.item(),
            "loss_kl":      l_kl.item(),
            "beta":         beta_val,
            "active_units": au,
        }

        return total, metrics
