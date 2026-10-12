"""Unified True Conditional Variational Autoencoder (CVAE v2) for VOIX.

Implements principled conditional latent variable modeling (Sohn et al., 2015):
  - Recognition Network q_phi(z | e_s, e_f): Learns joint posterior during training.
  - Prior Network p_psi(z | e_f): Predicts conditional prior distribution from facial features alone.
  - Conditional Decoder p_theta(e_s | z, e_f): Reconstructs 192-D speaker embeddings
    conditioned jointly on sampled latent z and input face e_f.
  - sample_voices(): Generates K diverse, plausible candidate voices for a single face image.

Architecture notes:
  - GroupNorm instead of BatchNorm: works at any batch size including batch_size=1 at inference.
  - No 'encoder' alias: the prior_net is the only encoder. Tests must reference prior_net directly.
"""

from __future__ import annotations

from typing import NamedTuple, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class CVAEOutput(NamedTuple):
    """Container for CVAE forward pass outputs."""
    e_s_hat: torch.Tensor          # Reconstructed speaker embedding (B, 192)
    mu: torch.Tensor               # Latent mean (B, 192)
    logvar: torch.Tensor           # Latent log-variance log(sigma^2) (B, 192)
    z: torch.Tensor                # Sampled latent vector (B, 192)
    prior_mu: Optional[torch.Tensor] = None      # Conditional prior mean (B, 192)
    prior_logvar: Optional[torch.Tensor] = None  # Conditional prior logvar (B, 192)


def _make_mlp_block(in_dim: int, out_dim: int, dropout: float = 0.0) -> nn.Sequential:
    """Build a single MLP block: Linear -> GroupNorm -> GELU -> Dropout.

    Uses GroupNorm (not BatchNorm) so it works at batch_size=1 during inference.
    num_groups is clamped so that out_dim is divisible by it.
    """
    num_groups = min(32, out_dim // 8)
    # Ensure out_dim is divisible by num_groups
    while out_dim % num_groups != 0 and num_groups > 1:
        num_groups -= 1
    layers: list[nn.Module] = [
        nn.Linear(in_dim, out_dim),
        nn.GroupNorm(num_groups=num_groups, num_channels=out_dim),
        nn.GELU(),
    ]
    if dropout > 0.0:
        layers.append(nn.Dropout(p=dropout))
    return nn.Sequential(*layers)


class CVAEPriorNetwork(nn.Module):
    """Conditional prior network p_psi(z | e_f).

    Predicts prior distribution parameters (mu_prior, logvar_prior) from facial representation e_f.
    """

    def __init__(
        self,
        face_dim: int = 560,
        latent_dim: int = 192,
        hidden_dims: Tuple[int, ...] = (512, 384, 256),
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        blocks: list[nn.Module] = []
        prev_dim = face_dim
        for h_dim in hidden_dims:
            blocks.append(_make_mlp_block(prev_dim, h_dim, dropout))
            prev_dim = h_dim

        self.trunk = nn.Sequential(*blocks)
        self.fc_mu = nn.Linear(prev_dim, latent_dim)
        self.fc_logvar = nn.Linear(prev_dim, latent_dim)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.normal_(self.fc_logvar.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.fc_logvar.bias)

    def forward(self, e_f: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feat = self.trunk(e_f)
        mu = self.fc_mu(feat)
        logvar = torch.clamp(self.fc_logvar(feat), min=-10.0, max=10.0)
        return mu, logvar


class CVAERecognitionNetwork(nn.Module):
    """Recognition posterior network q_phi(z | e_s, e_f).

    Jointly encodes speaker embedding and face representation to produce
    the recognition posterior used during training.
    """

    def __init__(
        self,
        in_dim: int = 560 + 192,  # face_dim + speaker_dim
        latent_dim: int = 192,
        hidden_dims: Tuple[int, ...] = (512, 384, 256),
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        blocks: list[nn.Module] = []
        prev_dim = in_dim
        for h_dim in hidden_dims:
            blocks.append(_make_mlp_block(prev_dim, h_dim, dropout))
            prev_dim = h_dim

        self.trunk = nn.Sequential(*blocks)
        self.fc_mu = nn.Linear(prev_dim, latent_dim)
        self.fc_logvar = nn.Linear(prev_dim, latent_dim)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.normal_(self.fc_logvar.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.fc_logvar.bias)

    def forward(self, joint_in: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feat = self.trunk(joint_in)
        mu = self.fc_mu(feat)
        logvar = torch.clamp(self.fc_logvar(feat), min=-10.0, max=10.0)
        return mu, logvar


class CVAEDecoder(nn.Module):
    """Marginal decoder network p_theta(e_s | z).

    Reconstructs 192-D speaker embedding from latent z ONLY.
    Face features e_f are NOT passed to the decoder — they condition the prior
    and recognition networks only. This prevents posterior collapse via skip shortcut.
    """

    def __init__(
        self,
        in_dim: int = 192,  # latent_dim only — no e_f skip connection
        speaker_dim: int = 192,
        hidden_dims: Tuple[int, ...] = (512, 384, 256),  # larger since no skip
        normalize_output: bool = False,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.speaker_dim = speaker_dim
        self.normalize_output = normalize_output

        blocks: list[nn.Module] = []
        prev_dim = in_dim
        for h_dim in hidden_dims:
            blocks.append(_make_mlp_block(prev_dim, h_dim))
            prev_dim = h_dim

        self.net = nn.Sequential(*blocks)
        self.output_proj = nn.Linear(prev_dim, speaker_dim)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, cond_z: torch.Tensor) -> torch.Tensor:
        e_s_hat = self.output_proj(self.net(cond_z))
        if self.normalize_output:
            e_s_hat = F.normalize(e_s_hat, p=2, dim=-1)
        return e_s_hat


class CVAE(nn.Module):
    """Unified True Conditional Variational Autoencoder (CVAE v2).

    Args:
        face_dim: Input face feature dimensionality (default: 560).
        speaker_dim: Target speaker embedding dimensionality (default: 192).
        latent_dim: Latent space dimensionality (default: 192).
        dropout: Encoder dropout rate (default: 0.0).
        normalize_output: Whether to L2-normalize e_s_hat (default: False).
    """

    def __init__(
        self,
        face_dim: int = 560,
        speaker_dim: int = 192,
        latent_dim: int = 192,
        dropout: float = 0.0,
        normalize_output: bool = False,
    ) -> None:
        super().__init__()
        self.face_dim = face_dim
        self.speaker_dim = speaker_dim
        self.latent_dim = latent_dim

        # Conditional Prior p_psi(z | e_f)
        self.prior_net = CVAEPriorNetwork(
            face_dim=face_dim,
            latent_dim=latent_dim,
            hidden_dims=(512, 384, 256),
            dropout=dropout,
        )
        # Recognition Posterior q_phi(z | e_s, e_f)
        self.recognition_net = CVAERecognitionNetwork(
            in_dim=face_dim + speaker_dim,
            latent_dim=latent_dim,
            hidden_dims=(512, 384, 256),
            dropout=dropout,
        )
        # Marginal Decoder p_theta(e_s | z) — NO e_f skip connection.
        # Face information enters ONLY via prior/recognition networks.
        # Forcing e_s to be reconstructed from z alone prevents the decoder
        # from bypassing z (posterior collapse via skip shortcut).
        self.decoder = CVAEDecoder(
            in_dim=latent_dim,
            speaker_dim=speaker_dim,
            hidden_dims=(512, 384, 256),  # larger to compensate for removed skip
            normalize_output=normalize_output,
        )

    def reparameterize(
        self,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        temperature: float = 1.0,
        deterministic: bool = False,
    ) -> torch.Tensor:
        """Sample from latent distribution with reparameterization trick.

        Stochastic by default — deterministic=True ONLY when explicitly requested
        (e.g., for mode prediction). The self.training flag is intentionally NOT
        used here so that val/test evaluation can use stochastic prior samples.
        """
        if deterministic:
            return mu
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std * temperature

    def forward(
        self,
        e_f: torch.Tensor,
        e_s: Optional[torch.Tensor] = None,
        deterministic: bool = False,
        temperature: float = 1.0,
    ) -> CVAEOutput:
        """Forward pass.

        Args:
            e_f: Input face representations (B, 560).
            e_s: Optional ground-truth speaker embeddings (B, 192). If provided,
                 uses joint posterior q(z|e_s, e_f). If None, uses prior p(z|e_f).
            deterministic: If True, uses mean mu without sampling noise.
            temperature: Sampling temperature scaling.

        Returns:
            CVAEOutput namedtuple.
        """
        prior_mu, prior_logvar = self.prior_net(e_f)

        if e_s is not None:
            # Training: use recognition network for tighter posterior
            joint_in = torch.cat([e_s, e_f], dim=-1)
            mu, logvar = self.recognition_net(joint_in)
        else:
            # Inference: sample from learned conditional prior
            mu, logvar = prior_mu, prior_logvar

        z = self.reparameterize(mu, logvar, temperature=temperature, deterministic=deterministic)

        # Decode from z only — no e_f skip connection.
        # Face context is already encoded into z via the prior/recognition nets.
        e_s_hat = self.decoder(z)

        return CVAEOutput(
            e_s_hat=e_s_hat,
            mu=mu,
            logvar=logvar,
            z=z,
            prior_mu=prior_mu,
            prior_logvar=prior_logvar,
        )

    @torch.no_grad()
    def sample_voices(
        self,
        e_f: torch.Tensor,
        num_samples: int = 5,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Generate K diverse candidate voice embeddings for a single face image.

        Samples K distinct latents from the conditional prior p_psi(z | e_f),
        and decodes them conditioned on the input facial biometrics.

        Args:
            e_f: Face representation tensor of shape (1, 560) or (560,).
            num_samples: Number of diverse voice candidates K.
            temperature: Sampling temperature (higher = more diverse).

        Returns:
            Tensor of shape (K, 192) containing K candidate speaker embeddings.
        """
        self.eval()
        if e_f.ndim == 1:
            e_f = e_f.unsqueeze(0)  # (1, 560)

        # Predict conditional prior distribution from face
        mu_prior, logvar_prior = self.prior_net(e_f)  # (1, 192)
        std_prior = torch.exp(0.5 * logvar_prior)

        # Expand for K candidate voices
        mu_exp  = mu_prior.expand(num_samples, -1)   # (K, 192)
        std_exp = std_prior.expand(num_samples, -1)  # (K, 192)
        eps = torch.randn_like(std_exp)              # (K, 192)

        z = mu_exp + eps * std_exp * temperature     # (K, 192)

        # Decode from z only — no e_f skip
        e_s_candidates = self.decoder(z)              # (K, 192)
        return e_s_candidates


# ---------------------------------------------------------------------------
# Backward-compatibility shim for any external code that imports CVAEEncoder
# ---------------------------------------------------------------------------
CVAEEncoder = CVAEPriorNetwork
