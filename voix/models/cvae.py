"""Conditional Variational Autoencoder (CVAE) Mapper (Phase 3).

Maps 560-D fused face representations (ArcFace 512-D + MediaPipe 32-D + Soft Demo 16-D)
into 192-D ECAPA-TDNN speaker embedding space via a latent space z in R^192.

System Pipeline:
    Face Image -> Unified Face Vector e_f (560-D)
               -> CVAE Encoder q_phi(z | e_f) -> mu, log_sigma^2 (192-D each)
               -> Latent Sampling: z = mu + sigma * epsilon, epsilon ~ N(0, I)
               -> CVAE Decoder p_theta(e_s_hat | z) -> Predicted Speaker Embedding e_s_hat (192-D)
               -> Downstream: StyleTTS2 Adapter -> Audio Waveform

Key Design Decisions (from architecture.md):
  - Latent dimension D_z = 192 matches ECAPA speaker embedding space.
  - Encoder uses BatchNorm1d + GELU activations.
  - Decoder reconstructs the 192-D ECAPA speaker vector.
  - Reparameterization includes variance clamping [-10, 10] to prevent numerical instability.
  - sample_voices() generates K diverse voice candidates for a single face at varying temperatures.
"""

from __future__ import annotations

from typing import NamedTuple, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class CVAEOutput(NamedTuple):
    """Container for CVAE forward pass outputs."""
    e_s_hat: torch.Tensor     # Reconstructed speaker embedding (B, 192)
    mu: torch.Tensor          # Latent mean (B, 192)
    logvar: torch.Tensor      # Latent log-variance log(sigma^2) (B, 192)
    z: torch.Tensor           # Sampled latent vector (B, 192)


class CVAEEncoder(nn.Module):
    """Encoder network q_phi(z | e_f).

    Maps 560-D face representation to latent Gaussian distribution parameters (mu, log_sigma^2).
    """

    def __init__(
        self,
        face_dim: int = 560,
        latent_dim: int = 192,
        hidden_dims: Tuple[int, ...] = (512, 384, 256),
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.face_dim = face_dim
        self.latent_dim = latent_dim

        # Trunk layers: 560 -> 512 -> 384 -> 256
        layers: list[nn.Module] = []
        prev_dim = face_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.GELU(),
            ])
            if dropout > 0.0:
                layers.append(nn.Dropout(p=dropout))
            prev_dim = h_dim

        self.trunk = nn.Sequential(*layers)

        # Projection heads
        self.fc_mu = nn.Linear(prev_dim, latent_dim)
        self.fc_logvar = nn.Linear(prev_dim, latent_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        # Small weights for logvar to start close to N(0, 1)
        nn.init.normal_(self.fc_logvar.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.fc_logvar.bias)

    def forward(self, e_f: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            e_f: Tensor of shape (B, 560)

        Returns:
            mu: Tensor of shape (B, 192)
            logvar: Tensor of shape (B, 192) clamped to [-10, 10]
        """
        features = self.trunk(e_f)
        mu = self.fc_mu(features)
        logvar = self.fc_logvar(features)
        logvar = torch.clamp(logvar, min=-10.0, max=10.0)
        return mu, logvar


class CVAEDecoder(nn.Module):
    """Decoder network p_theta(e_s_hat | z).

    Reconstructs 192-D ECAPA-TDNN speaker embedding from latent vector z.
    """

    def __init__(
        self,
        latent_dim: int = 192,
        speaker_dim: int = 192,
        hidden_dim: int = 256,
        normalize_output: bool = False,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.speaker_dim = speaker_dim
        self.normalize_output = normalize_output

        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, speaker_dim),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent z into reconstructed speaker embedding e_s_hat."""
        e_s_hat = self.net(z)
        if self.normalize_output:
            e_s_hat = F.normalize(e_s_hat, p=2, dim=-1)
        return e_s_hat


class CVAE(nn.Module):
    """Unified Conditional Variational Autoencoder Mapper.

    Args:
        face_dim: Input face feature dimensionality (default: 560).
        speaker_dim: Target speaker embedding dimensionality (default: 192).
        latent_dim: Latent space dimensionality (default: 192).
        dropout: Encoder dropout rate (default: 0.0).
        normalize_output: Whether to L2-normalize e_s_hat (default: False, preserves raw scale).
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

        self.encoder = CVAEEncoder(
            face_dim=face_dim,
            latent_dim=latent_dim,
            hidden_dims=(512, 384, 256),
            dropout=dropout,
        )
        self.decoder = CVAEDecoder(
            latent_dim=latent_dim,
            speaker_dim=speaker_dim,
            hidden_dim=256,
            normalize_output=normalize_output,
        )

    def reparameterize(
        self,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        temperature: float = 1.0,
        deterministic: bool = False,
    ) -> torch.Tensor:
        """Sample from latent distribution q(z|e_f) with reparameterization trick.

        z = mu + exp(0.5 * logvar) * epsilon * temperature
        """
        if deterministic or not self.training:
            return mu

        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std * temperature

    def forward(
        self,
        e_f: torch.Tensor,
        deterministic: bool = False,
        temperature: float = 1.0,
    ) -> CVAEOutput:
        """Full forward pass: e_f -> (mu, logvar) -> z -> e_s_hat.

        Args:
            e_f: Input face representations (B, 560)
            deterministic: If True, uses mean mu without sampling noise
            temperature: Scaling factor for sampling std (default: 1.0)

        Returns:
            CVAEOutput namedtuple containing (e_s_hat, mu, logvar, z)
        """
        mu, logvar = self.encoder(e_f)
        z = self.reparameterize(mu, logvar, temperature=temperature, deterministic=deterministic)
        e_s_hat = self.decoder(z)
        return CVAEOutput(e_s_hat=e_s_hat, mu=mu, logvar=logvar, z=z)

    @torch.no_grad()
    def sample_voices(
        self,
        e_f: torch.Tensor,
        num_samples: int = 5,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Generate K diverse candidate voice embeddings for a single face image.

        Args:
            e_f: Single face embedding tensor (560,) or (1, 560)
            num_samples: Number of voice candidates K to generate
            temperature: Sampling temperature (higher = greater voice diversity)

        Returns:
            Tensor of shape (num_samples, 192) containing candidate speaker embeddings.
        """
        self.eval()
        if e_f.ndim == 1:
            e_f = e_f.unsqueeze(0)  # (1, 560)

        mu, logvar = self.encoder(e_f)  # (1, 192), (1, 192)
        std = torch.exp(0.5 * logvar)

        # Expand for K parallel samples
        mu_exp = mu.expand(num_samples, -1)      # (K, 192)
        std_exp = std.expand(num_samples, -1)    # (K, 192)
        eps = torch.randn_like(std_exp)          # (K, 192)

        z = mu_exp + eps * std_exp * temperature
        e_s_candidates = self.decoder(z)         # (K, 192)
        return e_s_candidates
