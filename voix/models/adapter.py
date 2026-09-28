"""Style Projection Adapter for VOIX (Phase 4).

Bridges the 192-D ECAPA-TDNN speaker embedding space produced by the CVAE
into the style conditioning vector space (e.g., 128-D or 256-D) expected by StyleTTS 2.

Architecture:
    Input:   e_s (192-D, ECAPA speaker embedding)
    Layer 1: Linear(192 -> 256) + LayerNorm(256) + GELU
    Layer 2: Linear(256 -> 256) + LayerNorm(256) + GELU
    Layer 3: Linear(256 -> style_dim)
    Output:  s_hat (StyleTTS 2 conditioning style vector)

Design Rationale:
  - Frozen StyleTTS 2: The diffusion backbone and vocoder remain completely frozen.
  - The adapter is the only trainable module in Phase 4 (<200k parameters).
  - LayerNorm provides stable gradients and prevents representation drift.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class StyleAdapter(nn.Module):
    """3-layer MLP projecting ECAPA-TDNN embeddings to StyleTTS 2 style space.

    Args:
        in_dim: Input speaker embedding dimension (default: 192 for ECAPA).
        hidden_dim: Hidden representation dimension (default: 256).
        out_dim: Target StyleTTS 2 style vector dimension (default: 128).
        dropout: Optional dropout rate (default: 0.0).
        normalize_output: Whether to L2-normalize the output style vector (default: False).
    """

    def __init__(
        self,
        in_dim: int = 192,
        hidden_dim: int = 256,
        out_dim: int = 128,
        dropout: float = 0.0,
        normalize_output: bool = False,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.normalize_output = normalize_output

        layers: list[nn.Module] = [
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        ]
        if dropout > 0.0:
            layers.append(nn.Dropout(p=dropout))

        layers.extend([
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        ])
        if dropout > 0.0:
            layers.append(nn.Dropout(p=dropout))

        layers.append(nn.Linear(hidden_dim, out_dim))

        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, e_s: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            e_s: Speaker embedding tensor of shape (B, 192) or (192,).

        Returns:
            s_hat: Predicted style vector of shape (B, out_dim) or (out_dim,).
        """
        is_1d = e_s.dim() == 1
        if is_1d:
            e_s = e_s.unsqueeze(0)

        s_hat = self.net(e_s)

        if self.normalize_output:
            s_hat = F.normalize(s_hat, p=2, dim=-1)

        if is_1d:
            s_hat = s_hat.squeeze(0)

        return s_hat
