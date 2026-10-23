"""Style Projection Adapter for VOIX (Phase 4 - Dual-Head v2).

Bridges the 192-D ECAPA-TDNN speaker embedding space produced by the CVAE
into the 256-D style conditioning vector space expected by StyleTTS 2.

Architecture:
    Input:        e_s (192-D, ECAPA speaker embedding)
    Shared Trunk: Linear(192 -> 256) + LayerNorm(256) + GELU
                  Linear(256 -> 256) + LayerNorm(256) + GELU
    Timbre Head:  Linear(256 -> 128) -> ref_s[:, :128] (vocal tract resonance & timbre)
    Prosody Head: Linear(256 -> 128) -> ref_s[:, 128:] (F0 dynamics & cadence)
    Output:       s_hat = [timbre | prosody] (256-D)

Design Rationale:
  - Anchor-Free Synthesis: Predicts both timbre and prosody directly from biometrics.
  - Frozen StyleTTS 2: The diffusion vocoder remains frozen; adapter is the lightweight bridge.
  - Backward Compatibility: Remaps legacy single-head checkpoints automatically.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class StyleAdapter(nn.Module):
    """Dual-head MLP projecting ECAPA-TDNN embeddings to 256-D StyleTTS 2 style space."""

    def __init__(
        self,
        in_dim: int = 192,
        hidden_dim: int = 256,
        out_dim: int = 256,
        dropout: float = 0.0,
        normalize_output: bool = False,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.normalize_output = normalize_output

        # Shared representation trunk (192 -> 256 -> 256)
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

        self.net = nn.Sequential(*layers)

        # Dual prediction heads
        self.timbre_head = nn.Linear(hidden_dim, 128)
        self.prosody_head = nn.Linear(hidden_dim, 128)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def load_state_dict(self, state_dict: dict, strict: bool = False):
        """Load state dict with automatic remap for legacy single-head checkpoints."""
        sd = dict(state_dict)
        # Handle state_dict wrapped in dict
        if "model_state_dict" in sd:
            sd = sd["model_state_dict"]
        elif "state_dict" in sd:
            sd = sd["state_dict"]

        # Remap legacy net.6 (single 128 head) to timbre_head and duplicate to prosody_head
        if "net.6.weight" in sd and "timbre_head.weight" not in sd:
            sd["timbre_head.weight"] = sd.pop("net.6.weight")
            sd["timbre_head.bias"] = sd.pop("net.6.bias")
            sd["prosody_head.weight"] = sd["timbre_head.weight"].clone()
            sd["prosody_head.bias"] = sd["timbre_head.bias"].clone()

        return super().load_state_dict(sd, strict=strict)

    def forward(self, e_s: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            e_s: Speaker embedding tensor of shape (B, 192) or (192,).

        Returns:
            s_hat: Predicted 256-D style vector of shape (B, 256) or (256,).
        """
        is_1d = (e_s.dim() == 1)
        if is_1d:
            e_s = e_s.unsqueeze(0)

        h = self.net(e_s)
        timbre = self.timbre_head(h)
        prosody = self.prosody_head(h)

        if self.normalize_output:
            timbre = F.normalize(timbre, p=2, dim=-1)
            prosody = F.normalize(prosody, p=2, dim=-1)

        s_hat = torch.cat([timbre, prosody], dim=-1)

        if is_1d:
            s_hat = s_hat.squeeze(0)

        return s_hat
