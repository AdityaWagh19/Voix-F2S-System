"""Unified multimodal face feature fusion layer (T2.4).

Implements FaceFusionLayer: concatenates three heterogeneous face
representations and projects them into a unified 560-D embedding space.

Input streams:
    e_id   (512-D): ArcFace L2-normalised identity embedding
    e_geo  ( 32-D): Craniofacial morphological ratio vector
    e_demo ( 16-D): Soft demographic prior distribution
    -------
    Total: 560-D -> Linear(560, 560) -> LayerNorm -> GELU -> 560-D output

Design decisions (from Phase 2 plan, Section 3):
  - Simple learned linear projection, not transformer cross-attention.
    Justified by fixed 560-D input size and GTX 1650 compute budget.
  - LayerNorm after projection stabilises training when sub-embeddings
    have very different numerical scales (ArcFace: L2 norm ~1.0;
    craniofacial ratios: range ~0.5-5.0; demographics: range 0.0-1.0).
  - GELU activation (not ReLU) preserves small negative activations,
    which carry meaningful information in normalized embedding spaces.
  - Weights are NOT auto-loaded at import time. Call load_canonical_weights()
    explicitly to load a saved checkpoint.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from voix.utils.seed import set_global_seed

DEFAULT_FUSION_CHECKPOINT = Path("D:/voix/checkpoints/face_fusion.pt")


class FaceFusionLayer(nn.Module):
    """Multimodal face feature fusion: [ArcFace | Craniofacial | Demographic] -> 560-D.

    Args:
        id_dim:   Dimensionality of ArcFace identity embedding. Default: 512.
        geo_dim:  Dimensionality of craniofacial ratio vector.  Default: 32.
        demo_dim: Dimensionality of demographic prior vector.   Default: 16.
        out_dim:  Output dimensionality. Default: 560 (= id_dim + geo_dim + demo_dim).
    """

    ID_DIM:   int = 512
    GEO_DIM:  int = 32
    DEMO_DIM: int = 16
    OUT_DIM:  int = 560

    def __init__(
        self,
        id_dim:   int = 512,
        geo_dim:  int = 32,
        demo_dim: int = 16,
        out_dim:  int = 560,
    ) -> None:
        super().__init__()
        in_dim = id_dim + geo_dim + demo_dim

        # Ensure deterministic canonical initialization
        set_global_seed(42)
        self.proj = nn.Linear(in_dim, out_dim, bias=True)
        self.norm = nn.LayerNorm(out_dim)
        self.act  = nn.GELU()

        # NOTE: Weights are intentionally NOT auto-loaded here.
        # Callers must explicitly call load_canonical_weights(path) to load
        # a saved checkpoint. This prevents import-time filesystem side effects.
        self._frozen = False

    def load_canonical_weights(self, path: str | None = None) -> None:
        """Explicitly load saved canonical weights from a checkpoint file.

        Args:
            path: Path to the .pt file containing state_dict.
                  If None, uses DEFAULT_FUSION_CHECKPOINT.

        Raises:
            FileNotFoundError: If the checkpoint file does not exist.
        """
        ckpt_path = path or str(DEFAULT_FUSION_CHECKPOINT)
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(
                f"FaceFusionLayer checkpoint not found: {ckpt_path}. "
                "Run the fusion pre-training step or provide a valid path."
            )
        state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        self.load_state_dict(state)

    def forward(
        self,
        e_id:   torch.Tensor,
        e_geo:  torch.Tensor,
        e_demo: torch.Tensor,
    ) -> torch.Tensor:
        """Fuse three face representations into a single 560-D vector.

        Args:
            e_id:   ArcFace identity embedding (..., 512).
            e_geo:  Craniofacial morphological ratio vector (..., 32).
            e_demo: Soft demographic prior distribution (..., 16).

        Returns:
            Fused face representation of shape (..., out_dim).
        """
        x = torch.cat([e_id, e_geo, e_demo], dim=-1)
        return self.act(self.norm(self.proj(x)))

    def freeze(self) -> None:
        """Freeze all fusion layer parameters (no gradient updates)."""
        for p in self.parameters():
            p.requires_grad = False
        self._frozen = True

    def unfreeze(self) -> None:
        """Unfreeze fusion layer parameters."""
        for p in self.parameters():
            p.requires_grad = True
        self._frozen = False

    @property
    def in_dim(self) -> int:
        """Total input dimensionality (512 + 32 + 16 = 560)."""
        return self.proj.in_features

    @property
    def out_dim(self) -> int:
        """Output dimensionality."""
        return self.proj.out_features
