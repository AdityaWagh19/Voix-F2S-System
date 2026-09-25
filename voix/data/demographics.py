"""Soft demographic prior estimator (16-D).

Predicts probabilistic age and biological sex distributions from ArcFace
identity embeddings. All outputs are soft probability distributions, never
hard demographic labels. Gradient gating prevents demographic collapse.
"""

from __future__ import annotations

import numpy as np


class SoftDemographicEstimator:
    """Soft demographic prior estimator predicting 16-D probability vectors.

    Produces soft distributions over apparent age and biological sex from a
    512-D ArcFace identity embedding. Outputs are NEVER used as hard labels.

    Output vector layout (16-D total):
        Indices 0-7:   Age bin soft distribution (8 bins: ~0-10, 10-20, ..., 70+)
        Indices 8-9:   Biological sex distribution [p_male, p_female]
        Indices 10-13: Vocal age grouping soft distribution (child, young, adult, senior)
        Indices 14-15: Apparent body build proxy [slender, broad] (softmax)

    Args:
        device: PyTorch device string.
        checkpoint_path: Optional path to trained demographic estimator weights.
                         If None, uses uniform priors (acceptable for Phase 1).
    """

    OUTPUT_DIM: int = 16

    def __init__(
        self,
        device: str = "cuda",
        checkpoint_path: str | None = None,
    ) -> None:
        self.device = device
        self.checkpoint_path = checkpoint_path
        self._model = None
        self._loaded = False
        self._use_uniform_prior = checkpoint_path is None

    def estimate(self, arcface_embedding: np.ndarray) -> np.ndarray:
        """Estimate 16-D soft demographic distribution from face embedding.

        Args:
            arcface_embedding: Float32 array of shape (512,) from ArcFace.

        Returns:
            Float32 array of shape (16,) representing soft demographic
            probability distributions. All sub-distributions sum to 1.0.
        """
        if self._use_uniform_prior:
            return self._uniform_prior()

        if not self._loaded:
            self._load_model()

        import torch

        with torch.no_grad():
            emb_tensor = torch.from_numpy(arcface_embedding).unsqueeze(0).to(self.device)
            output = self._model(emb_tensor).squeeze(0).cpu().numpy()

        return output.astype(np.float32)

    def _uniform_prior(self) -> np.ndarray:
        """Return a uniform demographic prior (equal probability across bins).

        Used during Phase 1 when no demographic estimator is trained.
        Replaced with learned estimator in Phase 2.
        """
        prior = np.zeros(self.OUTPUT_DIM, dtype=np.float32)
        prior[0:8] = 1.0 / 8        # Age bins: uniform
        prior[8:10] = 0.5           # Sex: 50/50
        prior[10:14] = 0.25         # Vocal age: uniform
        prior[14:16] = 0.5          # Body build: 50/50
        return prior

    def _load_model(self) -> None:
        """Load trained demographic estimator from checkpoint."""
        import torch
        import torch.nn as nn

        class DemographicHead(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.age_head = nn.Sequential(
                    nn.Linear(512, 128), nn.GELU(), nn.Linear(128, 8), nn.Softmax(dim=-1)
                )
                self.sex_head = nn.Sequential(
                    nn.Linear(512, 64), nn.GELU(), nn.Linear(64, 2), nn.Softmax(dim=-1)
                )
                self.vocal_age_head = nn.Sequential(
                    nn.Linear(512, 64), nn.GELU(), nn.Linear(64, 4), nn.Softmax(dim=-1)
                )
                self.build_head = nn.Sequential(
                    nn.Linear(512, 32), nn.GELU(), nn.Linear(32, 2), nn.Softmax(dim=-1)
                )

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                age = self.age_head(x)          # (B, 8)
                sex = self.sex_head(x)          # (B, 2)
                vocal_age = self.vocal_age_head(x)  # (B, 4)
                build = self.build_head(x)      # (B, 2)
                return torch.cat([age, sex, vocal_age, build], dim=-1)  # (B, 16)

        model = DemographicHead()
        state = torch.load(self.checkpoint_path, map_location=self.device, weights_only=True)
        model.load_state_dict(state)
        model.to(self.device)
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
        self._model = model
        self._loaded = True
