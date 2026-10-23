"""ContrastiveFaceVoiceNet -- face-to-voice embedding model (Phase 3).

Trained via InfoNCE contrastive loss on VoxCeleb2 face+voice pairs.
Maps 560-D fused face representation to 192-D ECAPA speaker embedding space.

Checkpoint: D:/voix/results/checkpoints/D_joint_infonce_best.pt
Architecture: backbone MLP (560->512->384->256) + voice_head (256->192)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveFaceVoiceNet(nn.Module):
    """MLP backbone + voice projection head.

    Takes 560-D fused face features (ArcFace 512 + craniofacial 32 + demographic 16)
    and projects them to 192-D speaker embedding space (ECAPA-TDNN space).
    """

    def __init__(self, face_dim: int = 560, out_dim: int = 192) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(face_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, 384),
            nn.LayerNorm(384),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(384, 256),
            nn.LayerNorm(256),
            nn.GELU(),
        )
        self.voice_head = nn.Linear(256, out_dim)

    def forward(self, e_f: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        """Map fused face features to speaker embedding space.

        Args:
            e_f: Fused face representation (B, 560).
            normalize: If True, L2-normalise output (default True, matches training).

        Returns:
            Speaker embedding prediction (B, 192).
        """
        z = self.backbone(e_f)
        e_s = self.voice_head(z)
        if normalize:
            e_s = F.normalize(e_s, p=2, dim=-1)
        return e_s

    @classmethod
    def from_checkpoint(cls, path: str, device: str = "cpu") -> "ContrastiveFaceVoiceNet":
        """Load from D_joint_infonce_best.pt checkpoint."""
        ckpt = torch.load(path, map_location=device, weights_only=True)
        state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        model = cls()
        # The checkpoint has no model_state_dict wrapper -- keys are flat backbone.*/voice_head.*
        model.load_state_dict(state, strict=True)
        model.eval()
        return model
