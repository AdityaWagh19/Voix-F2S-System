"""ContrastiveFaceVoiceNet -- face-to-voice embedding model.

Maps the canonical 560-D FaceFeatureV2 representation (or the 512-D raw
ArcFace diagnostic) to a 192-D ECAPA-style speaker embedding. The architecture
is configurable so checkpoints record exactly how they were built:

    backbone: 560 -> 512 -> 384 -> 256 (GELU, optional LayerNorm/dropout)
    voice_head: 256 -> 192
    optional residual skip: 560 -> 192

LayerNorm is optional because it was present in the legacy collapsed checkpoint
and is not required by the retrained model. Checkpoints written by the new
trainer carry an ``architecture`` block that ``from_checkpoint`` honours.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveFaceVoiceNet(nn.Module):
    """MLP backbone + voice projection head.

    Args:
        face_dim: Input face dimension (560 fused or 512 raw ArcFace).
        out_dim: Output speaker-embedding dimension (192).
        hidden_dims: Hidden layer widths.
        dropout: Dropout probability between hidden layers.
        use_layernorm: Whether to apply LayerNorm after each hidden linear.
        residual: Whether to add a learned linear skip from face to voice.
    """

    def __init__(
        self,
        face_dim: int = 560,
        out_dim: int = 192,
        hidden_dims: tuple[int, ...] = (512, 384, 256),
        dropout: float = 0.1,
        use_layernorm: bool = True,
        residual: bool = False,
    ) -> None:
        super().__init__()
        self.face_dim = int(face_dim)
        self.out_dim = int(out_dim)
        self.hidden_dims = tuple(int(h) for h in hidden_dims)
        self.dropout_p = float(dropout)
        self.use_layernorm = bool(use_layernorm)
        self.residual = bool(residual)

        layers: list[nn.Module] = []
        prev = self.face_dim
        for hidden in self.hidden_dims:
            layers.append(nn.Linear(prev, hidden))
            if self.use_layernorm:
                layers.append(nn.LayerNorm(hidden))
            layers.append(nn.GELU())
            if self.dropout_p > 0:
                layers.append(nn.Dropout(self.dropout_p))
            prev = hidden
        self.backbone = nn.Sequential(*layers)
        self.voice_head = nn.Linear(prev, self.out_dim)
        self.skip = nn.Linear(self.face_dim, self.out_dim, bias=False) if self.residual else None
        if self.skip is not None:
            nn.init.normal_(self.skip.weight, mean=0.0, std=0.01)

    def forward(self, e_f: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        """Map fused face features to speaker embedding space.

        Args:
            e_f: Face representation (B, face_dim).
            normalize: If True, L2-normalise output.

        Returns:
            Speaker embedding prediction (B, 192).
        """
        z = self.backbone(e_f)
        e_s = self.voice_head(z)
        if self.skip is not None:
            e_s = e_s + self.skip(e_f)
        if normalize:
            e_s = F.normalize(e_s, p=2, dim=-1)
        return e_s

    def architecture(self) -> Dict[str, Any]:
        """Serializable architecture description for checkpoint metadata."""
        return {
            "face_dim": self.face_dim,
            "out_dim": self.out_dim,
            "hidden_dims": list(self.hidden_dims),
            "dropout": self.dropout_p,
            "use_layernorm": self.use_layernorm,
            "residual": self.residual,
        }

    @classmethod
    def from_architecture(cls, config: Optional[Dict[str, Any]]) -> "ContrastiveFaceVoiceNet":
        if not config:
            # Legacy checkpoint: original LayerNorm MLP, no residual.
            return cls()
        return cls(
            face_dim=config.get("face_dim", 560),
            out_dim=config.get("out_dim", 192),
            hidden_dims=tuple(config.get("hidden_dims", (512, 384, 256))),
            dropout=float(config.get("dropout", 0.1)),
            use_layernorm=bool(config.get("use_layernorm", True)),
            residual=bool(config.get("residual", False)),
        )

    @classmethod
    def from_checkpoint(cls, path: str, device: str = "cpu") -> "ContrastiveFaceVoiceNet":
        """Load a checkpoint, honouring its recorded architecture and device.

        ``map_location`` only controls where the state dict is deserialized; it
        does not move the freshly constructed module.  The explicit ``.to()``
        is required or callers that request CUDA get a CPU model and fail on
        the first input tensor.
        """
        try:
            ckpt = torch.load(path, map_location=device, weights_only=True)
        except Exception:
            ckpt = torch.load(path, map_location=device, weights_only=False)
        state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        config = ckpt.get("architecture") if isinstance(ckpt, dict) else None
        model = cls.from_architecture(config)
        model.load_state_dict(state, strict=True)
        model.eval()
        return model.to(device)
