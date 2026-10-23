"""Craniofacial-to-Style Modulation for VOIX.

Maps craniofacial morphology ratios to style-space deltas that modulate
vocal tract resonance. This provides the anatomical link between face
geometry and voice timbre without modifying the StyleTTS 2 backend.

Anatomical-to-Acoustic Mapping:
  - Jaw width   → Oral cavity volume → Lower F2/F3 for wider jaw (darker timbre)
  - Face height → Vocal tract length → Lower F1/F2 for taller face (deeper voice)
  - Lip fullness → Oral aperture → High-frequency radiation (fuller lips = brighter)
  - Nose width   → Nasal cavity resonance → Nasal timbre (wider nose = more nasal)
  - Face width   → Head size resonance → Overall timbre coloration
  - Chin prominence → Mandibular depth → Lower formant dispersion

Implementation:
  - Each craniofacial ratio maps to a block of 8 style dimensions
  - Delta magnitude is proportional to deviation from neutral (ratio = 1.0)
  - Sign determines whether formants shift up or down
  - Applied as additive delta to StyleAdapter output before StyleTTS2 inference
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

# Craniofacial ratio indices (from morphology.py)
IDX_FACE_HEIGHT    = 0   # Face height / IOC
IDX_FACE_WIDTH     = 1   # Face width / IOC
IDX_JAW_WIDTH      = 2   # Jaw width / IOC
IDX_NOSE_WIDTH     = 4   # Nose width / IOC
IDX_LIP_WIDTH      = 11  # Lip width / IOC
IDX_UPPER_LIP_H    = 12  # Upper lip height / IOC
IDX_LOWER_LIP_H    = 13  # Lower lip height / IOC
IDX_CHIN_PROTRUSION = 30 # Chin protrusion (z-depth)

# Style dimension mapping (128-D style vector)
# Each craniofacial feature maps to a block of 8 style dimensions
STYLE_DIM_JAW_WIDTH      = (0, 8)    # Timbre: lower formants (F2/F3)
STYLE_DIM_FACE_HEIGHT    = (8, 16)   # Timbre: vocal tract length (F1/F2)
STYLE_DIM_LIP_FULLNESS   = (16, 24)  # Timbre: oral aperture / high-frequency energy
STYLE_DIM_NOSE_WIDTH     = (24, 32)  # Timbre: nasal cavity resonance
STYLE_DIM_FACE_WIDTH     = (32, 40)  # Timbre: head size resonance
STYLE_DIM_CHIN_PROMINENCE = (40, 48) # Timbre: mandibular depth

# Mapping coefficients: how much each craniofacial ratio deviation affects style
# These are empirically determined scaling factors
JAW_WIDTH_SCALE      = 0.15  # Wider jaw → more negative style delta (darker)
FACE_HEIGHT_SCALE    = 0.12  # Taller face → more negative style delta (deeper)
LIP_FULLNESS_SCALE   = 0.10  # Fuller lips → more positive style delta (brighter)
NOSE_WIDTH_SCALE     = 0.08  # Wider nose → slight positive style delta (nasal)
FACE_WIDTH_SCALE     = 0.08  # Wider face → slight negative style delta
CHIN_PROMINENCE_SCALE = 0.10  # More prominent chin → slight negative style delta


def craniofacial_to_style_delta(
    e_geo: torch.Tensor,
    strength: float = 1.0,
) -> torch.Tensor:
    """Map craniofacial morphology ratios to style-space delta.

    Converts craniofacial ratio deviations from neutral (ratio = 1.0) into
    style-space deltas that modulate vocal tract resonance.

    Args:
        e_geo: Craniofacial ratio vector of shape (B, 32) or (32,).
        strength: Blend strength in [0, 1]. 0 = no modulation, 1 = full modulation.

    Returns:
        Style delta tensor of shape (B, 128) or (128,).
    """
    if not isinstance(e_geo, torch.Tensor):
        e_geo = torch.tensor(e_geo, dtype=torch.float32)
    is_1d = e_geo.dim() == 1
    if is_1d:
        e_geo = e_geo.unsqueeze(0)

    batch_size = e_geo.size(0)
    style_delta = torch.zeros(batch_size, 128, device=e_geo.device, dtype=e_geo.dtype)

    # Jaw width (index 2) → Lower formants for wider jaw
    jaw_width = e_geo[:, IDX_JAW_WIDTH]
    jaw_delta = (jaw_width - 1.0) * JAW_WIDTH_SCALE * strength
    style_delta[:, STYLE_DIM_JAW_WIDTH[0]:STYLE_DIM_JAW_WIDTH[1]] = \
        jaw_width.unsqueeze(1).expand(-1, 8) * jaw_delta.unsqueeze(1).expand(-1, 8) * 0.1

    # Face height (index 0) → Lower formants for taller face (longer VTL)
    face_height = e_geo[:, IDX_FACE_HEIGHT]
    face_height_delta = (face_height - 1.0) * FACE_HEIGHT_SCALE * strength
    style_delta[:, STYLE_DIM_FACE_HEIGHT[0]:STYLE_DIM_FACE_HEIGHT[1]] = \
        face_height.unsqueeze(1).expand(-1, 8) * face_height_delta.unsqueeze(1).expand(-1, 8) * 0.1

    # Lip fullness (indices 12, 13) → Brighter timbre for fuller lips
    lip_fullness = (e_geo[:, IDX_UPPER_LIP_H] + e_geo[:, IDX_LOWER_LIP_H]) / 2.0
    lip_delta = (lip_fullness - 1.0) * LIP_FULLNESS_SCALE * strength
    style_delta[:, STYLE_DIM_LIP_FULLNESS[0]:STYLE_DIM_LIP_FULLNESS[1]] = \
        lip_fullness.unsqueeze(1).expand(-1, 8) * lip_delta.unsqueeze(1).expand(-1, 8) * 0.1

    # Nose width (index 4) → Nasal resonance
    nose_width = e_geo[:, IDX_NOSE_WIDTH]
    nose_delta = (nose_width - 1.0) * NOSE_WIDTH_SCALE * strength
    style_delta[:, STYLE_DIM_NOSE_WIDTH[0]:STYLE_DIM_NOSE_WIDTH[1]] = \
        nose_width.unsqueeze(1).expand(-1, 8) * nose_delta.unsqueeze(1).expand(-1, 8) * 0.05

    # Face width (index 1) → Head size resonance
    face_width = e_geo[:, IDX_FACE_WIDTH]
    face_width_delta = (face_width - 1.0) * FACE_WIDTH_SCALE * strength
    style_delta[:, STYLE_DIM_FACE_WIDTH[0]:STYLE_DIM_FACE_WIDTH[1]] = \
        face_width.unsqueeze(1).expand(-1, 8) * face_width_delta.unsqueeze(1).expand(-1, 8) * 0.05

    # Chin prominence (index 30) → Mandibular depth
    chin_prom = e_geo[:, IDX_CHIN_PROTRUSION]
    chin_delta = (chin_prom - 1.0) * CHIN_PROMINENCE_SCALE * strength
    style_delta[:, STYLE_DIM_CHIN_PROMINENCE[0]:STYLE_DIM_CHIN_PROMINENCE[1]] = \
        chin_prom.unsqueeze(1).expand(-1, 8) * chin_delta.unsqueeze(1).expand(-1, 8) * 0.08

    if is_1d:
        style_delta = style_delta.squeeze(0)

    return style_delta


def apply_craniofacial_to_style(
    style: torch.Tensor,
    e_geo: torch.Tensor,
    strength: float = 1.0,
) -> torch.Tensor:
    """Apply craniofacial morphology modulation to style vector.

    Adds craniofacial-derived delta to the StyleAdapter output before
    StyleTTS2 inference. This provides anatomical control over vocal tract
    resonance without modifying the StyleTTS2 backend.

    Args:
        style: Style vector from StyleAdapter, shape (B, 128), (B, 256), or 1D equivalent.
        e_geo: Craniofacial ratio vector, shape (B, 32) or (32,).
        strength: Modulation strength in [0, 1].

    Returns:
        Modified style tensor with craniofacial modulation applied.
    """
    delta = craniofacial_to_style_delta(e_geo, strength=strength).to(style.device)
    is_1d = (style.dim() == 1)
    if is_1d:
        style = style.unsqueeze(0)
    
    out = style.clone()
    if out.shape[-1] >= 256:
        out[:, :128] = out[:, :128] + delta
    else:
        out = out + delta

    return out.squeeze(0) if is_1d else out
