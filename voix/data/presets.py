"""Demographic Presets Rail for VOIX (Feature 2).

Provides diverse, ethically grounded reference-face biometric presets for
instant evaluation without requiring uploaded photographs.

All presets are represented purely as 560-D multimodal embeddings
[ArcFace 512-D | MediaPipe 32-D | Demographics 16-D], containing zero
identifying personal imagery or raw face photographs.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from voix.data.demographics import SoftDemographicEstimator
from voix.data.morphology import apply_craniofacial_sliders
from voix.models.fusion import FaceFusionLayer

PRESETS_DIR = Path("D:/voix/data/presets")


@dataclass
class DemographicPreset:
    """Canonical demographic reference preset representation."""

    preset_id: str
    display_name: str
    description: str
    gender: int  # 0 = Female, 1 = Male
    age: float
    pitch_target_hz: float
    craniofacial_traits: Dict[str, float] = field(default_factory=dict)
    embedding: Optional[torch.Tensor] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert preset metadata to dictionary (excluding tensor)."""
        return {
            "preset_id": self.preset_id,
            "display_name": self.display_name,
            "description": self.description,
            "gender": self.gender,
            "age": self.age,
            "pitch_target_hz": self.pitch_target_hz,
            "craniofacial_traits": dict(self.craniofacial_traits),
        }


# ---------------------------------------------------------------------------
# Canonical Preset Definitions
# ---------------------------------------------------------------------------

CANONICAL_PRESETS_DEF: List[Dict[str, Any]] = [
    {
        "preset_id": "young_female_resonant",
        "display_name": "Young Adult Female (Resonant)",
        "description": "High fundamental register (~220 Hz) with slender midface and full vermilion aperture.",
        "gender": 0,
        "age": 22.0,
        "pitch_target_hz": 222.0,
        "seed": 101,
        "craniofacial_traits": {
            "lip_fullness": 0.40,
            "midface_length": -0.15,
            "jaw_width": -0.10,
        },
    },
    {
        "preset_id": "adult_female_warm",
        "display_name": "Adult Female (Warm & Balanced)",
        "description": "Balanced vocal register (~210 Hz) with warm vocal tract acoustics and harmonious facial symmetry.",
        "gender": 0,
        "age": 34.0,
        "pitch_target_hz": 210.0,
        "seed": 102,
        "craniofacial_traits": {
            "jaw_width": 0.05,
            "lip_fullness": 0.15,
            "face_width": 0.05,
        },
    },
    {
        "preset_id": "mature_female_authoritative",
        "display_name": "Mature Female (Authoritative)",
        "description": "Deeper female fundamental register (~190 Hz) with lengthened vocal tract proxy and prominent structure.",
        "gender": 0,
        "age": 52.0,
        "pitch_target_hz": 192.0,
        "seed": 103,
        "craniofacial_traits": {
            "face_height": 0.25,
            "jaw_width": 0.15,
            "chin_prominence": 0.20,
        },
    },
    {
        "preset_id": "young_male_bright",
        "display_name": "Young Adult Male (Bright Tenor)",
        "description": "Agile tenor register (~138 Hz) with slender jaw structure and compact vertical cranial length.",
        "gender": 1,
        "age": 23.0,
        "pitch_target_hz": 138.0,
        "seed": 201,
        "craniofacial_traits": {
            "jaw_width": -0.15,
            "face_height": -0.10,
            "lip_width": 0.10,
        },
    },
    {
        "preset_id": "adult_male_baritone",
        "display_name": "Adult Male (Resonant Baritone)",
        "description": "Rich baritone register (~122 Hz) with broad mandibular cavity and expansive bizygomatic width.",
        "gender": 1,
        "age": 38.0,
        "pitch_target_hz": 122.0,
        "seed": 202,
        "craniofacial_traits": {
            "jaw_width": 0.35,
            "face_width": 0.25,
            "chin_prominence": 0.20,
        },
    },
    {
        "preset_id": "senior_male_deep",
        "display_name": "Senior Male (Deep Bass)",
        "description": "Low fundamental frequency (~108 Hz) with elongated vocal tract proxy and pronounced mandibular angle.",
        "gender": 1,
        "age": 64.0,
        "pitch_target_hz": 108.0,
        "seed": 203,
        "craniofacial_traits": {
            "face_height": 0.30,
            "jaw_width": 0.40,
            "chin_prominence": 0.30,
        },
    },
    {
        "preset_id": "androgynous_balanced",
        "display_name": "Androgynous (Neutral Register)",
        "description": "Intermediate fundamental register (~165 Hz) with centered craniofacial proportions.",
        "gender": 0,
        "age": 28.0,
        "pitch_target_hz": 165.0,
        "seed": 301,
        "craniofacial_traits": {
            "jaw_width": 0.0,
            "face_height": 0.0,
            "lip_fullness": 0.0,
        },
    },
    {
        "preset_id": "athletic_broad_build",
        "display_name": "Broad Build (Strong Resonance)",
        "description": "Expansive acoustic cavity coupling with wide bizygomatic arch and prominent chin structure.",
        "gender": 1,
        "age": 31.0,
        "pitch_target_hz": 118.0,
        "seed": 302,
        "craniofacial_traits": {
            "jaw_width": 0.45,
            "face_width": 0.40,
            "chin_prominence": 0.35,
            "nose_width": 0.25,
        },
    },
]


class PresetRegistry:
    """Registry managing demographic presets and their 560-D feature vectors."""

    _INSTANCE: Optional[PresetRegistry] = None

    def __init__(self, device: str = "cpu") -> None:
        self.device = torch.device(device)
        self.fusion = FaceFusionLayer().to(self.device)
        self.demog = SoftDemographicEstimator(device=str(self.device))
        self._presets: Dict[str, DemographicPreset] = {}
        self._initialize_presets()

    @classmethod
    def get_instance(cls, device: str = "cpu") -> PresetRegistry:
        """Get or initialize singleton registry instance."""
        if cls._INSTANCE is None:
            cls._INSTANCE = cls(device=device)
        return cls._INSTANCE

    def _generate_synthetic_face_vector(
        self,
        seed: int,
        gender: int,
        age: float,
        traits: Dict[str, float],
    ) -> torch.Tensor:
        """Deterministically generate fused 560-D face vector for a preset."""
        rng = np.random.default_rng(seed)

        # 1. 512-D ArcFace identity vector (L2 normalized)
        raw_id = rng.standard_normal(512).astype(np.float32)
        # Inject demographic correlation into dimensions 0 and 1
        raw_id[0] = 0.8 if gender == 1 else -0.8
        raw_id[1] = (age - 35.0) / 25.0
        e_id = raw_id / (np.linalg.norm(raw_id) + 1e-8)

        # 2. 32-D Craniofacial ratios (baseline + traits)
        e_geo = np.ones(32, dtype=np.float32)
        e_geo[0] = 2.2   # face height / IOC
        e_geo[1] = 1.8   # face width / IOC
        e_geo[2] = 1.4   # jaw width / IOC
        e_geo[3] = 85.0  # mandible angle degrees
        e_geo[4] = 0.55  # nose width / IOC
        e_geo[5] = 0.80  # nose height / IOC
        e_geo[6] = 0.70  # nose bridge / IOC
        e_geo[7] = 1.8   # bizygomatic / IOC
        e_geo[8] = 0.35  # inner eye sep / IOC
        e_geo[11] = 0.90 # lip width / IOC
        e_geo[12] = 0.18 # upper lip height / IOC
        e_geo[13] = 0.22 # lower lip height / IOC
        e_geo[24] = 1.22 # face aspect ratio
        e_geo[25] = 2.54 # mandible to nose ratio
        e_geo[26] = 0.64 # lip to jaw ratio
        e_geo[27] = 0.61 # nose to lip ratio
        e_geo[30] = 0.05 # chin protrusion

        if traits:
            e_geo = apply_craniofacial_sliders(e_geo, traits)

        # 3. 16-D Demographic probability vector
        e_demo = self.demog.estimate_from_attributes(gender=gender, age=age)

        # 4. Fusion 560-D
        with torch.no_grad():
            e_id_t = torch.from_numpy(e_id).unsqueeze(0).to(self.device)
            e_geo_t = torch.from_numpy(e_geo).unsqueeze(0).to(self.device)
            e_demo_t = torch.from_numpy(e_demo).unsqueeze(0).to(self.device)
            e_f = self.fusion(e_id_t, e_geo_t, e_demo_t)

        return e_f

    def _initialize_presets(self) -> None:
        """Load or generate all canonical presets."""
        PRESETS_DIR.mkdir(parents=True, exist_ok=True)

        for p_def in CANONICAL_PRESETS_DEF:
            pid = p_def["preset_id"]
            pt_path = PRESETS_DIR / f"{pid}.pt"

            if pt_path.exists():
                try:
                    e_f = torch.load(pt_path, map_location=self.device, weights_only=True)
                except Exception:
                    e_f = self._generate_synthetic_face_vector(
                        seed=p_def["seed"],
                        gender=p_def["gender"],
                        age=p_def["age"],
                        traits=p_def["craniofacial_traits"],
                    )
                    torch.save(e_f, pt_path)
            else:
                e_f = self._generate_synthetic_face_vector(
                    seed=p_def["seed"],
                    gender=p_def["gender"],
                    age=p_def["age"],
                    traits=p_def["craniofacial_traits"],
                )
                try:
                    torch.save(e_f, pt_path)
                except Exception:
                    pass

            preset = DemographicPreset(
                preset_id=pid,
                display_name=p_def["display_name"],
                description=p_def["description"],
                gender=p_def["gender"],
                age=p_def["age"],
                pitch_target_hz=p_def["pitch_target_hz"],
                craniofacial_traits=p_def["craniofacial_traits"],
                embedding=e_f,
            )
            self._presets[pid] = preset

    def list_presets(self) -> List[Dict[str, Any]]:
        """Return list of all preset metadata summaries."""
        return [p.to_dict() for p in self._presets.values()]

    def get_preset(self, preset_id: str) -> DemographicPreset:
        """Get preset by ID.

        Raises:
            KeyError: If preset_id is not recognized.
        """
        if preset_id not in self._presets:
            raise KeyError(
                f"Preset '{preset_id}' not found. Available presets: {list(self._presets.keys())}"
            )
        return self._presets[preset_id]


# Convenience functional API
def list_demographic_presets() -> List[Dict[str, Any]]:
    """Return list of all available demographic presets."""
    return PresetRegistry.get_instance().list_presets()


def get_demographic_preset(preset_id: str) -> DemographicPreset:
    """Retrieve demographic preset by ID."""
    return PresetRegistry.get_instance().get_preset(preset_id)
