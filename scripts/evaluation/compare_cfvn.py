"""Compare fused-560 and raw-ArcFace CFVN checkpoints on held-out speakers.

The serving contract is the 560-D FaceFeatureV2 representation because presets
and image inference both produce it.  Raw ArcFace is a diagnostic baseline: it
isolates identity from the geometry/demographic channels.  This script trains
nothing; it loads both checkpoints, evaluates each with its own scaler, and
writes a side-by-side report so the representation choice is measured rather
than assumed.

The recommendation switches away from fused only when raw ArcFace improves
both unrestricted and gender-controlled face-to-voice R@1 by at least 50% on
the official test split.  That is deliberately conservative: a small gain does
not justify a new serving representation and preset rebuild.

Usage:
    python scripts/evaluation/compare_cfvn.py \
        --fused results/checkpoints/cfvn_v2_best.pt \
        --arcface results/checkpoints/cfvn_v2_arcface.pt \
        --output results/cfvn_representation_comparison.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from voix.data.feature_contract import load_scaler_payload  # noqa: E402
from voix.evaluation.metrics import DEFAULT_SCALER_ARCFACE  # noqa: E402
from voix.models.contrastive_net import ContrastiveFaceVoiceNet  # noqa: E402
from voix.training.data import (  # noqa: E402
    DEFAULT_FEATURES_V2,
    DEFAULT_INDEX_V2,
    DEFAULT_SCALER_V2,
    CleanFeatureDataset,
    load_v2_dataset,
    normalize_speaker_features,
    validate_scaler_for_dataset,
)
from voix.training.trainer_cfvn import (  # noqa: E402
    evaluate,
    normalize_faces,
)

SWITCH_FACTOR = 1.5


def _metrics_for(
    checkpoint_path: Path,
    features: str | Path,
    index: str | Path,
    scaler_path: Path,
    device: str,
) -> Dict[str, Dict[str, float]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    face_key = str(checkpoint.get("face_key", "face_features"))
    if face_key not in {"face_features", "arcface"}:
        raise ValueError(f"{checkpoint_path} has unsupported face_key {face_key!r}")
    face_dim = int(checkpoint.get("face_dim", 560 if face_key == "face_features" else 512))
    # Each checkpoint must be evaluated on the representation it was trained
    # on.  Loading one shared 560-D dataset makes the 512-D ArcFace scaler
    # validation fail (and would silently evaluate the wrong features if the
    # dimension check were relaxed).
    dataset = load_v2_dataset(features, index, face_key=face_key)

    model = ContrastiveFaceVoiceNet.from_checkpoint(
        str(checkpoint_path), device=device
    ).to(device)
    model.eval()
    if model.face_dim != face_dim:
        raise ValueError(
            f"{checkpoint_path} declares face_dim={face_dim} but the model uses "
            f"{model.face_dim}"
        )

    scaler = validate_scaler_for_dataset(
        load_scaler_payload(
            scaler_path, expected_dim=face_dim, face_key=face_key
        ),
        dataset,
        face_key=face_key,
    )
    face_z = normalize_faces(dataset.face_raw, scaler)
    speaker_z = normalize_speaker_features(dataset.speaker_raw)
    identity = F.normalize(torch.from_numpy(dataset.face_raw).float(), dim=-1)

    output: Dict[str, Dict[str, float]] = {
        "checkpoint": str(checkpoint_path),
        "face_key": face_key,
        "face_dim": face_dim,
        "scaler": str(scaler_path),
        "epoch": float(checkpoint.get("epoch", -1)),
        "splits": {},
    }
    for split in ("val", "test"):
        subset = dataset.subset(split, allow_empty=True)
        if len(subset) == 0:
            output["splits"][split] = {"available": False}
            continue
        mask = dataset.split == split
        metrics = evaluate(
            model,
            subset,
            face_z[mask],
            speaker_z[mask],
            device,
            identity[mask],
        )
        chance = max(float(metrics["val_chance_r1"]), 1e-9)
        gender_chance = max(
            float(metrics["val_gender_controlled_chance_r1"]), 1e-9
        )
        output["splits"][split] = {
            "available": True,
            "clips": int(len(subset)),
            "speakers": int(len(set(subset.speaker_ids.tolist()))),
            **{key: float(value) for key, value in metrics.items()},
            "chance_multiple": float(metrics["val_r1"] / chance),
            "gender_chance_multiple": float(
                metrics["val_gender_controlled_r1"] / gender_chance
            ),
        }
    return output


def _recommend(
    test_fused: Dict[str, float], test_arc: Dict[str, float]
) -> Dict[str, object]:
    """Apply the conservative representation-switch rule."""
    if not test_fused.get("available") or not test_arc.get("available"):
        return {
            "recommended_face_key": "face_features",
            "rationale": "test split unavailable; keep the serving contract",
        }
    fused_r1 = float(test_fused["val_r1"])
    arc_r1 = float(test_arc["val_r1"])
    fused_g = float(test_fused["val_gender_controlled_r1"])
    arc_g = float(test_arc["val_gender_controlled_r1"])
    switch = (
        arc_r1 > 0
        and arc_g > 0
        and arc_r1 >= fused_r1 * SWITCH_FACTOR
        and arc_g >= fused_g * SWITCH_FACTOR
    )
    return {
        "test_r1_ratio_arcface_over_fused": float(
            arc_r1 / max(fused_r1, 1e-9)
        ),
        "test_gender_controlled_r1_ratio": float(arc_g / max(fused_g, 1e-9)),
        "switch_factor": SWITCH_FACTOR,
        "recommended_face_key": "arcface" if switch else "face_features",
        "rationale": (
            "raw ArcFace improves both test R@1 and gender-controlled R@1 "
            f"by at least {SWITCH_FACTOR:.1f}x"
            if switch
            else "raw ArcFace does not clear the conservative switch rule; "
            "keep the fused 560-D serving contract"
        ),
    }


def compare(
    fused_path: str | Path,
    arcface_path: str | Path,
    *,
    features: str | Path = DEFAULT_FEATURES_V2,
    index: str | Path = DEFAULT_INDEX_V2,
    fused_scaler: str | Path = DEFAULT_SCALER_V2,
    arcface_scaler: str | Path = DEFAULT_SCALER_ARCFACE,
    device: Optional[str] = None,
) -> dict:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    fused = _metrics_for(
        Path(fused_path), features, index, Path(fused_scaler), device
    )
    arcface = _metrics_for(
        Path(arcface_path), features, index, Path(arcface_scaler), device
    )

    return {
        "features": str(features),
        "index": str(index),
        "fused": fused,
        "arcface": arcface,
        "comparison": _recommend(
            fused["splits"].get("test", {}),
            arcface["splits"].get("test", {}),
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare CFVN representations")
    parser.add_argument("--fused", type=Path, required=True)
    parser.add_argument("--arcface", type=Path, required=True)
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES_V2)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX_V2)
    parser.add_argument("--fused-scaler", type=Path, default=DEFAULT_SCALER_V2)
    parser.add_argument(
        "--arcface-scaler", type=Path, default=DEFAULT_SCALER_ARCFACE
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = compare(
        args.fused,
        args.arcface,
        features=args.features,
        index=args.index,
        fused_scaler=args.fused_scaler,
        arcface_scaler=args.arcface_scaler,
        device=args.device,
    )
    text = json.dumps(report, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
        print(f"Wrote {args.output}")
    print(text)


if __name__ == "__main__":
    main()
