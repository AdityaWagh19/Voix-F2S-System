"""Fit the canonical FaceFeatureV2 z-score scaler on the train split.

The scaler is part of the serving contract: extraction, CFVN training, CVAE
training, and inference must all use the same artifact.  This script makes the
build step explicit and refuses to write a scaler whose ``face_key`` or
dimension does not match the corpus.

Usage:
    python scripts/data/build_face_scaler_v2.py \
        --features data/processed/voxceleb2_v2/features_v2.h5 \
        --index data/processed/voxceleb2_v2/index.csv \
        --output data/processed/voxceleb2_v2/face_scaler_v2.pt
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from voix.training.data import (  # noqa: E402
    DEFAULT_FEATURES_V2,
    DEFAULT_INDEX_V2,
    DEFAULT_SCALER_V2,
    build_face_scaler_v2,
    load_v2_dataset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a FaceFeatureV2 scaler")
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES_V2)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX_V2)
    parser.add_argument("--output", type=Path, default=DEFAULT_SCALER_V2)
    parser.add_argument(
        "--face-key",
        default="face_features",
        help="HDF5 dataset to normalize (face_features or arcface)",
    )
    parser.add_argument("--source", default="", help="Free-form provenance note")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    canonical = Path(DEFAULT_SCALER_V2).resolve()
    try:
        output = Path(args.output).resolve()
    except OSError:
        output = Path(args.output)
    if args.face_key != "face_features" and output == canonical:
        raise SystemExit(
            f"Refusing to overwrite the canonical 560-D scaler {canonical} with "
            f"a {args.face_key} scaler. Use a distinct --output such as "
            "data/processed/voxceleb2_v2/face_scaler_arcface_v2.pt."
        )
    dataset = load_v2_dataset(args.features, args.index, face_key=args.face_key)
    payload = build_face_scaler_v2(
        dataset,
        args.output,
        source=args.source,
        face_key=args.face_key,
    )
    summary = {
        key: (list(value.shape) if hasattr(value, "shape") else value)
        for key, value in payload.items()
    }
    print(json.dumps(summary, indent=2))
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
