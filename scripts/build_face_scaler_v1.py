"""Build the versioned, train-only FaceFeatureV1 scaler from existing features."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parent.parent


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, default=ROOT / "data/processed/features.h5")
    parser.add_argument("--index", type=Path, default=ROOT / "data/processed/clean_index.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "data/processed/face_scaler_v1.pt")
    args = parser.parse_args()

    index = pd.read_csv(args.index)
    required = {"hdf5_index", "speaker_id", "split"}
    missing = required.difference(index.columns)
    if missing:
        raise ValueError(f"Index is missing required columns: {sorted(missing)}")
    train = index[index["split"] == "train"].copy()
    if train.empty:
        raise ValueError("No training rows found in split column")
    if train["speaker_id"].isin(index[index["split"] != "train"]["speaker_id"]).any():
        raise ValueError("Speaker leakage detected between train and non-train splits")

    with h5py.File(args.features, "r") as store:
        features = store["face_features"]
        indices = train["hdf5_index"].to_numpy(dtype=np.int64)
        if indices.min() < 0 or indices.max() >= len(features):
            raise ValueError("hdf5_index is outside face_features bounds")
        matrix = np.asarray(features[indices], dtype=np.float32)

    if matrix.ndim != 2 or matrix.shape[1] != 560 or not np.isfinite(matrix).all():
        raise ValueError("Expected finite FaceFeatureV1 matrix with shape (N, 560)")
    mean = matrix.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = matrix.std(axis=0, dtype=np.float64).astype(np.float32)
    std = np.maximum(std, 1e-6)
    payload = {
        "schema_version": "FaceFeatureV1",
        "representation": "fused_raw",
        "face_mean": torch.from_numpy(mean),
        "face_std": torch.from_numpy(std),
        "n_train": int(len(matrix)),
        "n_train_speakers": int(train["speaker_id"].nunique()),
        "source_features_sha256": sha256(args.features),
        "source_index_sha256": sha256(args.index),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    manifest = args.output.with_suffix(".json")
    manifest.write_text(json.dumps({k: v for k, v in payload.items() if not isinstance(v, torch.Tensor)}, indent=2), encoding="utf-8")
    print(f"Wrote {args.output} for {payload['n_train']} clips / {payload['n_train_speakers']} speakers")


if __name__ == "__main__":
    main()
