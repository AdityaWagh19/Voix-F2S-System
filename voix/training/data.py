"""Shared, speaker-disjoint data access for VOIX trainers and evaluation.

Two corpora are supported:

* the archived V1 paired HDF5 (``clean_index.csv``), kept for reproducing the
  audit's failure analysis;
* the rebuilt FaceFeatureV2 corpus (``voxceleb2_v2/features_v2.h5``), which is
  the authoritative training and serving corpus.

Every trainer must use this module so train/validation/test speaker overlap is
impossible by construction.  The V2 loader additionally exposes raw ArcFace and
the fused 560-D representation from the same rows.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import h5py
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_FEATURES = ROOT / "data" / "processed" / "features.h5"
DEFAULT_INDEX = ROOT / "data" / "processed" / "clean_index.csv"
DEFAULT_SCALER = ROOT / "data" / "processed" / "face_scaler_v1.pt"

# FaceFeatureV2 corpus rebuilt from raw VoxCeleb2 video.  The HDF5 is
# authoritative for row order; the manifest is used for validation only.
DEFAULT_FEATURES_V2 = ROOT / "data" / "processed" / "voxceleb2_v2" / "features_v2.h5"
DEFAULT_INDEX_V2 = ROOT / "data" / "processed" / "voxceleb2_v2" / "index.csv"
DEFAULT_SCALER_V2 = ROOT / "data" / "processed" / "voxceleb2_v2" / "face_scaler_v2.pt"


@dataclass
class CleanFeatureDataset:
    """Clean-index-aligned face/voice features with speaker labels and splits."""

    face_raw: np.ndarray          # (N, D) face features (560 fused or 512 ArcFace)
    speaker_raw: np.ndarray       # (N, 192) ECAPA embeddings
    speaker_ids: np.ndarray       # (N,) str
    split: np.ndarray             # (N,) str
    clip_ids: np.ndarray          # (N,) str
    hdf5_index: np.ndarray        # (N,) int64
    gender: np.ndarray            # (N,) str

    def __len__(self) -> int:
        return len(self.face_raw)

    def subset(
        self,
        split: str,
        *,
        allow_empty: bool = False,
    ) -> "CleanFeatureDataset":
        mask = self.split == split
        if not mask.any() and not allow_empty:
            raise ValueError(f"No rows for split '{split}'")
        return CleanFeatureDataset(
            face_raw=self.face_raw[mask],
            speaker_raw=self.speaker_raw[mask],
            speaker_ids=self.speaker_ids[mask],
            split=self.split[mask],
            clip_ids=self.clip_ids[mask],
            hdf5_index=self.hdf5_index[mask],
            gender=self.gender[mask],
        )

    def select(self, indices) -> "CleanFeatureDataset":
        """Return a row-indexed view, preserving all columns and alignment."""
        idx = np.asarray(indices, dtype=np.int64)
        if idx.size and (idx.min() < 0 or idx.max() >= len(self)):
            raise IndexError("select() index is outside the dataset")
        return CleanFeatureDataset(
            face_raw=self.face_raw[idx],
            speaker_raw=self.speaker_raw[idx],
            speaker_ids=self.speaker_ids[idx],
            split=self.split[idx],
            clip_ids=self.clip_ids[idx],
            hdf5_index=self.hdf5_index[idx],
            gender=self.gender[idx],
        )

    def speaker_to_indices(self) -> Dict[str, np.ndarray]:
        mapping: Dict[str, list[int]] = {}
        for i, spk in enumerate(self.speaker_ids):
            mapping.setdefault(str(spk), []).append(i)
        return {spk: np.asarray(idx, dtype=np.int64) for spk, idx in mapping.items()}


def load_clean_dataset(
    features_path: str | Path = DEFAULT_FEATURES,
    index_path: str | Path = DEFAULT_INDEX,
) -> CleanFeatureDataset:
    """Load HDF5 features in the exact row order of clean_index.csv."""
    features_path = Path(features_path)
    index_path = Path(index_path)
    if not features_path.exists():
        raise FileNotFoundError(f"Features file not found: {features_path}")
    if not index_path.exists():
        raise FileNotFoundError(f"Clean index not found: {index_path}")

    index = pd.read_csv(index_path)
    required = {"hdf5_index", "clip_id", "speaker_id", "split", "gender"}
    missing = required.difference(index.columns)
    if missing:
        raise ValueError(f"clean_index.csv is missing columns: {sorted(missing)}")
    if index["hdf5_index"].duplicated().any():
        raise ValueError("clean_index.csv contains duplicate hdf5_index rows")

    hdf5_idx = index["hdf5_index"].to_numpy(dtype=np.int64)
    with h5py.File(features_path, "r") as store:
        n_total = int(store["face_features"].shape[0])
        if hdf5_idx.min() < 0 or hdf5_idx.max() >= n_total:
            raise ValueError("clean_index.csv hdf5_index is outside the HDF5 range")
        face = np.asarray(store["face_features"][hdf5_idx], dtype=np.float32)
        speaker = np.asarray(store["speaker_features"][hdf5_idx], dtype=np.float32)
        stored_ids = np.asarray(store["speaker_ids"][hdf5_idx])

    # Sanity check: the HDF5 integer speaker id must be consistent with the CSV
    # speaker string. The CSV is authoritative for the split; the HDF5 id is only
    # used to detect accidental row misalignment.
    csv_speaker_codes = pd.factorize(index["speaker_id"].astype(str), sort=True)[0]
    if len(np.unique(stored_ids)) == len(np.unique(csv_speaker_codes)):
        # Both encodings are arbitrary; verify the partitions match.
        stored_groups = {}
        for code, spk in zip(stored_ids, index["speaker_id"].astype(str)):
            stored_groups.setdefault(int(code), set()).add(spk)
        if any(len(group) != 1 for group in stored_groups.values()):
            raise ValueError("HDF5 speaker_ids do not align with clean_index speaker_id")

    if not np.isfinite(face).all():
        raise ValueError("face_features contains non-finite values")
    if not np.isfinite(speaker).all():
        raise ValueError("speaker_features contains non-finite values")

    dataset = CleanFeatureDataset(
        face_raw=face,
        speaker_raw=speaker,
        speaker_ids=index["speaker_id"].astype(str).to_numpy(),
        split=index["split"].astype(str).to_numpy(),
        clip_ids=index["clip_id"].astype(str).to_numpy(),
        hdf5_index=hdf5_idx,
        gender=index["gender"].astype(str).to_numpy(),
    )
    assert_speaker_disjoint(dataset)
    return dataset


def load_v2_dataset(
    features_path: str | Path = DEFAULT_FEATURES_V2,
    index_path: str | Path | None = DEFAULT_INDEX_V2,
    *,
    face_key: str = "face_features",
) -> CleanFeatureDataset:
    """Load a FaceFeatureV2 HDF5 built by ``extract_voxceleb2_v2.py``.

    The HDF5 is authoritative for row order.  When ``index_path`` is given,
    every extracted clip is checked against the manifest so a stale or
    misaligned corpus cannot silently enter training.
    """
    features_path = Path(features_path)
    if not features_path.exists():
        raise FileNotFoundError(f"Features file not found: {features_path}")

    with h5py.File(features_path, "r") as store:
        required = {"clip_ids", "speaker_ids", "split", "gender", face_key, "speaker_features"}
        missing = required.difference(store.keys())
        if missing:
            raise ValueError(
                f"{features_path} is not a FaceFeatureV2 corpus; missing {sorted(missing)}"
            )
        clip_ids = _decode_strings(store["clip_ids"][:])
        speaker_ids = _decode_strings(store["speaker_ids"][:])
        split = _decode_strings(store["split"][:])
        gender = _decode_strings(store["gender"][:])
        face = np.asarray(store[face_key][:], dtype=np.float32)
        speaker = np.asarray(store["speaker_features"][:], dtype=np.float32)

    n = len(clip_ids)
    if not (len(speaker_ids) == len(split) == len(gender) == len(face) == len(speaker) == n):
        raise ValueError("FaceFeatureV2 arrays have inconsistent lengths")
    if len(set(clip_ids.tolist())) != n:
        raise ValueError("FaceFeatureV2 contains duplicate clip_ids")
    if not np.isfinite(face).all():
        raise ValueError(f"{face_key} contains non-finite values")
    if not np.isfinite(speaker).all():
        raise ValueError("speaker_features contains non-finite values")

    if index_path is not None:
        index_path = Path(index_path)
        if index_path.exists():
            manifest = pd.read_csv(index_path)
            required_cols = {"clip_id", "speaker_id", "split", "gender"}
            missing_cols = required_cols.difference(manifest.columns)
            if missing_cols:
                raise ValueError(
                    f"Manifest {index_path} is missing columns: {sorted(missing_cols)}"
                )
            manifest = manifest.drop_duplicates("clip_id").set_index("clip_id")
            unknown = [cid for cid in clip_ids if cid not in manifest.index]
            if unknown:
                raise ValueError(
                    f"{len(unknown)} extracted clips are not in the manifest, "
                    f"e.g. {unknown[:3]}"
                )
            for i, cid in enumerate(clip_ids):
                row = manifest.loc[cid]
                if str(row["speaker_id"]) != speaker_ids[i]:
                    raise ValueError(f"speaker mismatch for {cid}")
                if str(row["split"]) != split[i]:
                    raise ValueError(f"split mismatch for {cid}")
                if str(row["gender"]) != gender[i]:
                    raise ValueError(f"gender mismatch for {cid}")

    dataset = CleanFeatureDataset(
        face_raw=face,
        speaker_raw=speaker,
        speaker_ids=speaker_ids,
        split=split,
        clip_ids=clip_ids,
        hdf5_index=np.arange(n, dtype=np.int64),
        gender=gender,
    )
    assert_speaker_disjoint(dataset)
    return dataset


def _decode_strings(values: Iterable) -> np.ndarray:
    return np.asarray(
        [v.decode("utf-8") if isinstance(v, (bytes, np.bytes_)) else str(v) for v in values],
        dtype=object,
    )


def validate_scaler_for_dataset(
    scaler: dict,
    dataset: CleanFeatureDataset,
    *,
    face_key: str = "face_features",
) -> dict:
    """Fail when a scaler does not describe this corpus's train split.

    A scaler built on shard 1 is still 560-D and still has the right key, so a
    dimension-only check silently accepts it after the corpus grows.  The row
    and speaker counts make that staleness visible instead of training the
    full model on the wrong normalization.
    """
    train = dataset.subset("train")
    expected_rows = len(train)
    expected_speakers = len({str(s) for s in train.speaker_ids})
    if scaler.get("schema_version") != "FaceFeatureV2":
        raise ValueError(
            f"Scaler schema is {scaler.get('schema_version')!r}, expected "
            "'FaceFeatureV2'"
        )
    if scaler.get("face_key", face_key) != face_key:
        raise ValueError(
            f"Scaler face_key is {scaler.get('face_key')!r}, expected {face_key!r}"
        )
    if tuple(scaler["face_mean"].shape) != (dataset.face_raw.shape[1],):
        raise ValueError(
            f"Scaler dimension {tuple(scaler['face_mean'].shape)} does not match "
            f"the corpus dimension {dataset.face_raw.shape[1]}"
        )
    if int(scaler.get("n_train", -1)) != expected_rows:
        raise ValueError(
            f"Scaler was fit on {scaler.get('n_train')} train rows but the corpus "
            f"has {expected_rows}; rebuild it with scripts/data/build_face_scaler_v2.py"
        )
    if int(scaler.get("n_train_speakers", -1)) != expected_speakers:
        raise ValueError(
            f"Scaler was fit on {scaler.get('n_train_speakers')} train speakers "
            f"but the corpus has {expected_speakers}; rebuild it with "
            "scripts/data/build_face_scaler_v2.py"
        )
    return scaler


def build_face_scaler_v2(
    dataset: CleanFeatureDataset,
    output_path: str | Path = DEFAULT_SCALER_V2,
    *,
    source: str = "",
    face_key: str = "face_features",
) -> dict:
    """Fit the FaceFeatureV2 z-score scaler on the train split and save it.

    ``face_key`` is recorded so the 560-D fused scaler can never be mistaken
    for the 512-D raw-ArcFace scaler (or vice versa).
    """
    train = dataset.subset("train")
    face = np.asarray(train.face_raw, dtype=np.float64)
    mean = face.mean(axis=0)
    std = face.std(axis=0, ddof=0)
    std = np.maximum(std, 1e-6)
    payload = {
        "face_mean": torch.tensor(mean, dtype=torch.float32),
        "face_std": torch.tensor(std, dtype=torch.float32),
        "schema_version": "FaceFeatureV2",
        "face_key": str(face_key),
        "feature_dim": int(face.shape[1]),
        "n_train": int(len(train)),
        "n_train_speakers": int(len(set(train.speaker_ids.tolist()))),
        "source": str(source),
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)
    return payload


def assert_speaker_disjoint(dataset: CleanFeatureDataset) -> None:
    """Fail if any speaker appears in more than one split."""
    splits = {}
    for split in sorted(set(dataset.split)):
        splits[split] = set(dataset.speaker_ids[dataset.split == split].tolist())
    names = list(splits)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            overlap = splits[names[i]] & splits[names[j]]
            if overlap:
                raise ValueError(
                    f"Speaker leakage between {names[i]} and {names[j]}: "
                    f"{sorted(overlap)[:5]} ({len(overlap)} speakers)"
                )


def split_summary(dataset: CleanFeatureDataset) -> Dict[str, Dict[str, int]]:
    return {
        split: {
            "clips": int((dataset.split == split).sum()),
            "speakers": int(len(set(dataset.speaker_ids[dataset.split == split].tolist()))),
        }
        for split in sorted(set(dataset.split))
    }


def normalize_speaker_features(speaker: np.ndarray | torch.Tensor) -> torch.Tensor:
    """L2-normalize ECAPA targets (they are unnormalized in the HDF5 corpus)."""
    tensor = torch.as_tensor(speaker, dtype=torch.float32)
    return torch.nn.functional.normalize(tensor, p=2, dim=-1)


class SpeakerBatchSampler:
    """Sample P speakers x K clips per batch for supervised contrastive training."""

    def __init__(
        self,
        speaker_to_indices: Dict[str, np.ndarray],
        speakers_per_batch: int = 32,
        clips_per_speaker: int = 4,
        seed: int = 42,
        drop_last: bool = False,
    ) -> None:
        if speakers_per_batch < 2:
            raise ValueError("speakers_per_batch must be >= 2 for contrastive learning")
        if clips_per_speaker < 2:
            raise ValueError("clips_per_speaker must be >= 2 to form positive pairs")
        self.mapping = speaker_to_indices
        self.speakers = sorted(speaker_to_indices)
        self.p = speakers_per_batch
        self.k = clips_per_speaker
        self.seed = seed
        self.drop_last = drop_last
        self.epoch = 0

    def __len__(self) -> int:
        n = len(self.speakers) // self.p
        if not self.drop_last and len(self.speakers) % self.p:
            n += 1
        return max(1, n)

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        self.epoch += 1
        order = rng.permutation(len(self.speakers))
        for start in range(0, len(order), self.p):
            chosen = order[start:start + self.p]
            if len(chosen) < 2:
                continue
            batch = []
            for speaker_idx in chosen:
                spk = self.speakers[int(speaker_idx)]
                indices = self.mapping[spk]
                replace = len(indices) < self.k
                picks = rng.choice(indices, size=self.k, replace=replace)
                batch.extend(int(i) for i in picks)
            yield batch
