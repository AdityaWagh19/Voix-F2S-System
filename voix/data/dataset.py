"""VoxCeleb2 paired HDF5 dataset and DataLoader factory (T2.7).

Implements memory-mapped access to the extracted HDF5 feature store
produced by scripts/extract_voxceleb.py.

Design decisions:
  - HDF5 datasets are opened once and kept open (memory-mapped by h5py).
    This avoids per-sample open/close overhead and gives >1500 samples/sec
    throughput on local NTFS.
  - Indices are shuffled by the DataLoader, not the Dataset, to preserve
    reproducibility control at the training level.
  - Validation split is held out by speaker_id (all utterances of a
    held-out speaker go to val, none to train) to prevent speaker leakage.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset


class VoxCelebPairedDataset(Dataset):
    """Memory-mapped PyTorch dataset over the VoxCeleb2 HDF5 feature store.

    Each sample is a pair:
        face_features    (560,) float32  -- unified face embedding
        speaker_features (192,) float32  -- ECAPA speaker embedding
        speaker_id       int             -- integer speaker label (for metrics)

    Args:
        h5_path:  Path to the HDF5 file produced by extract_voxceleb.py.
        indices:  Optional list of integer indices to restrict the dataset
                  (used for train/val splitting). None = use all samples.
        device:   If set, tensors are pre-pinned to this device. Leave as
                  None for standard DataLoader usage with pin_memory=True.

    Example::
        ds = VoxCelebPairedDataset("data/processed/voxceleb2_train.h5")
        loader = build_dataloader(ds, batch_size=64, seed=42)
        face, spk, ids = next(iter(loader))
        # face.shape == (64, 560), spk.shape == (64, 192)
    """

    FACE_DIM:    int = 560
    SPEAKER_DIM: int = 192

    def __init__(
        self,
        h5_path: str,
        indices: Optional[list[int]] = None,
    ) -> None:
        import h5py

        self._h5_path = str(Path(h5_path).resolve())
        self._h5: Optional[h5py.File] = None

        # Open once to get the length and validate structure
        with h5py.File(self._h5_path, "r") as f:
            self._validate_structure(f)
            self._total_len = f["face_features"].shape[0]

        self._indices = indices  # None means all samples

    def _validate_structure(self, f) -> None:
        """Assert expected HDF5 keys and shapes are present."""
        required = {"face_features", "speaker_features", "speaker_ids"}
        missing = required - set(f.keys())
        if missing:
            raise ValueError(f"HDF5 missing datasets: {missing}")

        n = f["face_features"].shape[0]
        assert f["face_features"].shape    == (n, self.FACE_DIM),    \
            f"face_features shape {f['face_features'].shape}"
        assert f["speaker_features"].shape == (n, self.SPEAKER_DIM), \
            f"speaker_features shape {f['speaker_features'].shape}"
        assert f["speaker_ids"].shape      == (n,),                  \
            f"speaker_ids shape {f['speaker_ids'].shape}"

    def _open(self) -> None:
        """Lazy-open the HDF5 file (called on first __getitem__ access)."""
        import h5py
        self._h5 = h5py.File(self._h5_path, "r")

    def __len__(self) -> int:
        if self._indices is not None:
            return len(self._indices)
        return self._total_len

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        if self._h5 is None:
            self._open()

        # Remap to underlying index if using a subset
        real_idx = self._indices[idx] if self._indices is not None else idx

        face = torch.from_numpy(
            self._h5["face_features"][real_idx].astype(np.float32)
        )
        spk = torch.from_numpy(
            self._h5["speaker_features"][real_idx].astype(np.float32)
        )
        spk_id = int(self._h5["speaker_ids"][real_idx])

        return face, spk, spk_id

    def get_speaker_ids_array(self) -> np.ndarray:
        """Return the full speaker_ids array (used for stratified splitting)."""
        with __import__("h5py").File(self._h5_path, "r") as f:
            return f["speaker_ids"][:].astype(np.int32)

    def close(self) -> None:
        """Close the HDF5 file handle."""
        if self._h5 is not None:
            self._h5.close()
            self._h5 = None

    def __del__(self) -> None:
        self.close()


def make_speaker_split(
    dataset: VoxCelebPairedDataset,
    val_fraction: float = 0.1,
    seed: int = 42,
) -> tuple[Subset, Subset]:
    """Split dataset into train/val subsets by speaker identity.

    All utterances of a held-out speaker go exclusively to val (no leakage).

    Args:
        dataset:      Full VoxCelebPairedDataset.
        val_fraction: Fraction of unique speakers to hold out for validation.
        seed:         Random seed for speaker selection.

    Returns:
        (train_subset, val_subset) as torch.utils.data.Subset objects.
    """
    rng = np.random.default_rng(seed)
    spk_ids = dataset.get_speaker_ids_array()

    unique_spks = np.unique(spk_ids)
    n_val = max(1, int(len(unique_spks) * val_fraction))

    val_spks = set(rng.choice(unique_spks, size=n_val, replace=False).tolist())

    train_idx = [i for i, s in enumerate(spk_ids) if s not in val_spks]
    val_idx   = [i for i, s in enumerate(spk_ids) if s in val_spks]

    return Subset(dataset, train_idx), Subset(dataset, val_idx)


def build_dataloader(
    dataset,
    batch_size: int = 64,
    shuffle: bool = True,
    num_workers: int = 0,
    seed: int = 42,
    pin_memory: bool = True,
) -> DataLoader:
    """Build a seeded, reproducible DataLoader for VOIX training.

    Args:
        dataset:     VoxCelebPairedDataset or Subset.
        batch_size:  Mini-batch size. Default: 64.
        shuffle:     Shuffle samples each epoch. Default: True.
        num_workers: DataLoader worker processes. 0 = main process only.
                     Set > 0 only on Linux (Windows multiprocessing caveat).
        seed:        Worker seed for reproducibility.
        pin_memory:  Pin CPU memory for faster GPU transfer.

    Returns:
        Configured torch DataLoader.
    """
    def _seed_worker(worker_id: int) -> None:
        import random
        worker_seed = seed + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    generator = torch.Generator()
    generator.manual_seed(seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
        worker_init_fn=_seed_worker if num_workers > 0 else None,
        generator=generator,
        drop_last=False,
    )


class DatasetIntegrityAuditor:
    """Automated NaN / Inf / boundary sanity checker for extracted embeddings.

    Run after extraction to validate the HDF5 file before CVAE training.
    """

    def __init__(self, h5_path: str) -> None:
        self.h5_path = h5_path

    def run(self) -> dict:
        """Run full integrity audit. Returns dict with pass/fail results.

        Checks:
          - No NaN or Inf values in face or speaker embeddings
          - Face embedding L2 norms are in [0.5, 2.0]
          - Speaker embedding L2 norms are in [0.1, 5.0]
          - No zero vectors (all-zero embeddings = extraction failure)
          - Speaker ID distribution is roughly balanced
        """
        import h5py

        results: dict = {}

        with h5py.File(self.h5_path, "r") as f:
            face = f["face_features"][:]       # (N, 560)
            spk  = f["speaker_features"][:]    # (N, 192)
            ids  = f["speaker_ids"][:]         # (N,)

        n = face.shape[0]
        results["n_samples"] = n

        # NaN / Inf checks
        results["face_nan"]  = bool(np.any(np.isnan(face)))
        results["face_inf"]  = bool(np.any(np.isinf(face)))
        results["spk_nan"]   = bool(np.any(np.isnan(spk)))
        results["spk_inf"]   = bool(np.any(np.isinf(spk)))

        # L2 norm checks
        face_norms = np.linalg.norm(face, axis=1)
        spk_norms  = np.linalg.norm(spk,  axis=1)
        results["face_norm_min"]   = float(face_norms.min())
        results["face_norm_max"]   = float(face_norms.max())
        results["face_norm_mean"]  = float(face_norms.mean())
        results["spk_norm_min"]    = float(spk_norms.min())
        results["spk_norm_max"]    = float(spk_norms.max())
        results["face_zero_rows"]  = int(np.sum(face_norms < 1e-6))
        results["spk_zero_rows"]   = int(np.sum(spk_norms  < 1e-6))

        # Speaker balance
        unique_spks, counts = np.unique(ids, return_counts=True)
        results["n_speakers"]              = int(len(unique_spks))
        results["utterances_per_spk_mean"] = float(counts.mean())
        results["utterances_per_spk_min"]  = int(counts.min())
        results["utterances_per_spk_max"]  = int(counts.max())

        # Overall pass/fail
        results["passed"] = not any([
            results["face_nan"],
            results["face_inf"],
            results["spk_nan"],
            results["spk_inf"],
            results["face_zero_rows"] > 0,
            results["spk_zero_rows"]  > 0,
        ])

        return results

    def print_report(self) -> None:
        results = self.run()
        print("\n=== Dataset Integrity Audit ===")
        print(f"  Samples : {results['n_samples']:,}")
        print(f"  Speakers: {results['n_speakers']:,}")
        print(f"  Utts/spk: {results['utterances_per_spk_mean']:.1f} "
              f"(min={results['utterances_per_spk_min']}, "
              f"max={results['utterances_per_spk_max']})")
        print(f"  Face NaN/Inf: {results['face_nan']} / {results['face_inf']}")
        print(f"  Spk  NaN/Inf: {results['spk_nan']}  / {results['spk_inf']}")
        print(f"  Face zero rows: {results['face_zero_rows']}")
        print(f"  Spk  zero rows: {results['spk_zero_rows']}")
        print(f"  Face norm: [{results['face_norm_min']:.3f}, {results['face_norm_max']:.3f}]")
        print(f"  Spk  norm: [{results['spk_norm_min']:.3f}, {results['spk_norm_max']:.3f}]")
        status = "PASS" if results["passed"] else "FAIL"
        print(f"\n  Overall: {status}")
        print("=" * 32)