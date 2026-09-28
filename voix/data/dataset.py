"""VoxCeleb2 paired HDF5 dataset and DataLoader factory (T2.7).

Implements memory-mapped and in-memory access to the extracted HDF5 feature store
produced by scripts/extract_voxceleb.py / run_pipeline.py.

Design decisions:
  - By default, in_memory=True preloads the compact feature matrices (~223 MB total)
    into RAM, delivering >50,000 samples/sec throughput and eliminating random
    disk seek bottleneck on NTFS / mechanical / external drives.
  - Optional normalisation: pass a scaler dict loaded from feature_scaler.pt
    to apply per-dimension z-score normalisation. When in_memory=True, normalisation
    is applied once during preload for zero runtime overhead.
  - The canonical split (train/val/test) is driven by training_index.csv
    produced by scripts/build_training_index.py. Speaker identity is
    derived from the 'metadata' HDF5 field (speaker_id/video_id/utt_id)
    rather than the stale integer speaker_ids array.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


# ---------------------------------------------------------------------------
# Default paths
# ---------------------------------------------------------------------------
_DEFAULT_HDF5   = Path("D:/voix/data/processed/features.h5")
_DEFAULT_INDEX  = Path("D:/voix/data/processed/training_index.csv")
_DEFAULT_SCALER = Path("D:/voix/data/processed/feature_scaler.pt")


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class VoxCelebPairedDataset(Dataset):
    """PyTorch dataset over the VoxCeleb2 HDF5 feature store.

    Each sample is a pair:
        face_features    (560,) float32  -- unified face embedding (e_f)
        speaker_features (192,) float32  -- ECAPA speaker embedding (e_s)
        speaker_int_id   int             -- dense integer speaker label

    Args:
        h5_path:    Path to features.h5.
        indices:    Optional list of HDF5 row indices for this split.
        spk_map:    Optional dict mapping speaker string IDs to dense ints.
        scaler:     Optional dict of normalisation tensors (face_mean, etc.).
        in_memory:  If True (default), preloads the slice into RAM for high throughput.
    """

    FACE_DIM:    int = 560
    SPEAKER_DIM: int = 192

    def __init__(
        self,
        h5_path: str,
        indices: Optional[list[int]] = None,
        spk_map: Optional[dict[str, int]] = None,
        scaler:  Optional[dict]           = None,
        in_memory: bool                   = True,
        _preloaded: Optional[tuple[torch.Tensor, torch.Tensor, list[int]]] = None,
    ) -> None:
        import h5py

        self._h5_path = str(Path(h5_path).resolve())
        self._h5: Optional[h5py.File] = None
        self._scaler = scaler
        self._in_memory = in_memory
        self._indices = indices

        if _preloaded is not None:
            self._face_data, self._spk_data, self._spk_ids = _preloaded
            self._total_len = len(self._face_data)
            self._spk_map = spk_map or {}
            self._row_spk = []
            return

        with h5py.File(self._h5_path, "r") as f:
            self._validate_structure(f)
            self._total_len = f["face_features"].shape[0]

            # Build or store the speaker-string -> int mapping
            if spk_map is not None:
                self._spk_map = spk_map
            else:
                meta = [
                    (m.decode() if isinstance(m, bytes) else m)
                    for m in f["metadata"][:]
                ]
                unique_spks = sorted({m.split("/")[0] for m in meta})
                self._spk_map = {s: i for i, s in enumerate(unique_spks)}

            # Store per-row speaker string for quick lookup
            self._row_spk = []
            if "metadata" in f:
                for m in f["metadata"][:]:
                    raw = m.decode() if isinstance(m, bytes) else m
                    self._row_spk.append(raw.split("/")[0])
            else:
                for sid in f["speaker_ids"][:]:
                    self._row_spk.append(str(int(sid)))

            if self._in_memory:
                idx_slice = slice(None) if self._indices is None else np.array(self._indices, dtype=np.int64)
                raw_face = torch.from_numpy(f["face_features"][idx_slice].astype(np.float32))
                raw_spk  = torch.from_numpy(f["speaker_features"][idx_slice].astype(np.float32))

                if self._scaler is not None:
                    raw_face = (raw_face - self._scaler["face_mean"]) / self._scaler["face_std"]
                    raw_spk  = (raw_spk  - self._scaler["spk_mean"])  / self._scaler["spk_std"]

                self._face_data = raw_face
                self._spk_data  = raw_spk
                rows = self._indices if self._indices is not None else range(self._total_len)
                self._spk_ids = [self._spk_map.get(self._row_spk[i], -1) for i in rows]
            else:
                self._face_data = None
                self._spk_data  = None
                self._spk_ids   = None

    # ------------------------------------------------------------------
    # Factory: load from training_index.csv
    # ------------------------------------------------------------------

    @classmethod
    def from_index_csv(
        cls,
        h5_path: str = str(_DEFAULT_HDF5),
        index_csv: str = str(_DEFAULT_INDEX),
        scaler_path: Optional[str] = None,
        in_memory: bool = True,
    ) -> tuple["VoxCelebPairedDataset", "VoxCelebPairedDataset", "VoxCelebPairedDataset"]:
        """Return (train_ds, val_ds, test_ds) loaded from training_index.csv.

        When in_memory=True, reads the compact feature store once into RAM
        and partitions into splits for maximum DataLoader throughput.
        """
        import h5py

        # Load scaler
        scaler = None
        if scaler_path is not None and Path(scaler_path).exists():
            scaler = torch.load(scaler_path, weights_only=True)
        elif Path(_DEFAULT_SCALER).exists() and scaler_path is None:
            scaler = torch.load(str(_DEFAULT_SCALER), weights_only=True)

        # Parse CSV
        split_indices: dict[str, list[int]] = defaultdict(list)
        spk_set: set[str] = set()

        with open(index_csv, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                split_indices[row["split"]].append(int(row["hdf5_index"]))
                spk_set.add(row["speaker_id"])

        spk_map = {s: i for i, s in enumerate(sorted(spk_set))}

        if in_memory:
            # Efficient one-pass read into RAM
            with h5py.File(h5_path, "r") as f:
                all_face = torch.from_numpy(f["face_features"][:].astype(np.float32))
                all_spk  = torch.from_numpy(f["speaker_features"][:].astype(np.float32))
                meta = [
                    (m.decode() if isinstance(m, bytes) else m)
                    for m in f["metadata"][:]
                ]
                row_spks = [m.split("/")[0] for m in meta]

            if scaler is not None:
                all_face = (all_face - scaler["face_mean"]) / scaler["face_std"]
                all_spk  = (all_spk  - scaler["spk_mean"])  / scaler["spk_std"]

            def _make_split(split_name: str) -> "VoxCelebPairedDataset":
                idxs = split_indices[split_name]
                idx_t = torch.tensor(idxs, dtype=torch.long)
                split_face = all_face[idx_t]
                split_spk  = all_spk[idx_t]
                split_ids  = [spk_map.get(row_spks[i], -1) for i in idxs]
                return cls(
                    h5_path=h5_path,
                    indices=idxs,
                    spk_map=spk_map,
                    scaler=scaler,
                    in_memory=True,
                    _preloaded=(split_face, split_spk, split_ids),
                )

            return _make_split("train"), _make_split("val"), _make_split("test")

        # Fallback disk-backed
        train_ds = cls(h5_path, indices=split_indices["train"],  spk_map=spk_map, scaler=scaler, in_memory=False)
        val_ds   = cls(h5_path, indices=split_indices["val"],    spk_map=spk_map, scaler=scaler, in_memory=False)
        test_ds  = cls(h5_path, indices=split_indices["test"],   spk_map=spk_map, scaler=scaler, in_memory=False)
        return train_ds, val_ds, test_ds

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _validate_structure(self, f) -> None:
        required = {"face_features", "speaker_features"}
        missing = required - set(f.keys())
        if missing:
            raise ValueError(f"HDF5 missing datasets: {missing}")

        n = f["face_features"].shape[0]
        assert f["face_features"].shape    == (n, self.FACE_DIM),    \
            f"face_features shape {f['face_features'].shape}"
        assert f["speaker_features"].shape == (n, self.SPEAKER_DIM), \
            f"speaker_features shape {f['speaker_features'].shape}"

        if "metadata" not in f:
            raise ValueError(
                "HDF5 is missing the 'metadata' dataset.  "
                "Ensure you used run_pipeline.py to produce features.h5."
            )

    def _open(self) -> None:
        import h5py
        self._h5 = h5py.File(self._h5_path, "r")

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        if self._in_memory:
            return len(self._face_data)
        if self._indices is not None:
            return len(self._indices)
        return self._total_len

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        if self._in_memory:
            return self._face_data[idx], self._spk_data[idx], self._spk_ids[idx]

        if self._h5 is None:
            self._open()

        real_idx = self._indices[idx] if self._indices is not None else idx

        face = torch.from_numpy(
            self._h5["face_features"][real_idx].astype(np.float32)
        )
        spk = torch.from_numpy(
            self._h5["speaker_features"][real_idx].astype(np.float32)
        )

        if self._scaler is not None:
            face = (face - self._scaler["face_mean"]) / self._scaler["face_std"]
            spk  = (spk  - self._scaler["spk_mean"])  / self._scaler["spk_std"]

        spk_str = self._row_spk[real_idx]
        spk_int = self._spk_map.get(spk_str, -1)

        return face, spk, spk_int

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def get_speaker_ids_array(self) -> np.ndarray:
        if self._in_memory:
            return np.array(self._spk_ids, dtype=np.int32)
        if self._indices is None:
            row_range = range(self._total_len)
        else:
            row_range = self._indices
        return np.array(
            [self._spk_map.get(self._row_spk[i], -1) for i in row_range],
            dtype=np.int32,
        )

    def n_speakers(self) -> int:
        if self._in_memory:
            return len(set(self._spk_ids))
        if self._indices is None:
            return len(self._spk_map)
        spks = {self._row_spk[i] for i in self._indices}
        return len(spks)

    def close(self) -> None:
        if self._h5 is not None:
            try:
                self._h5.close()
            except Exception:
                pass
            self._h5 = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Dataloader factory
# ---------------------------------------------------------------------------

def build_dataloader(
    dataset,
    batch_size: int = 64,
    shuffle: bool = True,
    num_workers: int = 0,
    seed: int = 42,
    pin_memory: bool = True,
) -> DataLoader:
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


# ---------------------------------------------------------------------------
# Integrity auditor
# ---------------------------------------------------------------------------

class DatasetIntegrityAuditor:
    def __init__(self, h5_path: str) -> None:
        self.h5_path = h5_path

    def run(self) -> dict:
        import h5py
        results: dict = {}

        with h5py.File(self.h5_path, "r") as f:
            face = f["face_features"][:]
            spk  = f["speaker_features"][:]
            meta = [
                (m.decode() if isinstance(m, bytes) else m)
                for m in f["metadata"][:]
            ]

        n = face.shape[0]
        results["n_samples"] = n
        results["face_nan"] = bool(np.any(np.isnan(face)))
        results["face_inf"] = bool(np.any(np.isinf(face)))
        results["spk_nan"]  = bool(np.any(np.isnan(spk)))
        results["spk_inf"]  = bool(np.any(np.isinf(spk)))

        face_norms = np.linalg.norm(face, axis=1)
        spk_norms  = np.linalg.norm(spk,  axis=1)
        results["face_norm_min"]  = float(face_norms.min())
        results["face_norm_max"]  = float(face_norms.max())
        results["face_norm_mean"] = float(face_norms.mean())
        results["spk_norm_min"]   = float(spk_norms.min())
        results["spk_norm_max"]   = float(spk_norms.max())
        results["face_zero_rows"] = int(np.sum(face_norms < 1e-6))
        results["spk_zero_rows"]  = int(np.sum(spk_norms  < 1e-6))

        results["dup_clips"]    = len(meta) - len(set(meta))
        bad_fmt = sum(1 for m in meta if len(m.split("/")) != 3)
        results["bad_meta_fmt"] = bad_fmt

        spks = [m.split("/")[0] for m in meta]
        unique_spks, counts = np.unique(spks, return_counts=True)
        results["n_speakers"]              = int(len(unique_spks))
        results["utterances_per_spk_mean"] = float(counts.mean())
        results["utterances_per_spk_min"]  = int(counts.min())
        results["utterances_per_spk_max"]  = int(counts.max())

        results["passed"] = not any([
            results["face_nan"],
            results["face_inf"],
            results["spk_nan"],
            results["spk_inf"],
            results["face_zero_rows"] > 0,
            results["spk_zero_rows"]  > 0,
            results["dup_clips"]      > 0,
            results["bad_meta_fmt"]   > 0,
        ])
        return results

    def print_report(self) -> None:
        results = self.run()
        print("\n=== Dataset Integrity Audit ===")
        print(f"  Samples   : {results['n_samples']:,}")
        print(f"  Speakers  : {results['n_speakers']:,}")
        print(f"  Utts/spk  : {results['utterances_per_spk_mean']:.1f} "
              f"(min={results['utterances_per_spk_min']}, "
              f"max={results['utterances_per_spk_max']})")
        print(f"  Face NaN/Inf : {results['face_nan']} / {results['face_inf']}")
        print(f"  Spk  NaN/Inf : {results['spk_nan']}  / {results['spk_inf']}")
        print(f"  Face zero rows : {results['face_zero_rows']}")
        print(f"  Spk  zero rows : {results['spk_zero_rows']}")
        print(f"  Duplicate clips: {results['dup_clips']}")
        print(f"  Bad metadata   : {results['bad_meta_fmt']}")
        print(f"  Face norm : [{results['face_norm_min']:.3f}, {results['face_norm_max']:.3f}]")
        print(f"  Spk  norm : [{results['spk_norm_min']:.3f}, {results['spk_norm_max']:.3f}]")
        status = "PASS" if results["passed"] else "FAIL"
        print(f"\n  Overall: {status}")
        print("=" * 34)
