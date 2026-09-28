"""Build deterministic training_index.csv and feature_scaler.pt (Pre-Phase 3).

Reads D:/voix/data/processed/features.h5 (80,832 clips, 4,978 speakers) and
produces:

  data/processed/training_index.csv  -- one row per clip with split label
  data/processed/feature_scaler.pt   -- mean/std tensors for e_f and e_s

Speaker-disjoint split strategy (all clips of a speaker go to exactly one split):
  - Test speakers  : 200 speakers held out  (100 M + 100 F, >= 15 clips each)
  - Val  speakers  : 300 speakers held out  (150 M + 150 F, >= 15 clips each)
  - Train speakers : remaining speakers that have >= 10 clips

Gender stratification is enforced at each split boundary so the test and val
pools are perfectly balanced even though the full dataset is male-heavy.

Why >= 15 clips for test/val?  SECS and Recall@K require at least one
enrolled reference embedding per speaker -- a speaker with only 1-2 clips
cannot be reliably evaluated.  We use the higher bar (>= 15) only for
test/val; training accepts speakers with >= 10 clips.

Usage:
    python scripts/build_training_index.py
    python scripts/build_training_index.py --test-speakers 200 --val-speakers 300
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT          = Path("D:/voix")
HDF5_PATH     = ROOT / "data/processed/features.h5"
META_CSV_PATH = ROOT / "data/raw/voxceleb2/vox2_meta.csv"
OUT_INDEX     = ROOT / "data/processed/training_index.csv"
OUT_SCALER    = ROOT / "data/processed/feature_scaler.pt"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_gender_map(meta_path: Path) -> dict[str, str]:
    """Return {speaker_id: 'm'|'f'} from vox2_meta.csv."""
    gmap: dict[str, str] = {}
    with open(meta_path, newline="", encoding="utf-8") as f:
        for row in csv.reader(f):
            if not row or row[0].startswith("VoxCeleb"):
                continue
            parts = [p.strip() for p in row]
            if len(parts) >= 3:
                gmap[parts[0]] = parts[2].lower()
    return gmap


def split_speakers(
    speaker_clip_counts: dict[str, int],
    gender_map: dict[str, str],
    n_test: int,
    n_val: int,
    min_clips_eval: int = 15,
    min_clips_train: int = 10,
    seed: int = 42,
) -> tuple[set[str], set[str], set[str]]:
    """Return (test_spks, val_spks, train_spks) disjoint sets.

    Strategy:
      1. Eligible eval speakers: >= min_clips_eval and known gender.
      2. From eligible, gender-stratify test then val pools.
      3. Everything remaining with >= min_clips_train goes to train.
    """
    rng = np.random.default_rng(seed)

    eligible = {
        s for s, cnt in speaker_clip_counts.items()
        if cnt >= min_clips_eval and gender_map.get(s) in ("m", "f")
    }

    males   = sorted(s for s in eligible if gender_map[s] == "m")
    females = sorted(s for s in eligible if gender_map[s] == "f")
    rng.shuffle(males)
    rng.shuffle(females)

    n_test_half = n_test // 2
    n_val_half  = n_val  // 2

    needed_m = n_test_half + n_val_half
    needed_f = n_test_half + n_val_half

    if len(males) < needed_m:
        raise RuntimeError(
            f"Not enough male speakers for eval: need {needed_m}, have {len(males)}"
        )
    if len(females) < needed_f:
        raise RuntimeError(
            f"Not enough female speakers for eval: need {needed_f}, have {len(females)}"
        )

    test_m = set(males[:n_test_half])
    test_f = set(females[:n_test_half])
    val_m  = set(males[n_test_half : n_test_half + n_val_half])
    val_f  = set(females[n_test_half : n_test_half + n_val_half])

    test_spks  = test_m | test_f
    val_spks   = val_m  | val_f
    eval_spks  = test_spks | val_spks

    train_spks = {
        s for s, cnt in speaker_clip_counts.items()
        if s not in eval_spks and cnt >= min_clips_train
    }

    return test_spks, val_spks, train_spks


def compute_scaler(
    face_feats: np.ndarray,
    spk_feats: np.ndarray,
    train_mask: np.ndarray,
) -> dict:
    """Compute mean/std from training samples only (no eval leakage)."""
    face_train = face_feats[train_mask]
    spk_train  = spk_feats[train_mask]

    return {
        "face_mean":     torch.from_numpy(face_train.mean(axis=0).astype(np.float32)),
        "face_std":      torch.from_numpy(face_train.std(axis=0).astype(np.float32) + 1e-8),
        "spk_mean":      torch.from_numpy(spk_train.mean(axis=0).astype(np.float32)),
        "spk_std":       torch.from_numpy(spk_train.std(axis=0).astype(np.float32) + 1e-8),
        "face_l2_mean":  float(np.linalg.norm(face_train, axis=1).mean()),
        "spk_l2_mean":   float(np.linalg.norm(spk_train,  axis=1).mean()),
        "n_train_clips": int(train_mask.sum()),
        "face_dim":      face_feats.shape[1],
        "spk_dim":       spk_feats.shape[1],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build VOIX training index and feature scaler"
    )
    parser.add_argument("--hdf5",            default=str(HDF5_PATH))
    parser.add_argument("--meta",            default=str(META_CSV_PATH))
    parser.add_argument("--out-index",       default=str(OUT_INDEX))
    parser.add_argument("--out-scaler",      default=str(OUT_SCALER))
    parser.add_argument("--test-speakers",   type=int, default=200)
    parser.add_argument("--val-speakers",    type=int, default=300)
    parser.add_argument("--min-clips-eval",  type=int, default=15)
    parser.add_argument("--min-clips-train", type=int, default=10)
    parser.add_argument("--seed",            type=int, default=42)
    args = parser.parse_args()

    import h5py

    print("=" * 60)
    print("  VOIX Pre-Phase-3: Build Training Index & Feature Scaler")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Load HDF5
    # ------------------------------------------------------------------
    print(f"\n[1/5] Loading HDF5: {args.hdf5}")
    with h5py.File(args.hdf5, "r") as f:
        face_feats = f["face_features"][:].astype(np.float32)     # (N, 560)
        spk_feats  = f["speaker_features"][:].astype(np.float32)  # (N, 192)
        meta_raw   = [
            (m.decode() if isinstance(m, bytes) else m)
            for m in f["metadata"][:]
        ]

    N = len(meta_raw)
    print(f"  Loaded {N:,} clips | face_dim={face_feats.shape[1]} | spk_dim={spk_feats.shape[1]}")

    # ------------------------------------------------------------------
    # 2. Parse metadata -> per-speaker clip inventory
    # ------------------------------------------------------------------
    print("\n[2/5] Parsing speaker metadata...")
    gender_map = load_gender_map(Path(args.meta))
    print(f"  Loaded gender map for {len(gender_map):,} speakers")

    spk_to_indices: dict[str, list[int]] = defaultdict(list)
    for i, m in enumerate(meta_raw):
        spk_to_indices[m.split("/")[0]].append(i)

    speaker_clip_counts = {s: len(idxs) for s, idxs in spk_to_indices.items()}
    print(f"  Unique speakers in HDF5: {len(speaker_clip_counts):,}")

    # ------------------------------------------------------------------
    # 3. Speaker-disjoint splits
    # ------------------------------------------------------------------
    print("\n[3/5] Building speaker-disjoint splits...")
    test_spks, val_spks, train_spks = split_speakers(
        speaker_clip_counts = speaker_clip_counts,
        gender_map          = gender_map,
        n_test              = args.test_speakers,
        n_val               = args.val_speakers,
        min_clips_eval      = args.min_clips_eval,
        min_clips_train     = args.min_clips_train,
        seed                = args.seed,
    )

    def gender_counts(spk_set: set[str]) -> tuple[int, int]:
        m = sum(1 for s in spk_set if gender_map.get(s) == "m")
        f = sum(1 for s in spk_set if gender_map.get(s) == "f")
        return m, f

    tm,  tf  = gender_counts(test_spks)
    vm,  vf  = gender_counts(val_spks)
    trm, trf = gender_counts(train_spks)

    print(f"  Test  : {len(test_spks):>4} speakers  (M={tm}, F={tf})")
    print(f"  Val   : {len(val_spks):>4} speakers  (M={vm}, F={vf})")
    print(f"  Train : {len(train_spks):>4} speakers  (M={trm}, F={trf})")

    # ------------------------------------------------------------------
    # 4. Build per-clip assignment and write CSV
    # ------------------------------------------------------------------
    print(f"\n[4/5] Writing training index -> {args.out_index}")

    rows: list[dict] = []
    split_clip_counts: dict[str, int] = {"train": 0, "val": 0, "test": 0, "excluded": 0}
    train_mask = np.zeros(N, dtype=bool)

    for i, m in enumerate(meta_raw):
        parts  = m.split("/")
        spk_id = parts[0]
        vid_id = parts[1] if len(parts) > 1 else ""
        utt_id = parts[2] if len(parts) > 2 else ""
        gender = gender_map.get(spk_id, "u")

        if spk_id in test_spks:
            split = "test"
        elif spk_id in val_spks:
            split = "val"
        elif spk_id in train_spks:
            split = "train"
            train_mask[i] = True
        else:
            split_clip_counts["excluded"] += 1
            continue  # speaker has < min_clips_train

        split_clip_counts[split] += 1
        rows.append({
            "hdf5_index":   i,
            "clip_id":      m,
            "speaker_id":   spk_id,
            "video_id":     vid_id,
            "utterance_id": utt_id,
            "gender":       gender,
            "split":        split,
        })

    Path(args.out_index).parent.mkdir(parents=True, exist_ok=True)
    fields = ["hdf5_index", "clip_id", "speaker_id", "video_id",
              "utterance_id", "gender", "split"]
    with open(args.out_index, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    total_indexed = sum(v for k, v in split_clip_counts.items() if k != "excluded")
    print(f"  train clips : {split_clip_counts['train']:>7,}")
    print(f"  val   clips : {split_clip_counts['val']:>7,}")
    print(f"  test  clips : {split_clip_counts['test']:>7,}")
    print(f"  excluded    : {split_clip_counts['excluded']:>7,}  (too few clips per speaker)")
    print(f"  Total indexed: {total_indexed:,}")

    # ------------------------------------------------------------------
    # 5. Compute and save feature scaler (train split only â€” no leakage)
    # ------------------------------------------------------------------
    print(f"\n[5/5] Computing feature scaler (train-split only) -> {args.out_scaler}")
    scaler = compute_scaler(face_feats, spk_feats, train_mask)

    print(f"  face_features: dim={scaler['face_dim']}, L2 mean={scaler['face_l2_mean']:.4f}")
    print(f"  spk_features:  dim={scaler['spk_dim']},  L2 mean={scaler['spk_l2_mean']:.4f}")
    print(f"  face_mean  range : [{scaler['face_mean'].min().item():.4f}, "
          f"{scaler['face_mean'].max().item():.4f}]")
    print(f"  face_std   range : [{scaler['face_std'].min().item():.4f}, "
          f"{scaler['face_std'].max().item():.4f}]")
    print(f"  spk_mean   range : [{scaler['spk_mean'].min().item():.4f}, "
          f"{scaler['spk_mean'].max().item():.4f}]")
    print(f"  spk_std    range : [{scaler['spk_std'].min().item():.4f}, "
          f"{scaler['spk_std'].max().item():.4f}]")

    Path(args.out_scaler).parent.mkdir(parents=True, exist_ok=True)
    torch.save(scaler, args.out_scaler)

    print("\n" + "=" * 60)
    print("  PRE-PHASE-3 INDEX BUILD COMPLETE")
    print(f"  Index  : {args.out_index}")
    print(f"  Scaler : {args.out_scaler}")
    print("=" * 60)


if __name__ == "__main__":
    main()
