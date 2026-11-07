"""Validate a FaceFeatureV2 corpus before spending GPU time on training.

The corpus is built incrementally, so the checks distinguish hard contract
violations from expected extraction gaps:

* hard: missing datasets, inconsistent lengths, duplicate clip ids, non-finite
  features, speaker leakage across splits, extracted clips absent from the
  manifest, inconsistent gender labels for a speaker;
* soft: extraction failures, manifest clips with no extracted row, speakers
  with fewer than two clips.

With ``--expect-complete`` every manifest clip must be either extracted or
recorded in the failures CSV; an unaccounted clip becomes a hard error.  The
failures CSV is deduplicated by clip id because resumed extraction runs append
to it, and a later successful extraction supersedes an earlier failure.

Usage:
    python scripts/audit/validate_v2_corpus.py \
        --features data/processed/voxceleb2_v2/features_v2.h5 \
        --index data/processed/voxceleb2_v2/index.csv \
        --failures data/processed/voxceleb2_v2/extraction_failures_v2.csv \
        --output data/processed/voxceleb2_v2/corpus_validation.json \
        --expect-complete
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, Optional

import h5py
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

REQUIRED_DATASETS = {
    "clip_ids",
    "speaker_ids",
    "split",
    "gender",
    "face_features",
    "arcface",
    "speaker_features",
}
FEATURE_DATASETS = ("face_features", "arcface", "speaker_features")


def _decode(values: Iterable) -> np.ndarray:
    return np.asarray(
        [
            value.decode("utf-8")
            if isinstance(value, (bytes, np.bytes_))
            else str(value)
            for value in values
        ],
        dtype=object,
    )


def _speaker_split_overlap(
    speakers: np.ndarray, splits: np.ndarray
) -> Dict[str, int]:
    split_speakers = {
        split: set(speakers[splits == split].tolist())
        for split in sorted(set(splits.tolist()))
    }
    names = sorted(split_speakers)
    overlap: Dict[str, int] = {}
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            count = len(split_speakers[left] & split_speakers[right])
            if count:
                overlap[f"{left}__{right}"] = int(count)
    return overlap


def _failure_summary(
    failures_path: Optional[Path],
    manifest: Optional[pd.DataFrame],
    extracted: set,
) -> Dict[str, object]:
    if failures_path is None or not failures_path.exists():
        return {
            "path": str(failures_path) if failures_path else None,
            "available": False,
            "rows": 0,
            "unique_clips": 0,
            "effective_failures": 0,
            "recovered_clips": 0,
            "by_reason": {},
        }

    frame = pd.read_csv(failures_path)
    if "clip_id" not in frame:
        raise ValueError(f"{failures_path} has no clip_id column")
    rows = len(frame)
    frame = frame.drop_duplicates("clip_id", keep="last")
    failure_ids = set(frame["clip_id"].astype(str))
    recovered = failure_ids & extracted
    effective = failure_ids - extracted
    by_reason = {
        str(reason): int(count)
        for reason, count in frame["reason"].value_counts().items()
    } if "reason" in frame else {}
    summary: Dict[str, object] = {
        "path": str(failures_path),
        "available": True,
        "rows": int(rows),
        "unique_clips": int(len(failure_ids)),
        "effective_failures": int(len(effective)),
        "recovered_clips": int(len(recovered)),
        "by_reason": by_reason,
    }
    if manifest is not None and len(manifest):
        denominator = max(1, len(manifest))
        summary["failure_rate"] = float(len(effective) / denominator)
        counts = frame[frame["clip_id"].astype(str).isin(effective)][
            "speaker_id"
        ].astype(str).value_counts() if "speaker_id" in frame else pd.Series(dtype=int)
        if len(counts):
            manifest_counts = (
                manifest["speaker_id"].astype(str).value_counts()
            )
            high = []
            for speaker, failed in counts.items():
                total = int(manifest_counts.get(speaker, 0))
                if total and failed / total >= 0.5:
                    high.append(
                        {
                            "speaker_id": str(speaker),
                            "failed": int(failed),
                            "manifest_clips": total,
                        }
                    )
            high.sort(key=lambda item: (-item["failed"], item["speaker_id"]))
            summary["speakers_with_ge_50pct_failed"] = high
    return summary


def validate_corpus(
    features_path: str | Path,
    index_path: Optional[str | Path] = None,
    *,
    failures_path: Optional[str | Path] = None,
    expect_complete: bool = False,
    min_clips_per_speaker: int = 2,
) -> dict:
    """Return a JSON-serializable validation report.

    Raises only for unreadable inputs; contract violations are reported in
    ``hard_errors`` so callers can decide whether to abort.
    """
    features_path = Path(features_path)
    hard_errors: list[str] = []
    warnings: list[str] = []

    with h5py.File(features_path, "r") as store:
        missing = REQUIRED_DATASETS.difference(store.keys())
        if missing:
            hard_errors.append(f"missing datasets: {sorted(missing)}")
            return {
                "features": str(features_path),
                "status": "fail",
                "hard_errors": hard_errors,
                "warnings": warnings,
            }
        clip_ids = _decode(store["clip_ids"][:])
        speakers = _decode(store["speaker_ids"][:])
        splits = _decode(store["split"][:])
        genders = _decode(store["gender"][:])
        features = {
            name: np.asarray(store[name][:], dtype=np.float32)
            for name in FEATURE_DATASETS
        }
        shards = (
            np.asarray(store["shard"][:], dtype=np.int32)
            if "shard" in store
            else None
        )

    n = len(clip_ids)
    lengths = {
        "speaker_ids": len(speakers),
        "split": len(splits),
        "gender": len(genders),
        **{name: len(values) for name, values in features.items()},
    }
    if shards is not None:
        lengths["shard"] = len(shards)
    inconsistent = {name: length for name, length in lengths.items() if length != n}
    if inconsistent:
        hard_errors.append(f"inconsistent lengths for n={n}: {inconsistent}")

    if len(set(clip_ids.tolist())) != n:
        duplicates = pd.Series(clip_ids).value_counts()
        hard_errors.append(
            "duplicate clip_ids: "
            + json.dumps(
                {str(k): int(v) for k, v in duplicates[duplicates > 1].head(10).items()}
            )
        )

    non_finite = {}
    for name, values in features.items():
        if values.ndim != 2 or values.shape[0] != n:
            hard_errors.append(
                f"{name} has shape {tuple(values.shape)}, expected ({n}, D)"
            )
            continue
        bad = int((~np.isfinite(values)).any(axis=1).sum())
        if bad:
            non_finite[name] = bad
    if non_finite:
        hard_errors.append(f"non-finite feature rows: {non_finite}")

    overlap = _speaker_split_overlap(speakers, splits)
    if overlap:
        hard_errors.append(f"speaker leakage across splits: {overlap}")

    gender_conflicts = []
    frame = pd.DataFrame(
        {"speaker_id": speakers, "gender": genders, "split": splits}
    )
    for speaker, group in frame.groupby("speaker_id"):
        known = group.loc[group["gender"].isin(["f", "m"]), "gender"]
        if known.nunique() > 1:
            gender_conflicts.append(str(speaker))
    if gender_conflicts:
        hard_errors.append(
            f"{len(gender_conflicts)} speakers have conflicting gender labels: "
            f"{gender_conflicts[:10]}"
        )

    manifest: Optional[pd.DataFrame] = None
    manifest_summary: Dict[str, object] = {"available": False}
    if index_path is not None and Path(index_path).exists():
        manifest = pd.read_csv(index_path)
        if "clip_id" not in manifest or "speaker_id" not in manifest:
            raise ValueError(f"{index_path} must contain clip_id and speaker_id")
        manifest = manifest.drop_duplicates("clip_id", keep="last")
        manifest_ids = set(manifest["clip_id"].astype(str))
        extracted = set(clip_ids.tolist())
        unknown = extracted - manifest_ids
        if unknown:
            hard_errors.append(
                f"{len(unknown)} extracted clips are absent from the manifest: "
                f"{sorted(unknown)[:10]}"
            )
        missing_clips = manifest_ids - extracted
        split_counts = (
            manifest["split"].astype(str).value_counts().to_dict()
            if "split" in manifest
            else {}
        )
        manifest_summary = {
            "available": True,
            "path": str(index_path),
            "clips": int(len(manifest)),
            "speakers": int(manifest["speaker_id"].nunique()),
            "splits": {str(k): int(v) for k, v in split_counts.items()},
            "extracted": int(len(extracted & manifest_ids)),
            "missing": int(len(missing_clips)),
        }
        if "split" in manifest:
            missing_by_split = (
                manifest[manifest["clip_id"].astype(str).isin(missing_clips)][
                    "split"
                ].astype(str).value_counts().to_dict()
            )
            manifest_summary["missing_by_split"] = {
                str(k): int(v) for k, v in missing_by_split.items()
            }
        if shards is not None:
            manifest_summary["shards_extracted"] = sorted(
                int(value) for value in np.unique(shards)
            )
            if "shard" in manifest:
                expected_shards = sorted(
                    int(value)
                    for value in pd.to_numeric(
                        manifest["shard"], errors="coerce"
                    ).dropna().unique()
                )
                manifest_summary["shards_expected"] = expected_shards
                missing_shards = sorted(
                    set(expected_shards) - set(manifest_summary["shards_extracted"])
                )
                if missing_shards:
                    manifest_summary["shards_missing"] = missing_shards
                    warnings.append(
                        f"manifest shards have no extracted rows: {missing_shards}"
                    )
    else:
        extracted = set(clip_ids.tolist())

    failure_summary = _failure_summary(
        Path(failures_path) if failures_path else None,
        manifest,
        extracted,
    )
    if manifest is not None:
        accounted = set(extracted)
        if failure_summary.get("available"):
            raw = pd.read_csv(Path(failures_path)).drop_duplicates(
                "clip_id", keep="last"
            )
            accounted |= set(raw["clip_id"].astype(str))
        unaccounted = set(manifest["clip_id"].astype(str)) - accounted
        failure_summary["unaccounted"] = int(len(unaccounted))
        failure_summary["unaccounted_sample"] = sorted(unaccounted)[:10]
        if unaccounted and expect_complete:
            hard_errors.append(
                f"{len(unaccounted)} manifest clips are neither extracted nor in "
                f"the failures CSV: {sorted(unaccounted)[:10]}"
            )
        effective = int(failure_summary.get("effective_failures", 0))
        rate = float(failure_summary.get("failure_rate", 0.0))
        if rate > 0.25:
            warnings.append(f"extraction failure rate {rate:.1%} exceeds 25%")
        elif effective:
            warnings.append(
                f"{effective} clips failed extraction ({rate:.1%} of manifest)"
            )

    per_speaker = pd.Series(speakers).value_counts()
    coverage = {
        "speakers": int(len(per_speaker)),
        "clips_per_speaker_min": int(per_speaker.min()) if len(per_speaker) else 0,
        "clips_per_speaker_median": float(per_speaker.median())
        if len(per_speaker)
        else 0.0,
        "clips_per_speaker_max": int(per_speaker.max()) if len(per_speaker) else 0,
        "speakers_below_min": int((per_speaker < min_clips_per_speaker).sum()),
    }
    if manifest is not None:
        manifest_speakers = set(manifest["speaker_id"].astype(str))
        extracted_speakers = set(speakers.tolist())
        zero = sorted(manifest_speakers - extracted_speakers)
        coverage["manifest_speakers"] = int(len(manifest_speakers))
        coverage["speakers_with_zero_clips"] = int(len(zero))
        coverage["speakers_with_zero_clips_sample"] = zero[:10]
        if zero:
            warnings.append(
                f"{len(zero)} manifest speakers have no extracted clips"
            )
    if coverage["speakers_below_min"]:
        warnings.append(
            f"{coverage['speakers_below_min']} speakers have fewer than "
            f"{min_clips_per_speaker} clips"
        )

    split_summary = {}
    for split in sorted(set(splits.tolist())):
        mask = splits == split
        split_summary[split] = {
            "clips": int(mask.sum()),
            "speakers": int(len(set(speakers[mask].tolist()))),
            "gender": {
                str(k): int(v)
                for k, v in pd.Series(genders[mask]).value_counts().items()
            },
        }

    return {
        "features": str(features_path),
        "status": "fail" if hard_errors else "pass",
        "expect_complete": bool(expect_complete),
        "clips": int(n),
        "speakers": int(len(set(speakers.tolist()))),
        "splits": split_summary,
        "coverage": coverage,
        "manifest": manifest_summary,
        "failures": failure_summary,
        "feature_dims": {
            name: int(values.shape[1]) if values.ndim == 2 else None
            for name, values in features.items()
        },
        "hard_errors": hard_errors,
        "warnings": warnings,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a FaceFeatureV2 corpus")
    parser.add_argument("--features", required=True, type=Path)
    parser.add_argument("--index", type=Path, default=None)
    parser.add_argument("--failures", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--expect-complete",
        action="store_true",
        help="Treat manifest clips missing from both the HDF5 and failures CSV as errors",
    )
    parser.add_argument("--min-clips-per-speaker", type=int, default=2)
    parser.add_argument(
        "--allow-failures",
        action="store_true",
        help="Exit 0 even when hard contract errors are present (diagnostic only)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = validate_corpus(
        args.features,
        args.index,
        failures_path=args.failures,
        expect_complete=args.expect_complete,
        min_clips_per_speaker=args.min_clips_per_speaker,
    )
    text = json.dumps(report, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
        print(f"Wrote {args.output}")
    print(text)
    if report["status"] != "pass" and not args.allow_failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
