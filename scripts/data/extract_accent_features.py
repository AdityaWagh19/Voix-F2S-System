"""Extract ECAPA + prosody/spectral features for the GLOBE accent baseline.

Labels are self-reported GLOBE accent classes.  ``predicted_accent`` is dataset
metadata and is never used.  The speaker-disjoint split column is
``speaker_split``; the manifest's ``split`` column is not a speaker split.

Examples::

    # Five trainable classes, all speaker splits
    python scripts/data/extract_accent_features.py

    # Smoke on 200 clips
    python scripts/data/extract_accent_features.py --limit 200

    # Include Irish as an evaluation-only class
    python scripts/data/extract_accent_features.py --include-irish
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_MANIFEST = ROOT / "data" / "processed" / "globe_accent_manifest.csv"
DEFAULT_AUDIO_ROOT = ROOT / "data" / "processed" / "globe_audio"
DEFAULT_OUTPUT = ROOT / "data" / "processed" / "globe_accent_features.npz"
DEFAULT_REPORT = ROOT / "results" / "accent" / "a0" / "feature_report.json"
TRAINABLE_CLASSES = (
    "us",
    "england",
    "australian",
    "canadian",
    "india_south_asia",
)
EVAL_ONLY_CLASSES = ("irish",)
FEATURE_SCHEMA = "voix.globe_accent_features.v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract GLOBE accent features")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--audio-root", type=Path, default=DEFAULT_AUDIO_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--classes", nargs="+", default=list(TRAINABLE_CLASSES))
    parser.add_argument("--include-irish", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--checkpoint-every", type=int, default=500)
    parser.add_argument("--reuse-cache", action="store_true")
    return parser.parse_args(argv)


def _audio_path(row: pd.Series, audio_root: Path) -> Path:
    return (
        audio_root
        / str(row["speaker_split"])
        / str(row["accent_class"])
        / f"{row['clip_id']}.wav"
    )


def _load_existing(path: Path) -> Optional[Dict[str, np.ndarray]]:
    if not path.is_file():
        return None
    cache = np.load(path, allow_pickle=True)
    return {key: cache[key] for key in cache.files}


def _write_cache(
    path: Path,
    *,
    clip_ids: List[str],
    paths: List[str],
    labels: List[str],
    speakers: List[str],
    splits: List[str],
    features: List[np.ndarray],
    feature_names: Sequence[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        clip_id=np.asarray(clip_ids, dtype=str),
        path=np.asarray(paths, dtype=str),
        label=np.asarray(labels, dtype=str),
        speaker_id=np.asarray(speakers, dtype=str),
        split=np.asarray(splits, dtype=str),
        features=np.asarray(features, dtype=np.float32),
        feature_names=np.asarray(list(feature_names), dtype=str),
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    from voix.data.speaker_extractor import ECAPAExtractor
    from voix.evaluation.accent_features import (
        PROSODY_FEATURE_NAMES,
        combine_features,
        extract_prosody_spectral_features,
        feature_names,
    )

    if not args.manifest.is_file():
        raise FileNotFoundError(f"GLOBE manifest not found: {args.manifest}")
    manifest = pd.read_csv(args.manifest)
    required = {"clip_id", "accent_class", "speaker_split", "client_id"}
    missing = sorted(required - set(manifest.columns))
    if missing:
        raise ValueError(f"manifest is missing columns: {missing}")
    if manifest["clip_id"].duplicated().any():
        raise ValueError("manifest contains duplicate clip_id values")

    classes = [str(item).strip().lower() for item in args.classes]
    if args.include_irish:
        classes += list(EVAL_ONLY_CLASSES)
    unknown = sorted(set(classes) - set(TRAINABLE_CLASSES) - set(EVAL_ONLY_CLASSES))
    if unknown:
        raise ValueError(f"unknown accent classes: {unknown}")

    splits = [str(item).strip().lower() for item in args.splits]
    frame = manifest[
        manifest["accent_class"].astype(str).isin(classes)
        & manifest["speaker_split"].astype(str).isin(splits)
    ].copy()
    if frame.empty:
        raise ValueError("no rows selected")

    # Speaker-disjointness is the whole point of this split.
    leaks = frame.groupby("client_id")["speaker_split"].nunique()
    leaking = sorted(leaks[leaks > 1].index.astype(str).tolist())
    if leaking:
        raise ValueError(f"speakers appear in multiple speaker splits: {leaking[:10]}")

    frame["audio_path"] = [
        str(_audio_path(row, args.audio_root)) for _, row in frame.iterrows()
    ]
    missing_audio = sorted(
        {path for path in frame["audio_path"] if not Path(path).is_file()}
    )
    if missing_audio:
        raise FileNotFoundError(f"missing GLOBE audio: {missing_audio[:5]}")
    if args.limit is not None:
        frame = frame.head(int(args.limit)).copy()
    if frame.empty:
        raise ValueError("no rows left after --limit")

    clip_ids: List[str] = []
    paths: List[str] = []
    labels: List[str] = []
    speakers: List[str] = []
    split_values: List[str] = []
    features: List[np.ndarray] = []
    done: set[str] = set()
    if args.reuse_cache:
        existing = _load_existing(args.output)
        if existing is not None:
            clip_ids = existing["clip_id"].astype(str).tolist()
            paths = existing["path"].astype(str).tolist()
            labels = existing["label"].astype(str).tolist()
            speakers = existing["speaker_id"].astype(str).tolist()
            split_values = existing["split"].astype(str).tolist()
            features = [row for row in existing["features"]]
            done = set(clip_ids)
            print(f"Reusing {len(done)} cached clips", flush=True)

    selected = frame[~frame["clip_id"].astype(str).isin(done)].copy()
    print(
        f"Extracting {len(selected)} clip(s) across {len(classes)} class(es) "
        f"and splits {splits}",
        flush=True,
    )

    print(f"Loading ECAPA on {args.device} ...", flush=True)
    ecapa = ECAPAExtractor(device=args.device)
    names = feature_names(ecapa.EMBEDDING_DIM)
    failures: List[Dict[str, str]] = []
    started = time.time()

    for index, (_, row) in enumerate(selected.iterrows(), start=1):
        clip_id = str(row["clip_id"])
        path = str(row["audio_path"])
        try:
            embedding = (
                ecapa.extract_from_path(path).detach().cpu().numpy().reshape(-1)
            )
            norm = float(np.linalg.norm(embedding))
            if norm > 0:
                embedding = embedding / norm
            prosody = extract_prosody_spectral_features(path)
            vector = combine_features(embedding, prosody)
        except Exception as exc:  # pragma: no cover - runtime guard
            failures.append(
                {"clip_id": clip_id, "path": path, "error": f"{type(exc).__name__}: {exc}"}
            )
            continue
        clip_ids.append(clip_id)
        paths.append(path)
        labels.append(str(row["accent_class"]))
        speakers.append(str(row["client_id"]))
        split_values.append(str(row["speaker_split"]))
        features.append(vector)
        if index % 100 == 0 or index == len(selected):
            elapsed = time.time() - started
            rate = index / elapsed if elapsed > 0 else 0.0
            print(
                f"  {index}/{len(selected)} ({rate:.2f}/s, {elapsed:.0f}s, "
                f"failures={len(failures)})",
                flush=True,
            )
        if args.checkpoint_every > 0 and index % args.checkpoint_every == 0:
            _write_cache(
                args.output,
                clip_ids=clip_ids,
                paths=paths,
                labels=labels,
                speakers=speakers,
                splits=split_values,
                features=features,
                feature_names=names,
            )

    if not features:
        raise RuntimeError("no features were extracted")
    _write_cache(
        args.output,
        clip_ids=clip_ids,
        paths=paths,
        labels=labels,
        speakers=speakers,
        splits=split_values,
        features=features,
        feature_names=names,
    )

    feature_matrix = np.asarray(features, dtype=np.float32)
    label_array = np.asarray(labels, dtype=str)
    split_array = np.asarray(split_values, dtype=str)
    counts: Dict[str, Any] = {}
    for split in splits:
        for label in classes:
            mask = (split_array == split) & (label_array == label)
            if not mask.any():
                continue
            counts[f"{split}:{label}"] = {
                "clips": int(mask.sum()),
                "speakers": int(len(set(np.asarray(speakers)[mask]))),
                "hours": round(
                    float(
                        feature_matrix[
                            mask, names.index("duration_seconds")
                        ].sum()
                    )
                    / 3600.0,
                    4,
                ),
            }

    report = {
        "schema_version": FEATURE_SCHEMA,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "level": "exploratory",
        "manifest": str(args.manifest),
        "manifest_sha256": _sha256(args.manifest),
        "audio_root": str(args.audio_root),
        "output": str(args.output),
        "output_sha256": _sha256(args.output),
        "classes": classes,
        "eval_only_classes": list(EVAL_ONLY_CLASSES) if args.include_irish else [],
        "splits": splits,
        "feature_dim": int(feature_matrix.shape[1]),
        "ecapa_dim": int(ecapa.EMBEDDING_DIM),
        "prosody_dim": len(PROSODY_FEATURE_NAMES),
        "feature_names": list(names),
        "f0_method": "librosa.yin",
        "ecapa_normalized": True,
        "counts": counts,
        "clips": int(len(clip_ids)),
        "speakers": int(len(set(speakers))),
        "failures": failures,
        "notes": [
            "Labels are self-reported GLOBE accent classes.",
            "speaker_split is the speaker-disjoint split; the manifest split column is not used.",
            "predicted_accent is dataset metadata and is not a label or evaluation.",
            "Irish, if included, is evaluation-only and must not be used for training.",
            "F0 uses YIN for throughput; pYIN is ~35x slower on this machine.",
        ],
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Wrote {args.output} ({len(clip_ids)} clips, {feature_matrix.shape[1]} dims)", flush=True)
    print(f"Wrote {args.report}", flush=True)
    if failures:
        print(f"WARNING: {len(failures)} clip(s) failed; see report", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
