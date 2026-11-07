"""Build the A3 class-contrast accent residual.

GLOBE gives each speaker one accent, so subtracting a speaker's own mean style
and then averaging by accent is mathematically zero: every speaker's centered
mean is exactly zero.  This module therefore estimates a speaker-balanced mean
style per accent class and subtracts the unweighted mean of those class means.
The result is a class-contrast direction, not a speaker-disentangled one.

Accent and speaker identity remain confounded because GLOBE has no
within-speaker cross-accent pairs.  The timbre half of the residual is
down-weighted at application time, and the A3 evaluator measures ECAPA drift
against the real style donor.  This is a simple baseline for the learned A4
adapter, not a serving checkpoint.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_CACHE = ROOT / "data" / "processed" / "globe_accent_style_cache.npz"
DEFAULT_MANIFEST = ROOT / "data" / "processed" / "globe_accent_manifest.csv"
DEFAULT_ARTIFACT = ROOT / "checkpoints" / "accent_residuals" / "class_mean.pt"
DEFAULT_REPORT = ROOT / "results" / "accent" / "a3" / "class_mean_report.json"
TRAINABLE = ("us", "england", "australian", "canadian", "india_south_asia")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build class-mean accent residuals")
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--split", default="train")
    parser.add_argument("--classes", nargs="+", default=list(TRAINABLE))
    parser.add_argument("--min-clips-per-speaker", type=int, default=2)
    return parser.parse_args(argv)


def build_class_means(
    frame: pd.DataFrame,
    classes: Sequence[str],
    *,
    min_clips_per_speaker: int = 2,
    split: str = "train",
) -> Dict[str, Any]:
    """Return class-contrast residual directions.

    Per-speaker centering is intentionally not used.  GLOBE speakers have one
    accent, so every speaker's centered mean is exactly zero and the class mean
    of those centered vectors would be zero.  Instead, each speaker's mean
    style is averaged into its accent class with speakers weighted equally, and
    the unweighted mean of the class means is subtracted.
    """
    required = {"clip_id", "speaker_id", "accent_class", "split", "style"}
    missing = required - set(frame.columns)
    if missing:
        raise KeyError(f"style frame is missing columns: {sorted(missing)}")
    work = frame[frame["accent_class"].isin(classes)].copy()
    work = work[work["split"] == split]
    if work.empty:
        raise RuntimeError(f"no rows for split {split!r} and the requested classes")

    work["style"] = work["style"].map(
        lambda value: np.asarray(value, dtype=np.float32).reshape(-1)
    )
    counts = work.groupby("speaker_id").size()
    eligible = set(counts[counts >= min_clips_per_speaker].index)
    work = work[work["speaker_id"].isin(eligible)]
    if work.empty:
        raise RuntimeError("no speaker had enough clips for a stable mean")

    speaker_means: Dict[str, np.ndarray] = {}
    speaker_classes: Dict[str, str] = {}
    for speaker, group in work.groupby("speaker_id"):
        labels = sorted(group["accent_class"].astype(str).unique())
        if len(labels) != 1:
            raise ValueError(
                f"speaker {speaker!r} spans accents {labels}; the class-contrast "
                "baseline assumes one accent per speaker"
            )
        speaker_means[str(speaker)] = np.stack(group["style"].to_list()).mean(axis=0)
        speaker_classes[str(speaker)] = labels[0]

    class_means: Dict[str, np.ndarray] = {}
    for accent in classes:
        vectors = [
            mean
            for speaker, mean in speaker_means.items()
            if speaker_classes[speaker] == accent
        ]
        if not vectors:
            continue
        class_means[str(accent)] = np.stack(vectors).mean(axis=0)

    if not class_means:
        raise RuntimeError("no class residuals were built")

    global_mean = np.stack(list(class_means.values())).mean(axis=0)
    residuals = {
        accent: torch.from_numpy((mean - global_mean).astype(np.float32))
        for accent, mean in class_means.items()
    }
    raw_norms = {
        accent: float(residual.norm().item())
        for accent, residual in residuals.items()
    }
    speaker_counts = {
        accent: sum(1 for label in speaker_classes.values() if label == accent)
        for accent in residuals
    }
    clip_counts = {
        accent: int((work["accent_class"].astype(str) == accent).sum())
        for accent in residuals
    }

    matrix = torch.stack([residuals[accent] for accent in residuals])
    cosine = torch.nn.functional.normalize(matrix, dim=-1) @ torch.nn.functional.normalize(
        matrix, dim=-1
    ).T
    between = float(
        np.var(np.stack(list(class_means.values())), axis=0).sum()
    )
    within_parts = []
    for accent in class_means:
        vectors = np.stack(
            [
                mean
                for speaker, mean in speaker_means.items()
                if speaker_classes[speaker] == accent
            ]
        )
        if len(vectors) >= 2:
            within_parts.append(float(np.var(vectors, axis=0).sum()))
    within = float(np.mean(within_parts)) if within_parts else 0.0
    return {
        "residuals": residuals,
        "raw_norms": raw_norms,
        "speaker_counts": speaker_counts,
        "clip_counts": clip_counts,
        "cosine": cosine,
        "between_class_variance": between,
        "within_class_variance": within,
        "separation_ratio": between / max(within, 1e-12),
        "n_speakers": int(work["speaker_id"].nunique()),
        "n_clips": int(len(work)),
        "global_mean": torch.from_numpy(global_mean.astype(np.float32)),
        "class_mean_norms": {
            accent: float(np.linalg.norm(mean))
            for accent, mean in class_means.items()
        },
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if not args.cache.is_file():
        raise FileNotFoundError(f"style cache not found: {args.cache}")
    cache = np.load(args.cache, allow_pickle=True)
    frame = pd.DataFrame(
        {
            "clip_id": cache["clip_id"].astype(str),
            "speaker_id": cache["speaker_id"].astype(str),
            "accent_class": cache["accent_class"].astype(str),
            "split": cache["split"].astype(str),
            "style": [row for row in cache["style"]],
        }
    )
    classes = list(dict.fromkeys(str(item).lower() for item in args.classes))
    result = build_class_means(
        frame,
        classes,
        min_clips_per_speaker=args.min_clips_per_speaker,
        split=str(args.split),
    )

    metadata = {
        "schema_version": "voix.accent_class_mean.v2",
        "track": "accent",
        "level": "exploratory",
        "role": "a3_baseline_not_serving",
        "residual_kind": "class_contrast",
        "style_cache": str(args.cache),
        "style_cache_sha256": _sha256(args.cache),
        "manifest": str(args.manifest),
        "manifest_sha256": _sha256(args.manifest),
        "split": str(args.split),
        "classes": list(result["residuals"]),
        "style_dim": int(next(iter(result["residuals"].values())).numel()),
        "speaker_centered": False,
        "centering": "class_contrast_global_mean",
        "min_clips_per_speaker": int(args.min_clips_per_speaker),
        "speaker_counts": result["speaker_counts"],
        "clip_counts": result["clip_counts"],
        "raw_norms": result["raw_norms"],
        "class_mean_norms": result["class_mean_norms"],
        "global_mean_norm": float(result["global_mean"].norm().item()),
        "between_class_variance": result["between_class_variance"],
        "within_class_variance": result["within_class_variance"],
        "separation_ratio": result["separation_ratio"],
        "n_speakers": result["n_speakers"],
        "n_clips": result["n_clips"],
        "notes": [
            "Class-contrast baseline; this is not a learned adapter.",
            "Per-speaker centering is not used: with one accent per GLOBE speaker, every speaker's centered mean and therefore every class mean is exactly zero.",
            "Each speaker's mean style is averaged into its accent class with speakers weighted equally, then the unweighted class-mean average is subtracted.",
            "Accent and speaker identity are confounded because GLOBE has no within-speaker cross-accent pairs; the timbre half is down-weighted and identity drift is measured.",
            "Trained on speaker_split=train only; val/test are not used.",
        ],
    }
    args.artifact.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": metadata["schema_version"],
            "metadata": metadata,
            "classes": list(result["residuals"]),
            "residuals": result["residuals"],
            "cosine": result["cosine"],
            "global_mean": result["global_mean"],
        },
        args.artifact,
    )
    metadata["artifact"] = str(args.artifact)
    metadata["artifact_sha256"] = _sha256(args.artifact)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(
        f"Wrote {args.artifact} ({len(result['residuals'])} classes, "
        f"{result['n_speakers']} speakers, {result['n_clips']} clips)"
    )
    print(f"Wrote {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
