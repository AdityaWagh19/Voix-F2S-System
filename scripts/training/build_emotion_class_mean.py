"""Build the E1 class-mean emotion residual from real CREMA-D train audio.

For every training actor and sentence, select one neutral clip and one clip per
emotion, extract the frozen StyleTTS2 style/predictor vector for each, and
average ``style(emotion) - style(neutral)`` per emotion with speakers weighted
equally.  Validation and test actors are excluded.

The output artifact is a baseline, not a trained model.  It exists so E2's
learned residual must beat a strong simple alternative before it can justify
its extra complexity.

Examples::

    python scripts/training/build_emotion_class_mean.py
    python scripts/training/build_emotion_class_mean.py --reuse-cache
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

DEFAULT_MANIFEST = ROOT / "data" / "processed" / "crema_d_manifest.csv"
DEFAULT_CACHE = ROOT / "data" / "processed" / "crema_d_style_cache_train.npz"
DEFAULT_ARTIFACT = ROOT / "checkpoints" / "emotion_residuals" / "class_mean.pt"
DEFAULT_REPORT = ROOT / "results" / "emotion" / "e1" / "class_mean_report.json"
REPORT_SCHEMA = "voix.emotion_class_mean_report.v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve(path: str) -> Path:
    candidate = Path(str(path))
    return candidate if candidate.is_absolute() else ROOT / candidate


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build class-mean emotion residuals")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--split", default="train")
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--reuse-cache", action="store_true")
    parser.add_argument("--n-resamples", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args(argv)


def _load_cache(path: Path) -> Dict[str, np.ndarray]:
    cache = np.load(path, allow_pickle=True)
    return {
        "paths": cache["paths"].astype(str),
        "styles": cache["styles"].astype(np.float32),
    }


def _save_cache(path: Path, paths: Sequence[str], styles: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, paths=np.asarray(paths, dtype=str), styles=styles)


def _compute_styles(
    paths: Sequence[str],
    *,
    device: str,
    cache_path: Path,
    reuse_cache: bool,
) -> tuple[np.ndarray, Dict[str, Any]]:
    import torch

    from voix.models.tts_backend import StyleTTS2Backend

    requested = list(dict.fromkeys(str(path) for path in paths))
    if reuse_cache and cache_path.is_file():
        cached = _load_cache(cache_path)
        mapping = {path: index for index, path in enumerate(cached["paths"])}
        if all(path in mapping for path in requested):
            styles = cached["styles"][[mapping[path] for path in requested]]
            return styles, {
                "status": "reused",
                "cache": str(cache_path),
                "cache_sha256": _sha256(cache_path),
                "clips": len(requested),
            }

    print(f"Loading StyleTTS2 on {device} ...", flush=True)
    backend = StyleTTS2Backend(device=device, use_neural=True, allow_fallback=False)
    backend.load()
    styles = np.zeros((len(requested), 256), dtype=np.float32)
    started = time.time()
    with torch.no_grad():
        for index, path in enumerate(requested, start=1):
            style = backend.compute_style(str(_resolve(path)))
            styles[index - 1] = style.detach().cpu().numpy().reshape(-1).astype(np.float32)
            if index % 100 == 0 or index == len(requested):
                elapsed = time.time() - started
                rate = index / elapsed if elapsed > 0 else 0.0
                print(
                    f"  style {index}/{len(requested)} ({rate:.1f}/s, {elapsed:.0f}s)",
                    flush=True,
                )
    _save_cache(cache_path, requested, styles)
    return styles, {
        "status": "computed",
        "cache": str(cache_path),
        "cache_sha256": _sha256(cache_path),
        "clips": len(requested),
        "seconds": round(time.time() - started, 2),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    import torch

    from voix.models.emotion_residual import (
        NON_NEUTRAL_EMOTIONS,
        apply_style_residual,
        build_pairs,
        class_mean_residuals,
        save_artifact,
        select_one_per_group,
    )

    if not args.manifest.is_file():
        raise FileNotFoundError(f"CREMA-D manifest not found: {args.manifest}")

    frame = pd.read_csv(args.manifest)
    required = {"actor_id", "sentence", "intended_emotion", "intensity", "clip_id", "path", "split"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"manifest is missing columns: {missing}")

    train = frame[frame["split"].astype(str) == args.split].copy()
    if train.empty:
        raise ValueError(f"no rows for split {args.split!r}")
    excluded = frame[frame["split"].astype(str) != args.split]
    overlap = set(train["actor_id"].astype(str)) & set(excluded["actor_id"].astype(str))
    if overlap:
        raise ValueError(f"split leaks actors into other splits: {sorted(overlap)}")

    clips = select_one_per_group(train)
    pairs = build_pairs(clips, emotions=NON_NEUTRAL_EMOTIONS)
    train_actors = sorted(clips["actor_id"].astype(str).unique())
    if len(train_actors) < 2:
        raise ValueError("class-mean estimation requires at least two training speakers")

    # Validate audio before touching the GPU.
    missing_audio = [str(path) for path in pairs["neutral_path"]]
    missing_audio += [str(path) for path in pairs["emotion_path"]]
    missing_audio = sorted(
        {path for path in missing_audio if not _resolve(path).is_file()}
    )
    if missing_audio:
        raise FileNotFoundError(f"missing paired audio: {missing_audio[:5]}")

    paths = list(pairs["neutral_path"].astype(str)) + list(
        pairs["emotion_path"].astype(str)
    )
    styles, cache_info = _compute_styles(
        paths,
        device=args.device,
        cache_path=args.cache,
        reuse_cache=args.reuse_cache,
    )
    style_by_path = {
        str(path): styles[index]
        for index, path in enumerate(dict.fromkeys(str(path) for path in paths))
    }

    records: List[Dict[str, Any]] = []
    for _, pair in pairs.iterrows():
        neutral = np.asarray(style_by_path[str(pair["neutral_path"])], dtype=np.float64)
        emotion = np.asarray(style_by_path[str(pair["emotion_path"])], dtype=np.float64)
        records.append(
            {
                "emotion": str(pair["emotion"]),
                "speaker_id": str(pair["actor_id"]),
                "sentence": str(pair["sentence"]),
                "delta": emotion - neutral,
            }
        )

    results = class_mean_residuals(
        records,
        prosody_only=True,
        n_resamples=args.n_resamples,
        seed=args.seed,
    )

    # Neutral-zero is an exact contract: intensity 0 and a zero residual must
    # return the input object unchanged.  Verify it on a real style vector.
    sample_style = next(iter(style_by_path.values()))
    sample_style = torch.from_numpy(sample_style).reshape(1, -1)
    neutral_zero_max = 0.0
    for emotion, stats in results.items():
        zero = apply_style_residual(sample_style, stats.delta, intensity=0.0)
        neutral_zero_max = max(
            neutral_zero_max,
            float((zero - sample_style).abs().max().item()),
        )
    if neutral_zero_max > 1e-7:
        raise RuntimeError(f"neutral-zero check failed: {neutral_zero_max:.3e}")

    metadata = {
        "split": args.split,
        "train_actors": train_actors,
        "n_train_actors": len(train_actors),
        "excluded_splits": sorted(excluded["split"].astype(str).unique()),
        "excluded_actors": sorted(excluded["actor_id"].astype(str).unique()),
        "source_manifest": str(args.manifest),
        "source_manifest_sha256": _sha256(args.manifest),
        "style_source": "styletts2_style_encoder",
        "pairing": "same actor and sentence; emotion style minus neutral style",
        "speaker_weighting": "each speaker contributes one mean vector",
        "scale": 0.28,
        "scale_note": (
            "Matches the fixed-delta effective scale; intensity multiplies this "
            "value. The raw class mean is stored, not pre-scaled."
        ),
    }
    save_artifact(args.artifact, results, metadata=metadata)

    report = {
        "schema_version": REPORT_SCHEMA,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "level": "exploratory",
        "split": args.split,
        "train_actors": train_actors,
        "excluded_actors": sorted(excluded["actor_id"].astype(str).unique()),
        "manifest": str(args.manifest),
        "manifest_sha256": _sha256(args.manifest),
        "artifact": str(args.artifact),
        "artifact_sha256": _sha256(args.artifact),
        "style_cache": cache_info,
        "selected_clips": int(len(clips)),
        "pairs": int(len(pairs)),
        "pairs_per_emotion": {
            str(emotion): int(count)
            for emotion, count in pairs["emotion"].value_counts().sort_index().items()
        },
        "residuals": {
            emotion: stats.to_dict() for emotion, stats in sorted(results.items())
        },
        "neutral_zero_max_abs_error": neutral_zero_max,
        "neutral_zero_passed": bool(neutral_zero_max <= 1e-7),
        "notes": [
            "Class-mean residual is estimated from train actors only; val/test actors are excluded.",
            "Each speaker contributes one mean vector, so speakers are equally weighted.",
            "The artifact stores both the full 256-D mean and a prosody-only variant.",
            "Fear and disgust are available here even though no fixed-delta artifact exists.",
            "This is a baseline, not a learned model, and no human MOS was measured.",
        ],
        "limitations": [
            "CREMA-D is acted speech and every class is below the 2 h confirmatory gate.",
            "The class mean mixes recording/timbre variation with emotion; the full variant may drift identity.",
            "A class-mean cannot model within-class or within-speaker variation.",
        ],
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Pairs: {len(pairs)} across {len(train_actors)} train actors", flush=True)
    for emotion, stats in sorted(results.items()):
        print(
            f"  {emotion}: pairs={stats.n_pairs} speakers={stats.n_speakers} "
            f"norm={stats.delta.norm().item():.3f} "
            f"timbre={stats.delta[:128].norm().item():.3f} "
            f"prosody={stats.delta[128:].norm().item():.3f}",
            flush=True,
        )
    print(f"Neutral-zero max abs error: {neutral_zero_max:.3e}", flush=True)
    print(f"Wrote {args.artifact}", flush=True)
    print(f"Wrote {args.report}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
