"""Build the AMI style cache used by the N0b prosody prototype.

The frozen StyleTTS2 style encoder is run on real AMI headset audio.  The cache
stores the 256-D style vector and objective delivery metrics for each clip.  The
N0b prototype is the mean prosody half (dims 128:256) of the training clips;
validation and test speakers provide the held-out reference distribution.

AMI is used only by the naturalness track.  This script never touches the CFVN,
CVAE, style adapter, emotion adapter, or accent adapter.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from voix.evaluation.naturalness import CLIP_METRICS, extract_naturalness_metrics  # noqa: E402

DEFAULT_MANIFEST = ROOT / "data" / "processed" / "ami_naturalness" / "manifest.csv"
DEFAULT_CACHE = ROOT / "data" / "processed" / "ami_style_cache.npz"
DEFAULT_REPORT = ROOT / "results" / "naturalness" / "n0b" / "style_cache_report.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract AMI style vectors")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--max-per-speaker", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--reuse-cache", action="store_true")
    return parser.parse_args(argv)


def _sample(frame: pd.DataFrame, max_per_speaker: int) -> pd.DataFrame:
    """Deterministically sample clips, spreading them across speakers."""
    pieces = []
    for _, group in frame.groupby("speaker_id", sort=True):
        pieces.append(group.sort_values("clip_id").head(max(1, int(max_per_speaker))))
    if not pieces:
        return frame.iloc[0:0]
    return pd.concat(pieces, ignore_index=True).sort_values("clip_id").reset_index(drop=True)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if not args.manifest.is_file():
        raise FileNotFoundError(f"AMI manifest not found: {args.manifest}")
    frame = pd.read_csv(args.manifest)
    required = {"clip_id", "speaker_id", "split", "wav_path", "text", "duration_sec"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"AMI manifest is missing columns: {missing}")
    splits = [str(item).lower() for item in args.splits]
    frame = frame[frame["split"].astype(str).isin(splits)].copy()
    if args.limit is not None:
        frame = frame.head(int(args.limit))
    frame = _sample(frame, args.max_per_speaker)
    if frame.empty:
        raise RuntimeError("no AMI clips selected")
    missing_audio = [
        str(path) for path in frame["wav_path"].astype(str) if not Path(path).is_file()
    ]
    if missing_audio:
        raise FileNotFoundError(f"missing AMI audio: {missing_audio[:5]}")
    print(
        f"Selected {len(frame)} clips from {frame['speaker_id'].nunique()} speakers "
        f"across splits {splits}",
        flush=True,
    )

    cached: Dict[str, np.ndarray] = {}
    if args.reuse_cache and args.cache.is_file():
        cache = np.load(args.cache, allow_pickle=True)
        cached = {key: cache[key] for key in cache.files}
        print(f"Reusing {len(cached.get('clip_id', []))} cached clip(s)", flush=True)
    done = {str(item) for item in cached.get("clip_id", [])}
    pending = frame[~frame["clip_id"].astype(str).isin(done)].reset_index(drop=True)

    clip_ids: List[str] = [str(item) for item in cached.get("clip_id", [])]
    speaker_ids: List[str] = [str(item) for item in cached.get("speaker_id", [])]
    split_values: List[str] = [str(item) for item in cached.get("split", [])]
    meeting_ids: List[str] = [str(item) for item in cached.get("meeting_id", [])]
    styles: List[np.ndarray] = [
        np.asarray(item, dtype=np.float32) for item in cached.get("style", [])
    ]
    metrics: Dict[str, List[Optional[float]]] = {
        name: [float(value) if value is not None else np.nan for value in cached.get(name, [])]
        for name in CLIP_METRICS
    }

    failures: List[Dict[str, str]] = []
    if len(pending):
        import torch

        from voix.models.tts_backend import StyleTTS2Backend

        print(f"Loading StyleTTS2 on {args.device} ...", flush=True)
        backend = StyleTTS2Backend(
            device=args.device, use_neural=True, allow_fallback=False
        )
        backend.load()
        started = time.time()
        with torch.no_grad():
            for index, (_, row) in enumerate(pending.iterrows(), start=1):
                path = str(row["wav_path"])
                try:
                    style = backend.compute_style(path)
                    values = extract_naturalness_metrics(
                        path, text=str(row["text"]), sample_rate=24000
                    )
                except Exception as exc:  # pragma: no cover - runtime guard
                    failures.append(
                        {
                            "clip_id": str(row["clip_id"]),
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
                    continue
                clip_ids.append(str(row["clip_id"]))
                speaker_ids.append(str(row["speaker_id"]))
                split_values.append(str(row["split"]))
                meeting_ids.append(str(row["meeting_id"]))
                styles.append(style.detach().cpu().numpy().reshape(-1).astype(np.float32))
                for name in CLIP_METRICS:
                    value = values.get(name)
                    metrics[name].append(float(value) if value is not None else np.nan)
                if index % 50 == 0 or index == len(pending):
                    elapsed = time.time() - started
                    print(
                        f"  style {index}/{len(pending)} "
                        f"({index / max(elapsed, 1e-6):.2f}/s, {elapsed:.0f}s, "
                        f"failures={len(failures)})",
                        flush=True,
                    )

    style_array = np.asarray(styles, dtype=np.float32)
    payload: Dict[str, Any] = {
        "clip_id": np.asarray(clip_ids, dtype=str),
        "speaker_id": np.asarray(speaker_ids, dtype=str),
        "split": np.asarray(split_values, dtype=str),
        "meeting_id": np.asarray(meeting_ids, dtype=str),
        "style": style_array,
    }
    for name in CLIP_METRICS:
        payload[name] = np.asarray(metrics[name], dtype=np.float32)
    args.cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.cache, **payload)

    summary: Dict[str, Any] = {
        "schema_version": "voix.ami_style_cache.v1",
        "track": "naturalness",
        "level": "exploratory",
        "manifest": str(args.manifest),
        "manifest_sha256": _sha256(args.manifest),
        "cache": str(args.cache),
        "cache_sha256": _sha256(args.cache),
        "splits": splits,
        "clips": int(len(clip_ids)),
        "speakers": int(len(set(speaker_ids))),
        "style_dim": int(style_array.shape[1]) if style_array.ndim == 2 else 0,
        "counts": {
            split: int(sum(1 for value in split_values if value == split))
            for split in splits
        },
        "speakers_by_split": {
            split: sorted(
                {
                    speaker
                    for speaker, value in zip(speaker_ids, split_values)
                    if value == split
                }
            )
            for split in splits
        },
        "metrics_by_split": {
            split: {
                name: float(
                    np.nanmean(
                        [
                            value
                            for value, value_split in zip(metrics[name], split_values)
                            if value_split == split
                        ]
                    )
                )
                if any(
                    value_split == split and np.isfinite(value)
                    for value, value_split in zip(metrics[name], split_values)
                )
                else None
                for name in CLIP_METRICS
            }
            for split in splits
        },
        "failures": failures,
        "notes": [
            "Style vectors come from the frozen StyleTTS2 style/predictor encoder.",
            "AMI is used only for naturalness; no identity or expression artifact is trained here.",
            "The N0b prototype uses train only; val/test are held-out references.",
        ],
        "mos": {"status": "unavailable", "reason": "no human ratings collected"},
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {args.cache} ({len(clip_ids)} styles)")
    print(f"Wrote {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
