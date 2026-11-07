"""Extract frozen StyleTTS2 style vectors for GLOBE accent clips.

The style encoder is the face-free identity base used by the accent research
track.  This cache is used only for accent residual experiments; it never
touches the canonical face adapter, CFVN, or CVAE.
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

DEFAULT_MANIFEST = ROOT / "data" / "processed" / "globe_accent_manifest.csv"
DEFAULT_AUDIO_ROOT = ROOT / "data" / "processed" / "globe_audio"
DEFAULT_CACHE = ROOT / "data" / "processed" / "globe_accent_style_cache.npz"
DEFAULT_REPORT = ROOT / "results" / "accent" / "a3" / "style_cache_report.json"
TRAINABLE = ("us", "england", "australian", "canadian", "india_south_asia")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract GLOBE style vectors")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--audio-root", type=Path, default=DEFAULT_AUDIO_ROOT)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--classes", nargs="+", default=list(TRAINABLE))
    parser.add_argument("--include-irish", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--reuse-cache", action="store_true")
    parser.add_argument("--checkpoint-every", type=int, default=500)
    return parser.parse_args(argv)


def _audio_path(row: pd.Series, root: Path) -> Path:
    return root / str(row["speaker_split"]) / str(row["accent_class"]) / f"{row['clip_id']}.wav"


def _load_cache(path: Path) -> Dict[str, np.ndarray]:
    cache = np.load(path, allow_pickle=True)
    return {key: cache[key] for key in cache.files}


def _save_cache(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    classes = list(dict.fromkeys(str(item).lower() for item in args.classes))
    if args.include_irish and "irish" not in classes:
        classes.append("irish")
    splits = [str(item).lower() for item in args.splits]
    frame = pd.read_csv(args.manifest)
    frame = frame[
        frame["speaker_split"].astype(str).isin(splits)
        & frame["accent_class"].astype(str).isin(classes)
    ].copy()
    frame["audio_path"] = [_audio_path(row, args.audio_root) for _, row in frame.iterrows()]
    frame = frame[frame["audio_path"].map(lambda path: Path(path).is_file())]
    frame = frame.sort_values(["speaker_split", "accent_class", "client_id", "clip_id"])
    if args.limit is not None:
        frame = frame.head(int(args.limit))
    if frame.empty:
        raise RuntimeError("no GLOBE clips selected")
    print(
        f"Extracting {len(frame)} style vector(s) across {len(classes)} class(es) "
        f"and splits {splits}",
        flush=True,
    )

    cached: Dict[str, np.ndarray] = {}
    if args.reuse_cache and args.cache.is_file():
        cached = _load_cache(args.cache)
        print(f"Reusing {len(cached.get('clip_id', []))} cached style(s)", flush=True)

    done = {str(item) for item in cached.get("clip_id", [])}
    pending = frame[~frame["clip_id"].astype(str).isin(done)]
    clip_ids: List[str] = [str(item) for item in cached.get("clip_id", [])]
    styles: List[np.ndarray] = [np.asarray(item, dtype=np.float32) for item in cached.get("style", [])]
    speaker_ids: List[str] = [str(item) for item in cached.get("speaker_id", [])]
    accents: List[str] = [str(item) for item in cached.get("accent_class", [])]
    split_values: List[str] = [str(item) for item in cached.get("split", [])]

    if len(pending):
        import torch

        from voix.models.tts_backend import StyleTTS2Backend

        print(f"Loading StyleTTS2 on {args.device} ...", flush=True)
        backend = StyleTTS2Backend(device=args.device, use_neural=True, allow_fallback=False)
        backend.load()
        started = time.time()
        failures: List[Dict[str, str]] = []
        with torch.no_grad():
            for index, (_, row) in enumerate(pending.iterrows(), start=1):
                try:
                    style = backend.compute_style(str(row["audio_path"]))
                    styles.append(style.detach().cpu().numpy().reshape(-1).astype(np.float32))
                    clip_ids.append(str(row["clip_id"]))
                    speaker_ids.append(str(row["client_id"]))
                    accents.append(str(row["accent_class"]))
                    split_values.append(str(row["speaker_split"]))
                except Exception as exc:  # pragma: no cover - runtime guard
                    failures.append(
                        {"clip_id": str(row["clip_id"]), "error": f"{type(exc).__name__}: {exc}"}
                    )
                if args.checkpoint_every > 0 and index % args.checkpoint_every == 0:
                    _save_cache(
                        args.cache,
                        {
                            "clip_id": np.asarray(clip_ids, dtype=str),
                            "speaker_id": np.asarray(speaker_ids, dtype=str),
                            "accent_class": np.asarray(accents, dtype=str),
                            "split": np.asarray(split_values, dtype=str),
                            "style": np.asarray(styles, dtype=np.float32),
                        },
                    )
                if index % 100 == 0 or index == len(pending):
                    elapsed = time.time() - started
                    print(
                        f"  style {index}/{len(pending)} "
                        f"({index / max(elapsed, 1e-6):.1f}/s, {elapsed:.0f}s, "
                        f"failures={len(failures)})",
                        flush=True,
                    )
        _save_cache(
            args.cache,
            {
                "clip_id": np.asarray(clip_ids, dtype=str),
                "speaker_id": np.asarray(speaker_ids, dtype=str),
                "accent_class": np.asarray(accents, dtype=str),
                "split": np.asarray(split_values, dtype=str),
                "style": np.asarray(styles, dtype=np.float32),
            },
        )
    else:
        failures = []

    style_array = np.asarray(styles, dtype=np.float32)
    report = {
        "schema_version": "voix.globe_accent_style_cache.v1",
        "track": "accent",
        "level": "exploratory",
        "manifest": str(args.manifest),
        "manifest_sha256": _sha256(args.manifest),
        "audio_root": str(args.audio_root),
        "cache": str(args.cache),
        "cache_sha256": _sha256(args.cache),
        "classes": classes,
        "splits": splits,
        "clips": int(len(clip_ids)),
        "style_dim": int(style_array.shape[1]) if style_array.ndim == 2 else 0,
        "failures": failures,
        "counts": {
            split: {
                accent: int(
                    sum(
                        1
                        for s, a in zip(split_values, accents)
                        if s == split and a == accent
                    )
                )
                for accent in classes
            }
            for split in splits
        },
        "notes": [
            "Style vectors come from the frozen StyleTTS2 style/predictor encoder.",
            "This cache is for accent residuals only; it is not face identity data.",
            "Irish, if included, is evaluation-only.",
        ],
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {args.cache} ({len(clip_ids)} styles)")
    print(f"Wrote {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
