"""Add frozen Wav2Vec2 features to the GLOBE accent cache.

Reads the ECAPA + prosody cache, extracts layer-8 mean-pooled Wav2Vec2
features for every clip, and writes one cache containing:

    base_features   (N, 222)  ECAPA + prosody
    ssl_features    (N, 768)  frozen Wav2Vec2
    features        (N, 990)  concatenation, in that order

A0 trains on ``features`` by default.  The ablation keys let the report show
whether SSL actually helps instead of assuming it does.

The extractor is frozen and research-only.  It is never used by serving.
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

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from voix.evaluation.accent_ssl import (  # noqa: E402
    DEFAULT_BUNDLE,
    DEFAULT_NUM_LAYERS,
    SSLAccentConfig,
    SSLAccentFeatureExtractor,
    ssl_feature_names,
)

DEFAULT_BASE = ROOT / "data" / "processed" / "globe_accent_features.npz"
DEFAULT_OUTPUT = ROOT / "data" / "processed" / "globe_accent_ssl_features.npz"
DEFAULT_REPORT = ROOT / "results" / "accent" / "a0" / "ssl_feature_report.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract frozen Wav2Vec2 accent features")
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--bundle", default=DEFAULT_BUNDLE)
    parser.add_argument("--num-layers", type=int, default=DEFAULT_NUM_LAYERS)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--checkpoint-every", type=int, default=500)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--reuse-cache", action="store_true")
    return parser.parse_args(argv)


def _write_cache(
    path: Path,
    *,
    base: Dict[str, np.ndarray],
    ssl: np.ndarray,
    config: SSLAccentConfig,
) -> None:
    base_features = np.asarray(base["features"], dtype=np.float32)
    base_names = [str(name) for name in base.get("feature_names", [])]
    if len(base_names) != base_features.shape[1]:
        base_names = [f"base_{index}" for index in range(base_features.shape[1])]
    ssl_names = ssl_feature_names(config)
    combined = np.concatenate([base_features, ssl], axis=1)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        clip_id=np.asarray(base["clip_id"], dtype=str),
        path=np.asarray(base["path"], dtype=str),
        label=np.asarray(base["label"], dtype=str),
        speaker_id=np.asarray(base["speaker_id"], dtype=str),
        split=np.asarray(base["split"], dtype=str),
        features=combined,
        feature_names=np.asarray(base_names + ssl_names, dtype=str),
        base_features=base_features,
        base_feature_names=np.asarray(base_names, dtype=str),
        ssl_features=ssl,
        ssl_feature_names=np.asarray(ssl_names, dtype=str),
        ssl_config=np.asarray(json.dumps(config.as_dict(), sort_keys=True)),
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if not args.base.is_file():
        raise FileNotFoundError(f"base accent cache not found: {args.base}")
    base = np.load(args.base, allow_pickle=True)
    base = {key: base[key] for key in base.files}
    clip_ids = [str(item) for item in base["clip_id"]]
    paths = [str(item) for item in base["path"]]
    if args.limit is not None:
        limit = max(0, int(args.limit))
        clip_ids = clip_ids[:limit]
        paths = paths[:limit]
        base["clip_id"] = np.asarray(clip_ids, dtype=str)
        base["path"] = np.asarray(paths, dtype=str)
        for key in ("label", "speaker_id", "split"):
            base[key] = np.asarray(base[key])[:limit]
        base["features"] = np.asarray(base["features"])[:limit]
    missing = [path for path in paths if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError(f"missing accent audio: {missing[:5]}")

    config = SSLAccentConfig(
        bundle=str(args.bundle),
        num_layers=int(args.num_layers),
    )
    dim = int(config.feature_dim)
    ssl = np.zeros((len(paths), dim), dtype=np.float32)
    done: Dict[str, np.ndarray] = {}
    if args.reuse_cache and args.output.is_file():
        previous = np.load(args.output, allow_pickle=True)
        previous_ids = [str(item) for item in previous["clip_id"]]
        previous_ssl = np.asarray(previous["ssl_features"], dtype=np.float32)
        done = {
            clip_id: previous_ssl[index]
            for index, clip_id in enumerate(previous_ids)
            if index < len(previous_ssl) and np.any(previous_ssl[index])
        }
        print(f"Reusing {len(done)} cached SSL feature row(s)", flush=True)

    pending = [
        (index, path)
        for index, (clip_id, path) in enumerate(zip(clip_ids, paths))
        if clip_id not in done
    ]
    failures: List[Dict[str, str]] = []
    if pending:
        print(
            f"Extracting Wav2Vec2 layer {config.num_layers} for "
            f"{len(pending)} clip(s) on {args.device} ...",
            flush=True,
        )
        extractor = SSLAccentFeatureExtractor(
            config, device=args.device, batch_size=args.batch_size
        )
        started = time.time()
        for offset in range(0, len(pending), max(1, args.batch_size)):
            batch = pending[offset : offset + max(1, args.batch_size)]
            try:
                features = extractor.extract_paths([path for _, path in batch])
            except Exception as exc:  # pragma: no cover - runtime guard
                for index, path in batch:
                    failures.append(
                        {
                            "clip_id": clip_ids[index],
                            "path": path,
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
                continue
            if features.shape[1] != dim:
                raise RuntimeError(
                    f"SSL feature dim {features.shape[1]} does not match "
                    f"configured dim {dim}"
                )
            for (index, _), row in zip(batch, features):
                ssl[index] = row
                done[clip_ids[index]] = row
            processed = offset + len(batch)
            if processed % 200 == 0 or processed == len(pending):
                elapsed = time.time() - started
                print(
                    f"  {processed}/{len(pending)} "
                    f"({processed / max(elapsed, 1e-6):.2f}/s, {elapsed:.0f}s, "
                    f"failures={len(failures)})",
                    flush=True,
                )
            if (
                args.checkpoint_every > 0
                and processed % int(args.checkpoint_every) == 0
            ):
                _write_cache(args.output, base=base, ssl=ssl, config=config)

    for index, clip_id in enumerate(clip_ids):
        if clip_id in done and not np.any(ssl[index]):
            ssl[index] = done[clip_id]

    _write_cache(args.output, base=base, ssl=ssl, config=config)
    report = {
        "schema_version": "voix.globe_accent_ssl_features.v1",
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "level": "exploratory",
        "track": "accent",
        "base_cache": str(args.base),
        "base_cache_sha256": _sha256(args.base),
        "output": str(args.output),
        "output_sha256": _sha256(args.output),
        "clips": int(len(clip_ids)),
        "speakers": int(len(set(str(item) for item in base["speaker_id"]))),
        "classes": sorted(set(str(item) for item in base["label"])),
        "splits": sorted(set(str(item) for item in base["split"])),
        "base_dim": int(np.asarray(base["features"]).shape[1]),
        "ssl_dim": dim,
        "combined_dim": int(np.asarray(base["features"]).shape[1] + dim),
        "ssl_config": config.as_dict(),
        "failures": failures,
        "notes": [
            "The Wav2Vec2 model is frozen; no fine-tuning was performed.",
            "Mean pooling over layer-8 hidden states, L2-normalized per clip.",
            "The combined cache is base features followed by SSL features.",
            "Irish remains evaluation-only and is not a training class.",
        ],
        "mos": {"status": "unavailable", "reason": "no human ratings collected"},
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {args.output} ({len(clip_ids)} rows, dim {report['combined_dim']})")
    print(f"Wrote {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
