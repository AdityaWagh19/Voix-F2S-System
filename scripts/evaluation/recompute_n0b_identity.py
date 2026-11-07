"""Recompute N0b generated-domain identity cosines from existing audio.

The first N0b report multiplied raw ECAPA embeddings, which are not unit
normalized, so the reported values were dot products rather than cosine
similarities.  This script reads the same generated WAVs, L2-normalizes the
embeddings, updates the identity block and the identity part of the N0b gate,
and leaves the acoustic/WER results untouched.

This is a correction script, not a new experiment.  It does not generate audio
and does not modify any checkpoint.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

DEFAULT_INDEX = ROOT / "results" / "naturalness" / "n0b" / "clips.csv"
DEFAULT_REPORT = ROOT / "results" / "naturalness" / "n0b" / "report.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64).reshape(-1)
    right = np.asarray(right, dtype=np.float64).reshape(-1)
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= 0.0:
        return 0.0
    return float(np.dot(left, right) / denominator)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if not args.index.is_file():
        raise FileNotFoundError(args.index)
    if not args.report.is_file():
        raise FileNotFoundError(args.report)
    frame = pd.read_csv(args.index)
    required = {"audio_id", "ami_clip_id", "variant", "alpha", "seed", "wav_path"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"index is missing columns: {missing}")

    from voix.data.speaker_extractor import ECAPAExtractor

    extractor = ECAPAExtractor(device=args.device)
    embeddings: Dict[str, np.ndarray] = {}
    for row in frame.itertuples():
        audio_id = str(row.audio_id)
        if audio_id in embeddings:
            continue
        path = Path(str(row.wav_path))
        if not path.is_file():
            raise FileNotFoundError(path)
        embeddings[audio_id] = (
            extractor.extract_from_path(str(path))
            .detach()
            .cpu()
            .numpy()
            .reshape(-1)
        )
    try:
        del extractor
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # pragma: no cover - environment-dependent
        pass

    identity: Dict[str, Any] = {}
    for variant in sorted(frame["variant"].unique()):
        subset = frame[frame["variant"] == variant]
        variant_block: Dict[str, Any] = {}
        alphas = sorted(
            float(value) for value in subset["alpha"].unique() if float(value) > 0
        )
        for alpha in alphas:
            values: List[float] = []
            for _, group in subset.groupby(["ami_clip_id", "seed"]):
                base = group[np.isclose(group["alpha"], 0.0)]
                changed = group[np.isclose(group["alpha"], alpha)]
                if len(base) and len(changed):
                    left = embeddings[str(base.iloc[0]["audio_id"])]
                    right = embeddings[str(changed.iloc[0]["audio_id"])]
                    values.append(_cosine(left, right))
            variant_block[f"alpha_{alpha:.2f}"] = {
                "n_pairs": len(values),
                "mean_cosine_to_alpha0": float(np.mean(values)) if values else None,
                "min_cosine_to_alpha0": float(np.min(values)) if values else None,
            }
        identity[variant] = variant_block

    report = json.loads(args.report.read_text(encoding="utf-8"))
    report.setdefault("metrics", {})["identity_generated_consistency"] = {
        "status": "computed",
        "value": identity,
    }

    gates = report.get("gates", {}).get("n0b", {})
    for variant, gate in gates.items():
        variant_identity = identity.get(variant, {})
        measured = any(block.get("n_pairs", 0) > 0 for block in variant_identity.values())
        identity_ok = measured
        if identity_ok:
            for block in variant_identity.values():
                value = block.get("mean_cosine_to_alpha0")
                if value is None or value < 0.95:
                    identity_ok = False
        gate["identity_measured"] = bool(measured)
        gate["identity_ok"] = bool(identity_ok)
        gate["passed"] = bool(
            gate.get("n_improved_metrics", 0) >= 2
            and identity_ok
            and gate.get("wer_ok", False)
        )

    report["identity_recompute"] = {
        "reason": (
            "The original N0b identity block used raw ECAPA dot products, not "
            "cosine similarity; values were recomputed from the same WAVs with "
            "L2-normalized embeddings."
        ),
        "index": str(args.index),
        "index_sha256": _sha256(args.index),
        "values": identity,
    }
    limitations = report.setdefault("limitations", [])
    note = (
        "Identity cosines were recomputed with L2-normalized ECAPA vectors; "
        "the original raw-dot values were not cosine similarities."
    )
    if note not in limitations:
        limitations.append(note)

    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(identity, indent=1))
    print(f"Updated {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
