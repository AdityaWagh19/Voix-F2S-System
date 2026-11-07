"""Build a small, gender-balanced bank of real StyleTTS2 reference styles.

The style-geometry audit shows the face-derived adapter output sits off the
real-speech style manifold (face mean vs real mean cosine ~0.54).  This script
builds a reproducible real-style bank from AMI headset clips so we can test
manifold calibration without touching the canonical checkpoints.

Usage:
    python scripts/data/build_real_style_bank.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_MANIFEST = ROOT / "data" / "processed" / "ami_naturalness" / "manifest.csv"
DEFAULT_OUT = ROOT / "results" / "audit" / "style_calibration"
FEMALE_SPEAKERS = ("FEE005", "FIE081", "FEE055")
MALE_SPEAKERS = (
    "MEE006",
    "MEE007",
    "MEE008",
    "MIO016",
    "MIO050",
    "MIO082",
    "MEE009",
    "MEE010",
    "MEE011",
    "MEE012",
    "MEE053",
    "MEE054",
    "MEE056",
)


def _select(manifest: pd.DataFrame, speakers: Sequence[str], per_speaker: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for speaker in speakers:
        subset = manifest[
            (manifest["speaker_id"] == speaker)
            & (manifest["duration_sec"] >= 3.0)
            & (manifest["duration_sec"] <= 10.0)
        ].sort_values("duration_sec", ascending=False)
        for _, row in subset.head(per_speaker).iterrows():
            rows.append(
                {
                    "clip_id": str(row["clip_id"]),
                    "speaker_id": str(row["speaker_id"]),
                    "gender": "female" if speaker in FEMALE_SPEAKERS else "male",
                    "duration_sec": float(row["duration_sec"]),
                    "text": str(row["text"]),
                    "wav_path": str(row["wav_path"]),
                }
            )
    return rows


def _stats(styles: np.ndarray) -> Dict[str, Any]:
    norms = np.linalg.norm(styles, axis=1)
    unit = styles / np.maximum(norms[:, None], 1e-12)
    cosine = unit @ unit.T
    values = cosine[np.triu_indices_from(cosine, k=1)]
    return {
        "n": int(styles.shape[0]),
        "norm_mean": float(norms.mean()),
        "norm_std": float(norms.std()),
        "norm_min": float(norms.min()),
        "norm_max": float(norms.max()),
        "pairwise_cosine_mean": float(values.mean()),
        "pairwise_cosine_min": float(values.min()),
        "pairwise_cosine_max": float(values.max()),
        "mean_vector_norm": float(np.linalg.norm(styles.mean(axis=0))),
        "centered_rms": float(np.sqrt(np.mean((styles - styles.mean(axis=0)) ** 2))),
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--per-speaker", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(args.manifest)
    female = _select(manifest, FEMALE_SPEAKERS, per_speaker=4)
    male = _select(manifest, MALE_SPEAKERS, per_speaker=args.per_speaker)
    selected = female + male
    print(f"[select] {len(female)} female + {len(male)} male clips", flush=True)

    from voix.models.tts_backend import StyleTTS2Backend

    tts = StyleTTS2Backend(device=args.device, use_neural=True, allow_fallback=False)
    tts.load()
    if tts._model is None or tts._use_fallback:
        raise RuntimeError("StyleTTS2 failed to load")

    entries: List[Dict[str, Any]] = []
    for row in selected:
        path = Path(row["wav_path"])
        if not path.is_file():
            print(f"[skip] missing {path}", flush=True)
            continue
        print(f"[style] {row['clip_id']} ({row['gender']})", flush=True)
        style = tts.compute_style(str(path))
        vector = style.detach().cpu().numpy().reshape(-1).astype(np.float64)
        entries.append(
            {
                **row,
                "wav_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "norm": float(np.linalg.norm(vector)),
                "style_vector": vector.tolist(),
            }
        )

    if len(entries) < 6:
        raise RuntimeError(f"too few real styles: {len(entries)}")

    by_gender: Dict[str, Any] = {}
    for gender in ("female", "male"):
        subset = [entry for entry in entries if entry["gender"] == gender]
        if len(subset) < 3:
            raise RuntimeError(f"too few {gender} styles: {len(subset)}")
        array = np.asarray([entry["style_vector"] for entry in subset])
        mean = array.mean(axis=0)
        centered = array - mean
        # Regularized PCA basis for projection experiments.  The bank is small;
        # keep only components with meaningful variance.
        _, singular, vt = np.linalg.svd(centered, full_matrices=False)
        explained = singular**2 / max(float(np.sum(singular**2)), 1e-12)
        by_gender[gender] = {
            "stats": _stats(array),
            "mean_vector": mean.tolist(),
            "pca_components": vt[: min(8, vt.shape[0])].tolist(),
            "pca_explained_variance_ratio": explained.tolist(),
            "n": len(subset),
        }

    report = {
        "schema_version": "voix.real_style_bank.v1",
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "level": "exploratory",
        "source": "AMI headset speech (data/processed/ami_naturalness)",
        "manifest": str(args.manifest),
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "n_styles": len(entries),
        "by_gender": by_gender,
        "entries": entries,
        "notes": [
            "The bank is diagnostic infrastructure for style-manifold calibration; "
            "it is not a serving artifact and does not modify canonical checkpoints.",
            "AMI is used only for naturalness/style reference, not for identity training.",
        ],
    }
    output = args.output_dir / "real_style_bank.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({g: by_gender[g]["stats"] for g in by_gender}, indent=2))
    print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
