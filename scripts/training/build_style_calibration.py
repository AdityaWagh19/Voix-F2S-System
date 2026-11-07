"""Build the research-demo style calibration artifact.

Reads the face-voice diversity report (canonical face-derived styles) and the
gender-balanced real AMI style bank, fits the gender-matched mean translation,
and writes a versioned JSON artifact.  The demo generator loads this artifact;
the canonical checkpoints are never modified.

Usage:
    python scripts/training/build_style_calibration.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from voix.models.style_calibration import (  # noqa: E402
    GENDERS,
    fit_translation,
    save_calibration,
)

DEFAULT_FACE_REPORT = (
    ROOT / "results" / "audit" / "face_voice_diversity" / "face_voice_diversity.json"
)
DEFAULT_BANK = ROOT / "results" / "audit" / "style_calibration" / "real_style_bank.json"
DEFAULT_OUTPUT = ROOT / "results" / "audit" / "style_calibration" / "calibration.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--face-report", type=Path, default=DEFAULT_FACE_REPORT)
    parser.add_argument("--real-bank", type=Path, default=DEFAULT_BANK)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--alpha", type=float, default=1.0)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    face_report = json.loads(args.face_report.read_text(encoding="utf-8"))
    bank = json.loads(args.real_bank.read_text(encoding="utf-8"))

    face_by_gender: Dict[str, List[List[float]]] = {g: [] for g in GENDERS}
    for entry in face_report["entries"]:
        gender = "female" if int(entry["gender"]) == 0 else "male"
        vector = np.asarray(entry["style_vector"], dtype=np.float64).reshape(-1)
        if vector.shape != (256,):
            raise ValueError(
                f"face style {entry.get('label')!r} has shape {vector.shape}"
            )
        face_by_gender[gender].append(vector.tolist())

    real_by_gender: Dict[str, List[List[float]]] = {g: [] for g in GENDERS}
    for entry in bank["entries"]:
        gender = str(entry["gender"])
        vector = np.asarray(entry["style_vector"], dtype=np.float64).reshape(-1)
        if vector.shape != (256,):
            raise ValueError(
                f"real style {entry.get('clip_id')!r} has shape {vector.shape}"
            )
        real_by_gender[gender].append(vector.tolist())

    for gender in GENDERS:
        if len(face_by_gender[gender]) < 2:
            raise RuntimeError(f"too few face styles for {gender}")
        if len(real_by_gender[gender]) < 2:
            raise RuntimeError(f"too few real styles for {gender}")

    calibration = fit_translation(
        face_by_gender,
        real_by_gender,
        alpha=float(args.alpha),
        face_source={
            "report": str(args.face_report),
            "sha256": _sha256(args.face_report),
            "n": len(face_report["entries"]),
            "description": "canonical face -> CFVN -> CVAE -> StyleAdapter styles",
        },
        real_source={
            "bank": str(args.real_bank),
            "sha256": _sha256(args.real_bank),
            "n": len(bank["entries"]),
            "description": "real AMI headset styles computed by the frozen StyleTTS2 style encoder",
        },
        notes=[
            "Gender-matched mean translation: s' = s + alpha * (mean_real - mean_face).",
            "Within a gender, every pairwise difference is preserved exactly.",
            "This is an exploratory demo calibration, not a serving path and not a checkpoint change.",
            "Real styles come from AMI headset speech, not from the demo identity.",
        ],
    )
    save_calibration(args.output, calibration)

    summary = {
        gender: {
            "n_face": calibration["by_gender"][gender]["face"]["n"],
            "n_real": calibration["by_gender"][gender]["real"]["n"],
            "face_mean_norm": calibration["by_gender"][gender]["face_mean_norm"],
            "real_mean_norm": calibration["by_gender"][gender]["real_mean_norm"],
            "delta_norm": calibration["by_gender"][gender]["delta_norm"],
            "cos_face_mean_real_mean": calibration["by_gender"][gender][
                "cos_face_mean_real_mean"
            ],
            "delta_sha256": calibration["by_gender"][gender]["delta_sha256"],
        }
        for gender in GENDERS
    }
    print(json.dumps(summary, indent=2))
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
